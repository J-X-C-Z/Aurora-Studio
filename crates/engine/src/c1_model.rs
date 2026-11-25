//! Capture One's own adjustment model, decoded from the `ZVARIANTLAYER` table.
//! Data source: the real C1 catalog `Capture One.cocatalogdb`, table `ZVARIANTLAYER` (163 columns).
//! This decodes only the *input parameters* Capture One stores for a variant layer.
//! It is **not** Capture One's rendering operator: the maths that turns these values
//! into pixels still has to be re-derived (or measured) separately.

#[path = "c1_crop_limit.rs"]
pub(crate) mod crop_limit;

use std::collections::HashMap;

use crate::adjust::{
    AspectRatio, BlackWhite, Clarity, ClarityMethod, ColorBalance, ColorBalanceGains, ColorBand, ColorEditor,
    ColorRange, ColorWheel, Curve, CurvePoint, Curves, Dehaze, Exposure, FilmGrain, Geometry,
    GrainKind, HighDynamicRange, HslShift, HueSector, Keystone, Levels, LevelsChannel,
    LensCorrection, ManufacturerCurves, Moire, NoiseReduction, NormRect, RadialBlur, Sharpening,
    SkinUniformity, SplitTone, ToneCurvePreset, VignetteMethod, Vignetting, WhiteBalance,
    WhiteBalanceMode,
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
    C1Column { name: "ZGEOMETRYKEYSTONEFOCALLENGTH", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZGEOMETRYKEYSTONESKEW", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZFLIP", encoding: C1Encoding::Scalar, components: 1 },
    // ── lens / base (confidence >= 0.55) ─────────────────────────────────────
    C1Column { name: "ZLENSCHROMABRENABLED", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSIDGUID", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSREVISIONIDGUID", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSAPERTURE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSFOCALLENGTH", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZICCPROFILE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZFILMCURVE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZTECHNICALPROCESSINGMETHOD", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZSTYLEUUID", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZSTYLESOURCE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZSTYLEMODIFIED", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZAICOLORGRADE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLAYERSELECTPOINT", encoding: C1Encoding::Semicolon, components: 2 },
    // ── inferred columns this model still carries ───────────────────────────
    // White balance: R;G;B gains, G is always 1.
    C1Column { name: "ZWHITEBALANCE", encoding: C1Encoding::Semicolon, components: 3 },
    // Levels: four channels (rgb, red, green, blue) of black / white points.
    C1Column { name: "ZLEVELSSHADOW", encoding: C1Encoding::Semicolon, components: 4 },
    C1Column { name: "ZLEVELSHIGHLIGHT", encoding: C1Encoding::Semicolon, components: 4 },
    C1Column { name: "ZLEVELSTARGETENABLE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLEVELSTARGETSHADOW", encoding: C1Encoding::Semicolon, components: 4 },
    C1Column { name: "ZLEVELSTARGETHIGHLIGHT", encoding: C1Encoding::Semicolon, components: 4 },
    C1Column { name: "ZDEHAZECOLOR", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZCALCULATEDDEHAZECOLOR", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZDEHAZELEVELHIGH", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZDEHAZELEVELLOW", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZMOIRE", encoding: C1Encoding::Semicolon, components: 2 },
    C1Column { name: "ZCANVASSIZE", encoding: C1Encoding::Semicolon, components: 2 },
    // ZSPOTS / ZRELATIVESPOTS: `selected;type,x,y,radius;...`, raw pixels or
    // numerator/denominator fractions; see `c1_spots`.
    C1Column { name: "ZSPOTS", encoding: C1Encoding::SemicolonComma, components: 4 },
    C1Column { name: "ZRELATIVESPOTS", encoding: C1Encoding::SemicolonComma, components: 4 },
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
    C1Column { name: "ZLENSLCCLIGHTFALLOFF", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSLCCLIGHTFALLOFFAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSDUSTREMOVALENABLED", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSLCC", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSLCCBITMAPHASH", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSRADIALBLUR", encoding: C1Encoding::Pipe, components: 4 },
    C1Column { name: "ZLENSTILT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSDIFFRACTIONCORRECTIONENABLED", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSDEFAULTCHROMABR", encoding: C1Encoding::Pipe, components: 2 },
    C1Column { name: "ZLENSCUSTOMCHROMABR", encoding: C1Encoding::Pipe, components: 2 },
    C1Column { name: "ZLENSCUSTOMCHROMABRNAME", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSCHROMABRORIGIN", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSCHROMAFRINGING", encoding: C1Encoding::Pipe, components: 6 },
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

    /// A spline from `|`-field `which`, normalized to its outer radius.
    ///
    /// Capture One's CA tables start at radius zero, so the first point cannot
    /// be the normalizer (it would divide by zero and drop the whole table).
    /// The largest radius is the table's measured reference; x and y share it,
    /// which leaves y as a fraction of the radius.
    pub fn scale(&self, which: usize) -> Vec<(f32, f32)> {
        let Some(points) = self.splines.get(which) else { return Vec::new() };
        let radius = points
            .iter()
            .map(|p| p.0)
            .filter(|x| x.is_finite())
            .fold(0.0f32, f32::max);
        if radius <= 0.0 {
            return Vec::new();
        }
        points.iter().map(|&(x, y)| (x / radius, y / radius)).collect()
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

/// Capture One's `IC_GradationCurve` holds 16 interior points plus a start and
/// an end point; `P1.C1.ViewModels` refuses arrays of 18 or more. A stored value
/// can therefore never exceed this, so extra pairs are a corrupt file.
pub const MAX_CURVE_POINTS: usize = 18;

/// Parses `in,out;in,out` control points. Pairs that are not two numbers are
/// skipped, and at most [`MAX_CURVE_POINTS`] pairs are kept. C1 sorts the points
/// on load but stores them in editor order, which is not always ascending.
pub fn parse_curve(raw: &str) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for pair in clean(raw).split(';').take(MAX_CURVE_POINTS) {
        let mut it = pair.split(',');
        let (Some(x), Some(y)) = (it.next().and_then(parse_scalar), it.next().and_then(parse_scalar))
        else {
            continue;
        };
        out.push((x, y));
    }
    out
}

/// Wire `ZLENSDISTORTION` into the runtime lens calibration.
///
/// `pitch` is `IC_SensorGeometry.rawPixelWidth/rawPixelHeight` in micrometres
/// per pixel, so the curve lands in source pixels — the frame `FrameMap` maps
/// through. The four catalog disable paths leave `distortion` empty, which
/// keeps the geometry stage on its identity radial path. This calibration is
/// runtime input, not document state: it never enters `AdjustmentSet`,
/// equality or a saved `.portrait`.
pub(crate) fn apply_lens_distortion(
    runtime: &mut crate::document::LensRuntime,
    lens: &C1Lens,
    pitch: [f32; 2],
) {
    runtime.distortion = crop_limit::RadialCurve::from_row(
        lens.distortion.points(),
        lens.distortion.param(0).unwrap_or(0.0),
        lens.distortion.param(1).unwrap_or(0.0) != 0.0,
        pitch,
    );
    runtime.optical_center = lens.optic_center;
}

/// The runtime lens calibration implied by a catalog row.
///
/// Only the optical centre is known before the sensor pitch is attached; the
/// radial curve stays empty, exactly as `to_adjustment_set` left it. Capture One
/// callers use this to seed `document::RuntimeInputs` instead of writing these
/// values back into the document.
pub(crate) fn lens_runtime_from_settings(
    s: &C1VariantSettings,
) -> crate::document::LensRuntime {
    crate::document::LensRuntime {
        distortion: crop_limit::RadialCurve::default(),
        optical_center: s.lens.optic_center,
        uniform_light_grid: None,
    }
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

/// Parses `ZLENSRADIALBLUR`: `radialCurve|transverseCurve|amount|valid`.
pub fn parse_radial_blur(raw: &str) -> (RadialBlur, f32) {
    let mut fields = clean(raw).split('|');
    let radial = fields.next().map(parse_curve).unwrap_or_default();
    let transverse = fields.next().map(parse_curve).unwrap_or_default();
    let amount = fields.next().and_then(parse_scalar).unwrap_or(0.0);
    let valid = fields.next().and_then(parse_scalar).is_some_and(|v| v != 0.0);
    (RadialBlur { radial, transverse, valid }, amount)
}

/// Parses `ZLENSCHROMAFRINGING`: six `x,y;…` curves ordered red shift, red
/// radius, red strength, blue shift, blue radius, blue strength. Any other
/// field count reads as empty, matching C1's own parser.
pub fn parse_chroma_fringing(raw: &str) -> ChromaFringing {
    let fields: Vec<&str> = clean(raw).split('|').collect();
    if fields.len() != 6 {
        return ChromaFringing::default();
    }
    ChromaFringing {
        red_shift: parse_curve(fields[0]),
        red_ra: parse_curve(fields[1]),
        red_strength: parse_curve(fields[2]),
        blue_shift: parse_curve(fields[3]),
        blue_ra: parse_curve(fields[4]),
        blue_strength: parse_curve(fields[5]),
    }
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
    /// Whether the group actually shifts colour; its band geometry alone is
    /// part of C1's editor state, not an edit.
    pub fn affects_color(&self) -> bool {
        self.hue != 0.0
            || self.saturation != 0.0
            || self.lightness != 0.0
            || self.homogeneity != [0.0; 3]
    }

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

/// `x;y` for `ZLAYERSELECTPOINT`, in C1's raw drawing pixels.
fn parse_point(raw: &str) -> Option<[f32; 2]> {
    let v = components(Some(raw), 2, f32::NAN);
    (v[0].is_finite() && v[1].is_finite()).then_some([v[0], v[1]])
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
    /// `ZTECHNICALPROCESSINGMETHOD`: C1 Base Characteristics mode.
    pub technical_processing_method: i32,
    /// `settings+0x11008`: C1's negative-pipeline switch. There is no `.cop`
    /// or catalog column for it, so it is derived from the mode.
    pub negative_switch: bool,

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
    /// `ZLEVELSTARGETENABLE`: the model/UI gate for the target-levels stage.
    /// It is not an `IC_ImageProcessSettings` field; disabling it leaves the
    /// target values stored but makes the stage identity.
    pub levels_target_enable: bool,

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
    /// `ZDEHAZECOLOR`; `0;0;0` means Auto (the colour is not black).
    pub dehaze_color: [f32; 3],
    /// `ZCALCULATEDDEHAZECOLOR`, written by C1's auto job; read-only for us.
    pub dehaze_calculated_color: [f32; 3],
    /// `ZDEHAZELEVELLOW`, auto level black points.
    pub dehaze_level_low: [f32; 3],
    /// `ZDEHAZELEVELHIGH`, auto level white points.
    pub dehaze_level_high: [f32; 3],

    pub black_white: C1BlackWhite,
    /// Master, shadow, midtone and highlight colour balance, three components each.
    pub color_balance: [[f32; 3]; 4],
    /// The nine `ZCOLORCORRECTIONS` groups.
    pub color_corrections: Vec<[f32; 18]>,
    /// Gradation curves: rgb, red, green, blue, yellow.
    pub gradation_curves: [Vec<(f32, f32)>; 5],

    /// Stored crop in pixels: raw center X/Y, canvas width/height.
    pub crop_px: [f32; 4],
    pub crop_outside_image: bool,
    pub crop_aspect_ratio: Option<String>,
    pub rotation: f32,
    pub keystone_amount: f32,
    pub keystone_aspect: f32,
    pub keystone_tilt_x: f32,
    pub keystone_tilt_y: f32,
    /// `focal_mm / sensor_diagonal_mm`; `0` when unknown.
    pub keystone_focal_length: f32,
    /// Degrees, -45 ..= 45.
    pub keystone_skew: f32,
    /// `ZFLIP`: 0 none, 1 horizontal, 2 vertical.
    pub flip: i32,
    pub geometry_canvas_clipping: i32,
    /// Canvas size in pixels, `w;h`; `0` when C1 did not store it on the layer.
    pub canvas_size: [f32; 2],

    pub lens: C1Lens,
    pub vignetting: [f32; 4],
    pub icc_profile: Option<String>,
    pub film_curve: Option<String>,
    /// `ZSTYLEUUID` / `ZSTYLESOURCE` / `ZSTYLEMODIFIED`: the applied style, if any.
    pub style_uuid: Option<String>,
    pub style_source: Option<String>,
    pub style_modified: bool,
    /// `ZAICOLORGRADE`: this layer is a Match Look / AI Color Grade layer.
    pub ai_color_grade: bool,
    /// `ZLAYERSELECTPOINT`: `x;y` in raw drawing pixels, not normalised.
    pub layer_select_point: Option<[f32; 2]>,

    pub brush: C1Brush,
    pub eraser: C1Eraser,
    pub surface_smoothness: f32,
    /// `ZSPOTS`, raw image pixels. The catalog reader maps it to
    /// [`crate::AdjustmentSet::spots`] once the source size and the final
    /// orientation are known.
    pub spots: crate::c1_spots::SpotList,
    /// `ZRELATIVESPOTS`, the area-fraction copy C1 writes for pasting.
    pub relative_spots: crate::c1_spots::SpotList,
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
    /// `ZCNRAMOUNT`, the Color slider (CNR).
    pub cnr: f32,
}

impl Default for C1NoiseReduction {
    fn default() -> Self {
        // C1's BASEDEFAULT `CNRAmount` is 50 like the other three amounts.
        Self { luminance: 50.0, details: 50.0, color: 50.0, method: 0, long_exposure: 0.0, cnr: 50.0 }
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

/// `ZLENSCHROMABRORIGIN`, C1's `ChromaticAberrationOrigin`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChromaAberOrigin {
    /// 0: no active source; the stored curves are not applied.
    #[default]
    None,
    /// 1: the lens database supplied the curves in `ZLENSDEFAULTCHROMABR`.
    LensDb,
    /// 2: `ZLENSCUSTOMCHROMABR` is the active curve.
    Custom,
}

impl ChromaAberOrigin {
    fn from_stored(raw: Option<&str>) -> Self {
        match raw.and_then(parse_scalar) {
            Some(v) if v == 1.0 => Self::LensDb,
            Some(v) if v == 2.0 => Self::Custom,
            _ => Self::None,
        }
    }
}

/// `ZLENSCHROMAFRINGING`: the lens profile's six lateral-fringing curves as
/// `x,y;x,y` control points, one `|`-separated field each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChromaFringing {
    pub red_shift: Vec<(f32, f32)>,
    pub red_ra: Vec<(f32, f32)>,
    pub red_strength: Vec<(f32, f32)>,
    pub blue_shift: Vec<(f32, f32)>,
    pub blue_ra: Vec<(f32, f32)>,
    pub blue_strength: Vec<(f32, f32)>,
}

impl ChromaFringing {
    /// Whether none of the six curves carries a point.
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

/// C1's lens calibration block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct C1Lens {
    /// `ZLENSDISTORTION`: radial spline plus amount / flag / amount / centre.
    pub distortion: SplineProfile,
    /// `ZLENSOPTICCENTER`: optical centre in source pixels.
    pub optic_center: [f32; 2],
    /// `ZLENSLIGHTFALLOFF`: radial spline plus amount / flag.
    pub light_falloff: SplineProfile,
    /// `ZLENSRADIALBLUR` fields 1 and 2 plus its `valid` flag.
    pub radial_blur: RadialBlur,
    /// `ZLENSRADIALBLUR`'s amount (field 3), 0 ..= 120.
    pub sharpness_falloff: f32,
    /// `ZLENSLCCLIGHTFALLOFF`: Lens Calibration Uniform Light gate.
    pub lcc_light_falloff: bool,
    /// `ZLENSLCCLIGHTFALLOFFAMOUNT`: Uniform Light amount, 0 ..= 120.
    pub lcc_light_falloff_amount: f32,
    /// `ZLENSDUSTREMOVALENABLED`: apply the dust patches stored in the LCC.
    pub dust_removal: bool,
    /// `ZLENSLCC`: the sidecar id of the lens cast calibration, used by
    /// [`crate::dust_lcc::sidecar_path`] to find `<raw>.<id>.lcc`.
    pub lcc_id: Option<String>,
    /// `ZLENSTILT`, degrees.
    pub tilt: f32,
    /// `ZLENSDEFAULTCHROMABR`: two radial splines (red and blue).
    pub chromatic_aberration: SplineProfile,
    pub chromatic_aberration_enabled: bool,
    /// `ZLENSCHROMABRORIGIN`: which of the two curves above is active.
    pub chromatic_aberration_origin: ChromaAberOrigin,
    /// `ZLENSCUSTOMCHROMABR`: user-drawn red | blue curves.
    pub chromatic_aberration_custom: SplineProfile,
    /// `ZLENSCUSTOMCHROMABRNAME`: label of the custom curve.
    pub chromatic_aberration_custom_name: Option<String>,
    /// `ZLENSCHROMAFRINGING`: lateral-fringing model from the lens profile.
    pub chroma_fringing: ChromaFringing,
    pub profile: Option<String>,
    pub revision_profile: Option<String>,
    pub aperture: f32,
    /// `ZLENSDIFFRACTIONCORRECTIONENABLED`: aperture-dependent Airy blur.
    pub diffraction_enabled: bool,
    pub focal_length: f32,
    /// Manual optical centre shift, `x;y` in percent.
    pub shift_xy: [f32; 2],
    pub purple_fringing: f32,
    /// Set only when `profile` names Capture One's built-in Manufacturer
    /// Profile; the curves above are then the profile's real calibration.
    pub manufacturer: Option<ManufacturerCurves>,
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
            technical_processing_method: 0,
            negative_switch: false,

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
            levels_target_enable: true,

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
            dehaze_calculated_color: [0.0, 0.0, 0.0],
            dehaze_level_low: [0.0, 0.0, 0.0],
            dehaze_level_high: [1.0, 1.0, 1.0],

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
            keystone_focal_length: 0.0,
            keystone_skew: 0.0,
            flip: 0,
            geometry_canvas_clipping: 2,
            canvas_size: [0.0, 0.0],

            lens: C1Lens {
                distortion: SplineProfile::default(),
                optic_center: [0.0, 0.0],
                light_falloff: SplineProfile::default(),
                radial_blur: RadialBlur::default(),
                sharpness_falloff: 0.0,
                lcc_light_falloff: false,
                lcc_light_falloff_amount: 0.0,
                dust_removal: false,
                lcc_id: None,
                tilt: 0.0,
                chromatic_aberration: SplineProfile::default(),
                chromatic_aberration_enabled: false,
                chromatic_aberration_origin: ChromaAberOrigin::None,
                chromatic_aberration_custom: SplineProfile::default(),
                chromatic_aberration_custom_name: None,
                chroma_fringing: ChromaFringing::default(),
                profile: None,
                revision_profile: None,
                // 0 = not recorded; only the diffraction stage consumes it.
                aperture: 0.0,
                diffraction_enabled: false,
                focal_length: 120.0,
                shift_xy: [0.0, 0.0],
                purple_fringing: 0.0,
                manufacturer: None,
            },
            vignetting: [0.0, 1.0, 0.0, 1.0],
            icc_profile: None,
            film_curve: None,
            style_uuid: None,
            style_source: None,
            style_modified: false,
            ai_color_grade: false,
            layer_select_point: None,

            brush: C1Brush::default(),
            eraser: C1Eraser::default(),
            surface_smoothness: 0.0,
            spots: crate::c1_spots::SpotList::default(),
            relative_spots: crate::c1_spots::SpotList::default(),
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
    s.technical_processing_method = get("ZTECHNICALPROCESSINGMETHOD")
        .and_then(parse_scalar)
        .map(|v| v as i32)
        .unwrap_or(s.technical_processing_method);
    // No column stores C1's `settings+0x11008` switch; Capture One derives it
    // from the mode and the binary only ever clears the byte (`0x1855D06`), so
    // mirror the derivation instead of inventing a column.
    s.negative_switch = crate::TechnicalProcessingMethod::from_code(s.technical_processing_method)
        .is_film_negative();

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
    s.levels_target_enable = flag(get("ZLEVELSTARGETENABLE"), s.levels_target_enable);

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
    let calculated = components(get("ZCALCULATEDDEHAZECOLOR"), 3, 0.0);
    s.dehaze_calculated_color = [calculated[0], calculated[1], calculated[2]];
    let level_high = components(get("ZDEHAZELEVELHIGH"), 3, 1.0);
    s.dehaze_level_high = [level_high[0], level_high[1], level_high[2]];
    let level_low = components(get("ZDEHAZELEVELLOW"), 3, 0.0);
    s.dehaze_level_low = [level_low[0], level_low[1], level_low[2]];

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
    s.keystone_focal_length =
        fixed(get("ZGEOMETRYKEYSTONEFOCALLENGTH"), s.keystone_focal_length);
    s.keystone_skew = fixed(get("ZGEOMETRYKEYSTONESKEW"), s.keystone_skew);
    s.flip = fixed(get("ZFLIP"), s.flip as f32) as i32;
    s.geometry_canvas_clipping =
        fixed(get("ZGEOMETRYCANVASCLIPPING"), s.geometry_canvas_clipping as f32) as i32;
    let canvas = components(get("ZCANVASSIZE"), 2, 0.0);
    s.canvas_size = [canvas[0], canvas[1]];

    s.lens.distortion = get("ZLENSDISTORTION").map(parse_spline).unwrap_or_default();
    if let Some(raw) = get("ZLENSOPTICCENTER") {
        // `ZLENSOPTICCENTER` is pipe-separated: `x|y`.
        let v: Vec<f32> = clean(raw).split('|').filter_map(parse_scalar).collect();
        if v.len() >= 2 && v[0] > 0.0 && v[1] > 0.0 {
            s.lens.optic_center = [v[0], v[1]];
        }
    }
    s.lens.light_falloff = get("ZLENSLIGHTFALLOFF").map(parse_spline).unwrap_or_default();
    s.lens.lcc_light_falloff = flag(get("ZLENSLCCLIGHTFALLOFF"), s.lens.lcc_light_falloff);
    s.lens.lcc_light_falloff_amount =
        fixed(get("ZLENSLCCLIGHTFALLOFFAMOUNT"), s.lens.lcc_light_falloff_amount);
    s.lens.dust_removal = flag(get("ZLENSDUSTREMOVALENABLED"), s.lens.dust_removal);
    s.lens.lcc_id = text(get("ZLENSLCC"));
    if let Some(raw) = get("ZLENSRADIALBLUR") {
        let (blur, amount) = parse_radial_blur(raw);
        s.lens.radial_blur = blur;
        s.lens.sharpness_falloff = amount;
    }
    s.lens.tilt = fixed(get("ZLENSTILT"), s.lens.tilt);
    s.lens.chromatic_aberration = get("ZLENSDEFAULTCHROMABR").map(parse_spline).unwrap_or_default();
    s.lens.chromatic_aberration_enabled =
        flag(get("ZLENSCHROMABRENABLED"), s.lens.chromatic_aberration_enabled);
    s.lens.chromatic_aberration_origin =
        ChromaAberOrigin::from_stored(get("ZLENSCHROMABRORIGIN"));
    s.lens.chromatic_aberration_custom =
        get("ZLENSCUSTOMCHROMABR").map(parse_spline).unwrap_or_default();
    s.lens.chromatic_aberration_custom_name = text(get("ZLENSCUSTOMCHROMABRNAME"));
    s.lens.chroma_fringing = get("ZLENSCHROMAFRINGING")
        .map(parse_chroma_fringing)
        .unwrap_or_default();
    s.lens.profile = text(get("ZLENSIDGUID"));
    s.lens.revision_profile = text(get("ZLENSREVISIONIDGUID"));
    s.lens.aperture = fixed(get("ZLENSAPERTURE"), s.lens.aperture);
    s.lens.diffraction_enabled = flag(
        get("ZLENSDIFFRACTIONCORRECTIONENABLED"),
        s.lens.diffraction_enabled,
    );
    s.lens.focal_length = fixed(get("ZLENSFOCALLENGTH"), s.lens.focal_length);
    let shift = components(get("ZLENSSHIFTXY"), 2, 0.0);
    s.lens.shift_xy = [shift[0], shift[1]];
    s.lens.purple_fringing = fixed(get("ZLENSPURPLEFRINGINGAMOUNT"), s.lens.purple_fringing);
    if matches!(
        s.lens.profile.as_deref().and_then(crate::lensdb::builtin_profile),
        Some(crate::lensdb::BuiltinProfile::Manufacturer)
    ) {
        s.lens.manufacturer = Some(ManufacturerCurves {
            distortion: s.lens.distortion.points().to_vec(),
            ca_red: s.lens.chromatic_aberration.splines.first().cloned().unwrap_or_default(),
            ca_blue: s.lens.chromatic_aberration.splines.get(1).cloned().unwrap_or_default(),
        });
    }

    let vig = components(get("ZLENSVIGNETTING"), 4, 0.0);
    s.vignetting = [vig[0], vig[1], vig[2], vig[3]];

    s.icc_profile = text(get("ZICCPROFILE"));
    s.film_curve = text(get("ZFILMCURVE"));
    s.style_uuid = text(get("ZSTYLEUUID"));
    s.style_source = text(get("ZSTYLESOURCE"));
    s.style_modified = flag(get("ZSTYLEMODIFIED"), s.style_modified);
    s.ai_color_grade = flag(get("ZAICOLORGRADE"), s.ai_color_grade);
    s.layer_select_point = get("ZLAYERSELECTPOINT").and_then(parse_point);

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
    s.spots = get("ZSPOTS").and_then(crate::c1_spots::parse).unwrap_or_default();
    s.relative_spots =
        get("ZRELATIVESPOTS").and_then(crate::c1_spots::parse_relative).unwrap_or_default();
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// White balance conversion
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::c1_wb_oracle::{
    gains_to_kelvin_tint, gains_to_kelvin_tint_for_body, kelvin_tint_to_gains,
    WbDisplayConfidence,
};

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
/// `local` separates the global group from slices; `kind` keeps advanced, skin
/// and max-smooth groups out of the eight basic slots.
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
            // C1's ColorCorrection has no value bounds (18 getters checked).
            value_low: 0.0,
            value_high: 1.0,
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
            } else if cc.kind == 3 {
                // C1's `pie_maxSmooth` runs the basic `pie_smooth` window; keep
                // it out of the eight-slot basic array so it cannot evict one.
                editor.max_smooth.push(range);
            } else {
                editor.advanced.push(range);
            }
            editor.advanced_bands.push((id, band));
        }
    }
    editor
}

/// The five `ZLEVELS*` columns in engine form.
pub fn levels_from_c1(s: &C1VariantSettings) -> Levels {
    // `ApplyTargetLevelsCommon` is one RGB-master affine map and reads only
    // the `.x` scalar of targetShadow/targetHighlight. The r/g/b components
    // are persisted by C1 but have no consuming op in this build.
    // `ZLEVELSTARGETENABLE` is the model/UI gate: disabled means identity.
    let (output_black, output_white) = if s.levels_target_enable {
        (
            clamp(s.levels_target_shadow[0], 0.0, 1.0),
            clamp(s.levels_target_highlight[0], 0.0, 1.0),
        )
    } else {
        (0.0, 1.0)
    };
    let channel = |i: usize| LevelsChannel {
        input_black: clamp(s.levels_shadow[i], 0.0, 1.0),
        input_white: clamp(s.levels_highlight[i], 0.0, 1.0),
        gamma: gamma_from_c1_midtone(s.levels_midtone[i]),
        output_black: 0.0,
        output_white: 1.0,
    };
    let mut rgb = channel(0);
    rgb.output_black = output_black;
    rgb.output_white = output_white;
    Levels { rgb, red: channel(1), green: channel(2), blue: channel(3) }
}

/// The five `ZGRADATIONCURVE*` columns in engine form.
pub fn curves_from_c1(s: &C1VariantSettings) -> Curves {
    Curves {
        rgb: curve_from(&s.gradation_curves[0]),
        red: curve_from(&s.gradation_curves[1]),
        green: curve_from(&s.gradation_curves[2]),
        blue: curve_from(&s.gradation_curves[3]),
        // `ZGRADATIONCURVEY` is C1's luminance curve: the same gradation curve
        // class, selected by `GradationCurveY` when the editor is on luma.
        luma: curve_from(&s.gradation_curves[4]),
    }
}

/// The `ZBW*` columns in engine form. Global variants and local mask rows share
/// them, so they share this mapping.
pub fn black_white_from_c1(s: &C1VariantSettings) -> BlackWhite {
    BlackWhite {
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
    }
}

/// `DehazeOperations::CalculateInitialAutoHazeColor`: `0.98` where the white
/// balance gain is below 1, `0.98 / gain` above it.
fn c1_auto_haze_color(wb: [f32; 3]) -> [f32; 3] {
    wb.map(|g| 0.98 / g.max(1.0))
}

/// The live red/blue CA splines: the database copy for `FromLensDB`, the user
/// copy for `Custom`, nothing for `None`.
fn active_ca_splines(lens: &C1Lens) -> (Vec<(f32, f32)>, Vec<(f32, f32)>) {
    let profile = match lens.chromatic_aberration_origin {
        ChromaAberOrigin::LensDb => &lens.chromatic_aberration,
        ChromaAberOrigin::Custom => &lens.chromatic_aberration_custom,
        ChromaAberOrigin::None => return (Vec::new(), Vec::new()),
    };
    // `ZLENSDEFAULTCHROMABR` is `red|blue`, the same order as
    // `IC_ChromAberCurve.RedShift/BlueShift`.
    (profile.scale(0), profile.scale(1))
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
        gains: Some(s.white_balance_gains),
    };

    set.hdr = HighDynamicRange {
        highlights: clamp(s.highlights_recovery, -100.0, 100.0),
        shadows: clamp(s.shadows_recovery, -100.0, 100.0),
        whites: clamp(s.whites_recovery, -100.0, 100.0),
        blacks: clamp(s.blacks_recovery, -100.0, 100.0),
    };

    set.levels = levels_from_c1(s);
    set.curves = curves_from_c1(s);

    set.clarity = Clarity {
        method: ClarityMethod::from_c1(s.clarity_method),
        clarity: clamp(s.clarity, -100.0, 100.0),
        structure: clamp(s.clarity_structure, -100.0, 100.0),
    };

    // `ZDEHAZECOLOR = 0;0;0` is C1's Auto sentinel, not black haze. Use the
    // auto-calculated colour when C1 wrote one, else its `0.98 / WB` fallback;
    // the closed-form auto estimator is not known.
    let dehaze_color = if s.dehaze_color == [0.0; 3] {
        if s.dehaze_calculated_color != [0.0; 3] {
            s.dehaze_calculated_color
        } else {
            [0.0; 3]
        }
    } else {
        s.dehaze_color
    };
    let dehaze_color = if dehaze_color == [0.0; 3] {
        c1_auto_haze_color(s.white_balance_gains)
    } else {
        dehaze_color
    };
    set.dehaze = Dehaze {
        amount: clamp(s.dehaze_amount, 0.0, 100.0),
        shadow_tone: dehaze_color,
        level_low: s.dehaze_level_low.map(|v| v.clamp(0.0, 1.0)),
        level_high: s.dehaze_level_high.map(|v| v.clamp(0.0, 1.0)),
    };

    set.vignetting = Vignetting {
        amount: clamp(s.vignetting[0], -4.0, 4.0),
        method: VignetteMethod::EllipticOnCrop,
        ..Vignetting::default()
    };

    // C1's three numbers per zone are RGB multipliers: 1;1;1 is neutral and the
    // values follow the preset names (Warm lowers blue, Cool lowers red).
    let zone = |v: [f32; 3]| {
        v.map(|x| if x.is_finite() { x.clamp(0.0, 8.0) } else { 1.0 })
    };
    let gains = ColorBalanceGains {
        master: zone(s.color_balance[0]),
        shadow: zone(s.color_balance[1]),
        midtone: zone(s.color_balance[2]),
        highlight: zone(s.color_balance[3]),
    };
    set.color_balance_gains = (!gains.is_neutral()).then_some(gains);

    set.color_editor = color_editor_from_c1(&s.color_corrections);

    set.black_white = black_white_from_c1(s);

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
        // The Color slider is CNR (`ZCNRAMOUNT`). `ZAIDENOISEAMOUNT` is the
        // separate AI denoise amount and has no renderer without its ONNX model.
        color: crate::param_meta::clamp("ColorNoiseReductionAmount", s.noise_reduction.cnr),
        single_pixel: crate::param_meta::clamp(
            "CleanNoiseAmount",
            s.noise_reduction.long_exposure,
        ),
    };

    let grain_density = crate::param_meta::clamp("FilmGrainDensity", s.film_grain.density);
    set.film_grain = FilmGrain {
        kind: GrainKind::from_c1_type(s.film_grain.kind),
        impact: clamp(s.film_grain.amount, 0.0, 100.0),
        granularity: clamp(s.film_grain.granularity, 0.0, 100.0),
        density: grain_density,
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
        sharpness_falloff: clamp(s.lens.sharpness_falloff, 0.0, 120.0),
        radial_blur: s.lens.radial_blur.clone(),
        light_falloff: clamp(s.lens.light_falloff.param(0).unwrap_or(0.0), 0.0, 100.0),
        uniform_light: clamp(s.lens.lcc_light_falloff_amount, 0.0, 120.0),
        uniform_light_enabled: s.lens.lcc_light_falloff,
        dust_removal_enabled: s.lens.dust_removal,
        // The LCC file itself carries the patches; without a catalog-side
        // attachment this stays empty and the stage is inert.
        dust_patches: Vec::new(),
        dust_gain: Vec::new(),
        chromatic_aberration: s.lens.chromatic_aberration_enabled,
        // A manufacturer profile's CA lives in `manufacturer`; the raw splines
        // are not normalized to a usable radius, so they must not override it.
        // The origin decides which pair is live: the lens database copy for
        // `FromLensDB`, the user's `ZLENSCUSTOMCHROMABR` for `Custom`, nothing
        // for `None` (C1 keeps the stored curves but does not apply them).
        ca_red_points: if s.lens.manufacturer.is_some() {
            Vec::new()
        } else {
            active_ca_splines(&s.lens).0
        },
        ca_blue_points: if s.lens.manufacturer.is_some() {
            Vec::new()
        } else {
            active_ca_splines(&s.lens).1
        },
        ca_origin: match s.lens.chromatic_aberration_origin {
            ChromaAberOrigin::None => crate::CaOrigin::None,
            ChromaAberOrigin::LensDb => crate::CaOrigin::LensDb,
            ChromaAberOrigin::Custom => crate::CaOrigin::Custom,
        },
        ca_fringing: crate::CaFringing {
            red_shift: s.lens.chroma_fringing.red_shift.clone(),
            red_ra: s.lens.chroma_fringing.red_ra.clone(),
            red_strength: s.lens.chroma_fringing.red_strength.clone(),
            blue_shift: s.lens.chroma_fringing.blue_shift.clone(),
            blue_ra: s.lens.chroma_fringing.blue_ra.clone(),
            blue_strength: s.lens.chroma_fringing.blue_strength.clone(),
        },
        purple_fringing: clamp(s.lens.purple_fringing, -100.0, 100.0),
        diffraction_enabled: s.lens.diffraction_enabled,
        aperture: s.lens.aperture.max(0.0),
        hide_distorted_areas: true,
        optic_center: (s.lens.optic_center != [0.0, 0.0]).then_some(s.lens.optic_center),
        shift_x: clamp(s.lens.shift_xy[0], -100.0, 100.0),
        shift_y: clamp(s.lens.shift_xy[1], -100.0, 100.0),
        tilt: clamp(s.lens.tilt, -45.0, 45.0),
        manufacturer: s.lens.manufacturer.clone(),
        mount: None,
    };

    let flip_horizontal = s.flip == 1;
    let flip_vertical = s.flip == 2;
    let mut keystone = Keystone {
        vertical: clamp(s.keystone_tilt_y, -100.0, 100.0),
        horizontal: clamp(s.keystone_tilt_x, -100.0, 100.0),
        amount: crate::param_meta::clamp("KeystoneAmount", s.keystone_amount),
        aspect: clamp(s.keystone_aspect, -50.0, 50.0),
        skew: clamp(s.keystone_skew, -45.0, 45.0),
        focal_length: if s.keystone_focal_length.is_finite() && s.keystone_focal_length > 0.0 {
            s.keystone_focal_length
        } else {
            0.0
        },
        // `IC_KeystoneRaw.flags` has no `.cos` column; C1 never writes it.
        flags: 0,
    };
    // Storage is tiltX/tiltY; the sliders show the rotated and flipped H/V frame.
    let (h, v) = keystone.hv(s.rotation, flip_horizontal, flip_vertical);
    keystone.horizontal = h;
    keystone.vertical = v;

    set.geometry = Geometry {
        rotation: split_rotation(s.rotation).0,
        orientation: split_rotation(s.rotation).1,
        flip_horizontal,
        flip_vertical,
        crop_outside_image: s.crop_outside_image,
        canvas: s.canvas_size,
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
        keystone,
        crop: NormRect::FULL,
    };
    set.geometry.crop = crop_on_canvas_with_clipping(s.crop_px, &set.geometry, None, s.geometry_canvas_clipping);

    // C1 stores file names (`NikonZ6-ProStandard.icm`, `NikonZ6-Auto.fcrv`),
    // but the renderer looks profiles and curves up by stem, so the extension
    // has to go or the camera profile and film curve are silently dropped.
    set.base.icc_profile = s.icc_profile.clone().filter(|p| p != "EMBEDDED").map(|p| stem_of(&p));
    set.base.film_curve = s.film_curve.clone().map(|p| stem_of(&p));
    set.base.processing_method = crate::TechnicalProcessingMethod::from_code(s.technical_processing_method);
    set.base.curve = ToneCurvePreset::from_film_curve(s.film_curve.as_deref().unwrap_or(""));

    // Kept for documents and callers that still read the raw extra key; the
    // typed field above is the contract.
    if grain_density != 0.0 {
        set.extra.insert("FilmGrainDensity".to_string(), grain_density);
    }

    set
}


#[cfg(test)]
pub(crate) fn crop_on_canvas(
    crop: [f32; 4],
    geometry: &Geometry,
    source: Option<(u32, u32)>,
) -> NormRect {
    crop_on_canvas_with_clipping(crop, geometry, source, 2)
}

pub(crate) fn crop_on_canvas_with_clipping(
    crop: [f32; 4],
    geometry: &Geometry,
    source: Option<(u32, u32)>,
    clipping: i32,
) -> NormRect {
    let [raw_x, raw_y, width, height] = crop;
    let [canvas_width, canvas_height] = geometry.canvas;
    if !crop.iter().chain(&geometry.canvas).all(|v| v.is_finite())
        || width <= 0.0 || height <= 0.0 || canvas_width <= 0.0 || canvas_height <= 0.0
    {
        return NormRect::FULL;
    }

    let turn = geometry.orientation % 4;
    // shortcut: use Aurora's affine frame map, replace when native lens/keystone
    // Raw2Canvas behavior is verified. Canvas dimensions approximate missing raw dimensions.
    let (source_width, source_height) = source
        .filter(|&(w, h)| w > 0 && h > 0)
        .map(|(w, h)| (w as f32, h as f32))
        .unwrap_or(if turn % 2 == 0 {
            (canvas_width, canvas_height)
        } else {
            (canvas_height, canvas_width)
        });
    let (raw_x, raw_y) = (raw_x - source_width / 2.0, raw_y - source_height / 2.0);
    let (mut x, mut y) = match turn {
        1 => (-raw_y, raw_x),
        2 => (-raw_x, -raw_y),
        3 => (raw_y, -raw_x),
        _ => (raw_x, raw_y),
    };
    if geometry.flip_horizontal {
        x = -x;
    }
    if geometry.flip_vertical {
        y = -y;
    }
    let (sin, cos) = geometry.rotation.to_radians().sin_cos();
    let (x, y) = (
        x * cos - y * sin + canvas_width / 2.0,
        x * sin + y * cos + canvas_height / 2.0,
    );

    let requested = crop_limit::Crop {
        center: [f64::from(x), f64::from(y)],
        size: [f64::from(width), f64::from(height)],
    };
    let canvas = [f64::from(canvas_width), f64::from(canvas_height)];
    let resolved = if geometry.crop_outside_image {
        requested
    } else {
        let bounds = if clipping & 256 != 0 {
            crop_limit::Bounds::Mode10 { size: canvas }
        } else if clipping & 2 != 0 {
            // shortcut: the catalog stores the canvas size, not the native
            // warped min/max box. Use the same rectangular canvas as before
            // until that runtime box can be reproduced.
            crop_limit::Bounds::Mode8 { min: [0.0; 2], max: canvas }
        } else if let Some((w, h)) = source.filter(|&(w, h)| w > 0 && h > 0) {
            crop_limit::Bounds::Mode1 { image: [w, h] }
        } else {
            crop_limit::Bounds::Mode10 { size: canvas }
        };
        let map = |mode, point: [f64; 2]| {
            if mode != 2 {
                return point;
            }
            let [x, y] = [point[0] - canvas[0] / 2.0, point[1] - canvas[1] / 2.0];
            let (sin, cos) = (f64::from(sin), f64::from(cos));
            let mut x_raw = x * cos + y * sin;
            let mut y_raw = y * cos - x * sin;
            if geometry.flip_horizontal {
                x_raw = -x_raw;
            }
            if geometry.flip_vertical {
                y_raw = -y_raw;
            }
            let (x, y) = match turn {
                1 => (y_raw, -x_raw),
                2 => (-x_raw, -y_raw),
                3 => (-y_raw, x_raw),
                _ => (x_raw, y_raw),
            };
            [x + f64::from(source_width) / 2.0, y + f64::from(source_height) / 2.0]
        };
        crop_limit::limit_center_keep_aspect(requested, bounds, map, |mode, points| {
            points.map(|point| map(mode, point))
        })
    };
    let [x, y] = resolved.center.map(|v| v as f32);
    let [width, height] = resolved.size.map(|v| v as f32);

    NormRect {
        x: (x - width / 2.0) / canvas_width,
        y: (y - height / 2.0) / canvas_height,
        width: width / canvas_width,
        height: height / canvas_height,
    }
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
    fn clean_noise_amount_reaches_single_pixel() {
        let set = to_adjustment_set(&from_layer_row(&row(&[(
            "ZCLEANLONGEXPOSUREAMOUNT",
            Some("37.5"),
        )])));
        assert_eq!(set.noise_reduction.single_pixel, 37.5);

        let off = to_adjustment_set(&from_layer_row(&row(&[])));
        assert_eq!(off.noise_reduction.single_pixel, 0.0);

        let over = to_adjustment_set(&from_layer_row(&row(&[(
            "ZCLEANLONGEXPOSUREAMOUNT",
            Some("150"),
        )])));
        assert_eq!(over.noise_reduction.single_pixel, 100.0);
    }

    #[test]
    fn cnr_amount_reaches_the_color_slider() {
        let set = to_adjustment_set(&from_layer_row(&row(&[("ZCNRAMOUNT", Some("37.5"))])));
        assert_eq!(set.noise_reduction.color, 37.5);
        assert!(
            !set.extra.contains_key("BandingReductionAmount"),
            "CNR must not be carried as banding"
        );

        let off = to_adjustment_set(&from_layer_row(&row(&[])));
        assert_eq!(off.noise_reduction.color, 50.0);

        let over = to_adjustment_set(&from_layer_row(&row(&[("ZCNRAMOUNT", Some("150"))])));
        assert_eq!(over.noise_reduction.color, 100.0);
    }

    #[test]
    fn ai_denoise_is_not_the_color_slider() {
        let set = to_adjustment_set(&from_layer_row(&row(&[(
            "ZAIDENOISEAMOUNT",
            Some("80"),
        )])));
        assert_eq!(set.noise_reduction.color, 50.0, "AI denoise stays out of CNR");
    }

    #[test]
    fn fine_grain_column_is_not_the_grain_amount() {
        let dead = to_adjustment_set(&from_layer_row(&row(&[(
            "ZFINEGRAINAMOUNT",
            Some("90"),
        )])));
        assert_eq!(dead.film_grain.impact, 0.0, "ZFINEGRAINAMOUNT is a dead column");

        let real = to_adjustment_set(&from_layer_row(&row(&[(
            "ZFILMGRAINAMOUNT",
            Some("90"),
        )])));
        assert_eq!(real.film_grain.impact, 90.0);
    }

    #[test]
    fn denoise_method_lives_in_zdenoisemethod() {
        let s = from_layer_row(&row(&[("ZDENOISEMETHOD", Some("1"))]));
        assert_eq!(s.noise_reduction.method, 1);
        let off = from_layer_row(&row(&[]));
        assert_eq!(off.noise_reduction.method, 0);
    }

    #[test]
    fn keystone_storage_is_rotated_and_flipped_into_the_slider_frame() {
        let s = from_layer_row(&row(&[
            ("ZROTATION", Some("90")),
            ("ZGEOMETRYKEYSTONETILTX", Some("30")),
            ("ZGEOMETRYKEYSTONETILTY", Some("-20")),
            ("ZGEOMETRYKEYSTONESKEW", Some("12")),
            ("ZGEOMETRYKEYSTONEFOCALLENGTH", Some("0.35")),
            ("ZFLIP", Some("1")),
        ]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.geometry.orientation, 1);
        assert!(set.geometry.flip_horizontal && !set.geometry.flip_vertical);
        let k = set.geometry.keystone;
        assert!((k.horizontal - 20.0).abs() < 1e-4, "h {}", k.horizontal);
        assert!((k.vertical + 30.0).abs() < 1e-4, "v {}", k.vertical);
        assert_eq!(k.skew, 12.0);
        assert_eq!(k.focal_length, 0.35);
    }

    #[test]
    fn a_variant_without_a_crop_keeps_the_full_frame() {
        let mut s = from_layer_row(&row(&[]));
        s.canvas_size = [6000.0, 4000.0];
        let set = to_adjustment_set(&s);
        assert_eq!(set.geometry.crop, crate::NormRect::FULL);
    }

    #[test]
    fn film_grain_fields_are_typed_and_density_keeps_its_extra_key() {
        let s = from_layer_row(&row(&[
            ("ZFILMGRAINAMOUNT", Some("100")),
            ("ZFILMGRAINDENSITY", Some("40")),
            ("ZFILMGRAINGRANULARITY", Some("50")),
            ("ZFILMGRAINTYPE", Some("2")),
        ]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.film_grain.kind, GrainKind::SoftGrain);
        assert_eq!(set.film_grain.impact, 100.0);
        assert_eq!(set.film_grain.granularity, 50.0);
        assert_eq!(set.film_grain.density, 40.0);
        assert_eq!(set.extra.get("FilmGrainDensity"), Some(&40.0));
        assert!(!set.film_grain.density_is_effective());
    }

    #[test]
    fn film_grain_density_is_effective_only_for_fine_grain() {
        for (kind, effective) in [
            ("0", true),
            ("1", false),
            ("2", false),
            ("3", false),
            ("4", false),
            ("5", false),
        ] {
            let s = from_layer_row(&row(&[("ZFILMGRAINTYPE", Some(kind))]));
            let set = to_adjustment_set(&s);
            assert_eq!(set.film_grain.density_is_effective(), effective, "type {kind}");
        }
    }

    #[test]
    fn a_default_layer_row_keeps_default_grain() {
        let set = to_adjustment_set(&from_layer_row(&row(&[])));
        assert_eq!(set.film_grain, FilmGrain::default());
        assert!(!set.extra.contains_key("FilmGrainDensity"));
    }

    /// `ZDEHAZECOLOR = 0;0;0` is Auto: the calculated colour and both levels win.
    #[test]
    fn dehaze_auto_uses_the_calculated_colour_and_levels() {
        let s = from_layer_row(&row(&[
            ("ZDEHAZEAMOUNT", Some("40")),
            ("ZDEHAZECOLOR", Some("0;0;0")),
            ("ZCALCULATEDDEHAZECOLOR", Some("0.91;0.88;0.84")),
            ("ZDEHAZELEVELLOW", Some("0.02;0.03;0.04")),
            ("ZDEHAZELEVELHIGH", Some("0.98;0.97;0.96")),
        ]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.dehaze.amount, 40.0);
        assert_eq!(set.dehaze.shadow_tone, [0.91, 0.88, 0.84]);
        assert_eq!(set.dehaze.level_low, [0.02, 0.03, 0.04]);
        assert_eq!(set.dehaze.level_high, [0.98, 0.97, 0.96]);
    }

    /// Without a calculated colour the `0.98 / max(WB, 1)` fallback is used.
    #[test]
    fn dehaze_auto_without_a_calculated_colour_falls_back_to_the_white_balance() {
        let s = from_layer_row(&row(&[
            ("ZDEHAZECOLOR", Some("0;0;0")),
            ("ZWHITEBALANCE", Some("0.5;1.0;2.0")),
        ]));
        let set = to_adjustment_set(&s);
        assert!((set.dehaze.shadow_tone[0] - 0.98).abs() < 1e-6);
        assert!((set.dehaze.shadow_tone[1] - 0.98).abs() < 1e-6);
        assert!((set.dehaze.shadow_tone[2] - 0.49).abs() < 1e-5);
        assert_eq!(set.dehaze.level_low, [0.0; 3]);
        assert_eq!(set.dehaze.level_high, [1.0; 3]);
    }

    /// A manual haze colour is used verbatim, even when the auto column is set.
    #[test]
    fn manual_dehaze_colour_wins_over_the_calculated_one() {
        let s = from_layer_row(&row(&[
            ("ZDEHAZECOLOR", Some("0.2;0.3;0.4")),
            ("ZCALCULATEDDEHAZECOLOR", Some("0.9;0.9;0.9")),
        ]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.dehaze.shadow_tone, [0.2, 0.3, 0.4]);
    }

    #[test]
    fn zspots_is_read_as_raw_pixels_and_not_as_a_mask() {
        let s = from_layer_row(&row(&[(
            "ZSPOTS",
            Some("-1;2,1000.000000,750.000000,60.000000;3,2000.000000,1500.000000,5.000000"),
        )]));
        assert_eq!(s.spots.spots.len(), 2);
        assert_eq!(s.spots.spots[0].kind, crate::c1_spots::KIND_SPOT);
        assert_eq!((s.spots.spots[0].x, s.spots.spots[0].radius), (1000.0, 60.0));
        assert!(s.relative_spots.is_empty());
        // to_adjustment_set leaves the mapping to the catalog reader, which knows
        // the source size and the final orientation.
        assert!(to_adjustment_set(&s).spots.is_empty());
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
            "ZGEOMETRYKEYSTONEFOCALLENGTH",
            "ZGEOMETRYKEYSTONESKEW",
            "ZFLIP",
            "ZHIGHLIGHTRECOVERY",
            "ZSHADOWRECOVERY",
            "ZWHITERECOVERY",
            "ZBLACKRECOVERY",
            "ZDEHAZEAMOUNT",
            "ZDEHAZECOLOR",
            "ZCALCULATEDDEHAZECOLOR",
            "ZDEHAZELEVELHIGH",
            "ZDEHAZELEVELLOW",
            "ZFILMGRAINGRANULARITY",
            "ZFILMGRAINTYPE",
            "ZBWENABLED",
            "ZLENSCHROMABRENABLED",
            "ZLENSIDGUID",
            "ZICCPROFILE",
            "ZFILMCURVE",
            "ZTECHNICALPROCESSINGMETHOD",
            "ZSTYLEUUID",
            "ZSTYLESOURCE",
            "ZSTYLEMODIFIED",
            "ZAICOLORGRADE",
            "ZLAYERSELECTPOINT",
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
        assert_eq!(column("ZLENSCHROMAFRINGING").unwrap().encoding, C1Encoding::Pipe);
        assert_eq!(column("ZLENSCHROMAFRINGING").unwrap().components, 6);
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

    /// Legacy `GCurve` values are stored in editor order and may descend; the
    /// parser keeps them verbatim so a write-back is byte-faithful. C1 sorts
    /// them itself when it loads the file (`SetGradation`).
    #[test]
    fn stored_curve_order_is_preserved() {
        let c = parse_curve("0.514,0.583;0.210,0.260");
        assert_eq!(c, vec![(0.514, 0.583), (0.210, 0.260)]);
    }

    #[test]
    fn curve_points_are_capped_at_c1_capacity() {
        let raw = (0..20).map(|i| format!("{},{}", i as f32 / 19.0, i as f32 / 19.0))
            .collect::<Vec<_>>()
            .join(";");
        assert_eq!(parse_curve(&raw).len(), MAX_CURVE_POINTS);
    }

    /// `ZGRADATIONCURVEY` is the luminance curve; its non-identity value comes
    /// from the installed `Presets60/Curve/Contrast Luma.copreset`.
    #[test]
    fn gradation_curve_y_reaches_the_luma_curve() {
        let s = from_layer_row(&row(&[(
            "ZGRADATIONCURVEY",
            Some("0.000,0.000;0.104,0.075;0.252,0.254;0.549,0.640;1.000,1.000"),
        )]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.curves.luma.points.len(), 5);
        assert!((set.curves.luma.points[3].y - 0.640).abs() < 1e-6);
        assert_eq!(set.curves.rgb, Curve::default(), "Y must not leak into RGB");
        assert!(!set.curves.luma.is_identity(), "the luma curve must survive");
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
        assert_eq!(ce.advanced.len(), 1, "kind 0 is the only advanced group");
        assert_eq!(ce.skin.len(), 1, "kind 2 is skin");
        assert_eq!(ce.max_smooth.len(), 1, "kind 3 is MaxSmooth");
        assert_eq!(ce.advanced_bands.len(), 3);
        assert_eq!(ce.skin[0].uniformity.expect("skin uniformity").lightness, 60.0);
        assert!((ce.global_shift.hue - 5.0).abs() < 1e-6);
    }

    /// The stored `kind` decides the list: 0 advanced, 1 basic, 2 skin,
    /// 3 max smooth. A regression here silently drops a C1 correction.
    #[test]
    fn every_kind_routes_to_its_own_list() {
        for kind in 0..=3 {
            let raw = format!("1,1,{kind},5,-10,0,255,0,0,-10,10,-100,100,4,0,0,0,0");
            let ce = color_editor_from_c1(&parse_color_corrections(&raw));
            match kind {
                0 => {
                    assert_eq!(ce.advanced.len(), 1, "kind 0 must be advanced");
                    assert!(ce.skin.is_empty() && ce.max_smooth.is_empty());
                    assert_eq!(ce.basic.iter().filter(|s| **s != HslShift::default()).count(), 0);
                }
                1 => {
                    assert_eq!(ce.basic[HueSector::Red as usize].hue, 5.0);
                    assert!(ce.advanced.is_empty() && ce.skin.is_empty() && ce.max_smooth.is_empty());
                }
                2 => {
                    assert_eq!(ce.skin.len(), 1, "kind 2 must be skin");
                    assert!(ce.advanced.is_empty() && ce.max_smooth.is_empty());
                    assert!(ce.skin[0].uniformity.is_some());
                }
                _ => {
                    assert_eq!(ce.max_smooth.len(), 1, "kind 3 must be max smooth");
                    assert!(ce.advanced.is_empty() && ce.skin.is_empty());
                    assert!(ce.max_smooth[0].uniformity.is_none());
                }
            }
        }
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
    fn manufacturer_profile_reads_the_variants_curves() {
        let adj = to_adjustment_set(&from_layer_row(&row(&[
            ("ZLENSIDGUID", Some("2420559E-FE2D-4B13-91BB-E488FFA530E1")),
            (
                "ZLENSDISTORTION",
                Some("0.0,0.0000;742.0,-43.3392;23742.6,146.6017|100.0|1|100|-1.00000"),
            ),
            (
                "ZLENSDEFAULTCHROMABR",
                Some("0.0,0.0000;13355.2,-3.9107|0.0,0.0000;13355.2,-2.0419"),
            ),
            ("ZLENSCHROMABRENABLED", Some("1")),
        ])));
        let curves = adj.lens.manufacturer.as_ref().expect("manufacturer curves");
        assert_eq!(curves.distortion.len(), 3);
        assert!((curves.ca_red[1].1 + 3.9107).abs() < 1e-4);
        assert!((curves.ca_blue[1].1 + 2.0419).abs() < 1e-4);
        // A manufacturer profile's CA rides in `manufacturer`; the raw splines
        // must not shadow it.
        assert!(adj.lens.ca_red_points.is_empty());
        assert!(adj.lens.ca_blue_points.is_empty());

        let unknown = to_adjustment_set(&from_layer_row(&row(&[
            ("ZLENSIDGUID", Some("6E1F8E5B-0000-0000-0000-000000000000")),
            ("ZLENSDISTORTION", Some("0.0,0.0000;742.0,-43.3392|100.0|1|100|-1.00000")),
        ])));
        assert!(unknown.lens.manufacturer.is_none());
    }

    #[test]
    fn lens_radial_blur_uniform_light_and_tilt_follow_their_columns() {
        let adj = to_adjustment_set(&from_layer_row(&row(&[
            ("ZLENSRADIALBLUR", Some("0,0;32000,20|0,0;32000,10|100.0|1")),
            ("ZLENSLCCLIGHTFALLOFF", Some("1")),
            ("ZLENSLCCLIGHTFALLOFFAMOUNT", Some("75")),
            ("ZLENSTILT", Some("12")),
        ])));
        assert_eq!(adj.lens.sharpness_falloff, 100.0);
        assert!(adj.lens.radial_blur.valid);
        assert_eq!(adj.lens.radial_blur.radial, vec![(0.0, 0.0), (32000.0, 20.0)]);
        assert_eq!(adj.lens.radial_blur.transverse, vec![(0.0, 0.0), (32000.0, 10.0)]);
        assert!(adj.lens.uniform_light_enabled);
        assert_eq!(adj.lens.uniform_light, 75.0);
        assert_eq!(adj.lens.tilt, 12.0);

        let gated = to_adjustment_set(&from_layer_row(&row(&[
            ("ZLENSLCCLIGHTFALLOFF", Some("0")),
            ("ZLENSLCCLIGHTFALLOFFAMOUNT", Some("75")),
        ])));
        assert!(!gated.lens.uniform_light_enabled);
        assert_eq!(gated.lens.radial_blur, RadialBlur::default());
        assert_eq!(gated.lens.sharpness_falloff, 0.0);
    }

    #[test]
    fn lens_lcc_id_reaches_the_session_path() {
        let s = from_layer_row(&row(&[("ZLENSLCC", Some("AB12CD34"))]));
        assert_eq!(s.lens.lcc_id.as_deref(), Some("AB12CD34"));
        let path = crate::dust_lcc::sidecar_path(
            std::path::Path::new("C:/session"),
            "_DSC5673.NEF",
            "AB12CD34",
        );
        assert!(path.ends_with("_DSC5673.NEF.AB12CD34.lcc"));
        assert_eq!(from_layer_row(&row(&[])).lens.lcc_id, None);
    }

    #[test]
    fn chroma_aberration_origin_maps_c1_values() {
        let origin = |raw: Option<&'static str>| {
            from_layer_row(&row(&[("ZLENSCHROMABRORIGIN", raw)]))
                .lens
                .chromatic_aberration_origin
        };
        assert_eq!(origin(Some("0")), ChromaAberOrigin::None);
        assert_eq!(origin(Some("1")), ChromaAberOrigin::LensDb);
        assert_eq!(origin(Some("2")), ChromaAberOrigin::Custom);
        // C1's SetChromOrigin treats every other value as "no origin".
        assert_eq!(origin(Some("3")), ChromaAberOrigin::None);
        assert_eq!(origin(None), ChromaAberOrigin::None);
    }

    #[test]
    fn custom_chroma_curves_and_name_parse() {
        let s = from_layer_row(&row(&[
            (
                "ZLENSCUSTOMCHROMABR",
                Some("0.0,0.0000;1000.0,-1.5000|0.0,0.0000;1000.0,-0.7500"),
            ),
            ("ZLENSCUSTOMCHROMABRNAME", Some("SamYang 14mm")),
            ("ZLENSCHROMABRORIGIN", Some("2")),
        ]));
        assert_eq!(s.lens.chromatic_aberration_custom.splines.len(), 2);
        assert!((s.lens.chromatic_aberration_custom.splines[0][1].1 + 1.5).abs() < 1e-6);
        assert!((s.lens.chromatic_aberration_custom.splines[1][1].1 + 0.75).abs() < 1e-6);
        assert_eq!(
            s.lens.chromatic_aberration_custom_name.as_deref(),
            Some("SamYang 14mm")
        );

        let empty = from_layer_row(&row(&[]));
        assert!(empty.lens.chromatic_aberration_custom.splines.is_empty());
        assert_eq!(empty.lens.chromatic_aberration_custom_name, None);
    }

    #[test]
    fn ca_origin_selects_the_live_two_curve_pair() {
        let default = "0.0,0.0000;1000.0,-2.0000|0.0,0.0000;1000.0,-1.0000";
        let custom = "0.0,0.0000;1000.0,-4.0000|0.0,0.0000;1000.0,-2.0000";
        let rows = |origin: &'static str| {
            to_adjustment_set(&from_layer_row(&row(&[
                ("ZLENSDEFAULTCHROMABR", Some(default)),
                ("ZLENSCUSTOMCHROMABR", Some(custom)),
                ("ZLENSCHROMABRORIGIN", Some(origin)),
                ("ZLENSCHROMABRENABLED", Some("1")),
            ])))
        };

        let lens_db = rows("1");
        assert_eq!(lens_db.lens.ca_origin, crate::CaOrigin::LensDb);
        assert!((lens_db.lens.ca_red_points[1].1 + 0.002).abs() < 1e-6);
        assert!((lens_db.lens.ca_blue_points[1].1 + 0.001).abs() < 1e-6);

        let custom_rows = rows("2");
        assert_eq!(custom_rows.lens.ca_origin, crate::CaOrigin::Custom);
        assert!((custom_rows.lens.ca_red_points[1].1 + 0.004).abs() < 1e-6);
        assert!((custom_rows.lens.ca_blue_points[1].1 + 0.002).abs() < 1e-6);

        let none = rows("0");
        assert_eq!(none.lens.ca_origin, crate::CaOrigin::None);
        assert!(none.lens.ca_red_points.is_empty() && none.lens.ca_blue_points.is_empty());
    }

    #[test]
    fn chroma_fringing_parses_six_named_curves() {
        let raw = "0.000,0.000;10.000,1.000|0.000,0.500|0.000,0.250|\
                   0.000,0.000;10.000,-1.000|0.000,-0.500|0.000,-0.250";
        let f = from_layer_row(&row(&[("ZLENSCHROMAFRINGING", Some(raw))]))
            .lens
            .chroma_fringing;
        assert_eq!(f.red_shift.len(), 2);
        assert_eq!(f.red_ra.len(), 1);
        assert!((f.blue_shift[1].1 + 1.0).abs() < 1e-6);
        assert!((f.blue_strength[0].1 + 0.25).abs() < 1e-6);
        assert!(!f.is_empty());

        // The real catalog stores "|||||" for a profile without fringing data.
        let empty =
            from_layer_row(&row(&[("ZLENSCHROMAFRINGING", Some("|||||"))]))
                .lens
                .chroma_fringing;
        assert!(empty.is_empty());
        // C1 ignores anything that is not exactly six fields.
        assert!(parse_chroma_fringing("0.000,0.000").is_empty());
    }

    #[test]
    fn purple_fringing_keeps_the_full_signed_range() {
        let adj = |raw: &'static str| {
            to_adjustment_set(&from_layer_row(&row(&[(
                "ZLENSPURPLEFRINGINGAMOUNT",
                Some(raw),
            )])))
            .lens
            .purple_fringing
        };
        assert_eq!(adj("-40.0"), -40.0);
        assert_eq!(adj("100.0"), 100.0);
        assert_eq!(
            to_adjustment_set(&from_layer_row(&row(&[])))
                .lens
                .purple_fringing,
            0.0
        );
    }

    #[test]
    fn diffraction_column_reaches_the_lens() {
        let lens = to_adjustment_set(&from_layer_row(&row(&[
            ("ZLENSDIFFRACTIONCORRECTIONENABLED", Some("1")),
            ("ZLENSAPERTURE", Some("11.0")),
        ])))
        .lens;
        assert!(lens.diffraction_enabled);
        assert_eq!(lens.aperture, 11.0);

        let off = to_adjustment_set(&from_layer_row(&row(&[]))).lens;
        assert!(!off.diffraction_enabled);
        assert_eq!(off.aperture, 0.0);
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
        // Only the `.x` component of the target pair is consumed; the stored
        // r/g/b components do not become per-channel output points.
        for channel in [&adj.levels.rgb, &adj.levels.red, &adj.levels.green, &adj.levels.blue] {
            assert!((channel.output_black - 0.0).abs() < 1e-6);
            assert!((channel.output_white - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn target_levels_use_only_the_master_x_component() {
        let s = from_layer_row(&row(&[
            ("ZLEVELSTARGETSHADOW", Some("0.100;0.200;0.300;0.400")),
            ("ZLEVELSTARGETHIGHLIGHT", Some("0.900;0.800;0.700;0.600")),
        ]));
        let levels = levels_from_c1(&s);
        assert!((levels.rgb.output_black - 0.1).abs() < 1e-6);
        assert!((levels.rgb.output_white - 0.9).abs() < 1e-6);
        for channel in [&levels.red, &levels.green, &levels.blue] {
            assert_eq!((channel.output_black, channel.output_white), (0.0, 1.0));
        }
        // The affine map itself: out = in*(0.9-0.1)+0.1.
        let out = levels.apply([0.0, 0.5, 1.0]);
        assert!((out[0] - 0.1).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn target_levels_enable_is_a_stage_gate() {
        let disabled = from_layer_row(&row(&[
            ("ZLEVELSTARGETENABLE", Some("0")),
            ("ZLEVELSTARGETSHADOW", Some("0.200;0.000;0.000;0.000")),
            ("ZLEVELSTARGETHIGHLIGHT", Some("0.800;1.000;1.000;1.000")),
        ]));
        assert!(!disabled.levels_target_enable);
        let levels = levels_from_c1(&disabled);
        assert_eq!((levels.rgb.output_black, levels.rgb.output_white), (0.0, 1.0));

        let enabled = from_layer_row(&row(&[
            ("ZLEVELSTARGETENABLE", Some("1")),
            ("ZLEVELSTARGETSHADOW", Some("0.200;0.000;0.000;0.000")),
            ("ZLEVELSTARGETHIGHLIGHT", Some("0.800;1.000;1.000;1.000")),
        ]));
        let levels = levels_from_c1(&enabled);
        assert!((levels.rgb.output_black - 0.2).abs() < 1e-6);
        assert!((levels.rgb.output_white - 0.8).abs() < 1e-6);
    }

    #[test]
    fn zflip_is_single_axis_and_three_is_a_noop() {
        for (text, horizontal, vertical) in
            [("0", false, false), ("1", true, false), ("2", false, true), ("3", false, false)]
        {
            let s = from_layer_row(&row(&[("ZFLIP", Some(text))]));
            let g = to_adjustment_set(&s).geometry;
            assert_eq!((g.flip_horizontal, g.flip_vertical), (horizontal, vertical));
        }
    }

    #[test]
    fn crop_outside_image_reaches_the_geometry_contract() {
        for clipping in [1, 2, 256] {
            for (outside, expected_width) in [("1", 0.8), ("0", 0.6)] {
                let s = from_layer_row(&row(&[
                    ("ZCROPOUTSIDEIMAGE", Some(outside)),
                    ("ZCROP", Some("30;70;80;80")),
                    ("ZCANVASSIZE", Some("100;100")),
                ]));
                let g = to_adjustment_set(&s).geometry;
                let crop = crop_on_canvas_with_clipping(s.crop_px, &g, Some((100, 100)), clipping);
                assert_eq!(g.crop_outside_image, outside == "1");
                assert!((crop.width - expected_width).abs() < 0.0001);
                assert_eq!(crop.width, crop.height);
                assert!((crop.x + crop.width / 2.0 - 0.3).abs() < 0.0001);
                assert!((crop.y + crop.height / 2.0 - 0.7).abs() < 0.0001);
            }

            let s = from_layer_row(&row(&[
                ("ZCROP", Some("110;70;80;80")),
                ("ZCANVASSIZE", Some("100;100")),
            ]));
            let g = to_adjustment_set(&s).geometry;
            let crop = crop_on_canvas_with_clipping(s.crop_px, &g, Some((100, 100)), clipping);
            assert_eq!((crop.x, crop.y, crop.width, crop.height), (1.1, 0.7, 0.0, 0.0));
        }
    }

    #[test]
    fn real_dsc5673_crop_transforms_the_center_before_normalizing() {
        let settings = from_layer_row(&row(&[
            ("ZROTATION", Some("270")),
            ("ZCROP", Some("2525.8076171875;2348.0876464844;3328.8486328125;4993.2734375000")),
            ("ZCANVASSIZE", Some("4043.054;6056.673")),
        ]));
        let mut g = to_adjustment_set(&settings).geometry;
        assert_eq!(settings.crop_px, [2525.8076, 2348.0876, 3328.8486, 4993.2734]);
        assert_eq!(g.orientation, 3);
        let rect = crop_on_canvas(settings.crop_px, &g, Some((6048, 4032)));
        assert!((rect.width * g.canvas[0] - 3328.8486).abs() < 0.01);
        assert!((rect.height * g.canvas[1] - 4993.2734).abs() < 0.01);
        let center = |r: NormRect| [r.x + r.width / 2.0, r.y + r.height / 2.0];
        let [x, y] = center(rect);
        assert!((x * g.canvas[0] - 2353.6147).abs() < 0.01);
        assert!((y * g.canvas[1] - 3526.5288).abs() < 0.01);

        g.flip_horizontal = true;
        let [flipped_x, flipped_y] = center(crop_on_canvas(settings.crop_px, &g, Some((6048, 4032))));
        assert!((flipped_x - (1.0 - x)).abs() < 1e-6);
        assert!((flipped_y - y).abs() < 1e-6);
    }

    #[test]
    fn real_dsc5673_crop_row_is_exact() {
        // Catalog row ZVARIANTLAYER.Z_PK=1 (also row 10): the only non-zero
        // ZCROP in the sample catalog. 6048x4032 raw, 270 degrees, clipping=2.
        let settings = from_layer_row(&row(&[
            ("ZROTATION", Some("270")),
            ("ZCROP", Some("2525.8076171875;2348.0876464844;3328.8486328125;4993.2734375000")),
            ("ZCANVASSIZE", Some("4043.054;6056.673")),
        ]));
        let geometry = to_adjustment_set(&settings).geometry;
        let rect = crop_on_canvas(settings.crop_px, &geometry, Some((6048, 4032)));
        // Pinned f32 results of the verified transform, not a re-derivation.
        assert_eq!(rect.x.to_bits(), 0x3e2e8dcf);
        assert_eq!(rect.y.to_bits(), 0x3e2e1fa2);
        assert_eq!(rect.width.to_bits(), 0x3f52c711);
        assert_eq!(rect.height.to_bits(), 0x3f530d87);
        assert_eq!((rect.x + rect.width / 2.0).to_bits(), 0x3f1506fc);
        assert_eq!((rect.y + rect.height / 2.0).to_bits(), 0x3f150eac);
    }

    #[test]
    fn real_dsc5673_lens_distortion_row_is_exact() {
        // The only non-zero ZLENSDISTORTION in the sample catalog: eight
        // `radius, delta` pairs followed by amount, enabled, crop type, scale.
        let settings = from_layer_row(&row(&[(
            "ZLENSDISTORTION",
            Some(
                "0.0,0.0000;742.0,-43.3392;1483.9,-86.4269;2225.9,-129.0114;\
                 5935.7,-325.4922;14097.2,-526.0558;20032.8,-279.2483;\
                 23742.6,146.6017|100.0|1|100|-1.00000",
            ),
        )]));
        let profile = &settings.lens.distortion;
        assert_eq!(profile.points().len(), 8);
        assert_eq!(profile.points()[0], (0.0, 0.0));
        assert_eq!(profile.points()[1], (742.0, -43.3392));
        assert_eq!(profile.points()[7], (23742.6, 146.6017));
        assert_eq!(profile.param(0), Some(100.0));
        assert_eq!(profile.param(1), Some(1.0));
        assert_eq!(profile.param(2), Some(100.0));
        assert_eq!(profile.param(3), Some(-1.0));

        // 0x14204E0: micrometre radii over sqrt((pitch.x² + pitch.y²)/2) give
        // the raw-pixel control points. Pinned native bits for node 1 and 7.
        let pitch = [
            (35.9f64 * 1000.0 / 6048.0) as f32,
            (23.9f64 * 1000.0 / 4032.0) as f32,
        ];
        let curve = crop_limit::RadialCurve::from_profile(
            profile.points(),
            profile.param(0).unwrap_or(0.0),
            pitch,
        );
        assert_eq!(curve.v.len(), 8);
        assert_eq!(curve.v[1].to_bits(), 0x405f45c7c0000000);
        assert_eq!(curve.x[1].to_bits(), 0x405d722c80000000);
        assert_eq!(curve.v[7].to_bits(), 0x40af454ee0000000);
        assert_eq!(curve.x[7].to_bits(), 0x40af76bce0000000);
    }

    #[test]
    fn real_dsc5673_lens_wires_into_the_runtime_calibration() {
        let settings = from_layer_row(&row(&[
            (
                "ZLENSDISTORTION",
                Some(
                    "0.0,0.0000;742.0,-43.3392;1483.9,-86.4269;2225.9,-129.0114;\
                     5935.7,-325.4922;14097.2,-526.0558;20032.8,-279.2483;\
                     23742.6,146.6017|100.0|1|100|-1.00000",
                ),
            ),
            ("ZLENSOPTICCENTER", Some("3067.851562|1920.515625")),
        ]));
        assert_eq!(settings.lens.optic_center, [3067.851_562, 1920.515_625]);
        // The document carries no calibration: neither the curve nor the centre.
        let geometry = to_adjustment_set(&settings).geometry;
        let runtime_json = serde_json::to_string(&geometry).unwrap();
        assert!(!runtime_json.contains("distortion") && !runtime_json.contains("optical_center"));
        let mut runtime = lens_runtime_from_settings(&settings);
        assert!(runtime.distortion.is_empty(), "the catalog alone has no pixel pitch");
        apply_lens_distortion(&mut runtime, &settings.lens, [5.935_846_3, 5.927_579_4]);
        assert_eq!(runtime.optical_center, [3067.851_562, 1920.515_625]);
        assert_eq!(runtime.distortion.v.len(), 8);
        assert_eq!(runtime.distortion.v[7].to_bits(), 0x40af454ee0000000);
        assert_eq!(runtime.distortion.x[7].to_bits(), 0x40af76bce0000000);
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
        assert_eq!(s.style_uuid, None);
        assert!(!s.ai_color_grade);
        assert!(s.layer_select_point.is_none());
        assert!(to_adjustment_set(&s).extra.is_empty());
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
        assert!((kelvin - 4779.55).abs() < 2.0, "kelvin {kelvin}");
        assert!((tint + 4.4875).abs() < 0.05, "tint {tint}");
        let back = kelvin_tint_to_gains(kelvin, tint);
        for (a, b) in gains.iter().zip(back.iter()) {
            assert!((a - b).abs() < 2e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn neutral_gains_match_c1_neutral() {
        let (kelvin, tint) = gains_to_kelvin_tint([1.0, 1.0, 1.0]);
        assert!((kelvin - 4325.11).abs() < 2.0);
        assert!((tint + 36.49).abs() < 0.5);
    }

    #[test]
    fn crop_is_normalized_when_the_canvas_is_known() {
        let mut s = C1VariantSettings::defaults();
        s.canvas_size = [6000.0, 4000.0];
        s.crop_px = [3000.0, 2000.0, 3000.0, 2000.0];
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

    #[test]
    fn technical_processing_method_reaches_base_characteristics() {
        for (text, expected) in [
            ("0", crate::TechnicalProcessingMethod::Photography),
            ("2", crate::TechnicalProcessingMethod::ReproFilmNegatives),
            ("3", crate::TechnicalProcessingMethod::ReproFilmPositives),
            ("4", crate::TechnicalProcessingMethod::StandardFilmNegatives),
        ] {
            let s = from_layer_row(&row(&[("ZTECHNICALPROCESSINGMETHOD", Some(text))]));
            assert_eq!(to_adjustment_set(&s).base.processing_method, expected);
        }
    }

    #[test]
    fn base_characteristics_columns_round_trip() {
        let s = from_layer_row(&row(&[
            ("ZTECHNICALPROCESSINGMETHOD", Some("3")),
            ("ZFILMCURVE", Some("NikonZ6III-Auto.fcrv")),
            ("ZSTYLEUUID", Some("4FD0E17F-B1FC-4E78-BEAC-92315B3B15E3")),
            ("ZSTYLESOURCE", Some("Exposure")),
            ("ZSTYLEMODIFIED", Some("1")),
            ("ZAICOLORGRADE", Some("0")),
            ("ZLAYERSELECTPOINT", Some("778.75641;2700.73657")),
        ]));
        assert_eq!(s.style_uuid.as_deref(), Some("4FD0E17F-B1FC-4E78-BEAC-92315B3B15E3"));
        assert_eq!(s.style_source.as_deref(), Some("Exposure"));
        assert!(s.style_modified);
        assert!(!s.ai_color_grade);
        assert_eq!(s.layer_select_point, Some([778.75641, 2700.73657]));

        let set = to_adjustment_set(&s);
        assert_eq!(set.base.processing_method, crate::TechnicalProcessingMethod::ReproFilmPositives);
        assert_eq!(set.base.film_curve.as_deref(), Some("NikonZ6III-Auto"));
        assert_eq!(set.base.curve, ToneCurvePreset::Auto);
    }

    #[test]
    fn processing_method_maps_to_c1_and_imgcore() {
        use crate::TechnicalProcessingMethod as M;
        assert_eq!(M::from_code(1), M::Scientific);
        assert_eq!(M::from_code(9), M::Photography);
        assert_eq!(M::Scientific.to_c1(), 1);
        assert_eq!(M::StandardFilmNegatives.imgcore_value(), 4);
        assert!(M::StandardFilmNegatives.is_film_negative());
        assert!(M::ReproFilmNegatives.inserts_film_negative_conversion());
        assert!(!M::ReproFilmPositives.is_film_negative());
        assert!(M::ReproFilmPositives.forces_color_balance());
        assert!(!M::Scientific.offered_in_base_characteristics());
        assert_eq!(M::ALL.len(), 5);
    }

    #[test]
    fn dng_film_curve_names_map_to_dng_presets() {
        assert_eq!(ToneCurvePreset::from_film_curve("DNG Standard"), ToneCurvePreset::DngStandard);
        assert_eq!(
            ToneCurvePreset::from_film_curve("Phase One DNG Tone Mapped"),
            ToneCurvePreset::DngToneMapped
        );
    }

    #[test]
    fn negative_switch_is_derived_from_the_mode() {
        for (text, negative) in [("0", false), ("1", false), ("2", true), ("3", false), ("4", true)] {
            let s = from_layer_row(&row(&[("ZTECHNICALPROCESSINGMETHOD", Some(text))]));
            assert_eq!(s.negative_switch, negative, "method {text}");
            assert_eq!(to_adjustment_set(&s).base.negative_switch(), negative);
        }
    }

    /// Truth table copied from `out/negative-chain.tsv` rows 4, 7 and 9.
    #[test]
    fn negative_chain_truth_table_matches_lane_spec() {
        use crate::{EngineGeneration, TechnicalProcessingMethod};
        for method in TechnicalProcessingMethod::ALL {
            assert_eq!(
                method.offered_in_base_characteristics(),
                !matches!(method, TechnicalProcessingMethod::Scientific)
            );
        }
        for version in [800u32, 1100, 1300, 1301, 2400] {
            let generation = EngineGeneration::from_raw(version);
            for (text, block, cbfn) in [
                ("0", false, Some(false)),
                ("1", false, Some(false)),
                ("2", true, Some(false)),
                ("3", false, Some(false)),
                ("4", true, if generation == EngineGeneration::After1300 { Some(false) } else { None }),
            ] {
                let s = from_layer_row(&row(&[("ZTECHNICALPROCESSINGMETHOD", Some(text))]));
                let chain = to_adjustment_set(&s).base.negative_chain(generation);
                let tag = format!("method {text} version {version}");
                assert_eq!(chain.film_negative_conversion, block, "{tag}");
                assert_eq!(chain.clamp_negative, block, "{tag}");
                assert_eq!(chain.color_balance_film_negative, cbfn, "{tag}");
            }
        }
    }

    /// Almost every file is method 0; nothing may enter the negative block and
    /// a fresh import must stay byte-for-byte what `AdjustmentSet::default()`
    /// is, so adding the gate cannot move a single pixel.
    #[test]
    fn default_mode_is_inert_and_unchanged() {
        let fresh = to_adjustment_set(&C1VariantSettings::defaults());
        let default = crate::AdjustmentSet::default();
        assert_eq!(default.base.processing_method, fresh.base.processing_method);
        assert_eq!(default.base.engine_version, fresh.base.engine_version);
        assert!(!fresh.base.negative_switch());
        let generation = crate::EngineGeneration::default();
        assert!(fresh.base.negative_chain(generation).is_inert());
        assert_eq!(fresh.base.negative_chain(generation), crate::NegativeChain::default());
        assert_eq!(default.base.negative_chain(generation), crate::NegativeChain::default());
    }

    #[test]
    fn engine_generation_boundaries_are_inclusive() {
        use crate::EngineGeneration;
        assert_eq!(EngineGeneration::from_raw(0x44B), EngineGeneration::Before1100);
        assert_eq!(EngineGeneration::from_raw(0x44C), EngineGeneration::V1100ToV1300);
        assert_eq!(EngineGeneration::from_raw(0x514), EngineGeneration::V1100ToV1300);
        assert_eq!(EngineGeneration::from_raw(0x515), EngineGeneration::After1300);
        assert!(EngineGeneration::Before1100.keeps_legacy_color_balance());
        assert!(!EngineGeneration::V1100ToV1300.keeps_legacy_color_balance());
    }
}
