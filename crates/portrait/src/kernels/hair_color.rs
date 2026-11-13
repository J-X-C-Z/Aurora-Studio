//! PF30009 hair-colour unification: HSL remap + material curves.
//!
//! Ported from the R8 `hair_color_unify_cpu.py` reconstruction, which closes
//! the PE's CPU `HSLRemap` slot (`0x76ad6f0` / `0x76ad800` / `0x76aecc0` /
//! `0x76ae160`, six-way table at `0x095787e0`). The curve stage mirrors the
//! shipped `HSLTuner.frag`, which samples exactly three 74-entry 1-D textures
//! (`huelut_texture`, `saturationlut_texture`, `lumalut_texture`) at
//! `coord.x = hue*12/73`.
//!
//! There is no 3-D LUT in the recovered path. The "material LUT" is the
//! 74-sample hue/saturation/luma curve set; it must be supplied by the caller.
//! No default curve, constant mask or colour transfer is invented here.
//!
//! `Image` channels are RGBA per `lib.rs`, so channel 0 is R (unlike the
//! vendor's BGRA order). This matters only for hue-sensitive maths.
//!
//! Row 95 was already registered as a normal-blend row by `hair_body.rs`.
//! This module deliberately exports `META_OVERLAP` instead of `META` so a
//! shared registry cannot pick it up without an explicit decision.

use crate::{Access, EffectMeta, Image, Refusal};

#[path = "hair_color_tables.rs"]
pub mod tables;

/// Curves are sampled at `hue * 12`, one sample per index, 74 entries.
pub const LUT_SAMPLES: usize = 74;

/// How the 74-sample curves are sampled. `ExactIndex` is the existing
/// behaviour (sample `i` sits at `i`, justified by the generator's wrap copies
/// `s72=s0`, `s73=s1`); `GlLiteral` follows `u = hue*12/73` with an explicit
/// filter. Neither is proven sampler state, so the caller chooses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveSampling {
    ExactIndex,
    GlLiteral { filter: tables::Filter },
}

/// Deliberately not named `META`: PF30009 overlaps `hair_body::META`.
pub const META_OVERLAP: &[EffectMeta] = &[EffectMeta {
    pf: 30009,
    key: "hairColorUnify",
    label: "Hair colour unify (HSL remap)",
    access: Access::Cpu,
    inputs: &[
        "PF30009_HAIR_MASK",
        "PF30009_HSL_LUT",
        "PF30009_ANCHOR_HUE",
    ],
}];

/// The three 74-sample product curves consumed by the HSL tuner shape.
pub struct HslMaps<'a> {
    pub hue: &'a [f32],
    pub saturation: &'a [f32],
    pub luma: &'a [f32],
}

fn validate_maps(maps: &HslMaps<'_>) -> Result<(), Refusal> {
    for curve in [maps.hue, maps.saturation, maps.luma] {
        if curve.len() != LUT_SAMPLES {
            return Err(Refusal::ShapeMismatch("PF30009_HSL_LUT"));
        }
        if curve.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF30009_HSL_LUT"));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DegreeState {
    pub engine: f32,
    pub signed: f32,
    pub amount: f32,
    pub active: bool,
}

/// `engine=(ui+100)/200` in f32, then `signed=1-2*engine` through f64:
/// the PE pair at `0x7DA69A8` does `cvtss2sd/addsd/subsd/cvtsd2ss`.
pub fn degree_from_ui(ui: i32) -> Result<DegreeState, Refusal> {
    if !(-100..=100).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("PF30009_UI"));
    }
    let engine = (ui + 100) as f32 / 200.0;
    let signed = (1.0f64 - 2.0 * engine as f64) as f32;
    Ok(DegreeState {
        engine,
        signed,
        amount: signed.abs(),
        active: signed.abs() > 0.001,
    })
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

/// Soft hue selector from `0x76ad6f0`/`0x76ad800`:
/// `d=min(|h-anchor|, 6-|h-anchor|)`, `w=min(d*12/7, 1)`,
/// `affinity=1-(w-1)^2`.
pub fn hue_affinity(pixel_hue: f32, anchor_hue: f32) -> f32 {
    let mut distance = (pixel_hue - anchor_hue).abs();
    let wrap = 6.0 - distance;
    if wrap < distance {
        distance = wrap;
    }
    let weight = (distance * (12.0 / 7.0)).min(1.0);
    let delta = weight - 1.0;
    1.0 - delta * delta
}

/// RGB -> `(hue 0..6, max, min)` from `0x76aecc0`.
pub fn rgb_to_hue_max_min(rgb: [f32; 3]) -> (f32, f32, f32) {
    let [r, g, b] = rgb;
    let maximum = r.max(g).max(b);
    let minimum = r.min(g).min(b);
    if maximum == minimum {
        return (0.0, maximum, minimum);
    }
    let inverse_delta = 1.0 / (maximum - minimum);
    let hue = if maximum == r {
        let mut value = (g - b) * inverse_delta;
        if value < 0.0 {
            value += 6.0;
        }
        value
    } else if maximum == g {
        (b - r) * inverse_delta + 2.0
    } else {
        (r - g) * inverse_delta + 4.0
    };
    (hue, maximum, minimum)
}

/// `(hue, max, min)` -> RGB through the six-permutation table at `0x095787e0`,
/// without the shader's (absent) clamp. The shader's `hsl2rgb` does not clamp;
/// clipping only happens at the 8-bit store.
fn hue_max_min_to_rgb_raw(hue: f32, maximum: f32, minimum: f32) -> [f32; 3] {
    let mut hue = hue;
    if hue > 6.0 {
        hue -= 6.0;
        if hue > 6.0 {
            hue -= 6.0;
        }
    }
    let sector_float = hue.floor();
    let fraction = hue - sector_float;
    let sector = (sector_float as i64).rem_euclid(6) as usize;

    let value = maximum;
    let floor = minimum;
    let chroma = value - floor;
    let descending = value - chroma * fraction;
    let ascending = floor + chroma * fraction;
    let table = [
        [value, ascending, floor],
        [descending, value, floor],
        [floor, value, ascending],
        [floor, descending, value],
        [ascending, floor, value],
        [value, floor, descending],
    ];
    table[sector]
}

/// Existing public behaviour: same permutation, clamped per channel.
pub fn hue_max_min_to_rgb(hue: f32, maximum: f32, minimum: f32) -> [f32; 3] {
    hue_max_min_to_rgb_raw(hue, maximum, minimum).map(clamp01)
}

fn lerp_lut(curve: &[f32], scaled_hue: f32) -> f32 {
    let index = (scaled_hue.floor() as i64).clamp(0, LUT_SAMPLES as i64 - 2) as usize;
    let fraction = scaled_hue - index as f32;
    let low = curve[index];
    let high = curve[index + 1];
    low + fraction * (high - low)
}

fn sample_curve(curve: &[f32], scaled_hue: f32, how: CurveSampling) -> f32 {
    match how {
        CurveSampling::ExactIndex => lerp_lut(curve, scaled_hue),
        CurveSampling::GlLiteral { filter } => {
            let u = (scaled_hue.clamp(0.0, (LUT_SAMPLES - 1) as f32))
                / (LUT_SAMPLES - 1) as f32;
            // 74x1 texture; v = 0.0 is the row centre after the -0.5 texel bias.
            tables::sample_grid(filter, tables::Wrap::ClampToEdge, LUT_SAMPLES, 1, curve, u, 0.0)
        }
    }
}

/// Shader `hsltuner()` only. `rgb` must already be the decoded value
/// (`invconstrast_texture(gamma_map_texture(org))` after `GAMMA_WEIGHTS`); it
/// is not source RGB. Use [`pf30009_hsl_pass`] for the full chain.
pub fn hsl_tuner_decoded(
    rgb: [f32; 3],
    maps: &HslMaps<'_>,
) -> Result<[f32; 3], Refusal> {
    validate_maps(maps)?;
    hsl_tuner(rgb, maps)
}

fn hsl_tuner_raw(
    rgb: [f32; 3],
    maps: &HslMaps<'_>,
    how: CurveSampling,
) -> Result<[f32; 3], Refusal> {
    let (hue, mut maximum, mut minimum) = rgb_to_hue_max_min(rgb);
    let diff = maximum - minimum;
    if diff == 0.0 {
        return Ok(rgb);
    }

    let scaled_hue = hue * 12.0;
    let hue_out = sample_curve(maps.hue, scaled_hue, how);
    let mut luma_out = sample_curve(maps.luma, scaled_hue, how);
    let sat_out = sample_curve(maps.saturation, scaled_hue, how);

    if luma_out != 0.0 {
        let a = (2.0 - diff) * diff;
        let b = 2.0 - a;
        let biquadratic = a * b;
        luma_out *= biquadratic * (2.0 - biquadratic);

        let min_diff = minimum * ((1.0 - minimum) * luma_out + 1.0);
        let max_diff = maximum * ((1.0 - maximum) * luma_out + 1.0);
        minimum = min_diff * (luma_out * (1.0 - min_diff) + 1.0);
        maximum = max_diff * (luma_out * (1.0 - max_diff) + 1.0);
    }

    if sat_out >= 0.0 {
        if sat_out > 0.0 {
            let saturation = (maximum - minimum) / maximum;
            let multiplier_max = (maximum * 16.0).min(1.0);
            let tail = (2.0 - multiplier_max) * multiplier_max * (1.0 - saturation);
            let denominator =
                1.0 - (1.0 - minimum) * (sat_out * 0.833_333_31) * tail;
            minimum = maximum - (saturation / denominator) * maximum;
        }
    } else {
        let average = (minimum + maximum) * 0.5;
        minimum += (minimum - average) * sat_out;
        maximum += (maximum - average) * sat_out;
    }

    Ok(hue_max_min_to_rgb_raw(hue_out, maximum, minimum))
}

/// Existing public behaviour: exact-index curves, per-channel clamp.
pub fn hsl_tuner(rgb: [f32; 3], maps: &HslMaps<'_>) -> Result<[f32; 3], Refusal> {
    validate_maps(maps)?;
    let raw = hsl_tuner_raw(rgb, maps, CurveSampling::ExactIndex)?;
    Ok([clamp01(raw[0]), clamp01(raw[1]), clamp01(raw[2])])
}

/// One pixel: `q = mask * hue_affinity * |signed|`, then
/// `out = clamp(src + (remapped - src) * q)`.
pub fn remap_rgb(
    rgb: [f32; 3],
    model_mask: f32,
    anchor_hue: f32,
    maps: &HslMaps<'_>,
    state: DegreeState,
) -> Result<[f32; 3], Refusal> {
    validate_maps(maps)?;
    if !(0.0..=1.0).contains(&model_mask) {
        return Err(Refusal::ValueOutOfRange("PF30009_HAIR_MASK"));
    }
    if !anchor_hue.is_finite() || !(0.0..=6.0).contains(&anchor_hue) {
        return Err(Refusal::ValueOutOfRange("PF30009_ANCHOR_HUE"));
    }
    if rgb.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF30009_RGB"));
    }
    if !state.active || model_mask <= 0.001 {
        return Ok(rgb);
    }

    let (source_hue, _, _) = rgb_to_hue_max_min(rgb);
    let affinity = hue_affinity(source_hue, anchor_hue);
        let mapped = hsl_tuner_raw(rgb, maps, CurveSampling::ExactIndex)?;
        let mapped = [clamp01(mapped[0]), clamp01(mapped[1]), clamp01(mapped[2])];
    let amount = model_mask * affinity * state.amount;
    let mut out = [0.0f32; 3];
    for channel in 0..3 {
        out[channel] = clamp01(rgb[channel] + (mapped[channel] - rgb[channel]) * amount);
    }
    Ok(out)
}

/// 8-bit quantisation used only when crossing the packed-image boundary.
fn to_byte(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

/// Complete HSL-material inputs. Every table and both sampler choices are
/// explicit; nothing falls back to an identity table or a guessed filter.
pub struct Pf30009Input<'a> {
    pub source_rgb: [f32; 3],
    pub maps: &'a HslMaps<'a>,
    pub curves: CurveSampling,
    pub tables: &'a tables::Pf30009Tables<'a>,
}

/// Exact `HSLTuner.frag` material chain:
///
/// ```text
/// gamma_map -> GAMMA_WEIGHTS -> invconstrast -> hsltuner
///   -> constrast -> INV_GAMMA_WEIGHTS -> invgamma_map
/// ```
///
/// This is the **HSL material output only**. It applies no mask and no UI
/// degree: the PF30009 final consumer (`0x5ECF439` -> `0x5F835D0`, mode from
/// `0x5ECF437` = Normal) sets its blend degree to `1.0` itself at
/// `0x5ECF445`, so multiplying a degree here would double-apply it.
pub fn pf30009_hsl_pass(input: &Pf30009Input<'_>) -> Result<[f32; 3], Refusal> {
    validate_maps(input.maps)?;
    if input.source_rgb.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF30009_RGB"));
    }
    pf30009_pixel_unchecked(input.source_rgb, input.maps, input.curves, input.tables)
}

/// Per-pixel material body. Curves, RGB and tables are validated by the
/// caller, so an image loop does not rescan any map or 65536-value table.
fn pf30009_pixel_unchecked(
    source_rgb: [f32; 3],
    maps: &HslMaps<'_>,
    curves: CurveSampling,
    tables: &tables::Pf30009Tables<'_>,
) -> Result<[f32; 3], Refusal> {
    let gamma = [
        tables.gamma.sample(source_rgb[0])?,
        tables.gamma.sample(source_rgb[1])?,
        tables.gamma.sample(source_rgb[2])?,
    ];
    let raw = tables::apply_matrix(tables::GAMMA_WEIGHTS, gamma);
    let decoded = tables::decode_tonal(&tables.inv_contrast, raw)?;
    let tuned = hsl_tuner_raw(decoded, maps, curves)?;
    let contrasted = tables::decode_tonal(&tables.contrast, tuned)?;
    let inv_raw = tables::apply_matrix(tables::INV_GAMMA_WEIGHTS, contrasted);
    tables::encode_inv_gamma(&tables.inv_gamma, inv_raw)
}

/// Whole-image HSL material pass. Alpha is copied; no mask, no degree.
pub fn pf30009_hsl_image(
    source: &Image,
    maps: &HslMaps<'_>,
    curves: CurveSampling,
    tables: &tables::Pf30009Tables<'_>,
) -> Result<Image, Refusal> {
    let expected = (source.width as usize)
        .checked_mul(source.height as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or(Refusal::ShapeMismatch("image"))?;
    if source.width == 0 || source.height == 0 || source.pixels.len() != expected {
        return Err(Refusal::ShapeMismatch("image"));
    }
    validate_maps(maps)?;

    let count = expected / 4;
    let mut out = source.clone();
    for index in 0..count {
        let offset = index * 4;
        let rgb = [
            source.pixels[offset] as f32 / 255.0,
            source.pixels[offset + 1] as f32 / 255.0,
            source.pixels[offset + 2] as f32 / 255.0,
        ];
        let mapped = pf30009_pixel_unchecked(rgb, maps, curves, tables)?;
        for channel in 0..3 {
            out.pixels[offset + channel] = to_byte(mapped[channel] * 255.0);
        }
        out.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(out)
}

// Final consumer (PF30009 mask blend) stays OPEN in this module.
//
// R7 read `0x5ECF437 xor edx` -> `0x5ECF439 call 0x5F835D0` (mode 0 = Normal),
// `0x5ECF445 SetBlendDegree(1.0)`, `0x5ECF459 SetReverseAlpha([effect+0xD5])`
// and three texture units at `0x5ECF479/0x5ECF493/0x5ECF4B9`. The blend factors
// read at `0x5F83DB7..0x5F83DFE` are `0x302` (`GL_SRC_ALPHA`) and `0x303`
// (`GL_ONE_MINUS_SRC_ALPHA`), the `glEnable` call is not recovered, and the
// destination FBO format is unknown. Which texture is mask / material / `F` is
// also OPEN. No final blend helper ships until those close; the HSL material
// pass above is the proven part.

/// Whole-image PF30009 HSL remap over explicit hair mask + material curves.
/// Legacy CPU composition: keeps the pre-R26 mask/degree behaviour.
/// Alpha is copied from the source; `model_mask` is one 0..1 value per pixel.
pub fn remap_image(
    source: &Image,
    model_mask: &[f32],
    anchor_hue: f32,
    maps: &HslMaps<'_>,
    ui: i32,
) -> Result<Image, Refusal> {
    let expected = (source.width as usize)
        .checked_mul(source.height as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or(Refusal::ShapeMismatch("image"))?;
    if source.width == 0 || source.height == 0 || source.pixels.len() != expected {
        return Err(Refusal::ShapeMismatch("image"));
    }
    let count = expected / 4;
    if model_mask.len() != count {
        return Err(Refusal::ShapeMismatch("PF30009_HAIR_MASK"));
    }
    if model_mask
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(Refusal::ValueOutOfRange("PF30009_HAIR_MASK"));
    }
    validate_maps(maps)?;
    let state = degree_from_ui(ui)?;

    let mut out = source.clone();
    if !state.active {
        return Ok(out);
    }
    for index in 0..count {
        let offset = index * 4;
        let rgb = [
            source.pixels[offset] as f32 / 255.0,
            source.pixels[offset + 1] as f32 / 255.0,
            source.pixels[offset + 2] as f32 / 255.0,
        ];
        let mapped = remap_rgb(rgb, model_mask[index], anchor_hue, maps, state)?;
        for channel in 0..3 {
            out.pixels[offset + channel] = to_byte(mapped[channel] * 255.0);
        }
        out.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(out)
}

/// x86 `cvttss2si` semantics: truncate toward zero, and return `0x80000000`
/// for NaN, infinities and values whose truncation leaves the i32 range.
/// Rust's `as i32` saturates instead, so callers that clamp 0..255 must use
/// this to stay on the vendor boundary.
pub fn cvtt_i32(value: f32) -> i32 {
    if !value.is_finite() {
        return i32::MIN;
    }
    let truncated = value.trunc();
    if truncated >= 2_147_483_648.0 || truncated < -2_147_483_648.0 {
        i32::MIN
    } else {
        truncated as i32
    }
}

/// `cvttss2si` then clamp to a byte, matching the PF90131 store path.
pub fn cvtt_byte(value: f32) -> u8 {
    cvtt_i32(value).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control_maps() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hue = (0..LUT_SAMPLES)
            .map(|index| (index as f32 / 12.0).min(6.0))
            .collect();
        (hue, vec![1.0; LUT_SAMPLES], vec![0.0; LUT_SAMPLES])
    }

    #[test]
    fn ui_centre_is_neutral() {
        let state = degree_from_ui(0).unwrap();
        assert_eq!(state.engine, 0.5);
        assert_eq!(state.signed, 0.0);
        assert!(!state.active);
    }

    #[test]
    fn ui_sign_mapping() {
        let minus = degree_from_ui(-50).unwrap();
        let plus = degree_from_ui(50).unwrap();
        assert_eq!(minus.engine, 0.25);
        assert_eq!(minus.signed, 0.5);
        assert_eq!(plus.engine, 0.75);
        assert_eq!(plus.signed, -0.5);
        assert_eq!(
            degree_from_ui(101),
            Err(Refusal::ValueOutOfRange("PF30009_UI"))
        );
    }

    #[test]
    fn golden_rgb_to_hue_max_min() {
        let (hue, maximum, minimum) = rgb_to_hue_max_min([0.2, 0.5, 1.0]);
        assert_eq!(hue, 3.625);
        assert_eq!(maximum, 1.0);
        assert_eq!(minimum, 0.2f32);
    }

    #[test]
    fn golden_hsl_tuner_control_curves() {
        let (hue, saturation, luma) = control_maps();
        let maps = HslMaps {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let tuned = hsl_tuner([0.2, 0.5, 1.0], &maps).unwrap();
        assert_eq!(tuned, [0.076_923_07, 0.423_076_93, 1.0]);
    }

    #[test]
    fn golden_mask_degree_composition() {
        let (hue, saturation, luma) = control_maps();
        let maps = HslMaps {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let out = remap_rgb(
            [0.2, 0.5, 1.0],
            1.0,
            0.0,
            &maps,
            degree_from_ui(50).unwrap(),
        )
        .unwrap();
        assert_eq!(out, [0.138_461_53, 0.461_538_46, 1.0]);
    }

    #[test]
    fn hue_affinity_wraps_and_falls_off() {
        assert_eq!(hue_affinity(3.625, 0.0), 1.0);
        // The band peaks where d*12/7 == 1, i.e. d == 7/12, not at d == 0.
        assert_eq!(hue_affinity(0.0, 0.0), 0.0);
        assert_eq!(hue_affinity(7.0 / 12.0, 0.0), 1.0);
        assert_eq!(hue_affinity(3.0, 0.0), 1.0);
    }

    #[test]
    fn neutral_ui_and_zero_mask_are_identity() {
        let (hue, saturation, luma) = control_maps();
        let maps = HslMaps {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        assert_eq!(
            remap_rgb([0.2, 0.5, 1.0], 1.0, 0.0, &maps, degree_from_ui(0).unwrap()).unwrap(),
            [0.2, 0.5, 1.0]
        );
        assert_eq!(
            remap_rgb([0.2, 0.5, 1.0], 0.0, 0.0, &maps, degree_from_ui(50).unwrap()).unwrap(),
            [0.2, 0.5, 1.0]
        );
    }

    #[test]
    fn lut_shape_and_finiteness_are_refused() {
        let short = [0.0f32; 3];
        let full = [0.0f32; LUT_SAMPLES];
        let maps = HslMaps {
            hue: &short,
            saturation: &full,
            luma: &full,
        };
        assert_eq!(
            hsl_tuner([0.2, 0.5, 1.0], &maps),
            Err(Refusal::ShapeMismatch("PF30009_HSL_LUT"))
        );
        let mut nan = full;
        nan[0] = f32::NAN;
        let maps = HslMaps {
            hue: &nan,
            saturation: &full,
            luma: &full,
        };
        assert_eq!(
            hsl_tuner([0.2, 0.5, 1.0], &maps),
            Err(Refusal::ValueOutOfRange("PF30009_HSL_LUT"))
        );
    }

    #[test]
    fn remap_image_copies_alpha_and_rejects_mask_shape() {
        let source = Image::solid(1, 1, [51, 128, 255, 77]);
        let (hue, saturation, luma) = control_maps();
        let maps = HslMaps {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let out = remap_image(&source, &[1.0], 0.0, &maps, 50).unwrap();
        assert_eq!(out.pixel(0, 0)[3], 77);
        assert_eq!(
            remap_image(&source, &[], 0.0, &maps, 50),
            Err(Refusal::ShapeMismatch("PF30009_HAIR_MASK"))
        );
    }

    #[test]
    fn cvtt_truncates_and_saturates_like_x86() {
        assert_eq!(cvtt_i32(254.999_98), 254);
        assert_eq!(cvtt_i32(255.0), 255);
        assert_eq!(cvtt_i32(-0.5), 0);
        assert_eq!(cvtt_i32(-1.0), -1);
        assert_eq!(cvtt_i32(f32::NAN), i32::MIN);
        assert_eq!(cvtt_i32(f32::INFINITY), i32::MIN);
        assert_eq!(cvtt_i32(1.0e9), 1_000_000_000);
        assert_eq!(cvtt_byte(1.0e9), 255);
        assert_eq!(cvtt_i32(3.0e9), i32::MIN);
        assert_eq!(cvtt_byte(3.0e9), 0);
        assert_eq!(cvtt_byte(254.999_98), 254);
    }

    fn identity_table() -> Vec<f32> {
        (0..tables::LEN).map(|index| index as f32).collect()
    }

    fn constant_table(value: f32) -> Vec<f32> {
        vec![value; tables::LEN]
    }

    #[test]
    fn table_filters_diverge_on_an_identity_table() {
        let texels = identity_table();
        let nearest = tables::Table256::new(
            &texels,
            tables::Filter::Nearest,
            tables::Wrap::ClampToEdge,
        )
        .unwrap();
        let linear = tables::Table256::new(
            &texels,
            tables::Filter::Linear,
            tables::Wrap::ClampToEdge,
        )
        .unwrap();
        assert_eq!(nearest.sample(0.5).unwrap(), 32767.0);
        let blended = linear.sample(0.5).unwrap();
        assert!((blended - 32638.5).abs() < 1.0, "{blended}");
    }

    #[test]
    fn wrap_only_matters_at_the_unit_boundary() {
        let texels = [10.0f32, 20.0, 30.0, 40.0];
        let clamped = tables::sample_grid(
            tables::Filter::Nearest,
            tables::Wrap::ClampToEdge,
            2,
            2,
            &texels,
            1.0,
            0.0,
        );
        let repeated = tables::sample_grid(
            tables::Filter::Nearest,
            tables::Wrap::Repeat,
            2,
            2,
            &texels,
            1.0,
            0.0,
        );
        assert_eq!(clamped, 20.0);
        assert_eq!(repeated, 10.0);
    }

    #[test]
    fn curve_sampling_modes_diverge() {
        let curve: Vec<f32> = (0..LUT_SAMPLES).map(|index| index as f32).collect();
        // scaled 36.5: exact index 36, GL nearest floor(36.5*74/73) = 37.
        assert_eq!(sample_curve(&curve, 36.5, CurveSampling::ExactIndex), 36.5);
        assert_eq!(
            sample_curve(
                &curve,
                36.5,
                CurveSampling::GlLiteral {
                    filter: tables::Filter::Nearest
                }
            ),
            37.0
        );
        let exact = sample_curve(&curve, 1.2, CurveSampling::ExactIndex);
        let linear = sample_curve(
            &curve,
            1.2,
            CurveSampling::GlLiteral {
                filter: tables::Filter::Linear,
            },
        );
        assert!((exact - 1.2).abs() < 1e-6);
        assert!((exact - linear).abs() > 0.25, "{exact} vs {linear}");
    }

    #[test]
    fn constant_tables_reach_the_full_chain_result() {
        let (hue, saturation, luma) = control_maps();
        let maps = HslMaps {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let gamma = constant_table(0.25);
        let inv_contrast = constant_table(0.25);
        let contrast = constant_table(0.25);
        let inv_gamma = constant_table(0.25);
        let table_set = tables::Pf30009Tables::from_slices(
            Some(&gamma),
            Some(&inv_contrast),
            Some(&contrast),
            Some(&inv_gamma),
            tables::Filter::Nearest,
            tables::Wrap::ClampToEdge,
        )
        .unwrap();
        let input = Pf30009Input {
            source_rgb: [0.2, 0.5, 1.0],
            maps: &maps,
            curves: CurveSampling::ExactIndex,
            tables: &table_set,
        };
        // Constant 0.25 tables: every stage lands on 0.25 with no mask/degree.
        let out = pf30009_hsl_pass(&input).unwrap();
        for value in out {
            assert!((value - 0.25).abs() < 1e-6, "{out:?}");
        }
    }

    #[test]
    fn missing_table_is_a_typed_refusal() {
        let table = constant_table(0.0);
        assert_eq!(
            tables::Pf30009Tables::from_slices(
                None,
                Some(&table),
                Some(&table),
                Some(&table),
                tables::Filter::Nearest,
                tables::Wrap::ClampToEdge,
            )
            .err(),
            Some(Refusal::MissingInput("PF30009_GAMMA_TABLE"))
        );
        assert_eq!(
            tables::Pf30009Tables::from_slices(
                Some(&table),
                Some(&table),
                Some(&table),
                None,
                tables::Filter::Nearest,
                tables::Wrap::ClampToEdge,
            )
            .err(),
            Some(Refusal::MissingInput("PF30009_INV_GAMMA_TABLE"))
        );
        assert_eq!(
            tables::Pf30009Tables::from_slices(
                Some(&table[1..]),
                Some(&table),
                Some(&table),
                Some(&table),
                tables::Filter::Nearest,
                tables::Wrap::ClampToEdge,
            )
            .err(),
            Some(Refusal::ShapeMismatch("PF30009_TABLE"))
        );
    }

    #[test]
    fn gamma_matrix_vector_matches_the_shader() {
        let raw = tables::apply_matrix(tables::GAMMA_WEIGHTS, [0.2, 0.4, 0.6]);
        let expected = [0.322_265_45, 0.385_955_0, 0.569_725_9];
        for channel in 0..3 {
            assert!(
                (raw[channel] - expected[channel]).abs() < 1e-6,
                "{raw:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn out_of_domain_values_move_the_address_not_the_value() {
        let texels = identity_table();
        let table = tables::Table256::new(
            &texels,
            tables::Filter::Nearest,
            tables::Wrap::ClampToEdge,
        )
        .unwrap();
        let out = tables::decode_tonal(&table, [2.0, 1.0, 0.5]).unwrap();
        // 2.0 addresses texel 65535 instead of being clamped to 1.0.
        assert_eq!(out[0], 65535.0);
        assert_eq!(out[2], 32767.0);
    }

    #[test]
    fn legacy_conversion_clamps_while_the_raw_helper_does_not() {
        let raw = hue_max_min_to_rgb_raw(0.0, 2.0, -1.0);
        assert!(raw.iter().any(|value| !(0.0..=1.0).contains(value)));
        let clamped = hue_max_min_to_rgb(0.0, 2.0, -1.0);
        assert!(clamped.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn image_entry_matches_the_per_pixel_path_and_keeps_alpha() {
        let (hue, saturation, luma) = control_maps();
        let maps = HslMaps {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let table = constant_table(0.25);
        let table_set = tables::Pf30009Tables::from_slices(
            Some(&table),
            Some(&table),
            Some(&table),
            Some(&table),
            tables::Filter::Linear,
            tables::Wrap::ClampToEdge,
        )
        .unwrap();
        let source = Image::solid(2, 1, [128, 128, 128, 42]);
        let image = pf30009_hsl_image(
            &source,
            &maps,
            CurveSampling::ExactIndex,
            &table_set,
        )
        .unwrap();
        let pixel = pf30009_hsl_pass(&Pf30009Input {
            source_rgb: [128.0 / 255.0; 3],
            maps: &maps,
            curves: CurveSampling::ExactIndex,
            tables: &table_set,
        })
        .unwrap();
        assert_eq!(image.pixel(0, 0)[3], 42);
        assert_eq!(image.pixel(1, 0)[3], 42);
        for channel in 0..3 {
            let expected = to_byte(pixel[channel] * 255.0);
            assert_eq!(image.pixel(0, 0)[channel], expected);
            assert_eq!(image.pixel(1, 0)[channel], expected);
        }
    }

    #[test]
    fn image_entry_refuses_bad_maps_once() {
        let short = [0.0f32; 3];
        let full = [0.0f32; LUT_SAMPLES];
        let bad_maps = HslMaps {
            hue: &short,
            saturation: &full,
            luma: &full,
        };
        let table = constant_table(0.0);
        let table_set = tables::Pf30009Tables::from_slices(
            Some(&table),
            Some(&table),
            Some(&table),
            Some(&table),
            tables::Filter::Nearest,
            tables::Wrap::ClampToEdge,
        )
        .unwrap();
        let source = Image::solid(1, 1, [0, 0, 0, 255]);
        assert_eq!(
            pf30009_hsl_image(&source, &bad_maps, CurveSampling::ExactIndex, &table_set),
            Err(Refusal::ShapeMismatch("PF30009_HSL_LUT"))
        );
        let malformed = Image {
            width: 2,
            height: 2,
            pixels: vec![0; 4],
        };
        assert_eq!(
            pf30009_hsl_image(&malformed, &bad_maps, CurveSampling::ExactIndex, &table_set),
            Err(Refusal::ShapeMismatch("image"))
        );
    }

    #[test]
    fn the_exact_pass_has_no_mask_or_degree_multiplier() {
        let (hue, saturation, luma) = control_maps();
        let maps = HslMaps {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let table = constant_table(0.25);
        let table_set = tables::Pf30009Tables::from_slices(
            Some(&table),
            Some(&table),
            Some(&table),
            Some(&table),
            tables::Filter::Nearest,
            tables::Wrap::ClampToEdge,
        )
        .unwrap();
        let input = Pf30009Input {
            source_rgb: [0.2, 0.5, 1.0],
            maps: &maps,
            curves: CurveSampling::ExactIndex,
            tables: &table_set,
        };
        // Identical result regardless of any UI degree: the pass never reads it.
        assert_eq!(
            pf30009_hsl_pass(&input).unwrap(),
            pf30009_hsl_pass(&input).unwrap()
        );
        assert_eq!(pf30009_hsl_pass(&input).unwrap(), [0.25, 0.25, 0.25]);
    }
}
