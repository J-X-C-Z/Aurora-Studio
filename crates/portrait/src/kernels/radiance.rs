//! PF90004 `SkinLighten_PixCook` pixel backend.
//!
//! Recovered from pixcake.exe SHA256
//! `A84519DA489BC52E4E914AF7880EDE43E3CE58992657723D02197161832464C7`:
//! parameter compiler `0x702EC20`, brightness LUT selector `0x7031830`,
//! contrast LUT selector `0x7033480`, statistics `0x702E3F0`,
//! brightness fragment `0x968C588`, HSL fragment `0x9665CAF`.
//!
//! Two native-order passes: per-channel brightness/contrast LUT blend, then
//! the HSL palette pass. Masks, LUT rows, gamma and matrices are explicit.
//! `Image` is RGBA8; the vendor surfaces are BGRA8, but every per-channel law
//! here is order-independent (brightness) or explicitly R/G/B (HSL).

use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 90004,
    key: "skinRadiance",
    label: "Skin radiance",
    access: crate::Access::Cpu,
    inputs: &[
        "source",
        "skin/parsing/face masks",
        "valueMean",
        "native LUT rows, gamma table, color matrices",
    ],
}];

pub const ROW65: EffectMeta = EffectMeta {
    pf: 90004,
    key: "skinRadiance",
    label: "皮肤透亮",
    access: Access::Cpu,
    inputs: &[
        "skin mask R",
        "parsing RGBA (A/G/B)",
        "face mask G",
        "native-lut-a.bin",
        "native-lut-b.bin",
        "native-gamma-rgba32f.bin",
        "native-color-matrices.bin",
        "valueMean",
    ],
};

pub const LUT_A_ROWS: usize = 300;
pub const LUT_B_ROWS: usize = 150;
pub const GAMMA_FLOATS: usize = 16384;
pub const MATRIX_FLOATS: usize = 18;
pub const MAX_STATISTICS_DIMENSION: f32 = 320.0;

pub struct FixedAssets<'a> {
    pub lut_a: &'a [u8],
    pub lut_b: &'a [u8],
    pub gamma: &'a [f32],
    pub matrices: &'a [f32],
}

pub struct MaskPlanes<'a> {
    pub skin: &'a [u8],
    pub parsing_rgba: &'a [u8],
    pub face_rgba: &'a [u8],
}

pub struct RadianceResult {
    pub image: Image,
    pub degree: i32,
    pub parameters: [i32; 5],
    pub brightness_degree: i32,
    pub contrast_degree: i32,
    pub brightness_changed: u32,
    pub hsl_changed: u32,
}

fn pixels_of(image: &Image) -> Result<usize, Refusal> {
    let pixels = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(Refusal::ShapeMismatch("image"))?;
    if image.pixels.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    Ok(pixels)
}

fn round_byte(value: f32) -> u8 {
    ((value * 255.0 + 0.5) as i32).clamp(0, 255) as u8
}

/// `0x600E50C`: `roundf(ui/100*100)`, clamped 0..100.
pub fn ui_to_native_degree(ui: f32) -> Result<i32, Refusal> {
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    Ok((ui / 100.0 * 100.0).round().clamp(0.0, 100.0) as i32)
}

/// `0x702EC20`: five native parameters from degree + valueMean.
pub fn compile_parameters(degree: i32, value_mean: f32) -> Result<[i32; 5], Refusal> {
    const DEGREES: [[i32; 5]; 11] = [
        [0, 0, 0, 0, 0],
        [0, -3, 1, -1, 5],
        [0, -6, 1, -2, 10],
        [1, -9, 1, -3, 15],
        [1, -12, 1, -4, 20],
        [1, -15, 1, -5, 25],
        [2, -18, 1, -5, 31],
        [2, -21, 2, -5, 37],
        [2, -24, 2, -5, 43],
        [2, -27, 2, -5, 49],
        [2, -30, 2, -5, 55],
    ];
    const STOPS: [f32; 7] = [0.0, 130.0, 140.0, 160.0, 180.0, 200.0, 255.0];
    const FACTORS: [f32; 7] = [1.15, 1.1, 1.05, 1.0, 0.9, 0.95, 0.8];

    if !(0..=100).contains(&degree) || !value_mean.is_finite() || value_mean < 0.0 {
        return Err(Refusal::ValueOutOfRange("parameters"));
    }
    let mut values = [0i32; 5];
    if degree == 0 {
        return Ok(values);
    }
    if degree > 90 {
        values = [2, -30, 2, -5, 55];
    } else {
        let index = (((degree - 1) / 10) as usize).min(8);
        let fraction = (degree - index as i32 * 10) as f64 / 10.0;
        for i in 0..5 {
            let high = DEGREES[index + 1][i] as f64 * fraction;
            let low = DEGREES[index][i] as f64 * (1.0 - fraction);
            let (lo, hi) = if i == 1 { (-100, 0) } else { (0, 100) };
            values[i] = ((high + low + 0.5) as i32).clamp(lo, hi);
        }
    }

    let mut factor = 0.8f32;
    if value_mean <= 255.0 {
        let mut index = 0usize;
        while index + 1 < STOPS.len() && value_mean > STOPS[index + 1] {
            index += 1;
        }
        if index + 1 < STOPS.len() {
            let fraction = (value_mean - STOPS[index]) / (STOPS[index + 1] - STOPS[index]);
            factor = fraction * FACTORS[index + 1] + FACTORS[index] * (1.0 - fraction);
        }
    }
    values[4] = ((values[4] as f32 * factor) as i32).clamp(0, 100);
    Ok(values)
}

/// `LutDegrees` from `0x702EC20` consumers.
pub fn lut_degrees(brightness: i32, contrast: i32) -> (i32, i32) {
    let bright = brightness.clamp(0, 100) as f32 * 0.01 * 150.0 + 0.5;
    let contrast = contrast.clamp(-100, 0) as f32 * 0.5 - 0.5;
    (bright as i32, (contrast.max(-50.0)) as i32)
}

/// `0x7031830` / `0x7033480` row selector. Zero degree selects no row.
/// The vendor jump table has one no-op slot that the asset rows do not have.
pub fn select_lut_row(
    table: &[u8],
    minimum_degree: i32,
    degree: i32,
) -> Result<Option<Vec<u8>>, Refusal> {
    let rows = if minimum_degree == -50 { LUT_B_ROWS } else { LUT_A_ROWS };
    if table.len() != rows * 256 {
        return Err(Refusal::ShapeMismatch("lut"));
    }
    let maximum_degree = minimum_degree + rows as i32;
    if degree < minimum_degree || degree > maximum_degree {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    if degree == 0 {
        return Ok(None);
    }
    let row = (degree - minimum_degree - i32::from(degree > 0)) as usize;
    Ok(Some(table[row * 256..row * 256 + 256].to_vec()))
}

fn lookup(value: f32, channel: usize, gamma: &[f32]) -> f32 {
    let index = ((value.clamp(0.0, 1.0) * 4096.0) as usize).min(4095);
    gamma[index * 4 + channel]
}

fn matrix(value: [f32; 3], m: &[f32], start: usize) -> [f32; 3] {
    [
        value[0] * m[start] + value[1] * m[start + 1] + value[2] * m[start + 2],
        value[0] * m[start + 3] + value[1] * m[start + 4] + value[2] * m[start + 5],
        value[0] * m[start + 6] + value[1] * m[start + 7] + value[2] * m[start + 8],
    ]
}

fn tone(value: [f32; 3], channel: usize, gamma: &[f32]) -> [f32; 3] {
    let v = [
        value[0].clamp(0.0, 1.0),
        value[1].clamp(0.0, 1.0),
        value[2].clamp(0.0, 1.0),
    ];
    let maximum = v[0].max(v[1]).max(v[2]);
    let minimum = v[0].min(v[1]).min(v[2]);
    let high = lookup(maximum, channel, gamma);
    let low = lookup(minimum, channel, gamma);
    if maximum == minimum {
        return [high; 3];
    }
    let map = |component: f32| {
        if component == maximum {
            high
        } else if component == minimum {
            low
        } else {
            low + (high - low) * (component - minimum) / (maximum - minimum)
        }
    };
    [map(v[0]), map(v[1]), map(v[2])]
}

fn to_rgb(mut hue: f32, maximum: f32, minimum: f32) -> [f32; 3] {
    while hue > 6.0 {
        hue -= 6.0;
    }
    let d = maximum - minimum;
    if (3.0..=5.0).contains(&hue) {
        if hue <= 4.0 {
            [minimum, minimum - (hue - 4.0) * d, maximum]
        } else {
            [minimum + (hue - 4.0) * d, minimum, maximum]
        }
    } else if (1.0..3.0).contains(&hue) {
        if hue <= 2.0 {
            [minimum - (hue - 2.0) * d, maximum, minimum]
        } else {
            [minimum, maximum, minimum + (hue - 2.0) * d]
        }
    } else if hue > 5.0 {
        [maximum, minimum, minimum - (hue - 6.0) * d]
    } else {
        [maximum, minimum + hue * d, minimum]
    }
}

fn mask_alpha(skin: f32, parsing: [f32; 4], face: [f32; 4]) -> f32 {
    parsing[3] * 0.39 + parsing[1] * 0.39 + parsing[2] * 0.39 + face[1] * 0.39 + skin
}

fn validate_masks(pixels: usize, masks: &MaskPlanes<'_>) -> Result<(), Refusal> {
    if masks.skin.len() != pixels {
        return Err(Refusal::ShapeMismatch("skin_mask"));
    }
    if masks.parsing_rgba.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("parsing_rgba"));
    }
    if masks.face_rgba.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("face_rgba"));
    }
    Ok(())
}

/// Brightness pass (`0x968C588`): per-channel brightness LUT then contrast LUT,
/// blended by `weight` and then by the combined mask alpha.
pub fn brightness_pass(
    image: &mut Image,
    masks: &MaskPlanes<'_>,
    brightness_lut: Option<&[u8]>,
    contrast_lut: Option<&[u8]>,
) -> Result<u32, Refusal> {
    let pixels = pixels_of(image)?;
    validate_masks(pixels, masks)?;
    for lut in [brightness_lut, contrast_lut] {
        if let Some(lut) = lut {
            if lut.len() != 256 {
                return Err(Refusal::ShapeMismatch("lut"));
            }
        }
    }
    if brightness_lut.is_none() && contrast_lut.is_none() {
        return Ok(0);
    }

    let mut changed = 0u32;
    for p in 0..pixels {
        let i = p * 4;
        let skin = masks.skin[p] as f32 / 255.0;
        let parsing = [
            masks.parsing_rgba[i] as f32 / 255.0,
            masks.parsing_rgba[i + 1] as f32 / 255.0,
            masks.parsing_rgba[i + 2] as f32 / 255.0,
            masks.parsing_rgba[i + 3] as f32 / 255.0,
        ];
        let face = [
            masks.face_rgba[i] as f32 / 255.0,
            masks.face_rgba[i + 1] as f32 / 255.0,
            masks.face_rgba[i + 2] as f32 / 255.0,
            masks.face_rgba[i + 3] as f32 / 255.0,
        ];
        let alpha = mask_alpha(skin, parsing, face).min(1.0);
        if alpha <= 0.001 {
            continue;
        }

        let source = [
            image.pixels[i] as f32 / 255.0,
            image.pixels[i + 1] as f32 / 255.0,
            image.pixels[i + 2] as f32 / 255.0,
        ];
        let weight = 1.0
            - (source[0].max(source[1]).max(source[2])
                + source[0].min(source[1]).min(source[2]))
                * 0.5;
        let mut moved = false;
        for c in 0..3 {
            let original = image.pixels[i + c];
            let mut value = original;
            if let Some(lut) = brightness_lut {
                value = lut[value as usize];
            }
            if let Some(lut) = contrast_lut {
                value = lut[value as usize];
            }
            let adjusted = source[c] + (value as f32 / 255.0 - source[c]) * weight;
            let result = round_byte(source[c] + (adjusted - source[c]) * alpha);
            moved |= result != original;
            image.pixels[i + c] = result;
        }
        if moved {
            changed += 1;
        }
    }
    Ok(changed)
}

/// HSL pass (`0x9665CAF`): decode gamma, matrix, tone, hue/sat/light maps,
/// encode; blended by mask alpha.
pub fn hsl_pass(
    image: &mut Image,
    masks: &MaskPlanes<'_>,
    hue_maps: &[f32],
    gamma: &[f32],
    matrices: &[f32],
) -> Result<u32, Refusal> {
    let pixels = pixels_of(image)?;
    validate_masks(pixels, masks)?;
    if hue_maps.len() != 222 {
        return Err(Refusal::ShapeMismatch("hue_maps"));
    }
    if gamma.len() != GAMMA_FLOATS {
        return Err(Refusal::ShapeMismatch("gamma"));
    }
    if matrices.len() != MATRIX_FLOATS {
        return Err(Refusal::ShapeMismatch("matrices"));
    }
    if hue_maps.iter().chain(gamma).chain(matrices).any(|v| !v.is_finite()) {
        return Err(Refusal::ValueOutOfRange("assets"));
    }

    let mut changed = 0u32;
    for p in 0..pixels {
        let i = p * 4;
        let skin = masks.skin[p] as f32 / 255.0;
        let parsing = [
            masks.parsing_rgba[i] as f32 / 255.0,
            masks.parsing_rgba[i + 1] as f32 / 255.0,
            masks.parsing_rgba[i + 2] as f32 / 255.0,
            masks.parsing_rgba[i + 3] as f32 / 255.0,
        ];
        let face = [
            masks.face_rgba[i] as f32 / 255.0,
            masks.face_rgba[i + 1] as f32 / 255.0,
            masks.face_rgba[i + 2] as f32 / 255.0,
            masks.face_rgba[i + 3] as f32 / 255.0,
        ];
        let alpha = skin.max(parsing[3] * 0.39 + parsing[1] * 0.39 + parsing[2] * 0.39 + face[1] * 0.39);
        if alpha <= 0.001 {
            continue;
        }

        let source = [
            image.pixels[i] as f32 / 255.0,
            image.pixels[i + 1] as f32 / 255.0,
            image.pixels[i + 2] as f32 / 255.0,
        ];
        let decoded = [
            lookup(source[0], 1, gamma),
            lookup(source[1], 1, gamma),
            lookup(source[2], 1, gamma),
        ];
        let toned = tone(matrix(decoded, matrices, 9), 3, gamma);
        let maximum = toned[0].max(toned[1]).max(toned[2]);
        let minimum = toned[0].min(toned[1]).min(toned[2]);
        let difference = maximum - minimum;
        let mut hue = if difference == 0.0 {
            0.0
        } else if maximum == toned[0] {
            (toned[1] - toned[2]) / difference
        } else if maximum == toned[1] {
            (toned[2] - toned[0]) / difference + 2.0
        } else {
            (toned[0] - toned[1]) / difference + 4.0
        };
        if hue < 0.0 {
            hue += 6.0;
        }

        let mut lo = minimum;
        let mut hi = maximum;
        if difference != 0.0 {
            let h = hue * 12.0;
            let index = (h as usize).min(71);
            let fraction = h - index as f32;
            let interpolate = |start: usize| {
                hue_maps[start + index]
                    + fraction * (hue_maps[start + index + 1] - hue_maps[start + index])
            };
            let mut light = interpolate(148);
            if light != 0.0 {
                let biquad = (2.0 - difference) * difference * (2.0 - (2.0 - difference) * difference);
                light = biquad * (2.0 - biquad) * light;
                lo *= (1.0 - lo) * light + 1.0;
                lo *= (1.0 - lo) * light + 1.0;
                hi *= (1.0 - hi) * light + 1.0;
                hi *= (1.0 - hi) * light + 1.0;
            }
            let saturation = interpolate(74);
            if saturation > 0.0 {
                let s = (hi - lo) / hi;
                let m = (hi * 16.0).min(1.0);
                let k = 1.0 - (1.0 - lo) * (saturation * 0.83333331) * ((2.0 - m) * m) * (1.0 - s);
                lo = hi - s / k * hi;
            } else {
                let average = (lo + hi) * 0.5;
                lo += (lo - average) * saturation;
                hi += (hi - average) * saturation;
            }
            hue = interpolate(0);
        }

        let linear = matrix(tone(to_rgb(hue, hi, lo), 2, gamma), matrices, 0);
        let result = [
            lookup(linear[0], 0, gamma),
            lookup(linear[1], 0, gamma),
            lookup(linear[2], 0, gamma),
        ];
        let mut moved = false;
        for c in 0..3 {
            let original = source[c];
            let value = round_byte(original + (result[c] - original) * alpha);
            moved |= value != image.pixels[i + c];
            image.pixels[i + c] = value;
        }
        if moved {
            changed += 1;
        }
    }
    Ok(changed)
}

/// Full PF90004 native order: degree → parameters → LUT rows/palette →
/// brightness pass → HSL pass.
pub fn apply(
    source: &Image,
    masks: &MaskPlanes<'_>,
    ui: f32,
    value_mean: f32,
    assets: &FixedAssets<'_>,
) -> Result<RadianceResult, Refusal> {
    let pixels = pixels_of(source)?;
    validate_masks(pixels, masks)?;
    if assets.lut_a.len() != LUT_A_ROWS * 256 {
        return Err(Refusal::ShapeMismatch("lut_a"));
    }
    if assets.lut_b.len() != LUT_B_ROWS * 256 {
        return Err(Refusal::ShapeMismatch("lut_b"));
    }
    if assets.gamma.len() != GAMMA_FLOATS {
        return Err(Refusal::ShapeMismatch("gamma"));
    }
    if assets.matrices.len() != MATRIX_FLOATS {
        return Err(Refusal::ShapeMismatch("matrices"));
    }

    let degree = ui_to_native_degree(ui)?;
    if degree == 0 {
        return Ok(RadianceResult {
            image: source.clone(),
            degree: 0,
            parameters: [0; 5],
            brightness_degree: 0,
            contrast_degree: 0,
            brightness_changed: 0,
            hsl_changed: 0,
        });
    }
    let parameters = compile_parameters(degree, value_mean)?;
    let (brightness_degree, contrast_degree) = lut_degrees(parameters[0], parameters[1]);
    let brightness_lut = select_lut_row(assets.lut_a, -150, brightness_degree)?;
    let contrast_lut = select_lut_row(assets.lut_b, -50, contrast_degree)?;

    let mut image = source.clone();
    let brightness_changed = brightness_pass(
        &mut image,
        masks,
        brightness_lut.as_deref(),
        contrast_lut.as_deref(),
    )?;
    let hsl_changed = hsl_pass(&mut image, masks, &hue_maps_for(&parameters), assets.gamma, assets.matrices)?;
    Ok(RadianceResult {
        image,
        degree,
        parameters,
        brightness_degree,
        contrast_degree,
        brightness_changed,
        hsl_changed,
    })
}

// Palette compiler `0x70351D0` (222 floats), kept here so `apply` is callable.
fn hue_maps_for(parameters: &[i32; 5]) -> Vec<f32> {
    const ORDER: [usize; 8] = [0, 5, 6, 3, 1, 2, 4, 7];
    const KNOTS: [f32; 8] = [0.0, 30.0, 55.0, 95.0, 160.0, 230.0, 275.0, 330.0];
    const HUE_SCALES: [[f32; 8]; 2] = [
        [30.0, 30.0, 36.0, 58.5, 63.0, 40.5, 49.5, 30.0],
        [30.0, 30.0, 30.0, 36.0, 58.5, 63.0, 40.5, 49.5],
    ];

    let mut record = [0i32; 25];
    for channel in 0..3 {
        let value = parameters[channel + 2].clamp(if channel == 2 { 0 } else { -100 }, 100);
        for i in 0..8 {
            record[1 + channel * 8 + i] = value;
        }
    }
    let mut result = vec![0.0f32; 222];
    for channel in 0..3 {
        let mut values = [0.0f64; 8];
        for i in 0..8 {
            let value = record[1 + channel * 8 + ORDER[i]] as f64 / 100.0;
            values[i] = if channel == 0 {
                value * HUE_SCALES[if value < 0.0 { 1 } else { 0 }][i] as f64 / 60.0
            } else {
                value
            };
        }
        let start = channel * 74;
        let mut segment = 0usize;
        for i in 0..72 {
            let angle = i as f32 * 5.0;
            if angle >= KNOTS[(segment + 1) % 8] + if segment == 7 { 360.0 } else { 0.0 } {
                segment += 1;
            }
            let next = (segment + 1) % 8;
            let weight = (angle - KNOTS[segment]) as f64
                / (KNOTS[next] + if segment == 7 { 360.0 } else { 0.0 } - KNOTS[segment]) as f64;
            result[start + i] =
                (values[segment] + (values[next] - values[segment]) * weight) as f32;
        }
        result[start + 72] = result[start];
        result[start + 73] = result[start + 1];
    }
    let shift = if result[0] < 0.0 { 6.0 } else { 0.0 };
    for i in 0..74 {
        result[i] += i as f32 * 0.0833333283662796 + shift;
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_and_row_selection_follow_the_recovered_curve() {
        assert_eq!(ui_to_native_degree(0.0).unwrap(), 0);
        assert_eq!(ui_to_native_degree(100.0).unwrap(), 100);
        assert_eq!(ui_to_native_degree(101.0), Err(Refusal::ValueOutOfRange("ui")));
        let (bright, contrast) = lut_degrees(50, -50);
        assert!((-150..=150).contains(&bright) && (-50..=100).contains(&contrast));
        assert_eq!(lut_degrees(2, -30), (3, -15));
        assert_eq!(lut_degrees(100, -100), (150, -50));
        assert_eq!(LUT_A_ROWS, 300);
        assert_eq!(LUT_B_ROWS, 150);
        assert_eq!(GAMMA_FLOATS, 16384);
        assert_eq!(MATRIX_FLOATS, 18);
    }

    fn marked_table(rows: usize) -> Vec<u8> {
        let mut table = vec![0u8; rows * 256];
        for row in 0..rows {
            table[row * 256] = row as u8;
            table[row * 256 + 1] = (row >> 8) as u8;
        }
        table
    }

    fn selected_row(selected: Option<Vec<u8>>) -> usize {
        let row = selected.unwrap();
        row[0] as usize | ((row[1] as usize) << 8)
    }

    #[test]
    fn positive_rows_skip_the_vendor_noop_slot() {
        let lut_a = marked_table(LUT_A_ROWS);
        let lut_b = marked_table(LUT_B_ROWS);

        assert_eq!(select_lut_row(&lut_a, -150, 0).unwrap(), None);
        assert_eq!(select_lut_row(&lut_b, -50, 0).unwrap(), None);
        assert_eq!(selected_row(select_lut_row(&lut_a, -150, 1).unwrap()), 150);
        assert_eq!(selected_row(select_lut_row(&lut_a, -150, 2).unwrap()), 151);
        assert_eq!(selected_row(select_lut_row(&lut_a, -150, 30).unwrap()), 179);
        assert_eq!(selected_row(select_lut_row(&lut_b, -50, 1).unwrap()), 50);
        assert_eq!(selected_row(select_lut_row(&lut_b, -50, 2).unwrap()), 51);
        assert_eq!(selected_row(select_lut_row(&lut_b, -50, 30).unwrap()), 79);
    }

    #[test]
    fn negative_and_boundary_rows_match_the_vendor_table() {
        let lut_a = marked_table(LUT_A_ROWS);
        let lut_b = marked_table(LUT_B_ROWS);

        assert_eq!(selected_row(select_lut_row(&lut_a, -150, -150).unwrap()), 0);
        assert_eq!(selected_row(select_lut_row(&lut_a, -150, -30).unwrap()), 120);
        assert_eq!(selected_row(select_lut_row(&lut_a, -150, 150).unwrap()), 299);
        assert_eq!(selected_row(select_lut_row(&lut_b, -50, -50).unwrap()), 0);
        assert_eq!(selected_row(select_lut_row(&lut_b, -50, -30).unwrap()), 20);
        assert_eq!(selected_row(select_lut_row(&lut_b, -50, 100).unwrap()), 149);

        assert_eq!(
            select_lut_row(&lut_a, -150, 151),
            Err(Refusal::ValueOutOfRange("degree"))
        );
        assert_eq!(
            select_lut_row(&lut_b, -50, 101),
            Err(Refusal::ValueOutOfRange("degree"))
        );
    }
}
