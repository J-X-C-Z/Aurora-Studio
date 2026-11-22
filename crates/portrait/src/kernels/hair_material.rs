//! `HSLTuner.frag` material stage: gamma/contrast maps, two 3x3 matrices and
//! the PF30009 HSL curve generator.
//!
//! Transcribed from `extract/pixcook/glsl/HSLTuner/HSLTuner.frag`
//! (`c48d82af...b02b080`) and the recovered R68/R69/R72 producers. Channel
//! order inside the shader is `.r/.g/.b`; the minified corpus variant uses the
//! equivalent `.s/.t/.p` swizzle. `Image` is RGBA, so R=pixels[0] and a BGRA
//! source must be converted before calling the image entry point.
//!
//! No 256x256 texture data and no eight-control curve data was recovered.
//! Both stay explicit inputs: missing tables are a typed refusal, never an
//! identity/default map.
//!
//! This is a pipeline stage, not a catalog row, so it exports no `META`
//! (same rule as `mask_compose`).

use crate::{Image, Refusal};

/// Forward gamma->raw matrix, row-major, output row = input channels
/// `[gamma_r, gamma_g, gamma_b]`.
pub const FORWARD_MATRIX: [[f32; 3]; 3] = [
    [0.529_258_31, 0.330_156_21, 0.140_585_48],
    [0.098_386_407, 0.873_452_07, 0.028_161_526],
    [0.016_864_598, 0.117_641_75, 0.865_493_66],
];

/// Inverse matrix applied after the contrast stage.
pub const INVERSE_MATRIX: [[f32; 3]; 3] = [
    [2.034_487_2, -0.727_695_41, -0.306_791_79],
    [-0.228_891_37, 1.231_791_83, -0.002_900_47],
    [-0.008_531_12, -0.153_251_1, 1.161_782_21],
];

/// Recovered PF30009 curve anchors in degrees (R72 `ProductAnchors8`).
pub const PRODUCT_ANCHORS_8: [f32; 8] = [0.0, 30.0, 55.0, 95.0, 160.0, 230.0, 275.0, 330.0];
pub const CURVE_SAMPLES: usize = 74;
pub const CURVE_LOOP_SAMPLES: usize = 72;
pub const CURVE_STEP_DEGREES: f32 = 5.0;

pub fn apply_matrix(matrix: &[[f32; 3]; 3], rgb: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for row in 0..3 {
        out[row] = matrix[row][0] * rgb[0] + matrix[row][1] * rgb[1] + matrix[row][2] * rgb[2];
    }
    out
}

/// The shader's 16-bit split: `hi=floor(cur/256)`, `lo=cur-256*hi`, both
/// clamped to `[0,256]` then divided by 256.
pub fn lut_coord_16(value: f32) -> (f32, f32) {
    let current = value * 65535.0;
    let high = ((current / 256.0).floor().min(256.0)).max(0.0);
    let y = high / 256.0;
    let low = (current.floor() - high * 256.0).min(256.0).max(0.0);
    (low / 256.0, y)
}

/// GL_LINEAR sampling of a 1-D row at normalized `u` (`u*width - 0.5`).
pub fn sample_row(table: &[f32], width: usize, u: f32) -> Result<f32, Refusal> {
    if width == 0 || table.len() < width {
        return Err(Refusal::ShapeMismatch("HSLTUNER_ROW"));
    }
    if table[..width].iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("HSLTUNER_ROW"));
    }
    let position = u * width as f32 - 0.5;
    let low = position.floor();
    let frac = position - low;
    let index0 = (low as i64).clamp(0, width as i64 - 1) as usize;
    let index1 = ((low as i64) + 1).clamp(0, width as i64 - 1) as usize;
    let a = table[index0];
    let b = table[index1];
    Ok(a + (b - a) * frac)
}

fn sample_bilinear_2d(table: &[f32], width: usize, height: usize, u: f32, v: f32) -> f32 {
    let px = u * width as f32 - 0.5;
    let py = v * height as f32 - 0.5;
    let x0f = px.floor();
    let y0f = py.floor();
    let tx = px - x0f;
    let ty = py - y0f;
    let x0 = (x0f as i64).clamp(0, width as i64 - 1) as usize;
    let x1 = ((x0f as i64) + 1).clamp(0, width as i64 - 1) as usize;
    let y0 = (y0f as i64).clamp(0, height as i64 - 1) as usize;
    let y1 = ((y0f as i64) + 1).clamp(0, height as i64 - 1) as usize;
    let at = |x: usize, y: usize| table[y * width + x];
    let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
    let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
    top + (bottom - top) * ty
}

/// 256x256 map lookup by a 0..1 value, through the shader's coordinate split.
pub fn sample_map_256(table: &[f32], value: f32) -> Result<f32, Refusal> {
    if table.len() != 256 * 256 {
        return Err(Refusal::ShapeMismatch("HSLTUNER_MAP_256"));
    }
    if !value.is_finite() {
        return Err(Refusal::ValueOutOfRange("HSLTUNER_MAP_256"));
    }
    let (x, y) = lut_coord_16(value);
    Ok(sample_bilinear_2d(table, 256, 256, x, y))
}

/// Explicit material surfaces. `None` is a refusal, never an identity map.
pub struct HsltunerTables<'a> {
    pub gamma_row: Option<&'a [f32]>,
    pub gamma_width: usize,
    pub inverse_contrast: Option<&'a [f32]>,
    pub contrast: Option<&'a [f32]>,
    pub inverse_gamma_row: Option<&'a [f32]>,
    pub inverse_gamma_width: usize,
}

/// The three 74-sample curves fed to `hsltuner`.
pub struct HsltunerCurves<'a> {
    pub hue: &'a [f32],
    pub saturation: &'a [f32],
    pub luma: &'a [f32],
}

fn curves_valid(curves: &HsltunerCurves<'_>) -> Result<(), Refusal> {
    for curve in [curves.hue, curves.saturation, curves.luma] {
        if curve.len() != CURVE_SAMPLES {
            return Err(Refusal::ShapeMismatch("HSLTUNER_CURVES"));
        }
        if curve.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("HSLTUNER_CURVES"));
        }
    }
    Ok(())
}

/// `hsltuner()` LUT access: texel `i` sits at `i/73`, so the index is
/// `floor(hue*12)`. The recovered executable uses this convention; see
/// UNCLOSED for the GL sampler-state question.
fn lerp_curve(curve: &[f32], scaled_hue: f32) -> f32 {
    let index = (scaled_hue.floor() as i64).clamp(0, CURVE_SAMPLES as i64 - 2) as usize;
    let frac = scaled_hue - index as f32;
    curve[index] + (curve[index + 1] - curve[index]) * frac
}

/// `CalMiddleMapRes`: keep the middle channel proportional inside
/// `[map_min, map_max]`, flipping direction when the map is descending.
fn cal_middle_map(
    maximum: f32,
    middle: f32,
    minimum: f32,
    map_maximum: f32,
    map_minimum: f32,
) -> f32 {
    let diff = maximum - minimum;
    if map_maximum > map_minimum {
        map_minimum + (middle - minimum) * (map_maximum - map_minimum) / diff
    } else {
        map_maximum + (middle - maximum) * (map_minimum - map_maximum) / diff
    }
}

/// `rgb2hsl` from the shader: sort booleans plus the 2/4/6 hue complements.
pub fn shader_rgb2hsl(rgb: [f32; 3]) -> [f32; 3] {
    let mut minimum = rgb[0];
    let mut middle = rgb[1];
    let mut maximum = rgb[2];

    let greater1;
    if maximum <= middle {
        greater1 = false;
        maximum = rgb[1];
        middle = rgb[2];
    } else {
        greater1 = true;
    }
    let greater2;
    if middle <= minimum {
        greater2 = false;
        core::mem::swap(&mut middle, &mut minimum);
    } else {
        greater2 = true;
    }
    let greater3;
    if maximum <= middle {
        greater3 = false;
        core::mem::swap(&mut maximum, &mut middle);
    } else {
        greater3 = true;
    }

    let mut hue = if maximum == minimum {
        0.0
    } else {
        (middle - minimum) / (maximum - minimum)
    };
    if greater3 {
        hue = 2.0 - hue;
    }
    if greater2 {
        hue = 4.0 - hue;
    }
    if greater1 {
        hue = 6.0 - hue;
    }
    [hue, maximum, minimum]
}

/// `hsl2rgb` from the shader: three halving comparisons and swaps.
pub fn shader_hsl2rgb(hsl: [f32; 3]) -> [f32; 3] {
    let mut minimum = hsl[2];
    let mut maximum = hsl[1];
    let mut hue = hsl[0];

    if hue > 6.0 {
        hue -= 6.0;
        if hue > 6.0 {
            hue -= 6.0;
        }
    }
    let greater1;
    if hue <= 3.0 {
        greater1 = false;
    } else {
        greater1 = true;
        hue = 6.0 - hue;
    }
    let greater2;
    if hue <= 2.0 {
        greater2 = false;
    } else {
        greater2 = true;
        hue = 4.0 - hue;
    }
    let greater3;
    if hue <= 1.0 {
        greater3 = false;
    } else {
        greater3 = true;
        hue = 2.0 - hue;
    }
    let mut middle = hue * (maximum - minimum) + minimum;
    if greater3 {
        core::mem::swap(&mut middle, &mut maximum);
    }
    if greater2 {
        core::mem::swap(&mut minimum, &mut maximum);
    }
    if greater1 {
        core::mem::swap(&mut middle, &mut minimum);
    }
    [maximum, middle, minimum]
}

/// `hsltuner()`: curve interpolation, luma quartic and both saturation
/// branches, exactly as written in the shader.
pub fn hsltuner(hsl: [f32; 3], curves: &HsltunerCurves<'_>) -> Result<[f32; 3], Refusal> {
    curves_valid(curves)?;
    let mut minimum = hsl[2];
    let mut maximum = hsl[1];
    let diff = maximum - minimum;
    if diff == 0.0 {
        return Ok(hsl);
    }

    let scaled_hue = hsl[0] * 12.0;
    let hue_out = lerp_curve(curves.hue, scaled_hue);
    let mut luma_out = lerp_curve(curves.luma, scaled_hue);
    let sat_out = lerp_curve(curves.saturation, scaled_hue);

    if luma_out != 0.0 {
        let biquadratic = (2.0 - diff) * diff * (2.0 - (2.0 - diff) * diff);
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
            let denominator = 1.0 - (1.0 - minimum) * (sat_out * 0.833_333_31) * tail;
            minimum = maximum - (saturation / denominator) * maximum;
        }
    } else {
        let average = (minimum + maximum) * 0.5;
        minimum += (minimum - average) * sat_out;
        maximum += (maximum - average) * sat_out;
    }
    Ok([hue_out, maximum, minimum])
}

fn map_branch(raw: [f32; 3], table: &[f32]) -> Result<[f32; 3], Refusal> {
    let maximum = raw[0].max(raw[1]).max(raw[2]);
    let minimum = raw[0].min(raw[1]).min(raw[2]);
    if maximum == minimum {
        let value = sample_map_256(table, minimum)?;
        return Ok([value, value, value]);
    }
    let map_maximum = sample_map_256(table, maximum)?;
    let map_minimum = sample_map_256(table, minimum)?;
    let mut out = [0.0f32; 3];
    if maximum == raw[0] {
        out[0] = map_maximum;
        if minimum == raw[1] {
            out[2] = cal_middle_map(maximum, raw[2], minimum, map_maximum, map_minimum);
            out[1] = map_minimum;
        } else {
            out[1] = cal_middle_map(maximum, raw[1], minimum, map_maximum, map_minimum);
            out[2] = map_minimum;
        }
    } else if maximum == raw[1] {
        out[1] = map_maximum;
        if minimum == raw[0] {
            out[2] = cal_middle_map(maximum, raw[2], minimum, map_maximum, map_minimum);
            out[0] = map_minimum;
        } else {
            out[0] = cal_middle_map(maximum, raw[0], minimum, map_maximum, map_minimum);
            out[2] = map_minimum;
        }
    } else {
        out[2] = map_maximum;
        if minimum == raw[1] {
            out[0] = cal_middle_map(maximum, raw[0], minimum, map_maximum, map_minimum);
            out[1] = map_minimum;
        } else {
            out[1] = cal_middle_map(maximum, raw[1], minimum, map_maximum, map_minimum);
            out[0] = map_minimum;
        }
    }
    Ok(out)
}

/// Full shader chain: gamma row -> forward matrix -> inverse-contrast map ->
/// HSLTuner -> contrast map -> inverse matrix -> inverse-gamma row.
pub fn run_pipeline(
    rgb: [f32; 3],
    tables: &HsltunerTables<'_>,
    curves: &HsltunerCurves<'_>,
) -> Result<[f32; 3], Refusal> {
    let gamma = tables
        .gamma_row
        .ok_or(Refusal::MissingInput("HSLTUNER_GAMMA_MAP"))?;
    let inverse_contrast = tables
        .inverse_contrast
        .ok_or(Refusal::MissingInput("HSLTUNER_INVERSE_CONTRAST"))?;
    let contrast = tables
        .contrast
        .ok_or(Refusal::MissingInput("HSLTUNER_CONTRAST"))?;
    let inverse_gamma = tables
        .inverse_gamma_row
        .ok_or(Refusal::MissingInput("HSLTUNER_INVERSE_GAMMA"))?;
    curves_valid(curves)?;
    if rgb.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("HSLTUNER_RGB"));
    }

    let gamma_rgb = [
        sample_row(gamma, tables.gamma_width, rgb[0])?,
        sample_row(gamma, tables.gamma_width, rgb[1])?,
        sample_row(gamma, tables.gamma_width, rgb[2])?,
    ];
    let raw = apply_matrix(&FORWARD_MATRIX, gamma_rgb);
    let mapped = map_branch(raw, inverse_contrast)?;

    let tuned = hsltuner(shader_rgb2hsl(mapped), curves)?;
    let stage = shader_hsl2rgb(tuned);
    let contrasted = map_branch(stage, contrast)?;
    let inverse_raw = apply_matrix(&INVERSE_MATRIX, contrasted);

    Ok([
        sample_row(inverse_gamma, tables.inverse_gamma_width, inverse_raw[0])?,
        sample_row(inverse_gamma, tables.inverse_gamma_width, inverse_raw[1])?,
        sample_row(inverse_gamma, tables.inverse_gamma_width, inverse_raw[2])?,
    ])
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// RGBA8 entry point. Alpha is copied; the shader only writes RGB.
pub fn run_pipeline_image(
    source: &Image,
    tables: &HsltunerTables<'_>,
    curves: &HsltunerCurves<'_>,
) -> Result<Image, Refusal> {
    let expected = (source.width as usize)
        .checked_mul(source.height as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or(Refusal::ShapeMismatch("image"))?;
    if source.width == 0 || source.height == 0 || source.pixels.len() != expected {
        return Err(Refusal::ShapeMismatch("image"));
    }
    let mut out = source.clone();
    for index in 0..expected / 4 {
        let offset = index * 4;
        let rgb = [
            source.pixels[offset] as f32 / 255.0,
            source.pixels[offset + 1] as f32 / 255.0,
            source.pixels[offset + 2] as f32 / 255.0,
        ];
        let result = run_pipeline(rgb, tables, curves)?;
        for channel in 0..3 {
            out.pixels[offset + channel] = to_byte(result[channel]);
        }
        out.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(out)
}

/// The PF30009 curve generator (`0x4CEFA50` / `0x5D36850`): eight anchors in
/// degrees, eight controls, 72 samples at 5 degrees, two wrap copies, and a
/// `+i/12` ramp for the hue curve.
pub fn generate_curve_map(
    anchors8: [f32; 8],
    controls8: [f32; 8],
    hue_ramp: bool,
) -> Result<[f32; CURVE_SAMPLES], Refusal> {
    if anchors8.iter().chain(controls8.iter()).any(|v| !v.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF30009_CURVE_CONTROLS"));
    }
    for index in 1..8 {
        if anchors8[index] <= anchors8[index - 1] {
            return Err(Refusal::ValueOutOfRange("PF30009_CURVE_ANCHORS"));
        }
    }

    let mut result = [0.0f32; CURVE_SAMPLES];
    let mut segment = 0usize;
    for sample in 0..CURVE_LOOP_SAMPLES {
        let angle = sample as f32 * CURVE_STEP_DEGREES;
        let mut start = anchors8[segment];
        let mut end = if segment == 7 {
            anchors8[0] + 360.0
        } else {
            anchors8[segment + 1]
        };
        while angle >= end && segment < 7 {
            segment += 1;
            start = anchors8[segment];
            end = if segment == 7 {
                anchors8[0] + 360.0
            } else {
                anchors8[segment + 1]
            };
        }
        let next = (segment + 1) & 7;
        let t = (angle - start) / (end - start);
        result[sample] = controls8[segment] + (controls8[next] - controls8[segment]) * t;
    }
    result[72] = result[0];
    result[73] = result[1];
    if hue_ramp {
        let wrap = if result[0] < 0.0 { 6.0 } else { 0.0 };
        for (sample, value) in result.iter_mut().enumerate() {
            *value += wrap + sample as f32 / 12.0;
        }
    }
    Ok(result)
}

/// PF30009 affinity op (`0x76AD6F0` / `0x76AD800`): multiply an existing
/// destination plane in place by `1-(w-1)^2` and, when a mask is supplied, by
/// `mask/255`. This is a per-pixel weight, not a mask inversion.
pub fn apply_affinity_weight(
    destination: &mut [f32],
    source_hue: &[f32],
    reference_hue: &[f32],
    mask: Option<&[u8]>,
) -> Result<(), Refusal> {
    if source_hue.len() != destination.len() || reference_hue.len() != destination.len() {
        return Err(Refusal::ShapeMismatch("PF30009_HUE_PLANES"));
    }
    if source_hue
        .iter()
        .chain(reference_hue.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF30009_HUE_PLANES"));
    }
    if let Some(values) = mask {
        if values.len() != destination.len() {
            return Err(Refusal::ShapeMismatch("PF30009_HAIR_MASK"));
        }
    }

    for index in 0..destination.len() {
        let mut distance = (source_hue[index] - reference_hue[index]).abs();
        let wrap = 6.0 - distance;
        if wrap < distance {
            distance = wrap;
        }
        let weight = (distance * (12.0 / 7.0)).min(1.0);
        let factor = 1.0 - (weight - 1.0) * (weight - 1.0);
        let mask_factor = mask.map_or(1.0, |values| values[index] as f32 / 255.0);
        destination[index] = factor * destination[index] * mask_factor;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_rows() -> (Vec<f32>, Vec<f32>) {
        let row = (0..256).map(|i| i as f32 / 255.0).collect::<Vec<_>>();
        let map = (0..256 * 256).map(|i| (i % 256) as f32 / 255.0).collect::<Vec<_>>();
        (row, map)
    }

    fn neutral_curves() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            (0..CURVE_SAMPLES).map(|i| i as f32 / 12.0).collect(),
            vec![0.0; CURVE_SAMPLES],
            vec![0.0; CURVE_SAMPLES],
        )
    }

    #[test]
    fn matrices_preserve_white() {
        for row in FORWARD_MATRIX.iter().chain(INVERSE_MATRIX.iter()) {
            let sum = row[0] + row[1] + row[2];
            assert!((sum - 1.0).abs() < 1e-6, "row sum {sum}");
        }
    }

    #[test]
    fn forward_and_inverse_are_inverses() {
        for row in 0..3 {
            for column in 0..3 {
                let mut value = 0.0f32;
                for k in 0..3 {
                    value += INVERSE_MATRIX[row][k] * FORWARD_MATRIX[k][column];
                }
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((value - expected).abs() < 1e-5, "[{row}][{column}]={value}");
            }
        }
    }

    #[test]
    fn lut_coord_splits_16_bit_value() {
        assert_eq!(lut_coord_16(0.0), (0.0, 0.0));
        let (x, y) = lut_coord_16(32768.0 / 65535.0);
        assert!((y * 256.0 - 128.0).abs() <= 1.0);
        assert!((x * 256.0) < 256.0);
        // value*65535 means the last addressable pair is (255,255)/256.
        let (x, y) = lut_coord_16(1.0);
        assert_eq!(y, 255.0 / 256.0);
        assert_eq!(x, 255.0 / 256.0);
        // Only an out-of-range value reaches the clamped (0,256) pair.
        let (x, y) = lut_coord_16(65536.0 / 65535.0);
        assert_eq!(y, 1.0);
        assert_eq!(x, 0.0);
        let (x, y) = lut_coord_16(65535.0 / 65535.0);
        assert_eq!(y, 255.0 / 256.0);
        assert_eq!(x, 255.0 / 256.0);
    }

    #[test]
    fn map_lookup_uses_half_texel_offsets() {
        // The table encodes its own 16-bit address so the split is visible.
        let table = (0..256 * 256).map(|i| i as f32).collect::<Vec<_>>();
        assert_eq!(sample_map_256(&table, 0.0).unwrap(), 0.0);
        // 0.5 -> hi=127, lo=255 -> sample (254.5, 126.5).
        assert_eq!(sample_map_256(&table, 0.5).unwrap(), 32638.5);
        // 1.0 -> hi=255, lo=255 -> sample (254.5, 254.5): top texels unreachable.
        assert_eq!(sample_map_256(&table, 1.0).unwrap(), 65406.5);
    }

    #[test]
    fn row_lookup_clamps_ends() {
        let row = (0..256).map(|i| i as f32).collect::<Vec<_>>();
        assert_eq!(sample_row(&row, 256, 0.0).unwrap(), 0.0);
        assert_eq!(sample_row(&row, 256, 1.0).unwrap(), 255.0);
    }

    #[test]
    fn missing_tables_are_refused() {
        let (hue, saturation, luma) = neutral_curves();
        let curves = HsltunerCurves {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let tables = HsltunerTables {
            gamma_row: None,
            gamma_width: 0,
            inverse_contrast: None,
            contrast: None,
            inverse_gamma_row: None,
            inverse_gamma_width: 0,
        };
        assert_eq!(
            run_pipeline([0.5, 0.5, 0.5], &tables, &curves),
            Err(Refusal::MissingInput("HSLTUNER_GAMMA_MAP"))
        );
    }

    #[test]
    fn identity_material_round_trips_gray() {
        let (row, map) = identity_rows();
        let (hue, saturation, luma) = neutral_curves();
        let curves = HsltunerCurves {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let tables = HsltunerTables {
            gamma_row: Some(&row),
            gamma_width: 256,
            inverse_contrast: Some(&map),
            contrast: Some(&map),
            inverse_gamma_row: Some(&row),
            inverse_gamma_width: 256,
        };
        let out = run_pipeline([0.5, 0.5, 0.5], &tables, &curves).unwrap();
        for channel in out {
            assert!((channel - 0.5).abs() < 0.01, "channel={channel}");
        }
    }

    #[test]
    fn curve_generator_wraps_and_ramps_hue() {
        let map = generate_curve_map(PRODUCT_ANCHORS_8, [0.0; 8], false).unwrap();
        assert_eq!(map[0], 0.0);
        assert_eq!(map[72], map[0]);
        assert_eq!(map[73], map[1]);
        let hue = generate_curve_map(PRODUCT_ANCHORS_8, [0.0; 8], true).unwrap();
        assert_eq!(hue[0], 0.0);
        assert!((hue[72] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn curve_generator_rejects_bad_anchors() {
        let mut anchors = PRODUCT_ANCHORS_8;
        anchors[3] = anchors[2];
        assert_eq!(
            generate_curve_map(anchors, [0.0; 8], false),
            Err(Refusal::ValueOutOfRange("PF30009_CURVE_ANCHORS"))
        );
    }

    #[test]
    fn affinity_zeroes_destination_at_matching_hue() {
        let mut destination = [1.0f32, 1.0];
        apply_affinity_weight(&mut destination, &[1.0, 1.0], &[1.0, 0.0], None).unwrap();
        assert_eq!(destination[0], 0.0);
        assert_eq!(destination[1], 1.0);
    }

    #[test]
    fn affinity_multiplies_mask_not_inverts_it() {
        let mut full = [1.0f32];
        apply_affinity_weight(&mut full, &[3.0], &[0.0], Some(&[255])).unwrap();
        let mut zero = [1.0f32];
        apply_affinity_weight(&mut zero, &[3.0], &[0.0], Some(&[0])).unwrap();
        assert_eq!(full[0], 1.0);
        assert_eq!(zero[0], 0.0);
    }

    #[test]
    fn image_entry_copies_alpha() {
        let (row, map) = identity_rows();
        let (hue, saturation, luma) = neutral_curves();
        let curves = HsltunerCurves {
            hue: &hue,
            saturation: &saturation,
            luma: &luma,
        };
        let tables = HsltunerTables {
            gamma_row: Some(&row),
            gamma_width: 256,
            inverse_contrast: Some(&map),
            contrast: Some(&map),
            inverse_gamma_row: Some(&row),
            inverse_gamma_width: 256,
        };
        let source = Image::solid(1, 1, [128, 128, 128, 42]);
        let out = run_pipeline_image(&source, &tables, &curves).unwrap();
        assert_eq!(out.pixel(0, 0)[3], 42);
    }

    #[test]
    fn hsl_round_trip_is_stable() {
        for rgb in [[0.2, 0.5, 1.0], [1.0, 0.2, 0.2], [0.3, 0.3, 0.3]] {
            let hsl = shader_rgb2hsl(rgb);
            let back = shader_hsl2rgb(hsl);
            for channel in 0..3 {
                assert!((back[channel] - rgb[channel]).abs() < 1e-5, "{rgb:?} -> {back:?}");
            }
        }
    }
}
