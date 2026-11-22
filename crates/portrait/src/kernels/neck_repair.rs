//! PF90015 model-output tails from P13 `NeckLineCore.cs`.
//! The original final blend after 0x5B7B390 remains unresolved. This module
//! exposes repair maps only; it does not substitute the source-over alternative.

use crate::Refusal;

pub const WIDTH: usize = 640;
pub const HEIGHT: usize = 512;
pub const PLANE: usize = WIDTH * HEIGHT;
pub const AUX_COLOR: usize = 128 * 128 * 3;
pub const COARSE: usize = 8 * 10;

#[derive(Debug)]
pub struct RepairMaps {
    pub blend_weight: Vec<f32>,
    /// Native 3x512x640 tensor order, without a guessed channel transpose.
    pub color_target: Vec<f32>,
    pub aux_color: Vec<f32>,
    pub coarse: Vec<f32>,
}

fn require(values: &[f32], length: usize, name: &'static str) -> Result<(), Refusal> {
    if values.len() != length {
        return Err(Refusal::ShapeMismatch(name));
    }
    if !values.iter().all(|v| v.is_finite()) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(())
}

/// Validates the model-supplied coarse output against exact 64x64 max pooling.
pub fn build_repair_maps(
    raw_blend: &[f32],
    raw_color: &[f32],
    raw_aux: &[f32],
    raw_coarse: &[f32],
) -> Result<RepairMaps, Refusal> {
    require(raw_blend, PLANE, "blend512x640")?;
    require(raw_color, PLANE * 3, "color3x512x640")?;
    require(raw_aux, AUX_COLOR, "aux3x128x128")?;
    require(raw_coarse, COARSE, "coarse8x10")?;
    let blend_weight: Vec<f32> = raw_blend
        .iter()
        .map(|v| 0.5 * v.clamp(-1.0, 1.0) + 0.5)
        .collect();
    for oy in 0..8 {
        for ox in 0..10 {
            let mut maximum = f32::NEG_INFINITY;
            for y in oy * 64..(oy + 1) * 64 {
                for x in ox * 64..(ox + 1) * 64 {
                    let value = blend_weight[y * WIDTH + x];
                    if value > maximum {
                        maximum = value;
                    }
                }
            }
            if maximum.to_bits() != raw_coarse[oy * 10 + ox].to_bits() {
                return Err(Refusal::ValueOutOfRange("coarse_max_pool"));
            }
        }
    }
    let color_target: Vec<f32> = raw_color.iter().map(|v| 127.5 * v + 127.5).collect();
    let aux_color: Vec<f32> = raw_aux.iter().map(|v| 127.5 * v).collect();
    if !color_target.iter().chain(&aux_color).all(|v| v.is_finite()) {
        return Err(Refusal::ValueOutOfRange("color_tail"));
    }
    Ok(RepairMaps {
        blend_weight,
        color_target,
        aux_color,
        coarse: raw_coarse.to_vec(),
    })
}

/// Native response aggregation: maximum and mean of responses above 1e-5.
pub fn aggregate_positive_response(values: &[f32]) -> Result<f32, Refusal> {
    let mut maximum = 0.0f32;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &value in values {
        if !value.is_finite() {
            return Err(Refusal::ValueOutOfRange("response"));
        }
        if value > 1e-5 {
            maximum = maximum.max(value);
            sum += f64::from(value);
            count += 1;
        }
    }
    Ok(if count == 0 {
        1e-5
    } else {
        ((f64::from(maximum) + sum / count as f64) * 0.5) as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_tails_and_coarse_pool_match_recovered_shapes() {
        let mut raw_blend = vec![0.0; PLANE];
        raw_blend[HEIGHT / 2 * WIDTH + WIDTH / 2] = 2.0;
        let mut coarse = vec![0.5; COARSE];
        coarse[4 * 10 + 5] = 1.0;
        let maps = build_repair_maps(
            &raw_blend,
            &vec![0.0; PLANE * 3],
            &vec![-1.0; AUX_COLOR],
            &coarse,
        )
        .unwrap();
        assert_eq!(maps.blend_weight[0], 0.5);
        assert_eq!(maps.color_target[0], 127.5);
        assert_eq!(maps.aux_color[0], -127.5);
        assert_eq!(maps.coarse[45], 1.0);
        coarse[45] = 0.5;
        assert!(matches!(
            build_repair_maps(
                &raw_blend,
                &vec![0.0; PLANE * 3],
                &vec![0.0; AUX_COLOR],
                &coarse
            ),
            Err(Refusal::ValueOutOfRange("coarse_max_pool"))
        ));
    }

    #[test]
    fn malformed_and_non_finite_model_outputs_are_refused() {
        assert!(matches!(
            build_repair_maps(&[], &[], &[], &[]),
            Err(Refusal::ShapeMismatch(_))
        ));
        assert!(aggregate_positive_response(&[f32::NAN]).is_err());
        assert_eq!(aggregate_positive_response(&[-1.0, 1e-6]).unwrap(), 1e-5);
        assert_eq!(aggregate_positive_response(&[0.25, 0.75]).unwrap(), 0.625);
    }
}
