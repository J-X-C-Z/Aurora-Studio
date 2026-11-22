#[derive(Clone, Copy)]
struct Segment {
    end: f64,
    center: f32,
    coefficients: [f32; 4],
}

fn segments(nodes: &[[f64; 3]]) -> Option<Vec<Segment>> {
    if nodes.len() == 1 || !nodes.iter().flatten().all(|value| value.is_finite()) {
        return None;
    }
    nodes.windows(2).map(|pair| {
        let [left, low, left_slope] = pair[0];
        let [right, high, right_slope] = pair[1];
        if right < left || (right > left && right as f32 <= left as f32) {
            return None;
        }
        let half = (right - left) * 0.5;
        if half == 0.0 {
            return (pair[0] == pair[1]).then(|| Segment {
                end: right,
                center: left as f32,
                coefficients: [low as f32, left_slope as f32, 0.0, 0.0],
            });
        }
        let inverse = 1.0 / half;
        let rise = high - low;
        let sum = right_slope + left_slope;
        let difference = right_slope - left_slope;
        Some(Segment {
            end: right,
            center: ((right + left) * 0.5) as f32,
            coefficients: [
                ((high + low) * 0.5 - (difference * 0.25) * half) as f32,
                (((rise * 0.75) * inverse) - (sum * 0.25)) as f32,
                ((inverse * 0.25) * difference) as f32,
                ((sum - rise * inverse) * ((inverse * 0.25) * inverse)) as f32,
            ],
        })
    }).collect()
}

#[derive(Clone, Copy)]
pub struct LegacyLutParams {
    pub pre_scale: f32,
    pub curve_black: f32,
    pub curve_white: f32,
    pub film_black: f32,
    pub film_white: f32,
    pub post_gain: f32,
    pub post_offset: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Legacy gamma 1.8 power law, kept for the old engine and its oracles.
    PowerLaw,
    /// Verified Capture One stage 3: gamma 2.0 with a linear toe.
    C1Piecewise,
}

impl Encoding {
    fn apply(self, value: f32) -> f32 {
        match self {
            Self::PowerLaw => value.max(0.0).powf(5.0 / 9.0),
            Self::C1Piecewise => c1_piecewise(value),
        }
    }
}

/// C1 stage 3, `0x13F5DA0`: linear -> encoded, with the 0.01947 toe.
pub fn c1_piecewise(value: f32) -> f32 {
    if value >= 1.0 {
        1.0
    } else if value > 0.01947 {
        1.075 * value.sqrt() - 0.07500005
    } else if value > 0.0 {
        3.8521 * value
    } else {
        0.0
    }
}

/// Uses the primary response of a v6 FCRV with full-range endpoints.
pub fn build_legacy_lut_from_fcrv(params: LegacyLutParams, bytes: &[u8]) -> Option<Box<[u16; 65536]>> {
    build_legacy_lut_from_fcrv_with_encoding(params, bytes, Encoding::PowerLaw)
}

/// Same FCRV table with an explicit transfer.
pub fn build_legacy_lut_from_fcrv_with_encoding(
    params: LegacyLutParams,
    bytes: &[u8],
    encoding: Encoding,
) -> Option<Box<[u16; 65536]>> {
    build_legacy_lut_with_encoding(params, &super::legacy_fcrv::primary_nodes(bytes)?, encoding)
}

/// Fixed-gamma, non-linearization branch with only the optional main film spline.
/// Nodes are prepared `[x, y, tangent]` values with the spline's endpoint copies, not raw FCRV controls.
/// This CPU reference takes explicit parameters; it does not resolve a Capture One recipe.
pub fn build_legacy_lut(params: LegacyLutParams, film_nodes: &[[f64; 3]]) -> Option<Box<[u16; 65536]>> {
    build_legacy_lut_with_encoding(params, film_nodes, Encoding::PowerLaw)
}

/// Same table with an explicit transfer; old callers stay on `PowerLaw`.
pub fn build_legacy_lut_with_encoding(
    params: LegacyLutParams,
    film_nodes: &[[f64; 3]],
    encoding: Encoding,
) -> Option<Box<[u16; 65536]>> {
    let LegacyLutParams { pre_scale, curve_black, curve_white, film_black, film_white, post_gain, post_offset } = params;
    if ![pre_scale, curve_black, curve_white, film_black, film_white, post_gain, post_offset].into_iter().all(f32::is_finite)
        || pre_scale <= 0.0 || curve_white <= curve_black || film_white <= film_black {
        return None;
    }
    let film = segments(film_nodes)?;
    let curve_gain = 1.0 / (curve_white - curve_black);
    let film_gain = 1.0 / (film_white - film_black);
    let knee = 0.5 / pre_scale;
    let remaining = 1.0 - knee;
    let shoulder = (remaining * pre_scale) * 2.0;
    let mut samples = vec![0.0_f32; 65536];
    let mut negative_begin = None;
    let mut high_end = None;
    for (block, slice) in samples.chunks_mut(2048).enumerate() {
        let mut segment = 0;
        let mut last_negative = 0;
        let mut first_high = 65535;
        for (in_block, sample) in slice.iter_mut().enumerate() {
            let index = block * 2048 + in_block;
            let mut value = index as f32 * (1.0 / 65535.0);
            if pre_scale > 1.0 {
                value = if value < knee {
                    value * pre_scale
                } else {
                    let distance = value - knee;
                    (distance * (shoulder * 0.5)) / (distance * (shoulder - 1.0) + remaining) + 0.5
                };
            }
            if !film.is_empty() {
                if in_block == 0 {
                    segment = film.partition_point(|piece| piece.end <= f64::from(value)).min(film.len() - 1);
                } else {
                    while segment + 1 < film.len() && value > film[segment].end as f32 {
                        segment += 1;
                    }
                }
                let piece = film[segment];
                let offset = value - piece.center;
                let [constant, linear, square, cube] = piece.coefficients;
                value = ((cube * offset + square) * offset + linear) * offset + constant;
            }
            value = encoding.apply(value);
            value = (value - curve_black) * curve_gain;
            value = (value - film_black) * film_gain;
            if value < 0.0 && index > last_negative {
                last_negative = index;
            } else if value > 1.0 && index < first_high {
                first_high = index;
            }
            *sample = value;
        }
        if last_negative != 0 {
            negative_begin = Some(negative_begin.map_or(last_negative, |begin: usize| begin.min(last_negative)));
        }
        if first_high != 65535 {
            high_end = Some(high_end.map_or(first_high, |end: usize| end.max(first_high)));
        }
    }
    if let Some(begin) = negative_begin {
        for index in begin..samples.len() {
            samples[index] = samples[index].max(samples[index - 1]);
        }
    }
    if let Some(end) = high_end {
        for index in (0..=end).rev() {
            samples[index] = samples[index].min(samples[index + 1]);
        }
    }
    let gain = (post_gain - post_offset) * 65535.0;
    let offset = post_offset * 65535.0 + 0.5;
    samples.into_iter().map(|value| (value * gain + offset).clamp(0.0, 65535.0) as u16)
        .collect::<Box<[_]>>().try_into().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone() -> LegacyLutParams {
        LegacyLutParams {
            pre_scale: 1.0,
            curve_black: 0.0,
            curve_white: 1.0,
            film_black: 0.0,
            film_white: 1.0,
            post_gain: 1.0,
            post_offset: 0.0,
        }
    }

    #[test]
    fn c1_piecewise_matches_verified_knots() {
        assert_eq!(c1_piecewise(-1.0), 0.0);
        assert_eq!(c1_piecewise(0.0), 0.0);
        assert!((c1_piecewise(0.01947) - 0.075000387).abs() < 1e-6);
        assert!((c1_piecewise(0.5) - 0.6851397).abs() < 1e-6);
        assert_eq!(c1_piecewise(1.0), 1.0);
        assert_eq!(c1_piecewise(2.0), 1.0);
    }

    #[test]
    fn explicit_encoding_selects_the_transfer() {
        let power = build_legacy_lut_with_encoding(tone(), &[], Encoding::PowerLaw).unwrap();
        let c1 = build_legacy_lut_with_encoding(tone(), &[], Encoding::C1Piecewise).unwrap();
        for index in [0usize, 1, 128, 32768, 65535] {
            let value = index as f32 / 65535.0;
            let power_expected = (value.max(0.0).powf(5.0 / 9.0) * 65535.0 + 0.5) as u16;
            let c1_expected = (c1_piecewise(value) * 65535.0 + 0.5) as u16;
            assert_eq!(power[index], power_expected, "power entry {index}");
            assert_eq!(c1[index], c1_expected, "c1 entry {index}");
        }
        assert_ne!(power[32768], c1[32768]);
    }

}
