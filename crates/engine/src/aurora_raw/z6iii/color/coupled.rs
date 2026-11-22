//! ApplyExposure specialization 22: linear/Auto pre-film coupling.
//!
//! Both inputs are the post-matrix camera device codes after inverse linear
//! transfer; the output returns through the same linear transfer. The formula
//! is the exact kernel 22 replay in
//! `z6iii-native-check-20261120/apply_exposure_22.py` with (0.5, 0.35).

use super::legacy_lut::{build_legacy_lut_from_fcrv_with_encoding, Encoding, LegacyLutParams};
use super::pipeline::LegacyColorChain;

pub struct CoupledColorChain {
    linear: LegacyColorChain,
    auto: LegacyColorChain,
    linear_lut: Box<[u16; 65536]>,
    inverse: Vec<f32>,
    float: Option<Exposure>,
}

struct Exposure {
    matrix: [[f32; 3]; 3],
    film: Vec<[f32; 3]>,
}

impl Exposure {
    fn camera(&self, rgb: [f32; 3]) -> [f32; 3] {
        let rgb = rgb.map(|code| (code / 65535.0).max(0.0));
        let before = self.matrix.map(|[red, green, blue]| {
            madd(blue, rgb[2], madd(red, rgb[0], green * rgb[1])).max(0.0)
        });
        let curved = before.map(|value| film_at(&self.film, value));
        let coupled = couple_float(before, curved);
        coupled.map(encode_camera)
    }
}

#[inline]
pub(super) fn encode_camera(sample: f32) -> f32 {
    if sample > 0.01947 { madd(1.075, sample.sqrt(), 1.0 - 1.075_f32) }
    else { 3.8521 * sample.max(0.0) }
}

#[inline]
fn madd(left: f32, right: f32, offset: f32) -> f32 {
    left * right + offset
}

pub(super) fn film_at(film: &[[f32; 3]], code: f32) -> f32 {
    if film.is_empty() { return code; }
    let right = film.partition_point(|node| node[0] <= code).clamp(1, film.len() - 1);
    let [x0, y0, m0] = film[right - 1];
    let [x1, y1, m1] = film[right];
    let dx = x1 - x0;
    if dx.abs() < 0.0001 { return madd(code - x0, m0 + m1, y0 + y1) * 0.5; }
    let t = (code - x0) / dx;
    let s = t - 1.0;
    let base = madd(t * t * (y1 - y0), madd(-2.0, t, 3.0), y0);
    madd(madd(m0, s, m1 * t) * t * s, dx, base)
}

pub(super) fn couple_float(before: [f32; 3], curved: [f32; 3]) -> [f32; 3] {
    const K: f32 = 4.0 / 256.0;
    const SLOPE: f32 = 1.0 / (1.0 - 0.5 * K);
    const TOE: f32 = SLOPE / (2.0 * K);
    const INTERCEPT: f32 = 1.0 - SLOPE;
    let encode = |code: f32| if code > K * K { madd(SLOPE, code.sqrt(), INTERCEPT) } else { TOE * code };
    let luma = |[red, green, blue]: [f32; 3]| madd(blue, 0.114, madd(red, 0.299, green * 0.587)).abs() + 1e-10;
    let range = |[r, g, b]: [f32; 3]| r.max(g.max(b)) - r.min(g.min(b));
    let source = before.map(encode);
    let mut target = curved.map(encode);
    let mut y_target = luma(target);
    if target.into_iter().any(|value| value > 1.0) {
        let limit = luma(target.map(|value| value.min(1.0)));
        target = target.map(|value| value * (limit / y_target));
        y_target = limit;
    }
    let source = source.map(|value| value * (y_target / luma(source)));
    let gain = madd((range(target) + 1e-6) / (range(source) + 1e-6), 0.5, 0.5);
    let mut result = source.map(|value| madd(value - y_target, gain, y_target));
    let distance = result[0].max(result[1].max(result[2])) - y_target;
    if distance > 0.0 && 1.0 - y_target < 2.0 * distance {
        let ratio = (1.0 - y_target) / distance;
        let edge = (1.0 - (ratio - 1.0).abs()).max(0.0);
        let smooth = madd(edge * edge * edge, -1.0 / 6.0, ratio.min(1.0));
        result = result.map(|value| madd(value - y_target, smooth, y_target));
    }
    std::array::from_fn(|channel| {
        let value = result[channel];
        let decoded = if value > K * K * TOE {
            let z = (value - INTERCEPT) * (1.0 / SLOPE);
            z * z
        } else { (1.0 / TOE) * value };
        madd(curved[channel], 0.35_f32, decoded * (1.0 - 0.35_f32))
    })
}

const K: f64 = 4.0 / 256.0;
const SLOPE: f64 = 1.0 / (1.0 - K * 0.5);
const TOE: f64 = SLOPE / (2.0 * K);
const INTERCEPT: f64 = 1.0 - SLOPE;

fn encode(value: f64) -> f64 {
    if value > K * K {
        SLOPE * value.max(0.0).sqrt() + INTERCEPT
    } else {
        TOE * value
    }
}

fn decode(value: f64) -> f64 {
    if value > K * K * TOE {
        ((value - INTERCEPT) / SLOPE).powi(2)
    } else {
        value / TOE
    }
}

fn luma(value: [f64; 3]) -> f64 {
    (value[0] * 0.299 + value[1] * 0.587 + value[2] * 0.114).abs() + 1e-10
}

fn ptp(value: [f64; 3]) -> f64 {
    value[0].max(value[1]).max(value[2]) - value[0].min(value[1]).min(value[2])
}

fn color_compensation(before: [f64; 3], curved: [f64; 3], chroma_mix: f64, bypass: f64) -> [f64; 3] {
    let source = before.map(encode);
    let target = curved.map(encode);
    let y0 = luma(source);
    let y1 = luma(target);
    let effective_y = if target[0].max(target[1]).max(target[2]) > 1.0 {
        luma(target.map(|value| value.min(1.0)))
    } else {
        y1
    };
    let target = target.map(|value| value * (effective_y / y1));
    let source = source.map(|value| value * (effective_y / y0));
    let ratio = (ptp(target) + 1e-6) / (ptp(source) + 1e-6);
    let result = source.map(|value| {
        effective_y + (value - effective_y) * (ratio * (1.0 - chroma_mix) + chroma_mix)
    });
    let distance = result[0].max(result[1]).max(result[2]) - effective_y;
    let gain = if distance > 0.0 {
        (1.0 - effective_y) / distance
    } else {
        1.0
    };
    let smooth = gain.min(1.0) - (1.0 - (gain - 1.0).abs()).max(0.0).powi(3) / 6.0;
    let result = result.map(|value| {
        if distance > 0.0 && (1.0 - effective_y) < 2.0 * distance {
            effective_y + (value - effective_y) * smooth
        } else {
            value
        }
    });
    std::array::from_fn(|channel| {
        curved[channel] * bypass + decode(result[channel]) * (1.0 - bypass)
    })
}

fn inverse_table(lut: &[u16; 65536]) -> Vec<f32> {
    let mut inverse = vec![0.0f32; 65536];
    let mut lower = 0usize;
    for code in 0..65536usize {
        while lower + 1 < 65536 && usize::from(lut[lower + 1]) < code {
            lower += 1;
        }
        let left = f64::from(lut[lower]);
        let right = f64::from(lut[(lower + 1).min(65535)]);
        let input = if right == left {
            lower as f64
        } else {
            lower as f64 + (code as f64 - left) / (right - left)
        };
        inverse[code] = (input / 65535.0) as f32;
    }
    inverse
}

fn forward(lut: &[u16; 65536], value: f64) -> u16 {
    let position = (value.clamp(0.0, 1.0) * 65535.0).min(65535.0);
    let index = position.floor() as usize;
    if index >= 65535 {
        return lut[65535];
    }
    let weight = position - index as f64;
    (f64::from(lut[index]) * (1.0 - weight) + f64::from(lut[index + 1]) * weight).round() as u16
}

impl CoupledColorChain {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        matrix: [[f32; 3]; 3],
        exposure_scale: f32,
        tone: LegacyLutParams,
        auto_fcrv: &[u8],
        linear_fcrv: &[u8],
        camera_profile: &[u8],
        photo_profile: Option<&[u8]>,
        points: u8,
        encoding: Encoding,
    ) -> Option<Self> {
        let auto = LegacyColorChain::new_with_encoding(
            matrix,
            exposure_scale,
            tone,
            auto_fcrv,
            camera_profile,
            photo_profile,
            points,
            encoding,
        )?;
        let linear = LegacyColorChain::new_with_encoding(
            matrix,
            exposure_scale,
            tone,
            linear_fcrv,
            camera_profile,
            photo_profile,
            points,
            encoding,
        )?;
        let linear_lut = build_legacy_lut_from_fcrv_with_encoding(tone, linear_fcrv, encoding)?;
        let inverse = inverse_table(&linear_lut);
        Some(Self { linear, auto, linear_lut, inverse, float: None })
    }

    pub fn with_float_exposure(mut self, matrix: [[f32; 3]; 3], auto_fcrv: &[u8]) -> Option<Self> {
        let film = super::legacy_fcrv::primary_nodes(auto_fcrv)?;
        self.float = Some(Exposure { matrix, film: film.into_iter().map(|node| node.map(|value| value as f32)).collect() });
        Some(self)
    }

    pub fn apply_float(&self, rgb: [f32; 3]) -> Option<[f32; 3]> {
        let output = self.float.as_ref()?.camera(rgb);
        output.into_iter().all(f32::is_finite).then_some(output)
    }

    pub fn float_to_working(&self, rgb: [f32; 3]) -> Option<[f32; 3]> {
        self.auto.camera_to_working(rgb)
    }

    pub fn apply(&self, rgb: [f32; 3]) -> Option<([u16; 3], [f32; 3], [f32; 3])> {
        self.auto.legacy_to_working(self.apply_legacy(rgb)?)
    }

    pub fn apply_legacy(&self, rgb: [f32; 3]) -> Option<[u16; 3]> {
        let linear_code = self.linear.apply_legacy(rgb)?;
        let auto_code = self.auto.apply_legacy(rgb)?;
        let before = std::array::from_fn(|channel| {
            f64::from(self.inverse[usize::from(linear_code[channel])])
        });
        let curved = std::array::from_fn(|channel| {
            f64::from(self.inverse[usize::from(auto_code[channel])])
        });
        let corrected = color_compensation(before, curved, 0.5, 0.35);
        Some(std::array::from_fn(|channel| forward(&self.linear_lut, corrected[channel])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_hermite_preserves_endpoint_headroom() {
        let identity = [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        assert_eq!(film_at(&identity, 1.25), 1.25);
        let unclipped = couple_float([0.85, 1.05, 1.38], [0.85, 1.05, 1.38]);
        let clipped = couple_float([0.85, 1.0, 1.0], [0.85, 1.0, 1.0]);
        assert!(unclipped[0] > clipped[0]);
        assert!((couple_float([0.02, 0.06, 0.13], [0.02, 0.06, 0.13])[1] - 0.06).abs() < 1e-7);
    }

    #[test]
    fn identity_when_before_and_curved_match() {
        let value = [0.02, 0.06, 0.13];
        assert_eq!(color_compensation(value, value, 0.5, 1.0), value);
        let out = color_compensation(value, value, 0.5, 0.35);
        for (got, expected) in out.iter().zip(value) {
            assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
        }
    }

    #[test]
    fn inverse_and_forward_round_trip() {
        let lut: Box<[u16; 65536]> =
            (0..=65535u32).map(|code| code as u16).collect::<Vec<_>>().try_into().unwrap();
        let inverse = inverse_table(&lut);
        for code in [0usize, 1, 4095, 32768, 65535] {
            let value = f64::from(inverse[code]);
            assert_eq!(forward(&lut, value), code as u16, "code {code}");
        }
    }
}
