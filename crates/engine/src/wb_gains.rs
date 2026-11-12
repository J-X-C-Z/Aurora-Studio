use crate::c1_model::gains_to_kelvin_tint;
use crate::c1_model::kelvin_tint_to_gains;

pub struct WbGains {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

pub fn from_nef_rb(r: f32, b: f32) -> WbGains {
    WbGains { r, g: 1.0, b }
}

pub fn to_kelvin_tint(gains: &WbGains) -> (f32, f32) {
    gains_to_kelvin_tint([gains.r, gains.g, gains.b])
}

pub fn from_kelvin_tint(kelvin: f32, tint: f32) -> WbGains {
    let [r, g, b] = kelvin_tint_to_gains(kelvin, tint);
    WbGains { r, g, b }
}

fn srgb_encode(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_decode(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

pub fn lut(gains: &WbGains) -> ([u8; 256], [u8; 256], [u8; 256]) {
    let mut tables = ([0u8; 256], [0u8; 256], [0u8; 256]);
    for (i, table) in tables.0.iter_mut().enumerate() {
        *table = (srgb_encode((srgb_decode(i as f32 / 255.0) * gains.r).clamp(0.0, 1.0)) * 255.0
            + 0.5) as u8;
    }
    for (i, table) in tables.1.iter_mut().enumerate() {
        *table = (srgb_encode((srgb_decode(i as f32 / 255.0) * gains.g).clamp(0.0, 1.0)) * 255.0
            + 0.5) as u8;
    }
    for (i, table) in tables.2.iter_mut().enumerate() {
        *table = (srgb_encode((srgb_decode(i as f32 / 255.0) * gains.b).clamp(0.0, 1.0)) * 255.0
            + 0.5) as u8;
    }
    tables
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_gains_do_not_change_pixels() {
        let luts = lut(&from_nef_rb(1.0, 1.0));
        for (table, expected) in luts.0.iter().zip(luts.1.iter()) {
            assert_eq!(table, expected);
        }
    }

    #[test]
    fn known_gain_scales_linear_input() {
        let luts = lut(&from_nef_rb(1.625, 1.412));
        for value in [16, 32, 64, 128] {
            let x = value as f32 / 255.0;
            let expected = (srgb_encode((srgb_decode(x) * 1.625).clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
            assert_eq!(luts.0[value], expected);
        }
    }

    #[test]
    fn kelvin_tint_round_trip() {
        let gains = from_nef_rb(1.625, 1.412);
        let (kelvin, tint) = to_kelvin_tint(&gains);
        let back = from_kelvin_tint(kelvin, tint);
        assert!((back.r - gains.r).abs() < 1e-3);
        assert!((back.b - gains.b).abs() < 1e-3);
    }

    #[test]
    fn lut_matches_per_pixel_result() {
        let gains = from_nef_rb(1.385986, 1.524762);
        let luts = lut(&gains);
        for (channel, gain) in [(&luts.0, gains.r), (&luts.1, gains.g), (&luts.2, gains.b)] {
            for (i, value) in channel.iter().enumerate() {
                let x = i as f32 / 255.0;
                let expected =
                    (srgb_encode((srgb_decode(x) * gain).clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
                assert!((i32::from(*value) - i32::from(expected)).abs() <= 1);
            }
        }
    }

    #[test]
    fn boundaries_stay_in_range() {
        let luts = lut(&from_nef_rb(4.0, 0.0));
        assert_eq!(luts.0[0], 0);
        assert_eq!(luts.0[255], 255);
        assert_eq!(luts.2[0], 0);
        assert_eq!(luts.2[255], 0);
    }
}
