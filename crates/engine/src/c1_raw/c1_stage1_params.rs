//! Parameter initialization for the scalar C1v2012 first-stage reference.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage1Params {
    pub red_gain: f32,
    pub blue_gain: f32,
    pub noise_floor: f32,
    pub black_pivot: f32,
}

impl Stage1Params {
    pub fn prepare(rgb: [f32; 3], noise_fields: [f32; 2], black: u32, bayer_already_subtracted: bool) -> Self {
        let [red, green, blue] = rgb;
        let [first, second] = noise_fields;
        let variance = ((first * 4.0) * first) / 100.0 + second * second;
        Self {
            red_gain: red / green,
            blue_gain: blue / green,
            noise_floor: ((variance.sqrt() + second) * 50.0).min(65536.0),
            black_pivot: if bayer_already_subtracted { 0.0 } else { black as f32 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Stage1Params;

    #[test]
    fn black_pivot_and_noise_cap_are_independent() {
        let rgb = [3.0, 2.0, 4.0];
        let params = Stage1Params::prepare(rgb, [10.0, 0.0], 4032, false);
        assert_eq!(params, Stage1Params { red_gain: 1.5, blue_gain: 2.0, noise_floor: 100.0, black_pivot: 4032.0 });
        let params = Stage1Params::prepare(rgb, [1e6, 0.0], 4032, true);
        assert_eq!(params.noise_floor, 65536.0);
        assert_eq!(params.black_pivot, 0.0);
    }
}
