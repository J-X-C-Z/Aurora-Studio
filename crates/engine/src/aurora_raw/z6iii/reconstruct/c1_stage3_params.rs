#[derive(Clone, Copy, Debug)]
pub struct Z6iiiStage3 {
    pub(crate) gains: [f32; 3],
    pub(crate) offset: f32,
    pub(crate) noise: f32,
    pub(crate) noise_floor: f32,
    pub(crate) bounded_offset: f32,
    pub(crate) flags: [bool; 5],
    finish_control: i32,
}

impl Z6iiiStage3 {
    pub fn offset_for_noise([floor, poisson]: [f32; 2], noise_control: (i32, f32)) -> f32 {
        let width = ((100.0 - (noise_control.0 as f32 + noise_control.1)) * 0.2).max(1.0);
        let span = width * width;
        let variance = f64::from((floor * 4.0) * floor) / (f64::from(span) + 0.000001)
            + f64::from(poisson * poisson);
        (((variance as f32).sqrt() + poisson) * (span * 0.5)).min(65535.0)
    }

    pub fn prepare(
        gains: [f32; 3],
        noise_fields: [f32; 2],
        noise_control: (i32, f32),
        version: i32,
        raw_mode: i32,
        cpu_features: u64,
        finish_control: i32,
    ) -> Option<Self> {
        let [floor, poisson] = noise_fields;
        if !gains.into_iter().all(|gain| gain.is_finite() && gain > 0.0)
            || !noise_fields.into_iter().all(|value| value.is_finite() && value >= 0.0)
            || !noise_control.1.is_finite() {
            return None;
        }
        let bounded_offset = Self::offset_for_noise(noise_fields, noise_control);
        let ratio = 139_152_720.0_f32 / (floor * floor + poisson * f32::from_bits(0x4638_5134));
        let outer_finish = version >= 0x514 && ratio.log10() * 10.0 < 25.0;
        let first = cpu_features & 0x7f == 0x7f
            || cpu_features & 0x30_0000_0000_0000 == 0x30_0000_0000_0000
            || cpu_features & 0x50_0000_0000_0000 == 0x50_0000_0000_0000;
        let second = cpu_features as u8 == 0xff;
        let finish_mode = match finish_control {
            6 => false,
            value if value > 0 => true,
            _ => raw_mode != 0,
        };
        if !bounded_offset.is_finite() {
            return None;
        }
        Some(Self {
            gains, offset: bounded_offset.min(65536.0), noise: 1.0, noise_floor: floor, bounded_offset,
            flags: [first, second, outer_finish, finish_mode, !finish_mode],
            finish_control,
        })
    }

    pub fn gains(self) -> [f32; 3] { self.gains }
    pub fn offset(self) -> f32 { self.offset }
    pub fn interpolation_noise(self) -> f32 { self.noise }
    pub fn noise_fields(self) -> (f32, f32) { (self.noise_floor, self.bounded_offset) }
    pub fn flags(self) -> [bool; 5] { self.flags }

    pub fn supports_bare_planar(self) -> bool {
        (self.flags[0] || self.flags[1]) && !self.flags[2] && self.finish_control == 1
    }

    pub(crate) fn inner_finish(self) -> Option<(f32, i32)> {
        (self.flags[0] && !self.flags[1] && !self.flags[2] && self.finish_control != 1)
            .then_some((self.noise_floor, self.finish_control))
    }
}

#[cfg(test)]
mod tests {
    use super::Z6iiiStage3;

    #[test]
    fn gates_reject_each_unimplemented_route() {
        let prepare = |cpu, noise, control| Z6iiiStage3::prepare([2.0, 1.0, 1.25], noise, (40, 1.5), 0x514, 0, cpu, control).unwrap();
        assert!(prepare(0x7f, [12.8, 0.0], 1).supports_bare_planar());
        assert!(!prepare(0, [12.8, 0.0], 1).supports_bare_planar());
        assert!(!prepare(0xff, [12.8, 0.0], 0).supports_bare_planar());
        assert!(!prepare(0xff, [1024.0, 2.0], 1).supports_bare_planar());
    }

    #[test]
    #[ignore = "run verify_stage3_params.py"]
    fn stage3_settings_match_fixed_dll() {
        let rows = std::fs::read_to_string(std::env::var_os("C1_STAGE3_SETTINGS_ORACLE").unwrap()).unwrap();
        let mut count = 0;
        for row in rows.lines() {
            let words: Vec<_> = row.split_whitespace().map(|word| u64::from_str_radix(word, 16).unwrap()).collect();
            assert_eq!(words.len(), 18);
            let float = |index: usize| f32::from_bits(words[index] as u32);
            let params = Z6iiiStage3::prepare([float(0), float(1), float(2)], [float(3), float(4)],
                (words[5] as i32, float(6)), words[7] as i32, words[8] as i32, words[9], words[10] as i32).unwrap();
            let [original_floor, bounded_offset] = [params.noise_fields().0, params.noise_fields().1];
            let values = [params.offset(), params.interpolation_noise(), original_floor, bounded_offset];
            assert_eq!(params.gains().map(f32::to_bits).map(u64::from), words[..3]);
            assert_eq!(values.map(f32::to_bits).map(u64::from), words[11..15], "row {count}");
            let flags = params.flags().into_iter().enumerate().fold(0, |mask, (index, flag)| mask | (u64::from(flag) << index));
            assert_eq!(flags, words[15], "row {count}");
            if !params.flags()[2] { assert_eq!(words[16], 6, "row {count}"); }
            assert_eq!(u64::from(params.supports_bare_planar()), words[17], "row {count}");
            count += 1;
        }
        assert!(count >= 1000);
        println!("{count} Z6III stage 3 setup and route probes matched");
    }
}
