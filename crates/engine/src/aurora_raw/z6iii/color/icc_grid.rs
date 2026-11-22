//! RGB to RGB u16 CLUT sampling, before any optimizer white-point fixup.

pub(crate) fn saturate_word(scaled: f64) -> u16 {
    let biased = scaled + 0.5;
    if biased >= 65535.0 {
        return u16::MAX;
    }
    if biased <= 0.0 {
        return 0;
    }
    // The DLL's double-add rounds on a 2^-16 grid before extracting the word.
    let encoded = (biased - 32767.0) + 103079215104.0;
    ((encoded.to_bits() as i32 >> 16) + 32767) as u16
}

/// Converts one normalized pipeline output using the fixed x64 sampler.
pub fn quantize_output(value: f32) -> u16 {
    if value.is_nan() {
        // SSE widens and quiets the NaN while retaining its low payload bits.
        return ((value.to_bits() << 29) as i32 >> 16).wrapping_add(32767) as u16;
    }
    saturate_word(f64::from(value) * 65535.0)
}

fn grid_word(index: u8, samples: u8) -> u16 {
    saturate_word(f64::from(index) * 65535.0 / f64::from(samples - 1))
}

/// Visits RGB knots with blue fastest and writes contiguous RGB output words.
/// Axes must have at least two samples. An invalid shape leaves the table untouched.
pub fn sample_rgb(
    axes: [u8; 3],
    table: &mut [u16],
    mut evaluate: impl FnMut([f32; 3]) -> [f32; 3],
) -> bool {
    if axes.iter().any(|&axis| axis < 2)
        || table.len() != 3 * axes.into_iter().map(usize::from).product::<usize>()
    {
        return false;
    }

    let [red, green, blue] = axes;
    let mut knots = table.chunks_exact_mut(3);
    for r in 0..red {
        for g in 0..green {
            for b in 0..blue {
                let input = [grid_word(r, red), grid_word(g, green), grid_word(b, blue)]
                    .map(|word| (f64::from(word) / 65535.0) as f32);
                let output = evaluate(input).map(quantize_output);
                knots.next().unwrap().copy_from_slice(&output);
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_coordinates() {
        for count in 2..=255 {
            let denominator = u32::from(count - 1);
            for index in 0..count {
                let expected = (u32::from(index) * 65535 * 2 + denominator) / (2 * denominator);
                assert_eq!(u32::from(grid_word(index, count)), expected);
            }
        }
    }

    #[test]
    fn output_boundaries() {
        let below_half = f32::from_bits(0x37000000);
        assert_eq!(quantize_output(below_half), 1);
        assert_eq!(quantize_output(f32::from_bits(0x36fffe00)), 0);
        assert_eq!(quantize_output(0.5), 32768);
        assert_eq!(quantize_output(f32::NEG_INFINITY), 0);
        assert_eq!(quantize_output(f32::INFINITY), 65535);
        assert_eq!(quantize_output(f32::from_bits(0x7fc00000)), 32767);
        assert_eq!(quantize_output(f32::from_bits(0x7fc00001)), 40959);
        assert_eq!(quantize_output(f32::from_bits(0x7f800004)), 65535);
        assert_eq!(quantize_output(f32::from_bits(0xffc00007)), 24575);
        for word in 0..=u16::MAX {
            assert_eq!(quantize_output((f64::from(word) / 65535.0) as f32), word);
        }
    }

    #[test]
    fn traversal_and_channel_order() {
        let mut table = [0; 72];
        let mut index = 0;
        assert!(sample_rgb([2, 3, 4], &mut table, |[r, g, b]| {
            let expected = [
                [0, 65535][index / 12],
                [0, 32768, 65535][index / 4 % 3],
                [0, 21845, 43690, 65535][index % 4],
            ];
            assert_eq!([r, g, b], expected.map(|word| (f64::from(word) / 65535.0) as f32));
            index += 1;
            [b, r, g]
        }));
        assert_eq!(index, 24);
        assert_eq!(&table[..6], &[0, 0, 0, 21845, 0, 0]);
        assert_eq!(&table[69..], &[65535; 3]);
        let before = table;
        assert!(!sample_rgb([1, 3, 4], &mut table, |_| unreachable!()));
        assert!(!sample_rgb([2, 3, 4], &mut table[..71], |_| unreachable!()));
        assert_eq!(table, before);
    }
}
