//! Scalar C1v2012 direction selection for an interior Bayer site.

// The kernel uses the full gain at the reference center and Q10 at grid taps.
fn center_prediction(word: i32, gain: f32, bias: i32) -> i32 {
    (word as f32 * gain).floor() as i32 + bias
}

fn predictor(gain: f32, bias: i32) -> impl Fn(i32) -> i32 {
    let gain_q10 = (gain * 1024.0) as i32;
    move |word: i32| ((word.wrapping_mul(gain_q10) as u32 >> 10) as i32) + bias
}

pub fn direction_costs(patch: &[[u16; 11]; 11], gain: f32, black: f32) -> [u32; 6] {
    let at = |row: i32, col: i32| i32::from(patch[(row + 5) as usize][(col + 5) as usize]);
    let center = at(0, 0);
    let gain_scaled = gain * 1024.0;
    let curvature_gain = (gain_scaled * 1.6) as i32;
    let bias = (black - black * gain) as i32;
    let predicted = predictor(gain, bias);
    let reference_center = center_prediction(center, gain, bias);
    let dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    let reference = dirs.map(|(dy, dx)| at(dy, dx) - reference_center);
    let h = ((at(0, -2) + at(0, 2) - 2 * center).abs().wrapping_mul(curvature_gain) as u32 >> 10)
        + (at(-1, 0) - at(-1, -2)).unsigned_abs() + (at(-1, 0) - at(-1, 2)).unsigned_abs()
        + (at(1, 0) - at(1, -2)).unsigned_abs() + (at(1, 0) - at(1, 2)).unsigned_abs();
    let v = ((at(-2, 0) + at(2, 0) - 2 * center).abs().wrapping_mul(curvature_gain) as u32 >> 10)
        + (at(0, -1) - at(-2, -1)).unsigned_abs() + (at(0, -1) - at(2, -1)).unsigned_abs()
        + (at(0, 1) - at(-2, 1)).unsigned_abs() + (at(0, 1) - at(2, 1)).unsigned_abs();
    let mut costs = [h, h, v, v, h, v];
    for offset in [-4, 4] {
        for (base_row, base_col, first) in [(0, offset, 0), (offset, 0, 2)] {
            let target = predicted(at(base_row, base_col));
            let (dy0, dx0) = dirs[first];
            let (dy1, dx1) = dirs[first + 1];
            let first_sample = at(base_row + dy0, base_col + dx0);
            let second_sample = at(base_row + dy1, base_col + dx1);
            costs[first] += (first_sample - target - reference[first]).unsigned_abs();
            costs[first + 1] += (second_sample - target - reference[first + 1]).unsigned_abs();
            costs[4 + first / 2] += (first_sample - second_sample).unsigned_abs();
        }
    }
    for (row, col) in [-2, 0, 2].into_iter().flat_map(|row| [-2, 0, 2].map(|col| (row, col))) {
        let target = predicted(at(row, col));
        let nearby = dirs.map(|(dy, dx)| at(row + dy, col + dx));
        for index in 0..4 {
            costs[index] += (nearby[index] - target - reference[index]).unsigned_abs();
        }
        costs[4] += (nearby[0] - nearby[1]).unsigned_abs();
        costs[5] += (nearby[2] - nearby[3]).unsigned_abs();
    }
    costs
}

pub fn early_direction([left, right, up, down, horizontal, vertical]: [u32; 6], auxiliary_difference: f32) -> Option<u8> {
    let (h_min, h_max) = (left.min(right), left.max(right));
    let (v_min, v_max) = (up.min(down), up.max(down));
    let (horizontal_separates, vertical_separates) = if auxiliary_difference < 0.1 {
        (h_max < v_min, v_max < h_min)
    } else {
        let cube = (auxiliary_difference * auxiliary_difference) * auxiliary_difference;
        let weight = cube * 2.5 + 1.0;
        ((v_min as f32) * weight > h_max as f32, (h_min as f32) * weight > v_max as f32)
    };
    if horizontal_separates && horizontal < vertical {
        Some(1)
    } else if vertical_separates && vertical < horizontal {
        Some(2)
    } else {
        None
    }
}

pub fn low_curvature_direction(patch: &[[u16; 11]; 11], black: f32) -> Option<u8> {
    let [left, right, up, down] = [patch[5][4], patch[5][6], patch[4][5], patch[6][5]].map(i32::from);
    let curvature = (right + left - down - up).abs() / 2;
    let signal = right + left + down + up - (black * 4.0) as i32;
    (curvature.wrapping_mul(curvature) < signal).then_some(3)
}

pub fn extended_costs(patch: &[[u16; 11]; 11], gain: f32, black: f32) -> [u32; 4] {
    let at = |row: usize, col: usize| i32::from(patch[row][col]);
    let bias = (black - black * gain) as i32;
    let predicted = predictor(gain, bias);
    let offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    let neighbor = |row: usize, col: usize, (dy, dx)| {
        at(row.wrapping_add_signed(dy), col.wrapping_add_signed(dx))
    };
    let center = center_prediction(at(5, 5), gain, bias);
    let reference = offsets.map(|offset| neighbor(5, 5, offset) - center);
    let mut costs = [0; 4];
    for row in (1..=9).step_by(2) {
        for col in (1..=9).step_by(2) {
            let target = predicted(at(row, col));
            for (index, offset) in offsets.into_iter().enumerate() {
                costs[index] += (neighbor(row, col, offset) - target - reference[index]).unsigned_abs();
            }
        }
    }
    costs
}

pub fn extended_direction([left, right, up, down]: [u32; 4]) -> u8 {
    if left.max(right) < up.min(down) {
        1
    } else if up.max(down) < left.min(right) {
        2
    } else {
        3
    }
}
