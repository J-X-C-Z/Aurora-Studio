pub(super) fn apply(
    planes: [&mut [f32]; 3],
    stride: usize,
    width: usize,
    height: usize,
    gains: [f32; 3],
    offset: f32,
    noise: f32,
    control: i32,
) {
    let [red, green, blue] = planes;
    let red_scale = 1.0 / gains[0];
    let green_scale = 2.0 / gains[1];
    let native_green_scale = 1.0 / gains[1];
    let blue_scale = 1.0 / gains[2];
    let guide = |index: usize| {
        ((blue[index] * blue_scale + red[index] * red_scale)
            + (green[index] * green_scale + 1.0)).max(1.0)
    };
    let weight_scale = 1.0 / (offset * 3.0);
    let mut multipliers = Vec::with_capacity(width * height);
    for row in 6..6 + height {
        for column in 6..6 + width {
            let index = row * stride + column;
            let mut peak = f32::NEG_INFINITY;
            for delta_row in row - 1..=row + 1 {
                for delta_col in column - 1..=column + 1 {
                    let sampled = if (delta_row + delta_col) & 1 != 0 {
                        green[delta_row * stride + delta_col] * native_green_scale
                    } else {
                        0.0
                    };
                    peak = peak.max(sampled);
                }
            }
            let weight = match control {
                2 | 4 => 1.0,
                5 => 0.0,
                _ => ((offset * 4.0 - peak) * weight_scale).clamp(0.0, 1.0),
            } * 0.75;
            let center = guide(index);
            let diagonal_first = guide(index - stride + 1) + guide(index + stride - 1);
            let diagonal_second = guide(index - stride - 1) + guide(index + stride + 1);
            let doubled = center + center;
            let chosen = if (doubled - diagonal_second).abs() < (doubled - diagonal_first).abs() {
                diagonal_second
            } else {
                diagonal_first
            };
            let delta = (center - (chosen + doubled) * 0.25) * 20.0;
            let limited = center.min(noise);
            let blend = ((1.0 / (delta.abs() + limited) * limited) * delta) * weight;
            multipliers.push((blend + center) * (1.0 / center));
        }
    }
    for row in 0..height {
        for column in 0..width {
            let index = (row + 6) * stride + column + 6;
            let factor = multipliers[row * width + column];
            red[index] *= factor;
            green[index] *= factor;
            blue[index] *= factor;
        }
    }
}
