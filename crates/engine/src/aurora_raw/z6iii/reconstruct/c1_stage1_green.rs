//! C1v2012 scalar green apply, preserving the float32 operation order.

use super::c1_stage1_direction::{direction_costs, early_direction, extended_costs, extended_direction, low_curvature_direction};
use super::c1_stage1_params::Stage1Params;

fn add_product(value: f32, multiplier: f32, sum: f32) -> f32 {
    value * multiplier + sum
}

pub fn stage1_rgb(
    patch: &[[u16; 11]; 11],
    odd_row: bool,
    odd_col: bool,
    params: Stage1Params,
    auxiliary_difference: f32,
) -> ([f32; 3], Option<u8>) {
    let center = f32::from(patch[5][5]);
    if odd_row != odd_col {
        return ([0.0, center, 0.0], None);
    }
    let gain = if odd_row { params.blue_gain } else { params.red_gain };
    let black = params.black_pivot;
    let costs = direction_costs(patch, gain, black);
    let direction = early_direction(costs, auxiliary_difference)
        .or_else(|| low_curvature_direction(patch, black))
        .unwrap_or_else(|| extended_direction(extended_costs(patch, gain, black)));
    let green = interpolate_green(patch, direction, params.noise_floor, black, gain);
    let rgb = if odd_row { [0.0, green, center] } else { [center, green, 0.0] };
    (rgb, Some(direction))
}

fn interpolate_green(
    patch: &[[u16; 11]; 11],
    direction: u8,
    noise_floor: f32,
    black: f32,
    gain: f32,
) -> f32 {
    let at = |dy: isize, dx: isize| i32::from(patch[(5 + dy) as usize][(5 + dx) as usize]);
    let center = at(0, 0);
    let axis = |vertical: bool| {
        let sample = |across, along| {
            if vertical { at(along, across) } else { at(across, along) }
        };
        let near = [sample(0, -1), sample(0, 1)];
        let far = [sample(0, -2), sample(0, 2)];
        let weights = [-1, 1].map(|sign| {
            let integer = 1 + (near[0] - near[1]).abs()
                + (sample(-1, 2 * sign) - sample(-1, 0)).abs()
                + (sample(1, 2 * sign) - sample(1, 0)).abs()
                + (sample(0, 3 * sign) - sample(0, sign)).abs();
            let far_delta = (sample(0, 2 * sign) - center).abs();
            // Only the horizontal-only branch scales the entire right-hand cost.
            if direction == 1 && sign == 1 {
                (integer + far_delta) as f32 * gain
            } else {
                add_product(far_delta as f32, gain, integer as f32)
            }
        });
        (weights, near.map(|value| value as f32), far.map(|value| ((value + center) >> 1) as f32))
    };
    let (weight, numerator, denominator) = match direction {
        1 | 2 => {
            let ([negative, positive], near, mid) = axis(direction == 2);
            (
                positive + negative,
                add_product(near[0], positive, near[1] * negative),
                add_product(mid[0], positive, mid[1] * negative),
            )
        }
        3 => {
            let ([x_neg, x_pos], x_near, x_mid) = axis(false);
            let ([y_neg, y_pos], y_near, y_mid) = axis(true);
            let x_pair = x_pos * x_neg;
            let y_pair = y_pos * y_neg;
            let (left, right, up, down) = (y_pair * x_pos, y_pair * x_neg, x_pair * y_pos, x_pair * y_neg);
            let weighted = |vertical: [f32; 2], horizontal: [f32; 2]| {
                let sum = add_product(vertical[0], up, vertical[1] * down);
                let sum = add_product(horizontal[0], left, sum);
                add_product(horizontal[1], right, sum)
            };
            let weight = left + right + up + down;
            (weight, weighted(y_near, x_near), weighted(y_mid, x_mid))
        }
        _ => unreachable!("green interpolation requires direction 1, 2 or 3"),
    };
    let floor = weight * noise_floor;
    let black_weight = weight * black;
    let numerator = floor.max(numerator + floor - black_weight);
    let denominator = floor.max(denominator + floor - black_weight);
    (numerator / denominator) * (center as f32 - black + noise_floor) - noise_floor + black
}
