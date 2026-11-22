//! Non-CA C1v2012 basic green refinement, one interior R/B site.

pub(super) fn refine_green_basic(
    rgb: &[[f32; 3]],
    stride: usize,
    x: usize,
    y: usize,
    odd_row: bool,
) -> f32 {
    let sample = |row: isize, col: isize, channel| {
        let index = (y.wrapping_add_signed(row)) * stride + x.wrapping_add_signed(col);
        rgb[index][channel]
    };
    let primary = |row, col| sample(row, col, if odd_row { 2 } else { 0 });
    let opposite = |row, col| sample(row, col, if odd_row { 0 } else { 2 });
    let green = |row, col| sample(row, col, 1);
    let abs = |first: f32, second: f32| (first - second).abs();

    let center = green(0, 0);
    let (up, down, left, right) = (green(-1, 0), green(1, 0), green(0, -1), green(0, 1));
    let (ul, ur, dl, dr) = (green(-1, -1), green(-1, 1), green(1, -1), green(1, 1));
    let (uul, uur, ddl, ddr) = (green(-2, -1), green(-2, 1), green(2, -1), green(2, 1));
    let (ull, urr, dll, drr) = (green(-1, -2), green(-1, 2), green(1, -2), green(1, 2));
    let chroma = primary(0, 0);
    let vertical = abs(down, center) + abs(up, center);
    let horizontal = abs(right, center) + abs(left, center);
    let diagonal_back = abs(primary(2, -2), chroma) + abs(primary(-2, 2), chroma)
        + abs(opposite(1, -1), opposite(-1, 1)) + abs(up, left) + abs(down, right);
    let diagonal_slash = abs(primary(2, 2), chroma) + abs(primary(-2, -2), chroma)
        + abs(opposite(-1, -1), opposite(1, 1)) + abs(up, right) + abs(down, left);
    let gradients = [
        diagonal_back,
        diagonal_slash,
        abs(ddl, center) + abs(uur, center) + abs(up, dl) + abs(down, ur),
        abs(dll, center) + abs(urr, center) + abs(left, ur) + abs(right, dl),
        abs(ddr, center) + abs(uul, center) + abs(down, ul) + abs(up, dr),
        abs(drr, center) + abs(ull, center) + abs(right, ul) + abs(left, dr),
    ];
    let axial = vertical.min(horizontal);
    let minimum = gradients.into_iter().fold(axial, f32::min);
    if axial <= minimum {
        return center;
    }
    let one_third = f32::from_bits(0x3eaa_aaab);
    let axial_average = if vertical < horizontal {
        ((center + up) + down) * one_third
    } else {
        ((center + left) + right) * one_third
    };
    let candidates = [
        ((2.0 * center + ur) + dl) * 0.25,
        ((2.0 * center + ul) + dr) * 0.25,
        ((ddl + center) + uur) * one_third,
        ((dll + center) + urr) * one_third,
        ((center + uul) + ddr) * one_third,
        ((center + ull) + drr) * one_third,
    ];
    let selected = (0..gradients.len()).rev().find(|&index| gradients[index] <= minimum).unwrap();
    (candidates[selected] + axial_average) * 0.5
}
