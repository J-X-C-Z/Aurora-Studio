#![forbid(unsafe_code)]

fn chroma_cost(color: f32, green: f32, enabled: bool) -> f32 {
    if enabled {
        let denominator = ((color + green) + 1.0).max(1.0);
        (color - green).abs() * denominator.recip() + 0.03
    } else {
        1.0
    }
}

fn sample(plane: &[f32], stride: usize, row: usize, col: usize, dy: isize, dx: isize) -> f32 {
    let row = row.checked_add_signed(dy).unwrap();
    let col = col.checked_add_signed(dx).unwrap();
    assert!(col < stride);
    plane[row * stride + col]
}

pub fn green_position(
    first: &[f32],
    green: &[f32],
    second: &[f32],
    stride: usize,
    row: usize,
    col: usize,
    offset: f32,
) -> ([f32; 2], [bool; 2]) {
    let near = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let [gu, gd, gl, gr] = near.map(|(dy, dx)| sample(green, stride, row, col, dy, dx));
    let center = sample(green, stride, row, col, 0, 0);
    let vertical = (gu - gd).abs() + 1.0;
    let horizontal = (gr - gl).abs() + 1.0;
    let axes = [vertical, vertical, horizontal, horizontal];
    let far = near.map(|(dy, dx)| sample(green, stride, row, col, dy * 2, dx * 2));
    let norms: [f32; 4] = std::array::from_fn(|index| (far[index] - center).abs() + axes[index]);
    let green_sum = (gu + gd) + (gr + gl);
    let eval = |plane: &[f32]| {
        let [pu, pd, pl, pr] = near.map(|(dy, dx)| sample(plane, stride, row, col, dy, dx));
        let enabled = ((pr + pl) + (pd + pu)) * 0.75 < green_sum;
        let [cu, cd, cl, cr]: [f32; 4] = std::array::from_fn(|index| {
            chroma_cost([pu, pd, pl, pr][index], [gu, gd, gl, gr][index], enabled) * norms[index]
        });
        let wu = cd * (cl * cr);
        let wd = cu * (cl * cr);
        let wl = cr * (cd * cu);
        let wr = cl * (cd * cu);
        let floor = ((wl + wr) + (wd + wu)) * offset;
        let numerator = ((wd * pd + wu * pu) + floor) + (wr * pr + wl * pl);
        let denominator = ((gu * wu + wd * gd) + floor) + (wr * gr + wl * gl);
        let output = (denominator.max(floor).recip() * numerator) * (center + offset) - offset;
        (output, enabled)
    };
    let (first_value, first_mask) = eval(first);
    let (second_value, second_mask) = eval(second);
    ([first_value, second_value], [first_mask, second_mask])
}

pub fn diagonal_position(
    color: &[f32],
    green: &[f32],
    guide: &[f32],
    stride: usize,
    position: (usize, usize),
    offset: f32,
    noise: f32,
) -> (f32, [bool; 2]) {
    let (row, col) = position;
    let near = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
    let colors = near.map(|(dy, dx)| sample(color, stride, row, col, dy, dx));
    let greens = near.map(|(dy, dx)| sample(green, stride, row, col, dy, dx));
    let [pul, pur, pdl, pdr] = colors;
    let [gul, gur, gdl, gdr] = greens;
    let selected = (pul - pdr).abs() < (pdl - pur).abs();
    let enabled = ((pdr + pdl) + (pur + pul)) * 0.75 < (gdr + gdl) + (gur + gul);
    let [first, second] = if selected { [0, 3] } else { [2, 1] };
    let center = sample(green, stride, row, col, 0, 0);
    let guide_center = sample(guide, stride, row, col, 0, 0);
    let cost = |index: usize| {
        let (dy, dx) = near[index];
        let guide_far = sample(guide, stride, row, col, dy * 2, dx * 2);
        let norm = (guide_far - guide_center).abs() + ((greens[index] - center).abs() + noise);
        chroma_cost(colors[index], greens[index], enabled) * norm
    };
    let first_cost = cost(first);
    let second_cost = cost(second);
    let floor = (second_cost + first_cost) * offset;
    let denominator = (greens[second] * first_cost + second_cost * greens[first]) + floor;
    let numerator = (colors[second] * first_cost + second_cost * colors[first]) + floor;
    let output = (denominator.max(floor).recip() * numerator) * (offset + center) - offset;
    (output, [selected, enabled])
}

#[cfg(all(test, rb_golden))]
mod golden_tests;
