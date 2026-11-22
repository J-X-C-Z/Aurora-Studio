/// Input planes contain the reconstructed 1-pixel RGB neighbourhood around every
/// output sample. Stage 3 owns the wider (six-pixel) interpolation halo.
pub(super) fn core(
    planes: [&[f32]; 3], stride: usize, width: usize, height: usize,
    gains: [f32; 3], threshold: f32, noise_floor: f32,
) -> Vec<[f32; 3]> {
    let [red, green, blue] = planes;
    let [wr, wg, wb] = gains.map(f32::recip);
    let fused = |value: f32, weight: f32, accumulated: f32| {
        (f64::from(value) * f64::from(weight) + f64::from(accumulated)) as f32
    };
    let guide = |i: usize| {
        {
            let rg = fused(red[i], wr, (green[i] * 2.0) * wg);
            (fused(blue[i], wb, rg) + 1.0).max(1.0)
        }
    };
    let four_t = 4.0 * threshold;
    let rolloff = 1.0 / (3.0 * threshold);
    let mut output = Vec::with_capacity(width * height);
    for y in 6..height + 6 {
        for x in 6..width + 6 {
            let i = y * stride + x;
            let mut peak = 0.0f32;
            for row in y-1..=y+1 {
                for col in x-1..=x+1 {
                    if (row ^ col) & 1 == 1 {
                        peak = peak.max(green[row * stride + col] * wg);
                    }
                }
            }
            let pixel = [red[i], green[i], blue[i]];
            if peak >= four_t {
                output.push(pixel);
                continue;
            }
            let c = guide(i);
            let nw = guide(i - stride - 1);
            let ne = guide(i - stride + 1);
            let sw = guide(i + stride - 1);
            let se = guide(i + stride + 1);
            let twice = c + c;
            let diag = if (twice - nw - se).abs() < (twice - ne - sw).abs() {
                nw + se
            } else {
                ne + sw
            };
            let delta = (c - 0.25 * (diag + twice)) * 20.0;
            let cap = c.min(noise_floor);
            let limited = delta * cap / (cap + delta.abs());
            let weight = if peak < threshold { 1.0 } else { (four_t - peak) * rolloff };
            let gain = (limited * weight).mul_add(0.75, c) / c;
            output.push(pixel.map(|v| v * gain));
        }
    }
    output
}
