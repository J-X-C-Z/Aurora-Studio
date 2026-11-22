/// CPU CLUT record, with blue-fastest interleaved code values and no PCS conversion.
pub struct CameraClut<'a> {
    sizes: [usize; 3],
    nodes: &'a [[u16; 3]],
}

pub fn sample_icc_curves(rgb: [f32; 3], curves: [&[f32]; 3]) -> Option<[f32; 3]> {
    if curves.iter().any(|curve| curve.len() < 2) || !rgb.into_iter().all(f32::is_finite) {
        return None;
    }
    Some(std::array::from_fn(|channel| {
        let samples = curves[channel];
        let span = samples.len() - 1;
        let position = rgb[channel] * span as f32;
        let base = (position as usize).min(span - 1);
        let weight = position - base as f32;
        ((1.0 - weight) * samples[base] + weight * samples[base + 1]) * f32::from_bits(0x3780_0080)
    }))
}

pub fn lab_v2_to_v4(normalized: [f32; 3]) -> [f32; 3] {
    normalized.map(|value| value * (65535.0_f32 / 65280.0))
}

pub fn lab_v4_to_icc_xyz(normalized: [f32; 3]) -> [f32; 3] {
    let [lightness, a_star, b_star] = normalized.map(f64::from);
    let fy = (lightness * 100.0 + 16.0) / 116.0;
    let fx = fy + (a_star * 255.0 - 128.0) * 0.002;
    let fz = fy - (b_star * 255.0 - 128.0) * 0.005;
    let inverse = |value: f64| {
        if value > 6.0 / 29.0 {
            value * (value * value)
        } else {
            (value - 4.0 / 29.0) * (108.0 / 841.0)
        }
    };
    let xyz = [0.9642 * inverse(fx), inverse(fy), 0.8249 * inverse(fz)];
    xyz.map(|value| (value / (65535.0 / 32768.0)) as f32)
}

pub fn lab_v4_to_icc_xyz_type5(normalized: [f32; 3]) -> [f32; 3] {
    let [lightness, a_star, b_star] = normalized.map(|value| value * 65535.0);
    let common = lightness * f32::from_bits(0x375c_b16a);
    let fx = (common + a_star * f32::from_bits(0x3702_8fdf)) + f32::from_bits(0xbdf1_ce24);
    let fy = common + f32::from_bits(0x3e0d_3dcb);
    let fz = (common + b_star * f32::from_bits(0xb7a3_33d6)) + f32::from_bits(0x3f47_267d);
    let inverse = |value: f32| {
        if value > f32::from_bits(0x3e53_dcb1) {
            (value * value) * value
        } else {
            (value - f32::from_bits(0x3e0d_3dcb)) * f32::from_bits(0x3e03_8027)
        }
    };
    let xyz = [fx, fy, fz].map(inverse);
    let scale = [f32::from_bits(0x46f6_d5d0), 32768.0, f32::from_bits(0x46d3_2ca5)];
    std::array::from_fn(|channel| (xyz[channel] * scale[channel]) * f32::from_bits(0x3780_0080))
}

fn blend_code(low: f32, high: f32, weight: f32) -> f32 {
    if low == 0.0 || high == 0.0 {
        return (1.0 - weight) * low + weight * high;
    }
    if weight == 1.0 {
        return high;
    }
    let product = f64::from(weight) * f64::from(high - low);
    let addend = f64::from(low);
    let sum = product + addend;
    let rounded = sum as f32;
    let adjacent = if f64::from(rounded) < sum { rounded.next_up() } else { rounded.next_down() };
    let result = if sum == (f64::from(rounded) + f64::from(adjacent)) * 0.5 {
        let correction = if product.abs() >= addend.abs() {
            (product - sum) + addend
        } else {
            (addend - sum) + product
        };
        if correction > 0.0 { sum.next_up() as f32 }
        else if correction < 0.0 { sum.next_down() as f32 }
        else { rounded }
    } else {
        rounded
    };
    result.clamp(low.min(high), low.max(high))
}

impl<'a> CameraClut<'a> {
    pub fn new(sizes: [usize; 3], nodes: &'a [[u16; 3]]) -> Option<Self> {
        let count = sizes.into_iter().try_fold(1usize, |count, size| {
            (size >= 2).then(|| count.checked_mul(size)).flatten()
        })?;
        (sizes[0] == sizes[2] && count == nodes.len()).then_some(Self { sizes, nodes })
    }

    pub fn sample(&self, rgb: [f32; 3]) -> Option<[f32; 3]> {
        if !rgb.into_iter().all(|value| (0.0..=1.0).contains(&value)) {
            return None;
        }
        let axis: [(usize, f32); 3] = std::array::from_fn(|channel| {
            let span = self.sizes[channel] - 1;
            let position = rgb[channel] * span as f32;
            let base = (position as usize).min(span - 1);
            (base, position - base as f32)
        });
        let [(red, red_weight), (green, green_weight), (blue, blue_weight)] = axis;
        let node = |red, green, blue| self.nodes[(red * self.sizes[1] + green) * self.sizes[2] + blue];
        let corners = [node(red, green, blue), node(red, green, blue + 1),
                       node(red + 1, green, blue), node(red + 1, green, blue + 1),
                       node(red, green + 1, blue), node(red, green + 1, blue + 1),
                       node(red + 1, green + 1, blue), node(red + 1, green + 1, blue + 1)];
        Some(std::array::from_fn(|channel| {
            let [blue00, blue10, blue01, blue11] = std::array::from_fn(|pair| {
                blend_code(f32::from(corners[2 * pair][channel]), f32::from(corners[2 * pair + 1][channel]), blue_weight)
            });
            blend_code(blend_code(blue00, blue10, red_weight), blend_code(blue01, blue11, red_weight), green_weight)
                * f32::from_bits(0x3780_0080)
        }))
    }

    pub fn sample_type1(&self, rgb: [f32; 3]) -> Option<[f32; 3]> {
        if !rgb.into_iter().all(f32::is_finite) {
            return None;
        }
        let peak = rgb.into_iter().fold(1.0_f32, f32::max);
        let scale = 1.0 / peak;
        let [(red, red_fraction), (green, green_fraction), (blue, blue_fraction)] =
            std::array::from_fn(|channel| {
                let position = (rgb[channel] * scale).clamp(0.0, 1.0) * (self.sizes[channel] - 1) as f32;
                let base = position as usize;
                (base, position - base as f32)
            });
        let fractions = [red_fraction, green_fraction, blue_fraction];
        let mut order = [1, 0, 2];
        if red_fraction > green_fraction { order.swap(0, 1); }
        if fractions[2] > fractions[order[0]] { order.rotate_right(1); }
        else if fractions[2] >= fractions[order[1]] { order.swap(1, 2); }
        let [largest, middle, smallest] = order.map(|axis| fractions[axis]);
        let weights = [1.0 - largest, largest - middle, middle - smallest, smallest];
        let mut corner = [red, green, blue];
        let node = |[red, green, blue]: [usize; 3]| self.nodes[(red * self.sizes[1] + green) * self.sizes[2] + blue];
        let mut vertices = [node(corner); 4];
        for (slot, axis) in vertices[1..].iter_mut().zip(order) {
            corner[axis] = (corner[axis] + 1).min(self.sizes[axis] - 1);
            *slot = node(corner);
        }
        Some(std::array::from_fn(|channel| {
            let [base, first, second, last] = vertices.map(|vertex| f32::from(vertex[channel]));
            let code = ((base * weights[0] + first * weights[1]) + second * weights[2]) + last * weights[3];
            (code * peak) * f32::from_bits(0x3780_0080)
        }))
    }
}
