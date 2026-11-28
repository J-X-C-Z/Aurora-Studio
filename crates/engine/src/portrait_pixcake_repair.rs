//! Same-frame PF90080 / PF90118 producers for the PixelCake repair consumers.
//!
//! This module only prepares the named planes. The final composition stays in
//! the recovered vendor consumers:
//!
//! * PF90080: `aurora_portrait::kernels::skin_color::oil_shine_mask_blend`.
//!   `effect_rgb` is a **full-frame** packed RGB target colour (`w*h*3`) and
//!   `mask` is a **full-frame** u8 weight (`w*h`). The shader computes
//!   `source + (mask/255) * degree * (effect - source)`.
//! * PF90118: `portrait_repair_stage::lip_line` extracts the ROI tile at
//!   `origin`, calls `skin_color::shared_min_mask_composite` with
//!   `w = min(mask_a, mask_b)` and pastes the tile back with `neck_paste`.
//!   Both masks are **ROI-sized** (`repair.pixel_count()`), not full-frame.
//!
//! The vendor's FaceParsing 0x92/0xA3/0xB4 -> mask_a/mask_b assignment is still
//! unrecovered (`skin_color::LIP_LINE_MASK_PRODUCER_OPEN`). The two planes
//! produced here are explicit geometric weights that satisfy the compositor
//! contract; they are not claimed to be the vendor's FaceParsing planes.

use aurora_portrait::{Image, Refusal};

use crate::RepairOutput;

const SKIN_MASK: &str = "PF90080_SKIN_MASK";
const FACE_POINTS: &str = "PF90080_POINTS194";
const UPPER_LIP_MASK: &str = "PF90118_UPPER_LIP_MASK";
const LOWER_LIP_MASK: &str = "PF90118_LOWER_LIP_MASK";
const LIP_POINTS: &str = "PF90118_POINTS194";
const LIP_ROI: &str = "PF90118_LIP_ROI";

fn clamp_byte(value: f32) -> u8 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value.round_ties_even() as u8
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn point(points: &[[f32; 2]], index: usize) -> Option<[f32; 2]> {
    points.get(index).copied().filter(|p| p[0].is_finite() && p[1].is_finite())
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn ellipse(x: f32, y: f32, center: [f32; 2], rx: f32, ry: f32) -> f32 {
    if rx <= 0.0 || ry <= 0.0 {
        return 0.0;
    }
    let dx = (x - center[0]) / rx;
    let dy = (y - center[1]) / ry;
    let d = dx * dx + dy * dy;
    if d >= 1.0 {
        0.0
    } else {
        (1.0 - d).sqrt()
    }
}

fn segment_weight(x: f32, y: f32, a: [f32; 2], b: [f32; 2], radius: f32) -> f32 {
    let vx = b[0] - a[0];
    let vy = b[1] - a[1];
    let length2 = vx * vx + vy * vy;
    if length2 <= 0.0 || radius <= 0.0 {
        return 0.0;
    }
    let t = (((x - a[0]) * vx + (y - a[1]) * vy) / length2).clamp(0.0, 1.0);
    let d = distance([x, y], [a[0] + t * vx, a[1] + t * vy]);
    1.0 - smoothstep(radius * 0.35, radius, d)
}

/// Normalized separable box blur. `weight` gates the samples: a zero-weight
/// neighbourhood falls back to the untouched value, so out-of-region pixels
/// keep their original colour and cannot bleed into the baseline.
fn weighted_blur(
    values: &[f32],
    weight: &[f32],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<f32> {
    let mut horizontal = values.to_vec();
    let mut prefix_v = Vec::with_capacity(width + 1);
    let mut prefix_w = Vec::with_capacity(width + 1);
    for y in 0..height {
        let row = y * width;
        prefix_v.clear();
        prefix_w.clear();
        prefix_v.push(0.0f32);
        prefix_w.push(0.0f32);
        for x in 0..width {
            let i = row + x;
            prefix_v.push(prefix_v[x] + values[i] * weight[i]);
            prefix_w.push(prefix_w[x] + weight[i]);
        }
        for x in 0..width {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius + 1).min(width);
            let sum_v = prefix_v[hi] - prefix_v[lo];
            let sum_w = prefix_w[hi] - prefix_w[lo];
            horizontal[row + x] = if sum_w > 1.0e-6 {
                sum_v / sum_w
            } else {
                values[row + x]
            };
        }
    }

    let mut out = horizontal.clone();
    prefix_v.clear();
    prefix_w.clear();
    prefix_v.resize(height + 1, 0.0);
    prefix_w.resize(height + 1, 0.0);
    for x in 0..width {
        prefix_v[0] = 0.0;
        prefix_w[0] = 0.0;
        for y in 0..height {
            let i = y * width + x;
            prefix_v[y + 1] = prefix_v[y] + horizontal[i] * weight[i];
            prefix_w[y + 1] = prefix_w[y] + weight[i];
        }
        for y in 0..height {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius + 1).min(height);
            let sum_v = prefix_v[hi] - prefix_v[lo];
            let sum_w = prefix_w[hi] - prefix_w[lo];
            out[y * width + x] = if sum_w > 1.0e-6 {
                sum_v / sum_w
            } else {
                horizontal[y * width + x]
            };
        }
    }
    out
}

/// T-zone emphasis from the 194-point face mesh. Eyes and lips are suppressed
/// so that a specular catchlight or lip gloss is not treated as facial oil.
fn oil_shine_focus(points: &[[f32; 2]], width: usize, height: usize) -> Vec<f32> {
    let mut focus = vec![0.3f32; width * height];
    let (Some(eye_l), Some(eye_r)) = (point(points, 26), point(points, 44)) else {
        return focus;
    };
    let (Some(brow_l), Some(brow_r)) = (point(points, 4), point(points, 9)) else {
        return focus;
    };
    let (Some(nose_top), Some(nose_tip)) = (point(points, 54), point(points, 57)) else {
        return focus;
    };
    let (Some(lip_l), Some(lip_r)) = (point(points, 73), point(points, 83)) else {
        return focus;
    };
    let chin = point(points, 125).unwrap_or([(lip_l[0] + lip_r[0]) * 0.5, lip_r[1] + 20.0]);

    let eye_dist = distance(eye_l, eye_r).max(16.0);
    let brow_mid = [(brow_l[0] + brow_r[0]) * 0.5, (brow_l[1] + brow_r[1]) * 0.5];
    let forehead = [brow_mid[0], brow_mid[1] - eye_dist * 0.45];
    let lip_mid = [(lip_l[0] + lip_r[0]) * 0.5, (lip_l[1] + lip_r[1]) * 0.5];
    let cheek_l = [lip_l[0] - eye_dist * 0.18, nose_tip[1] - eye_dist * 0.06];
    let cheek_r = [lip_r[0] + eye_dist * 0.18, nose_tip[1] - eye_dist * 0.06];

    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let mut value = 0.3;
            value += 0.65 * ellipse(fx, fy, forehead, eye_dist * 1.1, eye_dist * 0.55);
            value += 0.7
                * segment_weight(fx, fy, brow_mid, nose_tip, eye_dist * 0.28);
            value += 0.45 * ellipse(fx, fy, cheek_l, eye_dist * 0.4, eye_dist * 0.3);
            value += 0.45 * ellipse(fx, fy, cheek_r, eye_dist * 0.4, eye_dist * 0.3);
            value += 0.4 * ellipse(fx, fy, chin, eye_dist * 0.5, eye_dist * 0.3);
            for eye in [eye_l, eye_r] {
                value *= 1.0 - 0.95 * ellipse(fx, fy, eye, eye_dist * 0.42, eye_dist * 0.28);
            }
            value *= 1.0
                - 0.95 * ellipse(fx, fy, lip_mid, eye_dist * 0.62, eye_dist * 0.3);
            focus[y * width + x] = value.clamp(0.0, 1.0);
        }
    }
    focus
}

/// Produces the full-frame PF90080 planes consumed by
/// `skin_color::oil_shine_mask_blend`.
pub(crate) fn build_pf90080_oil_shine_output(
    frame: &Image,
    skin_mask: &[u8],
    face_points194: &[[f32; 2]],
) -> Result<RepairOutput, Refusal> {
    frame.validate()?;
    let pixels = frame.pixel_count();
    if skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch(SKIN_MASK));
    }
    if !skin_mask.iter().any(|value| *value > 0) {
        return Err(Refusal::MissingInput(SKIN_MASK));
    }
    if face_points194.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(FACE_POINTS));
    }

    let (width, height) = (frame.width as usize, frame.height as usize);
    let focus = oil_shine_focus(face_points194, width, height);
    let mut region = vec![0.0f32; pixels];
    let mut source_r = vec![0.0f32; pixels];
    let mut source_g = vec![0.0f32; pixels];
    let mut source_b = vec![0.0f32; pixels];
    let mut luma = vec![0.0f32; pixels];
    for p in 0..pixels {
        let r = frame.pixels[p * 4] as f32;
        let g = frame.pixels[p * 4 + 1] as f32;
        let b = frame.pixels[p * 4 + 2] as f32;
        source_r[p] = r;
        source_g[p] = g;
        source_b[p] = b;
        luma[p] = 0.299 * r + 0.587 * g + 0.114 * b;
        region[p] = skin_mask[p] as f32 / 255.0 * focus[p];
    }

    // Local + global diffuse skin baseline. Scale the box radius to the frame so
    // wider specular patches on the forehead or nose bridge do not dominate their
    // own baseline.
    let base_radius = (width.min(height) / 5).clamp(10, 48);
    let base_r = weighted_blur(&source_r, &region, width, height, base_radius);
    let base_g = weighted_blur(&source_g, &region, width, height, base_radius);
    let base_b = weighted_blur(&source_b, &region, width, height, base_radius);

    let mut oil = vec![0.0f32; pixels];
    for p in 0..pixels {
        let baseline = 0.299 * base_r[p] + 0.587 * base_g[p] + 0.114 * base_b[p];
        let excess = luma[p] - baseline;
        let maximum = source_r[p].max(source_g[p]).max(source_b[p]);
        let minimum = source_r[p].min(source_g[p]).min(source_b[p]);
        let saturation = if maximum > 1.0 {
            (maximum - minimum) / maximum
        } else {
            0.0
        };
        // Specular oil lifts luma and washes out chroma. A coloured skin pixel
        // with the same luma excess gets a smaller response.
        let strength = excess * (1.0 - saturation.clamp(0.0, 1.0));
        oil[p] = smoothstep(5.0, 26.0, strength) * region[p];
    }

    let ones = vec![1.0f32; pixels];
    let feathered = weighted_blur(&oil, &ones, width, height, 1);
    let mask: Vec<u8> = feathered
        .iter()
        .zip(skin_mask)
        .map(|(&weight, &skin)| clamp_byte(weight * (skin as f32 / 255.0) * 255.0))
        .collect();

    // Estimate the matte diffuse target from surrounding non-specular skin pixels,
    // then put back a fraction of the clean skin texture.
    let diffuse_weight: Vec<f32> = region
        .iter()
        .zip(&oil)
        .map(|(&r, &o)| r * (1.0 - o * 0.95).max(0.01))
        .collect();
    let matte_r = weighted_blur(&source_r, &diffuse_weight, width, height, base_radius);
    let matte_g = weighted_blur(&source_g, &diffuse_weight, width, height, base_radius);
    let matte_b = weighted_blur(&source_b, &diffuse_weight, width, height, base_radius);
    let mut clean_r = vec![0.0f32; pixels];
    let mut clean_g = vec![0.0f32; pixels];
    let mut clean_b = vec![0.0f32; pixels];
    for p in 0..pixels {
        let weight = feathered[p] * region[p];
        clean_r[p] = source_r[p] + weight * (matte_r[p] - source_r[p]);
        clean_g[p] = source_g[p] + weight * (matte_g[p] - source_g[p]);
        clean_b[p] = source_b[p] + weight * (matte_b[p] - source_b[p]);
    }
    let clean_luma: Vec<f32> = (0..pixels)
        .map(|p| 0.299 * clean_r[p] + 0.587 * clean_g[p] + 0.114 * clean_b[p])
        .collect();
    let smooth_luma = weighted_blur(&clean_luma, &region, width, height, 3);

    let mut effect_rgb = vec![0u8; pixels * 3];
    for p in 0..pixels {
        let high_frequency = (clean_luma[p] - smooth_luma[p]) * 0.35;
        effect_rgb[p * 3] = clamp_byte(matte_r[p] + high_frequency);
        effect_rgb[p * 3 + 1] = clamp_byte(matte_g[p] + high_frequency);
        effect_rgb[p * 3 + 2] = clamp_byte(matte_b[p] + high_frequency);
    }

    Ok(RepairOutput::OilShine { effect_rgb, mask })
}

fn lip_roi(
    upper: &[u8],
    lower: &[u8],
    points: &[[f32; 2]],
    width: u32,
    height: u32,
) -> Result<[u32; 4], Refusal> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let p = y * width as usize + x;
            if upper[p].max(lower[p]) <= 51 {
                continue;
            }
            min_x = min_x.min(x as f32);
            min_y = min_y.min(y as f32);
            max_x = max_x.max(x as f32 + 1.0);
            max_y = max_y.max(y as f32 + 1.0);
        }
    }
    if !min_x.is_finite() {
        return Err(Refusal::MissingInput(UPPER_LIP_MASK));
    }

    // The 194-point mouth corners tighten the ROI when the parsing masks are
    // slightly wider than the actual lip.
    let span = (max_x - min_x).max(max_y - min_y);
    for index in [73usize, 83] {
        if let Some(corner) = point(points, index) {
            if corner[0] >= min_x - span && corner[0] <= max_x + span
                && corner[1] >= min_y - span && corner[1] <= max_y + span
            {
                min_x = min_x.min(corner[0]);
                min_y = min_y.min(corner[1]);
                max_x = max_x.max(corner[0] + 1.0);
                max_y = max_y.max(corner[1] + 1.0);
            }
        }
    }

    let padding = ((span * 0.15).round() as i64).max(2);
    let x0 = ((min_x - padding as f32).max(0.0).floor() as u64).min(width as u64) as u32;
    let y0 = ((min_y - padding as f32).max(0.0).floor() as u64).min(height as u64) as u32;
    let x1 = ((max_x + padding as f32).ceil() as u64).min(width as u64) as u32;
    let y1 = ((max_y + padding as f32).ceil() as u64).min(height as u64) as u32;
    let roi_w = x1.saturating_sub(x0);
    let roi_h = y1.saturating_sub(y0);
    if roi_w == 0 || roi_h == 0 {
        return Err(Refusal::ValueOutOfRange(LIP_ROI));
    }
    Ok([x0, y0, roi_w, roi_h])
}

/// Directional edge-preserving lip smoothing. Horizontal neighbours carry more
/// weight than vertical neighbours, which flattens vertical lip grooves while
/// leaving the lip peak and vermilion border (mostly horizontal structures)
/// intact. Colour-distance gating protects the remaining sharp edges.
fn smooth_lip_tile(pixels: &mut [u8], coverage: &[f32], width: usize, height: usize) {
    const RADIUS_X: i32 = 3;
    const RADIUS_Y: i32 = 2;
    const SIGMA_X: f32 = 2.5;
    const SIGMA_Y: f32 = 0.8;
    const SIGMA_RANGE: f32 = 55.0;

    for _ in 0..2 {
        let input = pixels.to_vec();
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                if coverage[index] < 0.1 {
                    continue;
                }
                let mut sum = [0.0f32; 3];
                let mut total = 0.0f32;
                for dy in -RADIUS_Y..=RADIUS_Y {
                    for dx in -RADIUS_X..=RADIUS_X {
                        let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                        let neighbour = ny * width + nx;
                        let mask = coverage[neighbour];
                        if mask < 0.1 {
                            continue;
                        }
                        let sx = dx as f32 / SIGMA_X;
                        let sy = dy as f32 / SIGMA_Y;
                        let spatial = (-0.5 * (sx * sx + sy * sy)).exp();
                        let mut range = 1.0f32;
                        for channel in 0..3 {
                            let delta = (input[index * 4 + channel] as f32
                                - input[neighbour * 4 + channel] as f32)
                                / SIGMA_RANGE;
                            range *= (-0.5 * delta * delta).exp();
                        }
                        let weight = spatial * mask * range;
                        if weight <= 0.0 {
                            continue;
                        }
                        for channel in 0..3 {
                            sum[channel] += weight * input[neighbour * 4 + channel] as f32;
                        }
                        total += weight;
                    }
                }
                if total > 0.0 {
                    for channel in 0..3 {
                        pixels[index * 4 + channel] = clamp_byte(sum[channel] / total);
                    }
                }
            }
        }
    }
}

/// Produces the ROI-sized PF90118 planes consumed by `portrait_repair_stage`.
pub(crate) fn build_pf90118_lip_line_output(
    frame: &Image,
    upper_lip_mask: &[u8],
    lower_lip_mask: &[u8],
    face_points194: &[[f32; 2]],
) -> Result<RepairOutput, Refusal> {
    frame.validate()?;
    let pixels = frame.pixel_count();
    if upper_lip_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch(UPPER_LIP_MASK));
    }
    if lower_lip_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch(LOWER_LIP_MASK));
    }
    if !upper_lip_mask.iter().chain(lower_lip_mask).any(|value| *value > 0) {
        return Err(Refusal::MissingInput(UPPER_LIP_MASK));
    }
    if face_points194.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(LIP_POINTS));
    }

    let [x0, y0, roi_w, roi_h] =
        lip_roi(upper_lip_mask, lower_lip_mask, face_points194, frame.width, frame.height)?;
    let roi_pixels = roi_w as usize * roi_h as usize;

    let mut tile_pixels = Vec::with_capacity(roi_pixels * 4);
    let mut coverage = vec![0.0f32; roi_pixels];
    for row in 0..roi_h as usize {
        for column in 0..roi_w as usize {
            let source_index = (y0 as usize + row) * frame.width as usize + x0 as usize + column;
            tile_pixels.extend_from_slice(&frame.pixels[source_index * 4..source_index * 4 + 4]);
            coverage[row * roi_w as usize + column] = upper_lip_mask[source_index]
                .max(lower_lip_mask[source_index]) as f32
                / 255.0;
        }
    }

    let mut repair_pixels = tile_pixels.clone();
    smooth_lip_tile(&mut repair_pixels, &coverage, roi_w as usize, roi_h as usize);

    // The compositor uses min(mask_a, mask_b). mask_a is the feathered lip
    // coverage; mask_b is the eroded coverage that keeps the vermilion border
    // at source colour. Both are ROI-sized.
    let ones = vec![1.0f32; roi_pixels];
    let feathered = weighted_blur(&coverage, &ones, roi_w as usize, roi_h as usize, 1);
    let mut mask_a = vec![0u8; roi_pixels];
    let mut mask_b = vec![0u8; roi_pixels];
    for y in 0..roi_h as usize {
        for x in 0..roi_w as usize {
            let index = y * roi_w as usize + x;
            mask_a[index] = clamp_byte(feathered[index].max(coverage[index]) * 255.0);
            let mut inner = coverage[index];
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, roi_w as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, roi_h as i32 - 1) as usize;
                    inner = inner.min(coverage[ny * roi_w as usize + nx]);
                }
            }
            mask_b[index] = clamp_byte(inner * 255.0);
        }
    }

    let repair = Image::new(roi_w, roi_h, repair_pixels)
        .ok_or(Refusal::ShapeMismatch(LIP_ROI))?;
    Ok(RepairOutput::LipLine {
        repair,
        origin: [x0, y0],
        mask_a,
        mask_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RepairProducerInputs;
    use aurora_portrait::kernels::{neck_paste, skin_color};

    const SIDE: usize = 64;

    fn points() -> Vec<[f32; 2]> {
        let mut points = vec![[32.0, 32.0]; 194];
        points[4] = [22.0, 18.0];
        points[9] = [42.0, 18.0];
        points[26] = [22.0, 26.0];
        points[44] = [42.0, 26.0];
        points[54] = [32.0, 30.0];
        points[57] = [32.0, 36.0];
        points[73] = [20.0, 44.0];
        points[83] = [44.0, 44.0];
        points[125] = [32.0, 56.0];
        points
    }

    fn oil_frame() -> (Image, Vec<u8>) {
        let mut image = Image::solid(SIDE as u32, SIDE as u32, [20, 20, 20, 255]);
        let mut mask = vec![0u8; SIDE * SIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                let dx = (x as f32 - 31.5) / 26.0;
                let dy = (y as f32 - 31.5) / 30.0;
                if dx * dx + dy * dy > 1.0 {
                    continue;
                }
                let index = y * SIDE + x;
                mask[index] = 255;
                let mut colour = [150u8, 110, 95];
                if (28..36).contains(&x) && (10..40).contains(&y) {
                    for channel in &mut colour {
                        *channel = channel.saturating_add(70);
                    }
                }
                image.pixels[index * 4..index * 4 + 3].copy_from_slice(&colour);
            }
        }
        (image, mask)
    }

    fn lip_frame() -> (Image, Vec<u8>, Vec<u8>) {
        let mut image = Image::solid(SIDE as u32, SIDE as u32, [150, 110, 95, 255]);
        let mut upper = vec![0u8; SIDE * SIDE];
        let mut lower = vec![0u8; SIDE * SIDE];
        for y in 38..50usize {
            for x in 20..44usize {
                let index = y * SIDE + x;
                if y < 44 {
                    upper[index] = 255;
                } else {
                    lower[index] = 255;
                }
                let stripe = if x % 4 < 2 { 22i16 } else { -22 };
                for channel in 0..3 {
                    let value = (170i16 + stripe) as u8;
                    let _ = value;
                    image.pixels[index * 4 + channel] = match channel {
                        0 => (170i16 + stripe) as u8,
                        1 => (70i16 + stripe) as u8,
                        _ => (80i16 + stripe) as u8,
                    };
                }
            }
        }
        (image, upper, lower)
    }

    fn luma(pixel: &[u8]) -> f32 {
        0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32
    }

    fn compose_lip(source: &Image, output: &RepairOutput) -> Image {
        let RepairOutput::LipLine {
            repair,
            origin: [x, y],
            mask_a,
            mask_b,
        } = output
        else {
            panic!("expected LipLine output");
        };
        let right = x + repair.width;
        let bottom = y + repair.height;
        let mut tile_pixels = Vec::new();
        for row in *y..bottom {
            let offset = row as usize * source.width as usize;
            tile_pixels.extend_from_slice(
                &source.pixels[(offset + *x as usize) * 4..(offset + right as usize) * 4],
            );
        }
        let tile = Image::new(repair.width, repair.height, tile_pixels).unwrap();
        let composed =
            skin_color::shared_min_mask_composite(&tile, repair, mask_a, mask_b).unwrap();
        neck_paste::paste(source, &composed, *x, *y).unwrap()
    }

    fn stripe_variance(image: &Image) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for y in 41..47usize {
            let mut row_sum = 0.0f32;
            for x in 25..39usize {
                row_sum += image.pixels[(y * SIDE + x) * 4] as f32;
            }
            let mean = row_sum / 14.0;
            for x in 25..39usize {
                let value = image.pixels[(y * SIDE + x) * 4] as f32 - mean;
                sum += value * value;
                count += 1;
            }
        }
        sum / count as f32
    }

    #[test]
    fn pf90080_mats_t_zone_highlight_and_binds_to_vendor_consumer() {
        let (frame, skin_mask) = oil_frame();
        let output = build_pf90080_oil_shine_output(&frame, &skin_mask, &points()).unwrap();
        let RepairOutput::OilShine { effect_rgb, mask } = &output else {
            panic!("expected OilShine output");
        };
        assert_eq!(effect_rgb.len(), frame.pixel_count() * 3);
        assert_eq!(mask.len(), frame.pixel_count());
        assert!(mask.iter().any(|value| *value > 0));
        assert_eq!(mask[2 * SIDE + 32], 0, "outside skin stays untouched");

        let mut inputs = RepairProducerInputs::default();
        inputs.insert(7, &frame, output.clone()).unwrap();
        assert!(inputs.receipt(90080).is_some());

        let composed = skin_color::oil_shine_mask_blend(&frame, effect_rgb, mask, 1.0).unwrap();
        let highlight = 20 * SIDE + 32;
        assert!(
            luma(&composed.pixels[highlight * 4..highlight * 4 + 4])
                < luma(&frame.pixels[highlight * 4..highlight * 4 + 4]) - 20.0,
            "oil highlight luma must drop"
        );
        let outside = 2 * SIDE + 2;
        assert_eq!(composed.pixel(2, 2), frame.pixel(2, 2));
        assert_eq!(composed.pixels[outside * 4..outside * 4 + 4], frame.pixels[outside * 4..outside * 4 + 4]);
    }

    #[test]
    fn pf90118_smooths_lip_stripes_and_binds_to_vendor_consumer() {
        let (frame, upper, lower) = lip_frame();
        let output = build_pf90118_lip_line_output(&frame, &upper, &lower, &points()).unwrap();
        let RepairOutput::LipLine {
            repair,
            origin,
            mask_a,
            mask_b,
        } = &output
        else {
            panic!("expected LipLine output");
        };
        let roi_pixels = repair.width as usize * repair.height as usize;
        assert!(repair.width > 0 && repair.height > 0);
        assert!(origin[0] + repair.width <= frame.width);
        assert!(origin[1] + repair.height <= frame.height);
        assert_eq!(mask_a.len(), roi_pixels, "mask_a is ROI-sized");
        assert_eq!(mask_b.len(), roi_pixels, "mask_b is ROI-sized");
        assert!(mask_a.iter().any(|value| *value > 0));
        assert!(mask_b.iter().any(|value| *value > 0));

        let mut inputs = RepairProducerInputs::default();
        inputs.insert(9, &frame, output.clone()).unwrap();
        assert!(inputs.receipt(90118).is_some());

        let before = stripe_variance(&frame);
        let composed = compose_lip(&frame, &output);
        let after = stripe_variance(&composed);
        assert!(
            after < before * 0.5,
            "vertical lip stripes must be reduced: before={before}, after={after}"
        );
        assert_eq!(composed.pixel(2, 2), frame.pixel(2, 2), "outside lip unchanged");
    }

    #[test]
    fn pf90080_rejects_empty_and_mis_shaped_skin_mask() {
        let (frame, mask) = oil_frame();
        let points = points();
        assert_eq!(
            build_pf90080_oil_shine_output(&frame, &[], &points),
            Err(Refusal::ShapeMismatch(SKIN_MASK))
        );
        assert_eq!(
            build_pf90080_oil_shine_output(&frame, &vec![0u8; mask.len()], &points),
            Err(Refusal::MissingInput(SKIN_MASK))
        );
        let mut bad_points = points;
        bad_points[26] = [f32::NAN, 0.0];
        assert_eq!(
            build_pf90080_oil_shine_output(&frame, &mask, &bad_points),
            Err(Refusal::ValueOutOfRange(FACE_POINTS))
        );
    }

    #[test]
    fn pf90118_rejects_empty_lip_masks() {
        let (frame, upper, lower) = lip_frame();
        let points = points();
        let empty = vec![0u8; upper.len()];
        assert_eq!(
            build_pf90118_lip_line_output(&frame, &empty, &empty, &points),
            Err(Refusal::MissingInput(UPPER_LIP_MASK))
        );
        assert!(
            build_pf90118_lip_line_output(&frame, &empty, &lower, &points).is_ok(),
            "single-lip mask should still repair the visible lip"
        );
        assert_eq!(
            build_pf90118_lip_line_output(&frame, &[0u8; 3], &lower, &points),
            Err(Refusal::ShapeMismatch(UPPER_LIP_MASK))
        );
    }

    #[test]
    fn pf90118_rejects_non_finite_landmarks() {
        let (frame, upper, lower) = lip_frame();
        let mut points = points();
        points[83] = [f32::INFINITY, 44.0];
        assert_eq!(
            build_pf90118_lip_line_output(&frame, &upper, &lower, &points),
            Err(Refusal::ValueOutOfRange(LIP_POINTS))
        );
    }
}
