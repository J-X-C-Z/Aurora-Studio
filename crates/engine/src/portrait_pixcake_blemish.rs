//! Same-frame PixCake producers for PF30259 / PF90008 / PF90089 / PF90034 /
//! PF100008.
//!
//! This module only builds the named planes consumed by the recovered business
//! logic. A coordinator must mount it as a child of `portrait_stage` and wire
//! each returned row/payload into `RepairProducerInputs`,
//! `HistoricalSkinExternalInputs` and `HistoricalToneProviderPayload`.
//!
//! No original model weight is present in this repository. The model images
//! produced here are explicit CPU syntheses from the current frame plus a mask;
//! they are not claimed to be vendor network outputs.
//!
//! Blend-law note: `skin_adjust::masked_model_output` (used by both the StretchV3
//! ticket and the historical-tone BreastRepair row) composes `model * mask/255`
//! for `MaskOnly`. The returned model therefore already carries the full,
//! soft-edged repair, and the ROI mask is a full 255 selector inside a tight
//! rectangle. Outside the ROI the source frame is preserved byte-for-byte.

use aurora_portrait::kernels::double_chin;
use aurora_portrait::kernels::skin_adjust;
use aurora_portrait::kernels::stretch_v3::StretchV3StageTicket;
use aurora_portrait::{Image, Refusal};

use super::{HistoricalSkinData, HistoricalSkinExternalRow, RepairOutput};
use crate::portrait_historical_tone_provider::{
    HistoricalToneBreastRepairPayload, HistoricalToneProviderRow, HistoricalToneRowData,
    HistoricalToneRowJob,
};

const SWEAT_MASK: &str = "PF30259_SKIN_MASK";
const SWEAT_POINTS: &str = "PF30259_POINTS194";
const STRETCH_TICKET: &str = "PF90008_JOB_TICKET";
const STRETCH_MASK: &str = "PF90008_BODY_SKIN_MASK";
const CHILD_MASK: &str = "PF90089_SKIN_MASK";
const CHILD_DEGREE: &str = "PF90089_DEGREE";
const CHIN_POINTS: &str = "PF90034_POINTS194";
const BREAST_TICKET: &str = "PF100008_JOB_TICKET";
const BREAST_MASK: &str = "PF100008_BODY_SKIN_MASK";
const BREAST_POINTS: &str = "PF100008_POINTS194";

/// 194-point template: left/right jaw contour and chin bottom.
const JAW_LEFT: usize = 109;
const JAW_RIGHT: usize = 141;
const CHIN: usize = 125;

const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

fn clamp_byte(value: f32) -> u8 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value.round_ties_even() as u8
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn point(points: &[[f32; 2]], index: usize) -> Option<[f32; 2]> {
    points
        .get(index)
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn validate_points(points: &[[f32; 2]], field: &'static str) -> Result<(), Refusal> {
    if points.len() != 194 {
        return Err(Refusal::ShapeMismatch(field));
    }
    if points.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(field));
    }
    Ok(())
}

fn validate_mask(frame: &Image, mask: &[u8], field: &'static str) -> Result<(), Refusal> {
    frame.validate()?;
    if mask.len() != frame.pixel_count() {
        return Err(Refusal::ShapeMismatch(field));
    }
    if !mask.iter().any(|value| *value > 0) {
        return Err(Refusal::MissingInput(field));
    }
    Ok(())
}

fn channel_planes(frame: &Image) -> [Vec<f32>; 3] {
    let mut planes = [
        Vec::with_capacity(frame.pixel_count()),
        Vec::with_capacity(frame.pixel_count()),
        Vec::with_capacity(frame.pixel_count()),
    ];
    for pixel in frame.pixels.chunks_exact(4) {
        planes[0].push(pixel[0] as f32);
        planes[1].push(pixel[1] as f32);
        planes[2].push(pixel[2] as f32);
    }
    planes
}

fn luma_at(planes: &[Vec<f32>; 3], index: usize) -> f32 {
    LUMA[0] * planes[0][index] + LUMA[1] * planes[1][index] + LUMA[2] * planes[2][index]
}

fn luma_plane(planes: &[Vec<f32>; 3]) -> Vec<f32> {
    (0..planes[0].len()).map(|index| luma_at(planes, index)).collect()
}

/// Separable box blur with a scalar weight per pixel. Zero-weight samples do
/// not contribute; a window with no weight reproduces the original value, so no
/// colour leaks in from outside the masked region.
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

fn mask_box(
    response: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
    margin: usize,
    fallback_full: bool,
) -> Option<skin_adjust::Roi> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for y in 0..height {
        for x in 0..width {
            if response[y * width + x] > threshold {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x + 1);
                max_y = max_y.max(y + 1);
            }
        }
    }
    if min_x >= max_x || min_y >= max_y {
        if !fallback_full {
            return None;
        }
        return Some(skin_adjust::Roi {
            x: 0,
            y: 0,
            width: width as u32,
            height: height as u32,
        });
    }
    let x = min_x.saturating_sub(margin);
    let y = min_y.saturating_sub(margin);
    let right = (max_x + margin).min(width);
    let bottom = (max_y + margin).min(height);
    Some(skin_adjust::Roi {
        x: x as u32,
        y: y as u32,
        width: (right - x) as u32,
        height: (bottom - y) as u32,
    })
}

// ---------------------------------------------------------------------------
// PF30259: sweat / facial water-droplet specular removal.
// ---------------------------------------------------------------------------

/// Face-area focus from the 194-point mesh. Eyes, brows, nose tip/ala and lips
/// are suppressed so a natural catchlight or lip gloss is not treated as sweat.
fn sweat_focus(points: &[[f32; 2]], width: usize, height: usize) -> Vec<f32> {
    let mut focus = vec![0.0f32; width * height];
    let (Some(eye_l), Some(eye_r)) = (point(points, 26), point(points, 44)) else {
        return focus;
    };
    let (Some(brow_l), Some(brow_r)) = (point(points, 4), point(points, 9)) else {
        return focus;
    };
    let (Some(nose_tip), Some(chin)) = (point(points, 57), point(points, CHIN)) else {
        return focus;
    };
    let (Some(lip_l), Some(lip_r)) = (point(points, 73), point(points, 83)) else {
        return focus;
    };
    let eye_dist = distance(eye_l, eye_r).max(12.0);
    let brow_mid = [(brow_l[0] + brow_r[0]) * 0.5, (brow_l[1] + brow_r[1]) * 0.5];
    let lip_mid = [(lip_l[0] + lip_r[0]) * 0.5, (lip_l[1] + lip_r[1]) * 0.5];
    let face_height = (chin[1] - brow_mid[1]).max(eye_dist * 1.8);
    let cx = brow_mid[0] * 0.45 + nose_tip[0] * 0.35 + chin[0] * 0.20;
    let cy = brow_mid[1] + face_height * 0.42;
    let rx = eye_dist * 0.9;
    let ry = face_height * 0.5;

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;
            let dx = (fx - cx) / rx;
            let dy = (fy - cy) / ry;
            let mut value = smoothstep(1.0, 0.55, dx * dx + dy * dy);
            // Eyes and brows.
            for eye in [eye_l, eye_r] {
                let ex = (fx - eye[0]) / (eye_dist * 0.34);
                let ey = (fy - eye[1]) / (eye_dist * 0.20);
                value *= smoothstep(0.6, 1.1, ex * ex + ey * ey);
            }
            let lx = (fx - brow_mid[0]) / (eye_dist * 0.70);
            let ly = (fy - brow_mid[1]) / (eye_dist * 0.14);
            value *= smoothstep(0.5, 1.0, lx * lx + ly * ly);
            // Nose tip and ala keep their natural specular.
            let nx = (fx - nose_tip[0]) / (eye_dist * 0.42);
            let ny = (fy - (nose_tip[1] + face_height * 0.18)) / (eye_dist * 0.22);
            value *= smoothstep(0.5, 1.0, nx * nx + ny * ny);
            // Lips keep their gloss.
            let mx = (fx - lip_mid[0]) / (eye_dist * 0.65);
            let my = (fy - lip_mid[1]) / (eye_dist * 0.28);
            value *= smoothstep(0.5, 1.0, mx * mx + my * my);
            focus[y * width + x] = value.clamp(0.0, 1.0);
        }
    }
    focus
}

/// Builds the full-frame PF30259 `model` image consumed by
/// `sweat::apply` (model RGB replaces the layer; source alpha is kept).
///
/// Droplets are small bright, low-chroma kernels with a darker rim. Each core is
/// replaced by the surrounding diffuse-skin colour; every other pixel is the
/// untouched source.
#[allow(dead_code)]
pub(crate) fn build_pf30259_sweat_output(
    frame: &Image,
    skin_mask: &[u8],
    face_points194: &[[f32; 2]],
) -> Result<super::RepairOutput, Refusal> {
    validate_mask(frame, skin_mask, SWEAT_MASK)?;
    validate_points(face_points194, SWEAT_POINTS)?;

    let width = frame.width as usize;
    let height = frame.height as usize;
    let pixels = width * height;
    let focus = sweat_focus(face_points194, width, height);
    let region: Vec<f32> = (0..pixels)
        .map(|p| skin_mask[p] as f32 / 255.0 * focus[p])
        .collect();
    if !region.iter().any(|value| *value > 0.0) {
        return Err(Refusal::MissingInput(SWEAT_MASK));
    }

    let source = channel_planes(frame);
    let source_luma = luma_plane(&source);
    let radius = (width.min(height) / 5).clamp(6, 24);
    let base = [
        weighted_blur(&source[0], &region, width, height, radius),
        weighted_blur(&source[1], &region, width, height, radius),
        weighted_blur(&source[2], &region, width, height, radius),
    ];
    let base_luma: Vec<f32> = (0..pixels)
        .map(|p| 0.299 * base[0][p] + 0.587 * base[1][p] + 0.114 * base[2][p])
        .collect();
    let ones = vec![1.0f32; pixels];
    let local_luma = weighted_blur(&source_luma, &ones, width, height, 1);

    let mut core = vec![0.0f32; pixels];
    for y in 2..height.saturating_sub(2) {
        for x in 2..width.saturating_sub(2) {
            let p = y * width + x;
            if region[p] <= 0.0 {
                continue;
            }
            let maximum = source[0][p].max(source[1][p]).max(source[2][p]);
            let minimum = source[0][p].min(source[1][p]).min(source[2][p]);
            let saturation = if maximum > 1.0 {
                (maximum - minimum) / maximum
            } else {
                0.0
            };
            let excess = source_luma[p] - base_luma[p];
            let strength = excess * (1.0 - saturation.clamp(0.0, 1.0));
            if strength <= 0.0 {
                continue;
            }
            // A water bead has a darker rim than its core; this term only
            // modulates the response, it never zeros the core.
            let mut rim = 0.0f32;
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    rim = rim.max(local_luma[p] - local_luma[ny * width + nx]);
                }
            }
            let rim_gain = 0.6 + 0.4 * smoothstep(0.0, 16.0, rim);
            core[p] = smoothstep(8.0, 24.0, strength) * rim_gain * region[p];
        }
    }
    let feather_r = (radius / 4).clamp(1, 5);
    let feathered = weighted_blur(&core, &ones, width, height, feather_r);

    let mut pixels_out = frame.pixels.clone();
    for p in 0..pixels {
        let weight = feathered[p].clamp(0.0, 1.0);
        if weight <= 0.0 {
            continue;
        }
        let pore_detail = ((source_luma[p] - local_luma[p]) * 0.45).clamp(-10.0, 10.0);
        for channel in 0..3 {
            let target = (base[channel][p] + pore_detail).clamp(0.0, 255.0);
            pixels_out[p * 4 + channel] = clamp_byte(
                source[channel][p] + weight * (target - source[channel][p]),
            );
        }
    }
    let model = Image::new(frame.width, frame.height, pixels_out)
        .ok_or(Refusal::ShapeMismatch(SWEAT_MASK))?;
    Ok(RepairOutput::Sweat { model })
}

// ---------------------------------------------------------------------------
// PF90008: stretch marks (StretchV3 shared-oracle row).
// ---------------------------------------------------------------------------

/// Directional edge-preserving mean toward horizontal neighbours. Stretch marks
/// run roughly vertically, so the horizontal neighbours carry the surrounding
/// diffuse colour; the colour-range gate protects the ends and skin texture.
fn horizontal_mean(source: &[f32], weight: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut out = source.to_vec();
    for y in 1..height.saturating_sub(1) {
        for x in 2..width.saturating_sub(2) {
            let p = y * width + x;
            if weight[p] <= 0.0 {
                continue;
            }
            let mut sum = 0.0f32;
            let mut total = 0.0f32;
            for dx in -2i32..=2 {
                if dx == 0 {
                    continue;
                }
                let q = y * width + (x as i32 + dx) as usize;
                let delta = (source[p] - source[q]).abs();
                let range = (-0.5 * (delta / 45.0).powi(2)).exp();
                let spatial = 0.6f32.powi(dx.abs());
                let w = weight[q] * range * spatial;
                sum += w * source[q];
                total += w;
            }
            if total > 0.0 {
                out[p] = sum / total;
            }
        }
    }
    out
}

/// Builds the PF90008 historical-skin external row.
///
/// The full-frame `model` carries a soft directional repair of elongated,
/// low-frequency red/silver streaks inside the body-skin region. The ticket uses
/// a tight ROI and an all-255 selector mask, because the recovered shared
/// composer is `model * mask/255` in `MaskOnly` mode.
#[allow(dead_code)]
pub(crate) fn build_pf90008_stretch_mark_row(
    job_id: u64,
    frame: &Image,
    body_skin_mask: &[u8],
) -> Result<HistoricalSkinExternalRow, Refusal> {
    if job_id == 0 {
        return Err(Refusal::MissingInput(STRETCH_TICKET));
    }
    validate_mask(frame, body_skin_mask, STRETCH_MASK)?;

    let width = frame.width as usize;
    let height = frame.height as usize;
    let pixels = width * height;
    let source = channel_planes(frame);
    let source_luma = luma_plane(&source);
    let weights: Vec<f32> = body_skin_mask
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect();
    let radius = (width.min(height) / 6).clamp(4, 24);
    let base = [
        weighted_blur(&source[0], &weights, width, height, radius),
        weighted_blur(&source[1], &weights, width, height, radius),
        weighted_blur(&source[2], &weights, width, height, radius),
    ];
    let base_luma: Vec<f32> = (0..pixels)
        .map(|p| 0.299 * base[0][p] + 0.587 * base[1][p] + 0.114 * base[2][p])
        .collect();
    let local = weighted_blur(&source_luma, &weights, width, height, 2);

    // A streak is a low-frequency luma groove with anisotropic extent: the
    // vertical neighbourhood stays similar while the horizontal neighbourhood
    // differs. Red streaks additionally carry excess red over the local skin.
    let mut response = vec![0.0f32; pixels];
    for y in 2..height.saturating_sub(2) {
        for x in 2..width.saturating_sub(2) {
            let p = y * width + x;
            if weights[p] <= 0.0 {
                continue;
            }
            let horizontal_delta = (local[p] - local[y * width + x - 2])
                .abs()
                .max((local[p] - local[y * width + x + 2]).abs());
            let vertical_delta = (local[p] - local[(y - 2) * width + x])
                .abs()
                .max((local[p] - local[(y + 2) * width + x]).abs());
            let groove = smoothstep(2.5, 10.0, horizontal_delta - vertical_delta * 0.5);
            let level = smoothstep(2.0, 9.0, (base_luma[p] - source_luma[p]).abs());
            let red_excess = source[0][p] - source[1][p] - (base[0][p] - base[1][p]);
            let silver = smoothstep(4.0, 18.0, source_luma[p] - base_luma[p]);
            let red = smoothstep(3.0, 16.0, red_excess);
            response[p] = groove * level.max(red.max(silver)) * weights[p];
        }
    }
    let feathered = weighted_blur(&response, &vec![1.0f32; pixels], width, height, 1);

    let repaired = [
        horizontal_mean(&source[0], &response, width, height),
        horizontal_mean(&source[1], &response, width, height),
        horizontal_mean(&source[2], &response, width, height),
    ];
    let mut model_pixels = Vec::with_capacity(pixels * 4);
    for p in 0..pixels {
        let weight = feathered[p].clamp(0.0, 1.0);
        for channel in 0..3 {
            model_pixels.push(clamp_byte(
                source[channel][p] + weight * (repaired[channel][p] - source[channel][p]),
            ));
        }
        model_pixels.push(frame.pixels[p * 4 + 3]);
    }
    let model = Image::new(frame.width, frame.height, model_pixels)
        .ok_or(Refusal::ShapeMismatch(STRETCH_MASK))?;

    let margin = (width.min(height) / 30).max(2);
    let roi = mask_box(&response, width, height, 0.05, margin, true)
        .ok_or(Refusal::MissingInput(STRETCH_MASK))?;
    let roi_mask = vec![255u8; roi.width as usize * roi.height as usize];
    let ticket = StretchV3StageTicket::shared_oracle(model, roi_mask, roi, false, frame.width * 4);
    ticket.validate_for_size([frame.width, frame.height])?;

    Ok(HistoricalSkinExternalRow {
        pf: 90008,
        frame: frame.clone(),
        frame_sha256: super::sha256_32(&frame.pixels),
        ticket: job_id,
        // The producer has no degree argument; the row is built at full
        // strength. A coordinator that runs a partial `pc:90008` must overwrite
        // `row.control` with that 0..1 control before building the payload.
        control: 1.0,
        data: HistoricalSkinData::StretchMark(ticket),
    })
}

// ---------------------------------------------------------------------------
// PF90089: baby blemish / small red-spot repair on face and body.
// ---------------------------------------------------------------------------

/// Builds the full-frame PF90089 `model`, a tight detection ROI and the ROI
/// selector mask. Red spots are detected as an excess of red over both green
/// and the local diffuse baseline.
#[allow(dead_code)]
pub(crate) fn build_pf90089_body_blemish_child_output(
    frame: &Image,
    skin_mask: &[u8],
    degree: f32,
) -> Result<super::RepairOutput, Refusal> {
    validate_mask(frame, skin_mask, CHILD_MASK)?;
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange(CHILD_DEGREE));
    }

    let width = frame.width as usize;
    let height = frame.height as usize;
    let pixels = width * height;
    let weights: Vec<f32> = skin_mask
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect();
    let source = channel_planes(frame);
    let source_luma = luma_plane(&source);
    let radius = (width.min(height) / 6).clamp(4, 24);
    let base = [
        weighted_blur(&source[0], &weights, width, height, radius),
        weighted_blur(&source[1], &weights, width, height, radius),
        weighted_blur(&source[2], &weights, width, height, radius),
    ];

    let mut response = vec![0.0f32; pixels];
    for p in 0..pixels {
        if weights[p] <= 0.0 {
            continue;
        }
        let local_red = source[0][p] - source[1][p] - (base[0][p] - base[1][p]);
        let local_luma = source_luma[p] - (0.299 * base[0][p] + 0.587 * base[1][p] + 0.114 * base[2][p]);
        response[p] = smoothstep(3.0, 14.0, local_red - local_luma * 0.35) * weights[p];
    }
    let feathered = weighted_blur(&response, &vec![1.0f32; pixels], width, height, 1);

    let margin = (width.min(height) / 30).max(2);
    let roi = mask_box(&feathered, width, height, 0.02, margin, false)
        .ok_or(Refusal::MissingInput(CHILD_MASK))?;
    let roi_mask = vec![255u8; roi.width as usize * roi.height as usize];

    let mut model_pixels = Vec::with_capacity(pixels * 4);
    for p in 0..pixels {
        let weight = (feathered[p] * degree).clamp(0.0, 1.0);
        for channel in 0..3 {
            model_pixels.push(clamp_byte(
                source[channel][p] + weight * (base[channel][p] - source[channel][p]),
            ));
        }
        model_pixels.push(frame.pixels[p * 4 + 3]);
    }
    let model = Image::new(frame.width, frame.height, model_pixels)
        .ok_or(Refusal::ShapeMismatch(CHILD_MASK))?;

    Ok(RepairOutput::BodyBlemishChild {
        model,
        roi,
        roi_mask,
        native_per_face_degree: degree,
        alternate: false,
    })
}

// ---------------------------------------------------------------------------
// PF90034: double-chin displacement field.
// ---------------------------------------------------------------------------

/// Builds a unit-scale `Flow` field that tightens the soft tissue below the jaw
/// inward and upward. The field is dense (`width * height`), finite and smooth.
#[allow(dead_code)]
pub(crate) fn build_pf90034_double_chin_output(
    frame: &Image,
    face_points194: &[[f32; 2]],
) -> Result<super::RepairOutput, Refusal> {
    frame.validate()?;
    validate_points(face_points194, CHIN_POINTS)?;
    let chin = point(face_points194, CHIN).ok_or(Refusal::ValueOutOfRange(CHIN_POINTS))?;
    let jaw_left = point(face_points194, JAW_LEFT).unwrap_or(chin);
    let jaw_right = point(face_points194, JAW_RIGHT).unwrap_or(chin);
    let center_x = (jaw_left[0] + jaw_right[0]) * 0.5;
    let jaw_width = (jaw_right[0] - jaw_left[0]).abs().max(16.0);
    let face_scale = (chin[1] - (jaw_left[1] + jaw_right[1]) * 0.5)
        .abs()
        .max(jaw_width * 0.35)
        .max(12.0);

    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut field = vec![[0.0f32, 0.0f32]; width * height];
    for y in 0..height {
        let fy = y as f32 + 0.5;
        for x in 0..width {
            let fx = x as f32 + 0.5;
            let dx = (fx - center_x) / (jaw_width * 0.6);
            let dy = (fy - chin[1]) / (face_scale * 0.85);
            // The lobe lives below the chin point; it fades to zero above it,
            // below the neck band and at the jaw sides.
            if dy <= 0.0 || dy >= 2.2 || dx.abs() >= 1.0 {
                continue;
            }
            let vertical = smoothstep(0.0, 0.35, dy) * (1.0 - smoothstep(1.1, 2.2, dy));
            let lateral = (1.0 - dx * dx).powi(2);
            let weight = vertical * lateral;
            // Inward (toward the face midline) and upward (negative image y).
            field[y * width + x] = [-dx * weight * 4.0, -weight * 6.0];
        }
    }
    debug_assert!(field.iter().flatten().all(|value| value.is_finite()));
    Ok(RepairOutput::DoubleChin {
        field,
        mode: double_chin::Mode::Flow,
        scale: 1.0,
    })
}

// ---------------------------------------------------------------------------
// PF100008: breast repair (HistoricalToneRowData::BreastRepair).
// ---------------------------------------------------------------------------

/// Localizes the breast band from the face landmarks (1.2..2.8 face heights
/// below the chin), or from the upper body-skin band when no face is present.
fn breast_roi(
    frame: &Image,
    body_skin_mask: &[u8],
    face_points194: Option<&[[f32; 2]]>,
) -> Result<skin_adjust::Roi, Refusal> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;

    let (scan_x0, scan_y0, scan_x1, scan_y1) = if let Some(points) = face_points194 {
        validate_points(points, BREAST_POINTS)?;
        let (Some(brow_l), Some(brow_r), Some(chin)) =
            (point(points, 4), point(points, 9), point(points, CHIN))
        else {
            return Err(Refusal::ValueOutOfRange(BREAST_POINTS));
        };
        let face_height = (chin[1] - brow_l[1].min(brow_r[1])).max(16.0);
        let face_cx = (brow_l[0] + brow_r[0] + chin[0]) / 3.0;
        let top = chin[1] + face_height * 1.2;
        let bottom = chin[1] + face_height * 2.8;
        let half_width = face_height * 1.4;
        (
            (face_cx - half_width).max(0.0).floor() as usize,
            top.max(0.0).floor() as usize,
            (face_cx + half_width).min(width as f32).ceil() as usize,
            bottom.min(height as f32).ceil() as usize,
        )
    } else {
        (0, height / 4, width, height * 3 / 4)
    };
    if scan_x0 >= scan_x1 || scan_y0 >= scan_y1 {
        return Err(Refusal::MissingInput(BREAST_MASK));
    }

    for y in scan_y0..scan_y1 {
        for x in scan_x0..scan_x1 {
            if body_skin_mask[y * width + x] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x + 1);
                max_y = max_y.max(y + 1);
            }
        }
    }
    if min_x >= max_x || min_y >= max_y {
        return Err(Refusal::MissingInput(BREAST_MASK));
    }
    Ok(skin_adjust::Roi {
        x: min_x as u32,
        y: min_y as u32,
        width: (max_x - min_x) as u32,
        height: (max_y - min_y) as u32,
    })
}

/// Builds one PF100008 historical-tone provider row.
///
/// The full-frame `model` carries soft low-frequency skin equalization inside
/// the body mask; the ROI mask is an all-255 selector because the recovered
/// composer is `model * mask/255` in `MaskOnly` mode. `roi_value` is the full
/// native 0..100 degree (the producer has no degree argument); a coordinator
/// that needs partial strength must scale the model or overwrite this value.
#[allow(dead_code)]
pub(crate) fn build_pf100008_breast_repair_row(
    job_id: u64,
    frame: &Image,
    body_skin_mask: &[u8],
    face_points194: Option<&[[f32; 2]]>,
) -> Result<HistoricalToneProviderRow, Refusal> {
    if job_id == 0 {
        return Err(Refusal::MissingInput(BREAST_TICKET));
    }
    validate_mask(frame, body_skin_mask, BREAST_MASK)?;
    let roi = breast_roi(frame, body_skin_mask, face_points194)?;

    let width = frame.width as usize;
    let height = frame.height as usize;
    let pixels = width * height;
    let weights: Vec<f32> = body_skin_mask
        .iter()
        .map(|value| *value as f32 / 255.0)
        .collect();
    let source = channel_planes(frame);
    let source_luma = luma_plane(&source);
    let radius = (width.min(height) / 6).clamp(4, 24);
    let base = [
        weighted_blur(&source[0], &weights, width, height, radius),
        weighted_blur(&source[1], &weights, width, height, radius),
        weighted_blur(&source[2], &weights, width, height, radius),
    ];
    let smooth_luma = weighted_blur(&source_luma, &weights, width, height, 2);

    let mut response = vec![0.0f32; pixels];
    for p in 0..pixels {
        if weights[p] <= 0.0 {
            continue;
        }
        let correction = (smooth_luma[p] - source_luma[p]).abs();
        response[p] = smoothstep(2.0, 10.0, correction) * weights[p];
    }
    let feathered = weighted_blur(&response, &vec![1.0f32; pixels], width, height, 1);

    let mut model_pixels = Vec::with_capacity(pixels * 4);
    for p in 0..pixels {
        let weight = feathered[p].clamp(0.0, 1.0);
        for channel in 0..3 {
            model_pixels.push(clamp_byte(
                source[channel][p] + weight * (base[channel][p] - source[channel][p]),
            ));
        }
        model_pixels.push(frame.pixels[p * 4 + 3]);
    }
    let model = Image::new(frame.width, frame.height, model_pixels)
        .ok_or(Refusal::ShapeMismatch(BREAST_MASK))?;

    let roi_mask = vec![255u8; roi.width as usize * roi.height as usize];
    Ok(HistoricalToneProviderRow {
        frame: frame.clone(),
        job: HistoricalToneRowJob {
            source_sha256: super::sha256_32(&frame.pixels),
            size: [frame.width, frame.height],
            // Full-strength row; a coordinator running a partial `pc:100008`
            // must overwrite this 0..1 control before building the payload.
            control: 1.0,
        },
        data: HistoricalToneRowData::BreastRepair(HistoricalToneBreastRepairPayload {
            model,
            roi_mask,
            roi,
            roi_value: 100.0,
            has_body_skeleton: true,
            has_roi_points: true,
            alternate: false,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDE: usize = 64;

    fn face_points() -> Vec<[f32; 2]> {
        let mut points = vec![[32.0f32, 32.0]; 194];
        points[4] = [22.0, 18.0];
        points[9] = [42.0, 18.0];
        points[26] = [22.0, 26.0];
        points[44] = [42.0, 26.0];
        points[54] = [32.0, 30.0];
        points[57] = [32.0, 36.0];
        points[73] = [20.0, 44.0];
        points[83] = [44.0, 44.0];
        points[109] = [16.0, 46.0];
        points[125] = [32.0, 52.0];
        points[141] = [48.0, 46.0];
        points
    }

    fn ellipse_mask(
        width: usize,
        height: usize,
        cx: f32,
        cy: f32,
        rx: f32,
        ry: f32,
    ) -> Vec<u8> {
        let mut mask = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = (x as f32 + 0.5 - cx) / rx;
                let dy = (y as f32 + 0.5 - cy) / ry;
                if dx * dx + dy * dy <= 1.0 {
                    mask[y * width + x] = 255;
                }
            }
        }
        mask
    }

    fn frame_with(
        width: usize,
        height: usize,
        base: [u8; 3],
        spots: &[(usize, usize, [u8; 3])],
    ) -> Image {
        let mut frame = Image::solid(width as u32, height as u32, [base[0], base[1], base[2], 255]);
        for &(x, y, rgb) in spots {
            let at = (y * width + x) * 4;
            frame.pixels[at..at + 3].copy_from_slice(&rgb);
        }
        frame
    }

    fn luma_of(frame: &Image, x: usize, y: usize) -> f32 {
        let at = (y * frame.width as usize + x) * 4;
        LUMA[0] * frame.pixels[at] as f32
            + LUMA[1] * frame.pixels[at + 1] as f32
            + LUMA[2] * frame.pixels[at + 2] as f32
    }

    #[test]
    fn pf30259_replaces_sweat_highlights_and_keeps_outside_skin() {
        let spots: Vec<(usize, usize, [u8; 3])> = (26..34)
            .flat_map(|y| (28..36).map(move |x| (x, y, [248u8, 246, 240])))
            .collect();
        let frame = frame_with(SIDE, SIDE, [150, 110, 95], &spots);
        let mask = ellipse_mask(SIDE, SIDE, 32.0, 32.0, 24.0, 26.0);
        let output = build_pf30259_sweat_output(&frame, &mask, &face_points()).unwrap();
        let RepairOutput::Sweat { model } = output else {
            panic!("expected Sweat output");
        };
        assert_eq!(model.pixel_count(), frame.pixel_count());
        assert!(luma_of(&model, 32, 30) < luma_of(&frame, 32, 30) - 10.0);
        assert_eq!(model.pixel(1, 1), frame.pixel(1, 1));
    }

    #[test]
    fn pf30259_rejects_bad_inputs() {
        let frame = Image::solid(SIDE as u32, SIDE as u32, [0, 0, 0, 255]);
        let mask = vec![255u8; SIDE * SIDE];
        assert_eq!(
            build_pf30259_sweat_output(&frame, &[], &face_points()),
            Err(Refusal::ShapeMismatch(SWEAT_MASK))
        );
        assert_eq!(
            build_pf30259_sweat_output(&frame, &vec![0u8; SIDE * SIDE], &face_points()),
            Err(Refusal::MissingInput(SWEAT_MASK))
        );
        let mut points = face_points();
        points[26] = [f32::NAN, 26.0];
        assert_eq!(
            build_pf30259_sweat_output(&frame, &mask, &points),
            Err(Refusal::ValueOutOfRange(SWEAT_POINTS))
        );
    }

    #[test]
    fn pf90008_produces_a_valid_stretch_mark_row() {
        let mut spots = Vec::new();
        for y in 18..46 {
            for x in 18..22 {
                spots.push((x, y, [182u8, 118, 104]));
            }
        }
        let frame = frame_with(SIDE, SIDE, [150, 110, 95], &spots);
        let mask = ellipse_mask(SIDE, SIDE, 32.0, 32.0, 28.0, 28.0);
        let row = build_pf90008_stretch_mark_row(7, &frame, &mask).unwrap();
        assert_eq!(row.pf, 90008);
        assert_eq!(row.ticket, 7);
        assert_eq!(row.control, 1.0);
        assert_eq!(row.frame_sha256, crate::portrait_stage::sha256_32(&frame.pixels));
        let HistoricalSkinData::StretchMark(ticket) = &row.data else {
            panic!("expected StretchMark ticket");
        };
        assert_eq!(ticket.validate_for_size([frame.width, frame.height]), Ok(()));
        assert!(ticket.source_roi.width > 0 && ticket.source_roi.height > 0);
        assert_eq!(
            ticket.mask.data.len(),
            ticket.source_roi.width as usize * ticket.source_roi.height as usize
        );
        assert!(ticket.mask.data.iter().all(|value| *value == 255));
    }

    #[test]
    fn pf90008_rejects_bad_inputs() {
        let frame = Image::solid(SIDE as u32, SIDE as u32, [0, 0, 0, 255]);
        let mask = vec![255u8; SIDE * SIDE];
        assert_eq!(
            build_pf90008_stretch_mark_row(0, &frame, &mask),
            Err(Refusal::MissingInput(STRETCH_TICKET))
        );
        assert_eq!(
            build_pf90008_stretch_mark_row(7, &frame, &[]),
            Err(Refusal::ShapeMismatch(STRETCH_MASK))
        );
        assert_eq!(
            build_pf90008_stretch_mark_row(7, &frame, &vec![0u8; SIDE * SIDE]),
            Err(Refusal::MissingInput(STRETCH_MASK))
        );
    }

    #[test]
    fn pf90089_builds_a_tight_roi_and_smooth_model() {
        let spots: Vec<(usize, usize, [u8; 3])> = (28..34)
            .flat_map(|y| (28..34).map(move |x| (x, y, [196u8, 96, 92])))
            .collect();
        let frame = frame_with(SIDE, SIDE, [150, 110, 95], &spots);
        let mask = ellipse_mask(SIDE, SIDE, 32.0, 32.0, 28.0, 28.0);
        let output = build_pf90089_body_blemish_child_output(&frame, &mask, 0.8).unwrap();
        let RepairOutput::BodyBlemishChild {
            model,
            roi,
            roi_mask,
            native_per_face_degree,
            alternate,
        } = output
        else {
            panic!("expected BodyBlemishChild output");
        };
        assert_eq!(model.pixel_count(), frame.pixel_count());
        assert_eq!(roi_mask.len(), roi.width as usize * roi.height as usize);
        assert!(roi.width > 0 && roi.height > 0);
        assert!(roi_mask.iter().all(|value| *value == 255));
        assert!(roi.x <= 28 && roi.y <= 28);
        assert!(roi.x + roi.width >= 34 && roi.y + roi.height >= 34);
        assert_eq!(native_per_face_degree, 0.8);
        assert!(!alternate);
        assert!(model.pixels[(30 * SIDE + 30) * 4] < frame.pixels[(30 * SIDE + 30) * 4]);
    }

    #[test]
    fn pf90089_rejects_bad_inputs_and_clean_skin() {
        let frame = Image::solid(SIDE as u32, SIDE as u32, [150, 110, 95, 255]);
        let mask = vec![255u8; SIDE * SIDE];
        assert_eq!(
            build_pf90089_body_blemish_child_output(&frame, &[], 1.0),
            Err(Refusal::ShapeMismatch(CHILD_MASK))
        );
        assert_eq!(
            build_pf90089_body_blemish_child_output(&frame, &mask, f32::NAN),
            Err(Refusal::ValueOutOfRange(CHILD_DEGREE))
        );
        assert_eq!(
            build_pf90089_body_blemish_child_output(&frame, &mask, 1.0),
            Err(Refusal::MissingInput(CHILD_MASK))
        );
    }

    #[test]
    fn pf90034_field_is_dense_smooth_and_inward_upward() {
        let frame = Image::solid(SIDE as u32, SIDE as u32, [0, 0, 0, 255]);
        let output = build_pf90034_double_chin_output(&frame, &face_points()).unwrap();
        let RepairOutput::DoubleChin { field, mode, scale } = output else {
            panic!("expected DoubleChin output");
        };
        assert_eq!(field.len(), frame.pixel_count());
        assert_eq!(mode, double_chin::Mode::Flow);
        assert_eq!(scale, 1.0);
        assert!(field.iter().flatten().all(|value| value.is_finite()));
        let below = field[60 * SIDE + 32];
        assert!(below[1] < 0.0, "jaw tissue must move upward");
        let left = field[58 * SIDE + 22];
        let right = field[58 * SIDE + 42];
        assert!(left[0] > 0.0, "left tissue must move toward the midline");
        assert!(right[0] < 0.0, "right tissue must move toward the midline");
        let mut max_step = 0.0f32;
        for y in 1..SIDE - 1 {
            for x in 1..SIDE - 1 {
                let i = y * SIDE + x;
                max_step = max_step.max((field[i][0] - field[i - 1][0]).abs());
                max_step = max_step.max((field[i][1] - field[i - SIDE][1]).abs());
            }
        }
        assert!(max_step < 5.0, "discontinuity: {max_step}");
    }

    #[test]
    fn pf90034_rejects_bad_landmarks() {
        let frame = Image::solid(SIDE as u32, SIDE as u32, [0, 0, 0, 255]);
        assert_eq!(
            build_pf90034_double_chin_output(&frame, &[]),
            Err(Refusal::ShapeMismatch(CHIN_POINTS))
        );
        let mut points = face_points();
        points[125] = [f32::INFINITY, 52.0];
        assert_eq!(
            build_pf90034_double_chin_output(&frame, &points),
            Err(Refusal::ValueOutOfRange(CHIN_POINTS))
        );
    }

    #[test]
    fn pf100008_builds_a_bound_breast_repair_row_from_body_mask() {
        let mask = ellipse_mask(SIDE, SIDE, 32.0, 30.0, 20.0, 8.0);
        let frame = frame_with(SIDE, SIDE, [150, 110, 95], &[(32, 30, [188, 104, 96])]);
        let row = build_pf100008_breast_repair_row(9, &frame, &mask, None).unwrap();
        assert_eq!(row.pf(), 100008);
        assert_eq!(row.job.size, [frame.width, frame.height]);
        assert_eq!(row.job.control, 1.0);
        assert_eq!(
            row.job.source_sha256,
            crate::portrait_stage::sha256_32(&frame.pixels)
        );
        let HistoricalToneRowData::BreastRepair(payload) = &row.data else {
            panic!("expected BreastRepair payload");
        };
        assert_eq!(payload.roi.width as usize * payload.roi.height as usize, payload.roi_mask.len());
        assert!(payload.roi_mask.iter().all(|value| *value == 255));
        assert_eq!(payload.roi_value, 100.0);
        assert!(payload.has_body_skeleton && payload.has_roi_points);
        assert!(!payload.alternate);
        assert_eq!(payload.model.pixel_count(), frame.pixel_count());
    }

    #[test]
    fn pf100008_uses_the_face_band_below_the_chin() {
        let (width, height) = (96usize, 240usize);
        let mask = ellipse_mask(width, height, 48.0, 170.0, 30.0, 15.0);
        let frame = frame_with(width, height, [150, 110, 95], &[(48, 170, [188, 104, 96])]);
        let mut points = vec![[48.0f32, 32.0]; 194];
        points[4] = [40.0, 20.0];
        points[9] = [56.0, 20.0];
        points[125] = [48.0, 70.0];
        let row = build_pf100008_breast_repair_row(9, &frame, &mask, Some(&points)).unwrap();
        let HistoricalToneRowData::BreastRepair(payload) = &row.data else {
            panic!("expected BreastRepair payload");
        };
        assert!(payload.roi.y >= 130, "breast band starts below the chin");
        assert!(payload.roi.y + payload.roi.height <= height as u32);
    }

    #[test]
    fn pf100008_rejects_bad_inputs() {
        let frame = Image::solid(SIDE as u32, SIDE as u32, [0, 0, 0, 255]);
        let mask = vec![255u8; SIDE * SIDE];
        assert_eq!(
            build_pf100008_breast_repair_row(0, &frame, &mask, None),
            Err(Refusal::MissingInput(BREAST_TICKET))
        );
        assert_eq!(
            build_pf100008_breast_repair_row(9, &frame, &[], None),
            Err(Refusal::ShapeMismatch(BREAST_MASK))
        );
        assert_eq!(
            build_pf100008_breast_repair_row(9, &frame, &vec![0u8; SIDE * SIDE], None),
            Err(Refusal::MissingInput(BREAST_MASK))
        );
        let mut points = face_points();
        points[4] = [f32::NAN, 18.0];
        assert_eq!(
            build_pf100008_breast_repair_row(9, &frame, &mask, Some(&points)),
            Err(Refusal::ValueOutOfRange(BREAST_POINTS))
        );
    }
}
