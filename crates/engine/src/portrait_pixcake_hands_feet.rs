//! Same-frame PixCake producers for the four hand/foot local effects.
//!
//! This module is mounted as a child of `portrait_stage`; the stage owns the
//! final wiring. It only builds the named producer planes:
//!
//! * PF30030: `HistoricalToneRowData::HandVeins` -> `hair_body::hand_blue_vein`.
//!   `mask` and `correction_rgb` are full-frame; the correction is BGRA-channel
//!   ordered with neutral 80, exactly as the recovered kernel consumes it.
//! * PF44297: `RepairOutput::FootBlueVein` -> `hair_body::foot_blue_vein`.
//! * PF30187: `HistoricalToneRowData::HandGray` ->
//!   `hair_body::hand_neutral_gray` with `NeutralTarget::Bgra`.
//! * PF30190: `HistoricalToneRowData::HandShape` -> `hair_body::hand_shape`.
//!   `dx`/`dy` are full-frame shift fields in pixels; the kernel applies
//!   `native_parameter * 1.2 / 100` and the source scale.
//!
//! There is no `BodySubject` parameter in these producer signatures. Region
//! selection therefore uses the `body_skin_mask` minus `face_mask` difference
//! plus conservative geometry priors (hands in the upper/middle frame, feet in
//! the lower frame). With no pose and no face mask the full body-skin plane is
//! used; no row is fabricated from an empty mask and no vendor asset is read.
//! The detection and neutral-surface math below are geometric producers, not
//! a claim of metric parity with the vendor implementation.

use aurora_portrait::{Image, Refusal};

use crate::portrait_historical_tone_provider::{
    HistoricalToneHandGrayPayload, HistoricalToneHandShapePayload,
    HistoricalToneHandVeinsPayload, HistoricalToneProviderRow, HistoricalToneRowData,
    HistoricalToneRowJob,
};

use super::sha256_32;

const FRAME: &str = "hands-feet frame";
const SKIN_MASK: &str = "hands-feet body skin mask";
const FACE_MASK: &str = "hands-feet face mask";
const JOB: &str = "hands-feet job id";
const HAND_REGION: &str = "PF30030/PF30187/PF30190 hand skin region";
const FOOT_REGION: &str = "PF44297 foot skin region";
const DEGREE: &str = "PF30190 degree";

fn byte(value: f32) -> u8 {
    if !value.is_finite() {
        0
    } else {
        value.round().clamp(0.0, 255.0) as u8
    }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Non-face skin confidence, full frame. A missing face mask is not an error:
/// the body-skin plane is used as-is so the producers still work without pose
/// or face parsing evidence.
fn non_face_skin(
    frame: &Image,
    body_skin_mask: &[u8],
    face_mask: Option<&[u8]>,
) -> Result<Vec<f32>, Refusal> {
    frame.validate().map_err(|_| Refusal::ShapeMismatch(FRAME))?;
    let count = frame.pixel_count();
    if body_skin_mask.len() != count {
        return Err(Refusal::ShapeMismatch(SKIN_MASK));
    }
    if let Some(face) = face_mask {
        if face.len() != count {
            return Err(Refusal::ShapeMismatch(FACE_MASK));
        }
    }
    let mut skin = vec![0.0f32; count];
    for index in 0..count {
        let body = body_skin_mask[index] as f32 / 255.0;
        if body <= 0.0 {
            continue;
        }
        let face = face_mask.map_or(0.0, |plane| plane[index] as f32 / 255.0);
        skin[index] = body * (1.0 - face).clamp(0.0, 1.0);
    }
    if !skin.iter().any(|&value| value > 0.02) {
        return Err(Refusal::MissingInput(SKIN_MASK));
    }
    Ok(skin)
}

/// Conservative hand/foot priors applied to the non-face skin confidence.
fn region_weight(skin: &[f32], width: u32, height: u32, feet: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; skin.len()];
    for y in 0..height as usize {
        let fy = if height > 1 {
            y as f32 / (height - 1) as f32
        } else {
            0.0
        };
        let vertical = if feet {
            smoothstep(0.50, 0.68, fy)
        } else {
            1.0 - smoothstep(0.72, 0.88, fy)
        };
        for x in 0..width as usize {
            let index = y * width as usize + x;
            if vertical <= 0.0 {
                continue;
            }
            let fx = if width > 1 {
                x as f32 / (width - 1) as f32
            } else {
                0.0
            };
            let lateral = if feet {
                1.0
            } else {
                0.8 + 0.2 * smoothstep(0.01, 0.18, fx.min(1.0 - fx))
            };
            out[index] = skin[index] * vertical * lateral;
        }
    }
    out
}

#[derive(Clone, Debug)]
struct SkinComponent {
    cells: Vec<usize>,
    center_x: f32,
    center_y: f32,
    area: usize,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

/// 8-connected components of the region confidence. No box substitute: the
/// cell list is kept so later stages can build fields from the real domain.
fn skin_components(region: &[f32], width: u32, height: u32) -> Vec<SkinComponent> {
    let w = width as usize;
    let h = height as usize;
    let mut seen = vec![false; region.len()];
    let mut result = Vec::new();
    for start in 0..region.len() {
        if seen[start] || region[start] <= 0.20 {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = true;
        let mut cells = Vec::new();
        while let Some(index) = stack.pop() {
            cells.push(index);
            let x = index % w;
            let y = index / w;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let next = ny as usize * w + nx as usize;
                    if !seen[next] && region[next] > 0.20 {
                        seen[next] = true;
                        stack.push(next);
                    }
                }
            }
        }
        let area = cells.len();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        for &cell in &cells {
            let x = cell % w;
            let y = cell / w;
            sum_x += x as f32;
            sum_y += y as f32;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        result.push(SkinComponent {
            cells,
            center_x: sum_x / area as f32,
            center_y: sum_y / area as f32,
            area,
            min_x: min_x as u32,
            min_y: min_y as u32,
            max_x: max_x as u32,
            max_y: max_y as u32,
        });
    }
    result
}

/// Accepted-hand/foot plane. Rejects tiny noise blobs and implausibly large
/// merges (torso) by area, and applies the hand/foot vertical prior again to
/// the component centroid.
fn selected_gate(
    region: &[f32],
    components: &[SkinComponent],
    width: u32,
    height: u32,
    feet: bool,
) -> Vec<f32> {
    let pixels = region.len();
    let mut gate = vec![0.0f32; pixels];
    let min_area = 24usize.max(pixels / 20_000);
    let area_cap = if feet { pixels / 6 } else { pixels / 8 };
    let max_area = area_cap.max(min_area + 1);
    for component in components {
        if component.area < min_area || component.area > max_area {
            continue;
        }
        let cy = component.center_y / height.max(1) as f32;
        if feet && cy < 0.55 {
            continue;
        }
        if !feet && cy > 0.80 {
            continue;
        }
        let span_x = (component.max_x - component.min_x + 1) as f32;
        let span_y = (component.max_y - component.min_y + 1) as f32;
        let elongation = 1.0 - span_x.min(span_y) / span_x.max(span_y).max(1.0);
        let score = 0.75 + 0.25 * elongation;
        for &cell in &component.cells {
            gate[cell] = region[cell] * score;
        }
    }
    gate
}

/// Weighted separable box blur. A zero-weight neighbourhood keeps its own
/// value, so unmasked pixels cannot contribute a skin baseline.
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
            let index = row + x;
            prefix_v.push(prefix_v[x] + values[index] * weight[index]);
            prefix_w.push(prefix_w[x] + weight[index]);
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
            let index = y * width + x;
            prefix_v[y + 1] = prefix_v[y] + horizontal[index] * weight[index];
            prefix_w[y + 1] = prefix_w[y] + weight[index];
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

/// BGRA channel planes plus the weighted baseline of each channel.
fn skin_planes(
    frame: &Image,
    gate: &[f32],
    radius: usize,
) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
    let count = frame.pixel_count();
    let mut b = vec![0.0f32; count];
    let mut g = vec![0.0f32; count];
    let mut r = vec![0.0f32; count];
    for index in 0..count {
        let offset = index * 4;
        b[index] = frame.pixels[offset] as f32;
        g[index] = frame.pixels[offset + 1] as f32;
        r[index] = frame.pixels[offset + 2] as f32;
    }
    let width = frame.width as usize;
    let height = frame.height as usize;
    (
        [b.clone(), g.clone(), r.clone()],
        [
            weighted_blur(&b, gate, width, height, radius),
            weighted_blur(&g, gate, width, height, radius),
            weighted_blur(&r, gate, width, height, radius),
        ],
    )
}

/// Dark, blue-shifted, thin structures relative to the local skin baseline.
/// `dark` uses the R channel; `blue_excess` compares B/G against R relative to
/// that same baseline, so a globally cool frame is not mistaken for veins.
fn vein_strength(frame: &Image, gate: &[f32]) -> Vec<f32> {
    let count = frame.pixel_count();
    let width = frame.width as usize;
    let height = frame.height as usize;
    let (channels, baseline) = skin_planes(frame, gate, 6);
    let (b, g, r) = (&channels[0], &channels[1], &channels[2]);
    let (base_b, base_g, base_r) = (&baseline[0], &baseline[1], &baseline[2]);
    let mut seed = vec![0.0f32; count];
    for index in 0..count {
        if gate[index] <= 0.02 {
            continue;
        }
        let dark = base_r[index] - r[index];
        let blue_excess = ((b[index] + g[index]) * 0.5 - r[index])
            - ((base_b[index] + base_g[index]) * 0.5 - base_r[index]);
        seed[index] = gate[index]
            * smoothstep(5.0, 16.0, dark)
            * smoothstep(3.0, 12.0, blue_excess);
    }
    if width < 3 || height < 3 {
        return seed;
    }
    let axes: [[isize; 2]; 4] = [[1, 0], [0, 1], [1, 1], [1, -1]];
    let mut out = vec![0.0f32; count];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = y * width + x;
            if seed[index] <= 0.02 {
                continue;
            }
            let mut neighbours = 0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = (x as isize + dx) as usize;
                    let ny = (y as isize + dy) as usize;
                    if seed[ny * width + nx] > 0.15 {
                        neighbours += 1;
                    }
                }
            }
            let mut best_axis = 0;
            for [ax, ay] in axes {
                let mut along = 0;
                for step in [1isize, 2isize] {
                    let nx = x as isize + ax * step;
                    let ny = y as isize + ay * step;
                    if nx > 0 && ny > 0 && nx < width as isize - 1 && ny < height as isize - 1
                        && seed[ny as usize * width + nx as usize] > 0.15
                    {
                        along += 1;
                    }
                    let nx = x as isize - ax * step;
                    let ny = y as isize - ay * step;
                    if nx > 0 && ny > 0 && nx < width as isize - 1 && ny < height as isize - 1
                        && seed[ny as usize * width + nx as usize] > 0.15
                    {
                        along += 1;
                    }
                }
                best_axis = best_axis.max(along);
            }
            if neighbours >= 2 && neighbours <= 12 && best_axis >= 2 {
                out[index] = seed[index];
            }
        }
    }
    out
}

fn hand_or_foot_gate(
    frame: &Image,
    body_skin_mask: &[u8],
    face_mask: Option<&[u8]>,
    feet: bool,
) -> Result<Vec<f32>, Refusal> {
    let skin = non_face_skin(frame, body_skin_mask, face_mask)?;
    let region = region_weight(&skin, frame.width, frame.height, feet);
    let components = skin_components(&region, frame.width, frame.height);
    let gate = selected_gate(&region, &components, frame.width, frame.height, feet);
    if !gate.iter().any(|&value| value > 0.02) {
        return Err(Refusal::MissingInput(if feet { FOOT_REGION } else { HAND_REGION }));
    }
    Ok(gate)
}

fn row_for(
    job_id: u64,
    frame: &Image,
    data: HistoricalToneRowData,
    control: f32,
) -> Result<HistoricalToneProviderRow, Refusal> {
    if job_id == 0 {
        return Err(Refusal::ValueOutOfRange(JOB));
    }
    if !control.is_finite() || !(0.0..=1.0).contains(&control) {
        return Err(Refusal::ValueOutOfRange("hands-feet control"));
    }
    Ok(HistoricalToneProviderRow {
        frame: frame.clone(),
        job: HistoricalToneRowJob {
            source_sha256: sha256_32(&frame.pixels),
            size: [frame.width, frame.height],
            control,
        },
        data,
    })
}

/// PF30030. The caller installs the returned row with `HistoricalToneProviderPayload`
/// ticket set to the same `job_id`; the row carries the exact frame digest/size.
pub(crate) fn build_pf30030_hand_veins_row(
    job_id: u64,
    frame: &Image,
    body_skin_mask: &[u8],
    face_mask: Option<&[u8]>,
) -> Result<HistoricalToneProviderRow, Refusal> {
    let gate = hand_or_foot_gate(frame, body_skin_mask, face_mask, false)?;
    let strength = vein_strength(frame, &gate);
    let pixels = frame.pixel_count();
    let width = frame.width as usize;
    let height = frame.height as usize;

    let mut mask = vec![0u8; pixels];
    let mut rgb = vec![80u8; pixels * 3];
    let (mut min_x, mut min_y) = (width, height);
    let (mut max_x, mut max_y) = (0usize, 0usize);
    for index in 0..pixels {
        let value = byte(strength[index] * 255.0);
        if value == 0 {
            continue;
        }
        mask[index] = value;
        let amount = value as f32 / 255.0;
        rgb[index * 3] = byte(80.0 - 46.0 * amount); // B: suppress blue
        rgb[index * 3 + 1] = byte(80.0 - 30.0 * amount); // G: suppress cyan cast
        rgb[index * 3 + 2] = byte(80.0 + 38.0 * amount); // R: restore skin red
        let x = index % width;
        let y = index / width;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    // A no-detection row is a well-formed no-op; the kernel still needs a
    // strictly in-frame ROI, so fall back to one pixel.
    let (x, y, roi_width, roi_height) = if max_x >= min_x && max_y >= min_y {
        let x0 = min_x.saturating_sub(2).min(width.saturating_sub(1));
        let y0 = min_y.saturating_sub(2).min(height.saturating_sub(1));
        let x1 = (max_x + 2).min(width - 1);
        let y1 = (max_y + 2).min(height - 1);
        (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    } else {
        (0, 0, 1.min(width), 1.min(height))
    };
    let data = HistoricalToneRowData::HandVeins(HistoricalToneHandVeinsPayload {
        percent: 100,
        mask,
        correction_rgb: rgb,
        feather: 2,
        roi: (x as u32, y as u32, roi_width as u32, roi_height as u32),
    });
    row_for(job_id, frame, data, 1.0)
}

/// PF44297. `feather_radius` is >= 1 as required by `foot_blue_vein` and the
/// repair-stage validation; the repair image is the local skin baseline
/// composited into the source only where a vein was detected.
pub(crate) fn build_pf44297_foot_vein_output(
    frame: &Image,
    body_skin_mask: &[u8],
    face_mask: Option<&[u8]>,
) -> Result<super::RepairOutput, Refusal> {
    let gate = hand_or_foot_gate(frame, body_skin_mask, face_mask, true)?;
    let strength = vein_strength(frame, &gate);
    let pixels = frame.pixel_count();
    let (_channels, baseline) = skin_planes(frame, &gate, 6);
    let mut repair = frame.clone();
    let mut model_mask = vec![0u8; pixels];
    for index in 0..pixels {
        let weight = strength[index];
        if weight <= 0.0 {
            continue;
        }
        model_mask[index] = byte(weight * 255.0);
        let offset = index * 4;
        for channel in 0..3 {
            let src = frame.pixels[offset + channel] as f32;
            let target = baseline[channel][index];
            repair.pixels[offset + channel] = byte(src * (1.0 - weight) + target * weight);
        }
        repair.pixels[offset + 3] = frame.pixels[offset + 3];
    }
    Ok(super::RepairOutput::FootBlueVein {
        repair,
        model_mask,
        feather_radius: 3,
    })
}

/// PF30187. Low-frequency illumination is removed by a broad weighted blur
/// and replaced with a neutral luminance plane; the residual high frequency
/// (fine skin detail) is retained at 60%. `alpha` gates the flatter, less
/// mottled areas more strongly.
pub(crate) fn build_pf30187_hand_gray_row(
    job_id: u64,
    frame: &Image,
    body_skin_mask: &[u8],
    face_mask: Option<&[u8]>,
) -> Result<HistoricalToneProviderRow, Refusal> {
    let gate = hand_or_foot_gate(frame, body_skin_mask, face_mask, false)?;
    let pixels = frame.pixel_count();
    let width = frame.width as usize;
    let height = frame.height as usize;
    let (channels, low) = skin_planes(frame, &gate, 10);
    let broad = [
        weighted_blur(&low[0], &gate, width, height, 24),
        weighted_blur(&low[1], &gate, width, height, 24),
        weighted_blur(&low[2], &gate, width, height, 24),
    ];

    let mut target = frame.clone();
    let mut mask = vec![0u8; pixels];
    let mut alpha = vec![0u8; pixels];
    for index in 0..pixels {
        let offset = index * 4;
        let luma = 0.114 * broad[0][index] + 0.587 * broad[1][index] + 0.299 * broad[2][index];
        let shading = (broad[2][index] - low[2][index]).abs();
        let confidence = 0.5 + 0.5 * smoothstep(2.0, 12.0, shading);
        for channel in 0..3 {
            let detail = channels[channel][index] - low[channel][index];
            target.pixels[offset + channel] = byte(luma + 0.6 * detail);
        }
        target.pixels[offset + 3] = frame.pixels[offset + 3];
        mask[index] = byte(gate[index] * 255.0);
        alpha[index] = byte(gate[index] * confidence * 255.0);
    }
    let data = HistoricalToneRowData::HandGray(HistoricalToneHandGrayPayload {
        mask,
        alpha,
        target,
    });
    row_for(job_id, frame, data, 1.0)
}

fn principal_axis(component: &SkinComponent, width: usize) -> [f32; 2] {
    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    let mut sxy = 0.0f32;
    for &cell in &component.cells {
        let x = (cell % width) as f32 - component.center_x;
        let y = (cell / width) as f32 - component.center_y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    [angle.cos(), angle.sin()]
}

/// Distance to the component boundary, capped at three pixels. The field
/// fades at the silhouette so the wrist/palm boundary is not torn.
fn inner_distance(component: &SkinComponent, width: usize, height: usize) -> Vec<u8> {
    let mut occupied = vec![false; width * height];
    for &cell in &component.cells {
        occupied[cell] = true;
    }
    let mut distance = vec![0u8; component.cells.len()];
    for (slot, &cell) in component.cells.iter().enumerate() {
        let x = cell % width;
        let y = cell / width;
        let mut nearest = 3usize;
        for radius in 1..=3usize {
            let mut outside = false;
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    if dx.abs() != radius as isize && dy.abs() != radius as isize {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize
                        || !occupied[ny as usize * width + nx as usize]
                    {
                        outside = true;
                    }
                }
            }
            if outside {
                nearest = radius - 1;
                break;
            }
        }
        distance[slot] = nearest as u8;
    }
    distance
}

/// PF30190. Each accepted hand component gets a slimming field toward its
/// principal axis and a slight along-axis elongation. `scale` is left `None`
/// so the kernel derives the source-scale factor itself.
pub(crate) fn build_pf30190_hand_shape_row(
    job_id: u64,
    frame: &Image,
    body_skin_mask: &[u8],
    face_mask: Option<&[u8]>,
    degree: f32,
) -> Result<HistoricalToneProviderRow, Refusal> {
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange(DEGREE));
    }
    let skin = non_face_skin(frame, body_skin_mask, face_mask)?;
    let width = frame.width as usize;
    let height = frame.height as usize;
    let region = region_weight(&skin, frame.width, frame.height, false);
    let components = skin_components(&region, frame.width, frame.height);
    let gate = selected_gate(&region, &components, frame.width, frame.height, false);
    if !gate.iter().any(|&value| value > 0.02) {
        return Err(Refusal::MissingInput(HAND_REGION));
    }

    let pixels = frame.pixel_count();
    let mut dx = vec![0.0f32; pixels];
    let mut dy = vec![0.0f32; pixels];
    let mut mask = vec![0u8; pixels];
    let mut protect = vec![255u8; pixels];
    let min_area = 24usize.max(pixels / 20_000);
    let max_area = (pixels / 8).max(min_area + 1);
    for component in &components {
        if component.area < min_area || component.area > max_area {
            continue;
        }
        let cy = component.center_y / frame.height.max(1) as f32;
        if cy > 0.80 {
            continue;
        }
        let span_x = (component.max_x - component.min_x + 1) as f32;
        let span_y = (component.max_y - component.min_y + 1) as f32;
        if span_x.min(span_y) / span_x.max(span_y).max(1.0) >= 1.0 {
            continue;
        }
        let [ax, ay] = principal_axis(component, width);
        let distances = inner_distance(component, width, height);
        for (slot, &cell) in component.cells.iter().enumerate() {
            let x = (cell % width) as f32;
            let y = (cell / width) as f32;
            let offset_x = x - component.center_x;
            let offset_y = y - component.center_y;
            let along = offset_x * ax + offset_y * ay;
            let perpendicular = -offset_x * ay + offset_y * ax;
            let envelope = smoothstep(0.0, 2.5, distances[slot] as f32);
            let slim = -perpendicular * 0.22 * envelope;
            let lengthen = along * 0.02 * envelope;
            dx[cell] = (-ay * slim + ax * lengthen) * gate[cell];
            dy[cell] = (ax * slim + ay * lengthen) * gate[cell];
            mask[cell] = byte(gate[cell] * 255.0);
            protect[cell] = byte((1.0 - envelope) * 255.0);
        }
    }
    let native_parameter = (degree * 100.0).round().clamp(0.0, 100.0) as i32;
    let data = HistoricalToneRowData::HandShape(HistoricalToneHandShapePayload {
        native_parameter,
        dx,
        dy,
        mask,
        protect,
        scale: None,
    });
    row_for(job_id, frame, data, degree)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Image {
        let width = 64usize;
        let height = 48usize;
        let mut pixels = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 4;
                pixels[offset] = 90;
                pixels[offset + 1] = 110;
                pixels[offset + 2] = 150;
                pixels[offset + 3] = 255;
            }
        }
        Image::new(width as u32, height as u32, pixels).unwrap()
    }

    /// One hand patch above the waist and one foot patch near the bottom.
    fn masks() -> (Vec<u8>, Vec<u8>) {
        let width = 64usize;
        let height = 48usize;
        let mut body = vec![0u8; width * height];
        let mut face = vec![0u8; width * height];
        for y in 18..30 {
            for x in 4..18 {
                body[y * width + x] = 255;
            }
        }
        for y in 36..45 {
            for x in 4..20 {
                body[y * width + x] = 255;
            }
        }
        for y in 4..16 {
            for x in 24..40 {
                face[y * width + x] = 255;
                body[y * width + x] = 255;
            }
        }
        (body, face)
    }

    fn with_vein(frame: &mut Image, y: usize, x0: usize, x1: usize) {
        for x in x0..x1 {
            let offset = (y * frame.width as usize + x) * 4;
            frame.pixels[offset] = 120;
            frame.pixels[offset + 1] = 115;
            frame.pixels[offset + 2] = 80;
        }
    }

    #[test]
    fn pf30030_builds_full_frame_vein_planes() {
        let mut source = frame();
        with_vein(&mut source, 23, 6, 14);
        let (body, face) = masks();
        let row = build_pf30030_hand_veins_row(41, &source, &body, Some(&face)).unwrap();
        assert_eq!(row.pf(), 30030);
        assert_eq!(row.job.size, [source.width, source.height]);
        assert_eq!(row.job.source_sha256, sha256_32(&source.pixels));
        assert_eq!(row.job.control, 1.0);
        let HistoricalToneRowData::HandVeins(payload) = &row.data else {
            panic!("PF30030 payload");
        };
        assert_eq!(payload.percent, 100);
        assert_eq!(payload.mask.len(), source.pixel_count());
        assert_eq!(payload.correction_rgb.len(), source.pixel_count() * 3);
        assert!(payload.mask.iter().any(|&value| value > 0));
        assert_eq!(payload.correction_rgb[2], 80);
        let (x, y, w, h) = payload.roi;
        assert!(w > 0 && h > 0);
        assert!(x + w <= source.width && y + h <= source.height);
    }

    #[test]
    fn pf44297_builds_repair_output() {
        let mut source = frame();
        with_vein(&mut source, 40, 6, 16);
        let (body, face) = masks();
        let output = build_pf44297_foot_vein_output(&source, &body, Some(&face)).unwrap();
        assert_eq!(output.pf(), 44297);
        let super::super::RepairOutput::FootBlueVein {
            repair,
            model_mask,
            feather_radius,
        } = output
        else {
            panic!("PF44297 output");
        };
        assert_eq!((repair.width, repair.height), (source.width, source.height));
        assert_eq!(model_mask.len(), source.pixel_count());
        assert!(model_mask.iter().any(|&value| value > 0));
        assert!(feather_radius >= 1);
        let at = (40 * source.width as usize + 8) * 4;
        assert_ne!(repair.pixels[at], source.pixels[at]);
    }

    #[test]
    fn pf30187_builds_neutral_gray_planes() {
        let source = frame();
        let (body, face) = masks();
        let row = build_pf30187_hand_gray_row(41, &source, &body, Some(&face)).unwrap();
        assert_eq!(row.pf(), 30187);
        let HistoricalToneRowData::HandGray(payload) = &row.data else {
            panic!("PF30187 payload");
        };
        assert_eq!(payload.mask.len(), source.pixel_count());
        assert_eq!(payload.alpha.len(), source.pixel_count());
        assert_eq!((payload.target.width, payload.target.height), (source.width, source.height));
        assert_eq!(payload.target.pixels.len(), source.pixels.len());
        assert!(payload.mask.iter().any(|&value| value > 0));
        assert!(payload.alpha.iter().any(|&value| value > 0));
        // The neutral target keeps the source alpha and is a BGRA image of
        // the exact frame shape, with an actual hand-plane mask present.
        let index = (23 * source.width as usize + 8) * 4;
        assert_ne!(payload.target.pixels[index], 0);
        assert_eq!(payload.target.pixels[index + 3], 255);
    }

    #[test]
    fn pf30190_builds_principal_axis_field() {
        let source = frame();
        let (body, face) = masks();
        let row = build_pf30190_hand_shape_row(41, &source, &body, Some(&face), 0.8).unwrap();
        assert_eq!(row.pf(), 30190);
        assert_eq!(row.job.control, 0.8);
        let HistoricalToneRowData::HandShape(payload) = &row.data else {
            panic!("PF30190 payload");
        };
        assert_eq!(payload.native_parameter, 80);
        assert_eq!(payload.dx.len(), source.pixel_count());
        assert_eq!(payload.dy.len(), source.pixel_count());
        assert_eq!(payload.mask.len(), source.pixel_count());
        assert_eq!(payload.protect.len(), source.pixel_count());
        assert!(payload.dx.iter().all(|value| value.is_finite()));
        assert!(payload.dy.iter().all(|value| value.is_finite()));
        assert!(payload.dx.iter().any(|&value| value.abs() > 0.01));
        assert!(payload.mask.iter().any(|&value| value > 0));
        assert!(payload.protect.iter().any(|&value| value > 0));
        assert!(payload.scale.is_none());
    }

    #[test]
    fn missing_face_mask_is_accepted() {
        let source = frame();
        let (body, _) = masks();
        assert!(build_pf30030_hand_veins_row(41, &source, &body, None).is_ok());
        assert!(build_pf30187_hand_gray_row(41, &source, &body, None).is_ok());
        assert!(build_pf30190_hand_shape_row(41, &source, &body, None, 0.5).is_ok());
        assert!(build_pf44297_foot_vein_output(&source, &body, None).is_ok());
    }

    #[test]
    fn illegal_inputs_are_refused() {
        let source = frame();
        let (body, face) = masks();
        let short = vec![0u8; source.pixel_count() - 1];
        assert_eq!(
            build_pf30030_hand_veins_row(41, &source, &short, Some(&face)),
            Err(Refusal::ShapeMismatch(SKIN_MASK))
        );
        assert_eq!(
            build_pf44297_foot_vein_output(&source, &body, Some(&short)),
            Err(Refusal::ShapeMismatch(FACE_MASK))
        );
        assert_eq!(
            build_pf30187_hand_gray_row(0, &source, &body, None),
            Err(Refusal::ValueOutOfRange(JOB))
        );
        assert_eq!(
            build_pf30190_hand_shape_row(41, &source, &body, None, f32::NAN),
            Err(Refusal::ValueOutOfRange(DEGREE))
        );
        assert_eq!(
            build_pf30190_hand_shape_row(41, &source, &body, None, 1.5),
            Err(Refusal::ValueOutOfRange(DEGREE))
        );
        let blank = vec![0u8; source.pixel_count()];
        assert_eq!(
            build_pf30030_hand_veins_row(41, &source, &blank, None),
            Err(Refusal::MissingInput(SKIN_MASK))
        );
    }
}
