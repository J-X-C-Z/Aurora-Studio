//! Same-frame producers for the four recovered PixelCake hair / glasses rows:
//!
//! * `PF21026` beard removal -> `BeardInputs { mask_a, mask_b, repair_rgb }`
//!   consumed by `portrait_stage::apply_beard_cpu` /
//!   `aurora_portrait::kernels::beard::apply` (`w = min(mask_a, mask_b)`).
//! * `PF90088` stray hair -> `StrayHairProducerInputs` consumed by
//!   `resolve_stray_hair` and `hair_body::stray_hair_with_native` (CPU) or
//!   `cuda_stray_hair` (NVIDIA). The vendor's `raw_mask`, `hairfly_v1`,
//!   person/background planes and native scalars are all required.
//! * `PF90131` white hair -> `RepairOutput::WhiteHair { repair, raw_mask }`.
//!   The consumer law is `alpha = repair.a/255 * (1 - raw_mask/255) * degree`,
//!   so `raw_mask = 0` means "repair fully" and `255` means identity.
//! * `PF90130` glasses glare -> `SkinRepairData::Glare(GlareInputs)` consumed by
//!   `aurora_portrait::kernels::face_detail::glasses_glare_removal`. Its
//!   `effect_rgb` is a full-frame interleaved **[-1, 1]** plane (`0` encodes the
//!   neutral byte `128`) and `lens_mask` is a full-frame `[0, 1]` plane.
//!
//! This module does not run a vendor model and does not claim vendor parsing
//! planes. The inputs it accepts (`points194`, `skin_mask`, `hair_mask`,
//! `eye_mask`) are the engine's own same-frame planes. The produced planes are
//! explicit geometric / pixel-level estimates that satisfy the recovered
//! consumer contracts; where the vendor's own network output is the source of a
//! plane, that is stated in the module that owns the network.
//!
//! Mount point (owned by the coordinator):
//!
//! ```ignore
//! #[path = "portrait_pixcake_hair_glare.rs"]
//! mod portrait_pixcake_hair_glare;
//! ```
//!
//! `build_pf21026_beard_inputs` returns an owned type because `BeardInputs`
//! borrows its three planes. [`BeardProducerInputs::as_borrowed`] bridges the
//! owned result to the exact consumer struct.

use aurora_portrait::kernels::face_detail::GlareMaskEncoding;
use aurora_portrait::{Image, Refusal};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

const POINTS: &str = "PF21026_POINTS194";

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
    if edge1 <= edge0 {
        return if value >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn point(points: &[[f32; 2]], index: usize) -> Option<[f32; 2]> {
    points
        .get(index)
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// Normalised ellipse bump: `1` at the centre, `0` on and outside the rim.
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

fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

fn saturation(r: f32, g: f32, b: f32) -> f32 {
    let maximum = r.max(g).max(b);
    let minimum = r.min(g).min(b);
    if maximum > 1.0 {
        (maximum - minimum) / maximum
    } else {
        0.0
    }
}

/// Normalised separable box blur. `weight` gates every sample so out-of-region
/// pixels cannot bleed into the local baseline. A zero-weight neighbourhood
/// keeps the untouched value (callers always include a small floor when they
/// need a full-frame baseline).
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

#[allow(clippy::too_many_arguments)]
fn blur_rgb(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    weight: &[f32],
    width: usize,
    height: usize,
    radius: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        weighted_blur(r, weight, width, height, radius),
        weighted_blur(g, weight, width, height, radius),
        weighted_blur(b, weight, width, height, radius),
    )
}

fn feather_beard_mask(mask: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; mask.len()];
    for y in 0..height {
        for x in 0..width {
            let p = y * width + x;
            let center = f32::from(mask[p]);
            if center == 0.0 {
                continue;
            }
            let mut sum = 0u16;
            let mut count = 0u16;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    sum += u16::from(mask[ny * width + nx]);
                    count += 1;
                }
            }
            // Feather inward without spreading the repair across a protected edge.
            let mean = f32::from(sum) / f32::from(count);
            out[p] = clamp_byte(0.75 * center + 0.25 * center.min(mean));
        }
    }
    out
}

/// Approximate distance from every pixel to the nearest `mask > 0` pixel, in
/// units of 1/5 px (3-4 chamfer). Used for the PF90088 outer transition band.
fn distance_from_mask(mask: &[u8], width: usize, height: usize) -> Vec<u32> {
    const STRAIGHT: u32 = 5;
    const DIAGONAL: u32 = 7;
    const FAR: u32 = u32::MAX / 8;
    let mut dist = vec![FAR; width * height];
    for (slot, &m) in dist.iter_mut().zip(mask.iter()) {
        if m > 0 {
            *slot = 0;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let mut best = dist[i];
            if x > 0 {
                best = best.min(dist[i - 1].saturating_add(STRAIGHT));
            }
            if y > 0 {
                best = best.min(dist[i - width].saturating_add(STRAIGHT));
                if x > 0 {
                    best = best.min(dist[i - width - 1].saturating_add(DIAGONAL));
                }
                if x + 1 < width {
                    best = best.min(dist[i - width + 1].saturating_add(DIAGONAL));
                }
            }
            dist[i] = best;
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let i = y * width + x;
            let mut best = dist[i];
            if x + 1 < width {
                best = best.min(dist[i + 1].saturating_add(STRAIGHT));
            }
            if y + 1 < height {
                best = best.min(dist[i + width].saturating_add(STRAIGHT));
                if x + 1 < width {
                    best = best.min(dist[i + width + 1].saturating_add(DIAGONAL));
                }
                if x > 0 {
                    best = best.min(dist[i + width - 1].saturating_add(DIAGONAL));
                }
            }
            dist[i] = best;
        }
    }
    dist
}

// ---------------------------------------------------------------------------
// PF21026 beard removal
// ---------------------------------------------------------------------------

const BEARD_SKIN: &str = "PF21026_SKIN_MASK";

/// Owned counterpart of `portrait_stage::BeardInputs`. The consumer borrows its
/// three planes, so a producer that allocates them must hand back an owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BeardProducerInputs {
    /// Full-frame anatomical beard coverage, feathered.
    pub mask_a: Vec<u8>,
    /// Full-frame detected dark stubble with feathered coverage.
    pub mask_b: Vec<u8>,
    /// Full-frame repaired skin, same dimensions as the source frame.
    pub repair_rgb: Image,
    pub repair_source: Image,
}

impl BeardProducerInputs {
    /// Borrow this owned payload as the exact consumer struct.
    #[allow(dead_code)]
    pub(crate) fn as_borrowed(&self) -> super::BeardInputs<'_> {
        super::BeardInputs {
            mask_a: &self.mask_a,
            mask_b: &self.mask_b,
            repair_rgb: &self.repair_rgb,
        }
    }
}

/// Builds the two PF21026 masks and the repaired skin image.
///
/// The beard zone is anchored on the 194-point mouth corners (73/83), nose tip
/// (57), chin (125) and jaw contour (109/141): moustache above the upper lip,
/// chin patch below the lower lip, and the two jaw segments. The lip line,
/// mouth corners and nose base are guarded. `mask_b` follows fine dark ridges
/// with brighter skin on both sides.
pub(crate) fn build_pf21026_beard_inputs(
    frame: &Image,
    face_points194: &[[f32; 2]],
    skin_mask: Option<&[u8]>,
) -> Result<BeardProducerInputs, Refusal> {
    frame.validate()?;
    let pixels = frame.pixel_count();
    if let Some(mask) = skin_mask {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch(BEARD_SKIN));
        }
        if !mask.iter().any(|&value| value > 0) {
            return Err(Refusal::MissingInput(BEARD_SKIN));
        }
    }
    if face_points194
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange(POINTS));
    }

    let mouth_l = point(face_points194, 73)
        .or_else(|| point(face_points194, 83))
        .or_else(|| point(face_points194, 84));
    let mouth_r = point(face_points194, 83)
        .or_else(|| point(face_points194, 84))
        .or_else(|| point(face_points194, 73));
    let (Some(mouth_l), Some(mouth_r), Some(nose_tip), Some(chin)) =
        (mouth_l, mouth_r, point(face_points194, 57), point(face_points194, 125))
    else {
        return Err(Refusal::ValueOutOfRange(POINTS));
    };

    let mouth_mid = midpoint(mouth_l, mouth_r);
    let mouth_w = distance(mouth_l, mouth_r).max(6.0);
    let eye_dist = match (
        point(face_points194, 26),
        point(face_points194, 44),
    ) {
        (Some(a), Some(b)) => distance(a, b),
        _ => mouth_w * 1.6,
    }
    .max(mouth_w * 1.15)
    .max(16.0);

    let jaw_l = point(face_points194, 109);
    let jaw_r = point(face_points194, 141);

    let moustache_center = [
        nose_tip[0] + (mouth_mid[0] - nose_tip[0]) * 0.55,
        nose_tip[1] + (mouth_mid[1] - nose_tip[1]) * 0.55,
    ];
    let chin_center = [
        mouth_mid[0] + (chin[0] - mouth_mid[0]) * 0.35,
        mouth_mid[1] + (chin[1] - mouth_mid[1]) * 0.55,
    ];
    let chin_rx = mouth_w * 0.55;
    let chin_ry = (chin[1] - mouth_mid[1]).abs().max(eye_dist * 0.20) * 0.55;
    let lip_radius = eye_dist * 0.14;
    let nose_rx = (mouth_w * 0.32).max(eye_dist * 0.12);
    let nose_ry = eye_dist * 0.13;
    let jaw_radius = eye_dist * 0.20;
    let cheek_radius = eye_dist * 0.16;
    let lower_face = [chin[0] - nose_tip[0], chin[1] - nose_tip[1]];
    let lower_face_len = lower_face[0].hypot(lower_face[1]).max(f32::EPSILON);
    let down = [lower_face[0] / lower_face_len, lower_face[1] / lower_face_len];

    let (width, height) = (frame.width as usize, frame.height as usize);
    let mut zone = vec![0.0f32; pixels];
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let mut weight = ellipse(
                fx,
                fy,
                moustache_center,
                mouth_w * 0.72,
                (eye_dist * 0.16).max(2.0),
            );
            weight = weight.max(ellipse(fx, fy, chin_center, chin_rx, chin_ry));
            if let Some(jaw) = jaw_l {
                weight = weight.max(segment_weight(fx, fy, jaw, chin, jaw_radius));
                weight = weight.max(segment_weight(fx, fy, mouth_l, jaw, cheek_radius));
            }
            if let Some(jaw) = jaw_r {
                weight = weight.max(segment_weight(fx, fy, chin, jaw, jaw_radius));
                weight = weight.max(segment_weight(fx, fy, mouth_r, jaw, cheek_radius));
            }
            // Mouth-to-jaw bands can cross the lower eyelid on a turned face.
            let below_nose = (fx - nose_tip[0]) * down[0] + (fy - nose_tip[1]) * down[1];
            weight *= smoothstep(0.0, eye_dist * 0.12, below_nose);
            weight *= 1.0 - segment_weight(fx, fy, mouth_l, mouth_r, lip_radius);
            let across_nose = (fx - nose_tip[0]) * down[1] - (fy - nose_tip[1]) * down[0];
            let nose_guard = ellipse(
                across_nose,
                below_nose,
                [0.0, eye_dist * 0.03],
                nose_rx,
                nose_ry,
            );
            weight *= 1.0 - smoothstep(0.0, 0.65, nose_guard * nose_guard);
            zone[y * width + x] = weight.clamp(0.0, 1.0);
        }
    }
    if zone.iter().all(|&value| value <= 0.0) {
        return Err(Refusal::ValueOutOfRange("PF21026_BEARD_ZONE"));
    }

    let mut source_r = vec![0.0f32; pixels];
    let mut source_g = vec![0.0f32; pixels];
    let mut source_b = vec![0.0f32; pixels];
    let mut source_luma = vec![0.0f32; pixels];
    let mut region = vec![0.0f32; pixels];
    for p in 0..pixels {
        let r = frame.pixels[p * 4] as f32;
        let g = frame.pixels[p * 4 + 1] as f32;
        let b = frame.pixels[p * 4 + 2] as f32;
        source_r[p] = r;
        source_g[p] = g;
        source_b[p] = b;
        source_luma[p] = luma(r, g, b);
        let skin = skin_mask.map_or(1.0, |mask| mask[p] as f32 / 255.0);
        region[p] = zone[p] * skin;
    }

    // A fine dark ridge has brighter samples on both sides across its width.
    // A smooth shadow or a one-sided step lacks this contrast at hair scale.
    let probe = ((eye_dist * 0.012).round() as usize).clamp(2, 5);
    let diagonal = ((probe as f32 * std::f32::consts::FRAC_1_SQRT_2).round() as usize).max(1);
    let offsets = [
        probe,
        probe * width,
        diagonal * (width + 1),
        diagonal * (width - 1),
    ];
    let mut texture_drop = vec![0.0f32; pixels];
    for y in probe..height.saturating_sub(probe) {
        for x in probe..width.saturating_sub(probe) {
            let p = y * width + x;
            if region[p] <= 0.0 {
                continue;
            }
            let mut brighter_side = source_luma[p];
            for offset in offsets {
                brighter_side =
                    brighter_side.max(source_luma[p - offset].min(source_luma[p + offset]));
            }
            texture_drop[p] = brighter_side - source_luma[p];
        }
    }

    let radius = probe * 3;
    let mut clean_weight = vec![0.0f32; pixels];
    let mut mask_a = vec![0u8; pixels];
    let mut mask_b_raw = vec![0u8; pixels];
    for p in 0..pixels {
        let hair = smoothstep(4.0, 18.0, texture_drop[p]);
        clean_weight[p] = region[p] * (1.0 - 0.80 * hair);
        mask_a[p] = clamp_byte(255.0 * region[p]);
        mask_b_raw[p] = clamp_byte(255.0 * region[p] * hair);
    }
    let (base_r, base_g, base_b) =
        blur_rgb(&source_r, &source_g, &source_b, &clean_weight, width, height, radius);

    let mut repair_pixels = frame.pixels.clone();
    for p in 0..pixels {
        if mask_b_raw[p] == 0 {
            continue;
        }
        let lift = luma(base_r[p], base_g[p], base_b[p]) - source_luma[p];
        if lift <= 0.0 {
            continue;
        }
        // Keep the repair within the measured texture contrast near shadows.
        let mix = (0.85 * texture_drop[p] / lift).min(1.0);
        repair_pixels[p * 4] = clamp_byte(source_r[p] + mix * (base_r[p] - source_r[p]));
        repair_pixels[p * 4 + 1] = clamp_byte(source_g[p] + mix * (base_g[p] - source_g[p]));
        repair_pixels[p * 4 + 2] = clamp_byte(source_b[p] + mix * (base_b[p] - source_b[p]));
    }
    let mask_b = feather_beard_mask(&mask_b_raw, width, height);
    let repair_rgb =
        Image::new(frame.width, frame.height, repair_pixels).ok_or(Refusal::ShapeMismatch(
            "PF21026_REPAIR_RGB",
        ))?;
    Ok(BeardProducerInputs {
        mask_a,
        mask_b,
        repair_rgb,
        repair_source: frame.clone(),
    })
}

// ---------------------------------------------------------------------------
// PF90088 stray hair removal
// ---------------------------------------------------------------------------

const STRAY_HAIR_MASK: &str = "PF90088_HAIR_MASK";
const STRAY_SKIN_MASK: &str = "PF90088_SKIN_MASK";
const STRAY_BAND: &str = "PF90088_STRAY_BAND";

/// Builds the PF90088 same-job planes for the recovered consumer.
///
/// The stray-hair search band is the `4..=18 px` ring outside `hair_mask`.
/// A strand is accepted when it is significantly darker than the local
/// background **and** thinner than the surrounding structure (fine blur follows
/// the strand, medium blur recovers, coarse blur recovers more). A broad dark
/// region therefore fails the thinness test and is not rewritten.
///
/// `person_mask` / `background_mask` are the spatial partition derived from
/// `hair_mask` and the optional `skin_mask`. The recovered vendor scalars
/// PF90113/PF90114 are per-face model values this producer does not have, so
/// the two area scalars are `1.0` (the multiplicative identity) and the
/// per-pixel planes carry the spatial gate. This is stated instead of inventing
/// a model value.
pub(crate) fn build_pf90088_stray_hair_inputs(
    frame: &Image,
    hair_mask: &[u8],
    skin_mask: Option<&[u8]>,
) -> Result<super::StrayHairProducerInputs, Refusal> {
    frame.validate()?;
    let pixels = frame.pixel_count();
    if hair_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch(STRAY_HAIR_MASK));
    }
    if !hair_mask.iter().any(|&value| value > 0) {
        return Err(Refusal::MissingInput(STRAY_HAIR_MASK));
    }
    if let Some(mask) = skin_mask {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch(STRAY_SKIN_MASK));
        }
    }

    let (width, height) = (frame.width as usize, frame.height as usize);
    let distance_units = distance_from_mask(hair_mask, width, height);
    let mut band = vec![0.0f32; pixels];
    let mut band_pixels = 0usize;
    for p in 0..pixels {
        if hair_mask[p] > 0 {
            continue;
        }
        let px = distance_units[p] as f32 / 5.0;
        if (4.0..=18.0).contains(&px) {
            band[p] = 1.0;
            band_pixels += 1;
        }
    }
    if band_pixels == 0 {
        return Err(Refusal::MissingInput(STRAY_BAND));
    }

    let mut source_r = vec![0.0f32; pixels];
    let mut source_g = vec![0.0f32; pixels];
    let mut source_b = vec![0.0f32; pixels];
    let mut source_luma = vec![0.0f32; pixels];
    for p in 0..pixels {
        let r = frame.pixels[p * 4] as f32;
        let g = frame.pixels[p * 4 + 1] as f32;
        let b = frame.pixels[p * 4 + 2] as f32;
        source_r[p] = r;
        source_g[p] = g;
        source_b[p] = b;
        source_luma[p] = luma(r, g, b);
    }

    let medium = weighted_blur(&source_luma, &band, width, height, 6);
    let coarse = weighted_blur(&source_luma, &band, width, height, 14);

    let mut raw_mask = vec![0u8; pixels];
    let mut confidence = vec![0.0f32; pixels];
    for p in 0..pixels {
        if band[p] <= 0.0 {
            continue;
        }
        let drop = coarse[p] - source_luma[p];
        let dark = smoothstep(8.0, 28.0, drop);
        // Thin strands are followed by the fine blur but recovered by the
        // medium blur; a broad dark patch is already dark at the medium scale.
        let recovered = medium[p] - source_luma[p];
        let total = drop.max(1.0);
        let ratio = (recovered / total).clamp(0.0, 1.0);
        let thin = smoothstep(0.45, 0.85, ratio);
        let value = (dark * thin).clamp(0.0, 1.0);
        confidence[p] = value;
        raw_mask[p] = clamp_byte(255.0 * value);
    }

    // Clean background/skin target. Accepted strands keep a small vote so the
    // baseline itself is never pulled to the strand colour.
    let mut clean_weight = vec![0.0f32; pixels];
    for p in 0..pixels {
        if band[p] > 0.0 {
            clean_weight[p] = 0.15 + 0.85 * (1.0 - confidence[p]);
        }
    }
    let (base_r, base_g, base_b) =
        blur_rgb(&source_r, &source_g, &source_b, &clean_weight, width, height, 10);

    let mut repair_pixels = frame.pixels.clone();
    for p in 0..pixels {
        if band[p] <= 0.0 {
            continue;
        }
        repair_pixels[p * 4] = clamp_byte(base_r[p]);
        repair_pixels[p * 4 + 1] = clamp_byte(base_g[p]);
        repair_pixels[p * 4 + 2] = clamp_byte(base_b[p]);
    }
    let hairfly_v1 = Image::new(frame.width, frame.height, repair_pixels).ok_or(
        Refusal::ShapeMismatch("PF90088_HAIRFLY_V1"),
    )?;

    let mut person_mask = vec![0u8; pixels];
    for p in 0..pixels {
        let skin = skin_mask.map_or(0u8, |mask| mask[p]);
        person_mask[p] = hair_mask[p].max(skin);
    }
    let background_mask: Vec<u8> = person_mask.iter().map(|&value| 255 - value).collect();

    Ok(super::StrayHairProducerInputs {
        hairfly_v1: Some(hairfly_v1),
        raw_mask: Some(raw_mask),
        person_mask: Some(person_mask),
        background_mask: Some(background_mask),
        native_degree: Some(1.0),
        native_person_area: Some(1.0),
        native_background_area: Some(1.0),
    })
}

// ---------------------------------------------------------------------------
// PF90131 white hair
// ---------------------------------------------------------------------------

const WHITE_HAIR_MASK: &str = "PF90131_HAIR_MASK";

/// Dominant dark-hair luminance threshold: the 25th percentile of the hair
/// pixels, so a few grey strands cannot move the reference colour.
fn dark_hair_threshold(hair_mask: &[u8], luma_plane: &[f32]) -> f32 {
    let mut histogram = [0u32; 256];
    let mut total = 0u32;
    for (p, &mask) in hair_mask.iter().enumerate() {
        if mask > 0 {
            let bin = luma_plane[p].clamp(0.0, 255.0).round() as usize;
            histogram[bin.min(255)] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    let target = (total / 4).max(1);
    let mut accumulated = 0u32;
    for (value, &count) in histogram.iter().enumerate() {
        accumulated += count;
        if accumulated >= target {
            return value as f32;
        }
    }
    255.0
}

/// Recolours white/grey strands inside `hair_mask` to the dominant dark hair
/// colour while keeping the strand's own sheen.
///
/// The recovered consumer reads `raw_mask` inverted: `0` is full repair and
/// `255` is identity. Only detected low-saturation bright strands receive a
/// low value; every other pixel stays `255`, so the output is a no-op there.
pub(crate) fn build_pf90131_white_hair_output(
    frame: &Image,
    hair_mask: &[u8],
) -> Result<super::RepairOutput, Refusal> {
    frame.validate()?;
    let pixels = frame.pixel_count();
    if hair_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch(WHITE_HAIR_MASK));
    }
    if !hair_mask.iter().any(|&value| value > 0) {
        return Err(Refusal::MissingInput(WHITE_HAIR_MASK));
    }

    let (width, height) = (frame.width as usize, frame.height as usize);
    let mut source_luma = vec![0.0f32; pixels];
    for p in 0..pixels {
        source_luma[p] = luma(
            frame.pixels[p * 4] as f32,
            frame.pixels[p * 4 + 1] as f32,
            frame.pixels[p * 4 + 2] as f32,
        );
    }

    let threshold = dark_hair_threshold(hair_mask, &source_luma);
    let mut dominant_sum = [0.0f64; 3];
    let mut dominant_count = 0u64;
    for p in 0..pixels {
        if hair_mask[p] == 0 || source_luma[p] > threshold + 1.0 {
            continue;
        }
        dominant_sum[0] += frame.pixels[p * 4] as f64;
        dominant_sum[1] += frame.pixels[p * 4 + 1] as f64;
        dominant_sum[2] += frame.pixels[p * 4 + 2] as f64;
        dominant_count += 1;
    }
    if dominant_count == 0 {
        // No dark reference at all: every hair pixel is already light. There is
        // nothing to recolour to, so return the identity planes instead of
        // inventing a colour.
        let repair = frame.clone();
        return Ok(super::RepairOutput::WhiteHair {
            repair,
            raw_mask: vec![255u8; pixels],
        });
    }
    let dominant = [
        (dominant_sum[0] / dominant_count as f64) as f32,
        (dominant_sum[1] / dominant_count as f64) as f32,
        (dominant_sum[2] / dominant_count as f64) as f32,
    ];
    let dominant_luma = luma(dominant[0], dominant[1], dominant[2]);
    let white_threshold = (dominant_luma + 60.0).max(150.0);

    let hair_weight: Vec<f32> =
        hair_mask.iter().map(|&value| value as f32 / 255.0).collect();
    let smooth_luma = weighted_blur(&source_luma, &hair_weight, width, height, 2);

    let mut raw_mask = vec![255u8; pixels];
    let mut repair_pixels = frame.pixels.clone();
    for p in 0..pixels {
        if hair_mask[p] == 0 {
            continue;
        }
        let r = frame.pixels[p * 4] as f32;
        let g = frame.pixels[p * 4 + 1] as f32;
        let b = frame.pixels[p * 4 + 2] as f32;
        let bright = smoothstep(
            white_threshold - 10.0,
            white_threshold + 40.0,
            source_luma[p],
        );
        let low_saturation = 1.0 - smoothstep(0.10, 0.30, saturation(r, g, b));
        let confidence = (bright * low_saturation * (hair_mask[p] as f32 / 255.0))
            .clamp(0.0, 1.0);
        if confidence <= 0.0 {
            continue;
        }
        raw_mask[p] = clamp_byte(255.0 * (1.0 - confidence));

        // Darken toward the dominant hair colour, scaled by the strand's own
        // luminance so highlights keep their sheen instead of turning flat.
        let sheen = (source_luma[p] / 255.0).clamp(0.0, 1.0);
        let gain = 0.55 + 0.75 * sheen;
        let high_frequency = (source_luma[p] - smooth_luma[p]) * 0.25;
        repair_pixels[p * 4] = clamp_byte(dominant[0] * gain + high_frequency);
        repair_pixels[p * 4 + 1] = clamp_byte(dominant[1] * gain + high_frequency);
        repair_pixels[p * 4 + 2] = clamp_byte(dominant[2] * gain + high_frequency);
        repair_pixels[p * 4 + 3] = 255;
    }

    let repair = Image::new(frame.width, frame.height, repair_pixels)
        .ok_or(Refusal::ShapeMismatch("PF90131_REPAIR_RGB"))?;
    Ok(super::RepairOutput::WhiteHair { repair, raw_mask })
}

// ---------------------------------------------------------------------------
// PF90130 glasses glare
// ---------------------------------------------------------------------------

const GLARE_POINTS: &str = "PF90130_POINTS194";
const GLARE_EYE_MASK: &str = "PF90130_EYE_MASK";
const GLARE_LENS: &str = "PF90130_LENS_REGION";

struct Lens {
    center: [f32; 2],
    rx: f32,
    ry: f32,
    pupil_rx: f32,
    pupil_ry: f32,
}

fn lens_geometry(
    points: &[[f32; 2]],
    corner_a: usize,
    corner_b: usize,
    center_index: usize,
    upper_index: usize,
    lower_index: usize,
) -> Option<Lens> {
    let a = point(points, corner_a)?;
    let b = point(points, corner_b)?;
    let center = point(points, center_index).unwrap_or_else(|| midpoint(a, b));
    let eye_w = distance(a, b).max(6.0);
    let eye_h = match (point(points, upper_index), point(points, lower_index)) {
        (Some(upper), Some(lower)) => distance(upper, lower).max(eye_w * 0.35),
        _ => eye_w * 0.45,
    };
    let pupil = (eye_w * 0.13).max(2.0);
    Some(Lens {
        center,
        rx: eye_w * 0.95,
        ry: (eye_h * 1.7).max(eye_w * 0.70),
        pupil_rx: pupil,
        pupil_ry: pupil * 0.9,
    })
}

/// Full-frame PF90130 row for `SkinRepairProducerInputs::insert`.
///
/// `control` is `1.0`: the per-pixel lens plane carries the glare weight, so
/// the recovered consumer's `ui_percent` multiplier is left at the identity.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Pf90130GlareRow {
    pub ticket: u64,
    pub control: f32,
    pub data: super::SkinRepairData,
}

/// Encodes a byte-domain target as the consumer's `[-1, 1]` float plane
/// (`0.0` is the neutral byte `128`).
fn glare_encode(value: f32) -> f32 {
    (value / 127.5 - 1.0).clamp(-1.0, 1.0)
}

/// Builds the PF90130 `GlareInputs` row.
///
/// The lens search region is an ellipse around each eye (landmarks 18/22/26 and
/// 36/40/44), sized from the eye contour. Within that region a specular sample
/// must be both brighter than its local lens baseline and low-saturation. A
/// small disc around the pupil landmarks is excluded so the recovered
/// catchlight is never treated as lens glare. The repair target is the
/// surrounding lens / periocular colour, so the blend reads as a directional
/// fill rather than a flat patch.
///
/// `mask` is `None` and `lens_mask` carries the weight with
/// [`GlareMaskEncoding::UnormLens`]. That keeps the consumer's recovered
/// `knee()` step from quantising the feathered edge: `unit = 1`, so the final
/// byte is `round(255 * lens_weight * strength)`.
pub(crate) fn build_pf90130_glare_row(
    job_id: u64,
    frame: &Image,
    face_points194: &[[f32; 2]],
    eye_mask: Option<&[u8]>,
) -> Result<Pf90130GlareRow, Refusal> {
    if job_id == 0 {
        return Err(Refusal::ValueOutOfRange("PF90130_JOB_TICKET"));
    }
    frame.validate()?;
    let pixels = frame.pixel_count();
    if let Some(mask) = eye_mask {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch(GLARE_EYE_MASK));
        }
        if !mask.iter().any(|&value| value > 0) {
            return Err(Refusal::MissingInput(GLARE_EYE_MASK));
        }
    }
    if face_points194
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange(GLARE_POINTS));
    }

    let left = lens_geometry(face_points194, 18, 22, 26, 20, 24)
        .ok_or(Refusal::ValueOutOfRange(GLARE_POINTS))?;
    let right = lens_geometry(face_points194, 36, 40, 44, 38, 42)
        .ok_or(Refusal::ValueOutOfRange(GLARE_POINTS))?;

    let (width, height) = (frame.width as usize, frame.height as usize);
    let mut source_luma = vec![0.0f32; pixels];
    let mut source_r = vec![0.0f32; pixels];
    let mut source_g = vec![0.0f32; pixels];
    let mut source_b = vec![0.0f32; pixels];
    let mut lens_region = vec![0.0f32; pixels];
    let mut pupil_guard = vec![0.0f32; pixels];
    let mut any_lens = false;
    for y in 0..height {
        for x in 0..width {
            let p = y * width + x;
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let r = frame.pixels[p * 4] as f32;
            let g = frame.pixels[p * 4 + 1] as f32;
            let b = frame.pixels[p * 4 + 2] as f32;
            source_r[p] = r;
            source_g[p] = g;
            source_b[p] = b;
            source_luma[p] = luma(r, g, b);

            let mut member = ellipse(fx, fy, left.center, left.rx, left.ry);
            let mut pupil = ellipse(fx, fy, left.center, left.pupil_rx * 1.5, left.pupil_ry * 1.5);
            member = member.max(ellipse(fx, fy, right.center, right.rx, right.ry));
            pupil = pupil.max(ellipse(
                fx,
                fy,
                right.center,
                right.pupil_rx * 1.5,
                right.pupil_ry * 1.5,
            ));
            if member > 0.0 {
                any_lens = true;
            }
            lens_region[p] = member;
            pupil_guard[p] = pupil.clamp(0.0, 1.0);
        }
    }
    if !any_lens {
        return Err(Refusal::ValueOutOfRange(GLARE_LENS));
    }

    let coarse_radius = ((left.rx.max(right.rx)) * 0.8)
        .round()
        .clamp(4.0, 32.0) as usize;
    let baseline_luma =
        weighted_blur(&source_luma, &lens_region, width, height, coarse_radius);

    let eye_gate: Vec<f32> = match eye_mask {
        Some(mask) => mask.iter().map(|&value| value as f32 / 255.0).collect(),
        None => vec![1.0f32; pixels],
    };

    let mut glare_weight = vec![0.0f32; pixels];
    for p in 0..pixels {
        if lens_region[p] <= 0.0 {
            continue;
        }
        let excess = source_luma[p] - baseline_luma[p];
        let bright = smoothstep(12.0, 45.0, excess);
        let low_saturation =
            1.0 - smoothstep(0.12, 0.32, saturation(source_r[p], source_g[p], source_b[p]));
        let guard = if pupil_guard[p] > 0.6 {
            0.0
        } else {
            (1.0 - pupil_guard[p] / 0.6).clamp(0.0, 1.0)
        };
        glare_weight[p] =
            (lens_region[p] * bright * low_saturation * guard * eye_gate[p]).clamp(0.0, 1.0);
    }

    // Estimate the lens colour without the glare itself, then keep a little of
    // the source texture so the fill is not perfectly flat.
    let mut clean_weight = vec![0.0f32; pixels];
    for p in 0..pixels {
        if lens_region[p] > 0.0 {
            clean_weight[p] = lens_region[p] * (0.05 + 0.95 * (1.0 - glare_weight[p]));
        }
    }
    let (base_r, base_g, base_b) = blur_rgb(
        &source_r,
        &source_g,
        &source_b,
        &clean_weight,
        width,
        height,
        coarse_radius,
    );
    let fine_luma = weighted_blur(&source_luma, &lens_region, width, height, 1);

    let mut effect_rgb = vec![0.0f32; pixels * 3];
    for p in 0..pixels {
        let high_frequency = (source_luma[p] - fine_luma[p]) * 0.20;
        effect_rgb[p * 3] = glare_encode(base_r[p] + high_frequency);
        effect_rgb[p * 3 + 1] = glare_encode(base_g[p] + high_frequency);
        effect_rgb[p * 3 + 2] = glare_encode(base_b[p] + high_frequency);
    }

    Ok(Pf90130GlareRow {
        ticket: job_id,
        control: 1.0,
        data: super::SkinRepairData::Glare(super::GlareInputs {
            effect_rgb,
            mask: None,
            lens_mask: Some(glare_weight),
            encoding: GlareMaskEncoding::UnormLens,
        }),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::hair_body::StrayHairNative;
    use aurora_portrait::kernels::{beard, hair_body};

    const SIDE: u32 = 128;

    fn face_points() -> Vec<[f32; 2]> {
        let mut points = vec![[64.0f32, 64.0f32]; 194];
        // Brows.
        for i in 0..=8usize {
            let t = i as f32 / 8.0;
            points[i] = [40.0 - t * 20.0, 30.0];
            points[9 + i] = [88.0 + t * 20.0, 30.0];
        }
        // Left eye contour + centre.
        points[18] = [22.0, 50.0];
        points[22] = [42.0, 50.0];
        points[26] = [32.0, 50.0];
        points[20] = [32.0, 45.0];
        points[24] = [32.0, 55.0];
        // Right eye contour + centre.
        points[36] = [106.0, 50.0];
        points[40] = [86.0, 50.0];
        points[44] = [96.0, 50.0];
        points[38] = [96.0, 45.0];
        points[42] = [96.0, 55.0];
        // Nose.
        points[54] = [64.0, 64.0];
        points[57] = [64.0, 72.0];
        // Mouth corners.
        points[73] = [56.0, 88.0];
        points[83] = [72.0, 88.0];
        // Jaw contour and chin.
        points[109] = [32.0, 92.0];
        points[141] = [96.0, 92.0];
        points[125] = [64.0, 110.0];
        points
    }

    fn skin_frame(colour: [u8; 3]) -> Image {
        let mut pixels = Vec::with_capacity(SIDE as usize * SIDE as usize * 4);
        for _ in 0..SIDE as usize * SIDE as usize {
            pixels.extend_from_slice(&[colour[0], colour[1], colour[2], 255]);
        }
        Image::new(SIDE, SIDE, pixels).unwrap()
    }

    fn set_pixel(image: &mut Image, x: u32, y: u32, colour: [u8; 3]) {
        let i = (y as usize * SIDE as usize + x as usize) * 4;
        image.pixels[i] = colour[0];
        image.pixels[i + 1] = colour[1];
        image.pixels[i + 2] = colour[2];
    }

    fn pixel_luma(image: &Image, x: u32, y: u32) -> f32 {
        let i = (y as usize * SIDE as usize + x as usize) * 4;
        luma(
            image.pixels[i] as f32,
            image.pixels[i + 1] as f32,
            image.pixels[i + 2] as f32,
        )
    }

    // ---- PF21026 ----------------------------------------------------------

    #[test]
    fn pf21026_builds_masks_and_repairs_stubble() {
        let skin = [190u8, 155, 130];
        let stubble = [55u8, 45, 38];
        let mut frame = skin_frame(skin);
        for y in 79..83 {
            set_pixel(&mut frame, 59, y, stubble);
        }
        for y in 96..106 {
            set_pixel(&mut frame, 63, y, [178, 143, 118]);
            set_pixel(&mut frame, 64, y, stubble);
        }

        let mut points = face_points();
        points[109] = [38.0, 50.0];
        points[141] = [90.0, 50.0];
        let guarded_pixels = [(40, 54), (64, 75), (56, 88), (64, 88), (72, 88)];
        for (x, y) in guarded_pixels {
            set_pixel(&mut frame, x, y, stubble);
        }
        let index = |x: u32, y: u32| (y * SIDE + x) as usize;
        let chin_index = index(64, 100);
        frame.pixels[chin_index * 4 + 3] = 137;
        let owned = build_pf21026_beard_inputs(&frame, &points, None).unwrap();
        assert_eq!(owned.mask_a.len(), frame.pixel_count());
        assert_eq!(owned.mask_b.len(), frame.pixel_count());
        assert_eq!((owned.repair_rgb.width, owned.repair_rgb.height), (SIDE, SIDE));

        for (x, y) in guarded_pixels {
            assert_eq!(owned.mask_a[index(x, y)], 0, "anatomy at ({x}, {y})");
            assert_eq!(owned.mask_b[index(x, y)], 0);
        }
        assert!(owned.mask_b[index(59, 81)] > 0, "thin moustache survives");
        let soft_edge = owned.mask_b[index(63, 100)];
        assert!(soft_edge > 0 && soft_edge < owned.mask_b[chin_index]);
        assert_eq!(owned.mask_b[index(62, 100)], 0, "no spread onto skin");

        let borrowed = owned.as_borrowed();
        let repaired = beard::apply(
            &frame,
            borrowed.repair_rgb,
            borrowed.mask_a,
            borrowed.mask_b,
            1.0,
        )
        .unwrap();
        assert!(
            pixel_luma(&repaired, 64, 100) > pixel_luma(&frame, 64, 100) + 40.0,
            "stubble must be replaced by skin"
        );
        assert_eq!(repaired.pixels[chin_index * 4 + 3], 137);
        assert_eq!(
            repaired.pixel(2, 2),
            frame.pixel(2, 2),
            "outside the beard zone is untouched"
        );
    }

    #[test]
    fn pf21026_preserves_smooth_shadow_and_one_sided_edge() {
        let mut frame = skin_frame([175; 3]);
        for y in 0..SIDE {
            let shade = 112 + (y.abs_diff(100).min(24) / 3) as u8;
            for x in 0..48 {
                set_pixel(&mut frame, x, y, [shade; 3]);
            }
        }
        let owned = build_pf21026_beard_inputs(&frame, &face_points(), None).unwrap();
        assert!(owned.mask_a[(100 * SIDE + 47) as usize] > 0);
        assert!(owned.mask_b.iter().all(|&alpha| alpha == 0));
        assert_eq!(owned.repair_rgb, frame);
    }

    #[test]
    fn pf21026_rejects_bad_points_and_masks() {
        let frame = skin_frame([190, 155, 130]);
        let points = face_points();
        assert_eq!(
            build_pf21026_beard_inputs(&frame, &[], None),
            Err(Refusal::ValueOutOfRange(POINTS))
        );
        let mut bad = points.clone();
        bad[73] = [f32::NAN, 88.0];
        assert_eq!(
            build_pf21026_beard_inputs(&frame, &bad, None),
            Err(Refusal::ValueOutOfRange(POINTS))
        );
        assert_eq!(
            build_pf21026_beard_inputs(&frame, &points, Some(&[0u8; 5])),
            Err(Refusal::ShapeMismatch(BEARD_SKIN))
        );
        let empty = vec![0u8; frame.pixel_count()];
        assert_eq!(
            build_pf21026_beard_inputs(&frame, &points, Some(&empty)),
            Err(Refusal::MissingInput(BEARD_SKIN))
        );
    }

    // ---- PF90088 ----------------------------------------------------------

    #[test]
    fn pf90088_detects_thin_strands_in_the_outer_band() {
        let skin = [200u8, 170, 150];
        let mut frame = skin_frame(skin);
        let mut hair_mask = vec![0u8; SIDE as usize * SIDE as usize];
        // Hair blob on the left edge.
        for y in 10..118u32 {
            for x in 0..32u32 {
                hair_mask[y as usize * SIDE as usize + x as usize] = 255;
                set_pixel(&mut frame, x, y, [42, 35, 28]);
            }
        }
        // Single-pixel dark filaments inside the 4..18 px outer band.
        for x in [38u32, 44, 50] {
            for y in 20..108u32 {
                set_pixel(&mut frame, x, y, [25, 22, 20]);
            }
        }

        let inputs = build_pf90088_stray_hair_inputs(&frame, &hair_mask, None).unwrap();
        let raw = inputs.raw_mask.as_ref().unwrap();
        let filament = 60 * SIDE as usize + 38;
        assert!(raw[filament] > 30, "thin dark filament must be in raw_mask");
        assert_eq!(raw[2 * SIDE as usize + 100], 0, "far background stays clear");

        let repair = inputs.hairfly_v1.as_ref().unwrap();
        assert!(
            pixel_luma(repair, 38, 60) > pixel_luma(&frame, 38, 60) + 60.0,
            "repair plane must hold surrounding skin, not the filament"
        );

        assert!(inputs.person_mask.is_some());
        assert!(inputs.background_mask.is_some());
        assert_eq!(inputs.native_degree, Some(1.0));

        let result = hair_body::stray_hair_with_native(
            &frame,
            100.0,
            StrayHairNative {
                degree: inputs.native_degree,
                person_area: inputs.native_person_area,
                background_area: inputs.native_background_area,
            },
            inputs.hairfly_v1.as_ref(),
            inputs.raw_mask.as_deref(),
            inputs.person_mask.as_deref(),
            inputs.background_mask.as_deref(),
        )
        .unwrap();
        assert!(
            pixel_luma(&result, 38, 60) > pixel_luma(&frame, 38, 60) + 40.0,
            "consumer must blend the filament toward the repair plane"
        );
    }

    #[test]
    fn pf90088_rejects_missing_or_mis_shaped_hair_masks() {
        let frame = skin_frame([200, 170, 150]);
        assert_eq!(
            build_pf90088_stray_hair_inputs(&frame, &[0u8; 3], None),
            Err(Refusal::ShapeMismatch(STRAY_HAIR_MASK))
        );
        assert_eq!(
            build_pf90088_stray_hair_inputs(&frame, &vec![0u8; frame.pixel_count()], None),
            Err(Refusal::MissingInput(STRAY_HAIR_MASK))
        );
        let mut hair = vec![0u8; frame.pixel_count()];
        hair[0] = 255;
        assert_eq!(
            build_pf90088_stray_hair_inputs(&frame, &hair, Some(&[255u8; 4])),
            Err(Refusal::ShapeMismatch(STRAY_SKIN_MASK))
        );
    }

    // ---- PF90131 ----------------------------------------------------------

    #[test]
    fn pf90131_recolours_white_strands_to_the_dominant_hair_colour() {
        let mut frame = skin_frame([200, 170, 150]);
        let mut hair_mask = vec![0u8; SIDE as usize * SIDE as usize];
        // Dark hair block on the right, with white strands.
        for y in 8..120u32 {
            for x in 70..126u32 {
                hair_mask[y as usize * SIDE as usize + x as usize] = 255;
                set_pixel(&mut frame, x, y, [44, 38, 32]);
            }
        }
        for x in [80u32, 90, 100, 110] {
            for y in 12..116u32 {
                set_pixel(&mut frame, x, y, [226, 224, 216]);
            }
        }

        let output = build_pf90131_white_hair_output(&frame, &hair_mask).unwrap();
        let super::super::RepairOutput::WhiteHair { repair, raw_mask } = &output else {
            panic!("expected WhiteHair output");
        };
        assert_eq!(raw_mask.len(), frame.pixel_count());
        assert_eq!((repair.width, repair.height), (SIDE, SIDE));

        let strand = 60 * SIDE as usize + 80;
        let dark = 60 * SIDE as usize + 74;
        assert!(raw_mask[strand] < 120, "white strand must be repaired");
        assert_eq!(raw_mask[dark], 255, "dark hair is identity");
        assert!(
            pixel_luma(repair, 80, 60) < pixel_luma(&frame, 80, 60) - 80.0,
            "white strand must take the dominant dark colour"
        );

        let mut inputs = super::super::RepairProducerInputs::default();
        inputs.insert(7, &frame, output.clone()).unwrap();
        assert!(inputs.receipt(90131).is_some());

        let result =
            hair_body::white_hair(&frame, 100.0, raw_mask, repair).unwrap();
        assert!(
            pixel_luma(&result, 80, 60) < pixel_luma(&frame, 80, 60) - 60.0,
            "consumer must blend toward the recoloured strand"
        );
        assert_eq!(
            result.pixel(2, 2),
            frame.pixel(2, 2),
            "outside the hair mask is untouched"
        );
    }

    #[test]
    fn pf90131_rejects_missing_or_mis_shaped_hair_masks() {
        let frame = skin_frame([200, 170, 150]);
        assert_eq!(
            build_pf90131_white_hair_output(&frame, &[0u8; 2]),
            Err(Refusal::ShapeMismatch(WHITE_HAIR_MASK))
        );
        assert_eq!(
            build_pf90131_white_hair_output(&frame, &vec![0u8; frame.pixel_count()]),
            Err(Refusal::MissingInput(WHITE_HAIR_MASK))
        );
    }

    // ---- PF90130 ----------------------------------------------------------

    fn glare_frame() -> Image {
        let mut frame = skin_frame([180, 150, 130]);
        // Highlight inside the left lens, away from the pupil guard.
        for y in 41..50u32 {
            for x in 24..33u32 {
                set_pixel(&mut frame, x, y, [246, 244, 240]);
            }
        }
        // Pupil catchlight: must stay out of the lens weight.
        set_pixel(&mut frame, 32, 50, [255, 255, 255]);
        frame
    }

    #[test]
    fn pf90130_builds_lens_weight_and_binds_to_the_skin_repair_consumer() {
        let frame = glare_frame();
        let row = build_pf90130_glare_row(11, &frame, &face_points(), None).unwrap();
        assert_eq!(row.ticket, 11);
        assert_eq!(row.control, 1.0);

        let super::super::SkinRepairData::Glare(glare) = &row.data else {
            panic!("expected a Glare row");
        };
        assert_eq!(glare.effect_rgb.len(), frame.pixel_count() * 3);
        assert!(glare.mask.is_none());
        let lens = glare.lens_mask.as_ref().unwrap();
        assert_eq!(lens.len(), frame.pixel_count());
        assert!(
            lens[45 * SIDE as usize + 28] > 0.3,
            "specular inside the lens must be detected"
        );
        assert!(
            lens[50 * SIDE as usize + 32] < 0.05,
            "pupil catchlight is guarded"
        );
        assert_eq!(lens[5 * SIDE as usize + 5], 0.0, "outside the lens is clear");
        assert!(glare.effect_rgb.iter().all(|value| value.is_finite()
            && (-1.0..=1.0).contains(value)));

        let mut inputs = super::super::SkinRepairProducerInputs::default();
        inputs
            .insert(row.ticket, &frame, row.control, row.data.clone())
            .unwrap();
        let output = super::super::run_skin_repair(
            &frame,
            Some(row.ticket),
            90130,
            row.control,
            Some(&inputs),
        )
        .unwrap()
        .expect("glare stage must run");
        let super::super::SkinRepairOutput::Glare { rgba8 } = output else {
            panic!("expected Glare output");
        };
        assert!(
            pixel_luma(&rgba8, 28, 45) < pixel_luma(&frame, 28, 45) - 25.0,
            "lens specular must be filled with surrounding lens colour"
        );
        assert_eq!(
            rgba8.pixel(5, 5),
            frame.pixel(5, 5),
            "outside the lens is untouched"
        );
        assert_eq!(
            rgba8.pixel(32, 50),
            frame.pixel(32, 50),
            "pupil catchlight must be preserved"
        );
    }

    #[test]
    fn pf90130_rejects_job_points_and_mask_errors() {
        let frame = glare_frame();
        let points = face_points();
        assert_eq!(
            build_pf90130_glare_row(0, &frame, &points, None),
            Err(Refusal::ValueOutOfRange("PF90130_JOB_TICKET"))
        );
        assert_eq!(
            build_pf90130_glare_row(11, &frame, &[], None),
            Err(Refusal::ValueOutOfRange(GLARE_POINTS))
        );
        let mut bad = points.clone();
        bad[18] = [f32::INFINITY, 50.0];
        assert_eq!(
            build_pf90130_glare_row(11, &frame, &bad, None),
            Err(Refusal::ValueOutOfRange(GLARE_POINTS))
        );
        assert_eq!(
            build_pf90130_glare_row(11, &frame, &points, Some(&[255u8; 3])),
            Err(Refusal::ShapeMismatch(GLARE_EYE_MASK))
        );
        assert_eq!(
            build_pf90130_glare_row(
                11,
                &frame,
                &points,
                Some(&vec![0u8; frame.pixel_count()])
            ),
            Err(Refusal::MissingInput(GLARE_EYE_MASK))
        );
    }
}
