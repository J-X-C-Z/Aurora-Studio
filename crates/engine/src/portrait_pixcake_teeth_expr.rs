//! Geometric producers for PF30025 / PF30026 / PF21076 / PF100007.
//!
//! These four rows are consumed by already-recovered vendor kernels:
//!
//! * PF30025 `face_detail::tooth_repair_same_job`: packed BGRA repair target,
//!   one f32 gate per pixel and an input ROI.
//! * PF30026 `teeth_blend::apply_alignment_final_blend`: full-frame effect RGB,
//!   full-frame u8 teeth mask and two f32 edge planes feeding the A+B gate.
//! * PF21076 `close_smile::apply`: a full-frame model RGB that replaces the
//!   source RGB while the paste gate is on.
//! * PF100007 `smile_warp::pf100007_apply`: full-frame lip/mouth masks and a
//!   per-pixel displacement field, gated by the `Grinning` material.
//!
//! This module does **not** read a vendor model tensor for any of the four
//! rows. It derives explicit geometric planes from the 194-point face template
//! and the caller's masks so the existing consumers can run on the same frame.
//! The returned receipts therefore describe this producer under the recovered
//! consumer contract; they are not captured vendor readbacks and must not be
//! reported as vendor parity.
//!
//! The coordinator still has to:
//! * mount this file as a child module of `portrait_stage`;
//! * write PF30025 with `ProducerInputs::write_pf30025`;
//! * write PF30026 with `ProducerInputs::write_teeth_alignment` (the stage also
//!   requires its `roi` to equal the dispatch `region`, so this producer uses
//!   `NormRect::FULL`);
//! * bind PF21076 with `RepairProducerInputs::insert` / `insert_same_frame`;
//! * write PF100007 into `ProducerInputs::smile`. `SmileProducerInputs` has no
//!   direct `model_dx`/`model_dy` fields: the typed route is
//!   `channel_split: Some(SmileChannelSplit { .. })`, whose `spec.job_id` must
//!   equal `ticket` for `PortraitJobInputs::from_renderer_job` to promote the
//!   planes into `SmileInputs::model_dx`/`model_dy`.

use aurora_portrait::kernels::face_detail::Rect as ToothRect;
use aurora_portrait::kernels::smile_warp::Rect as SmileRect;
use aurora_portrait::{Image, Refusal};

use crate::NormRect;

/// Mouth landmarks verified against `constants.json["template"]`.
const MOUTH_LEFT: usize = 73;
const UPPER_OUTER_CENTER: usize = 78;
const MOUTH_RIGHT: usize = 83;
const LOWER_OUTER_CENTER: usize = 89;
const INNER_LEFT: usize = 95;
const INNER_UPPER_CENTER: usize = 98;
const INNER_RIGHT: usize = 101;
const INNER_LOWER_CENTER: usize = 104;

/// Fixed PF100007/PF30025 model canvas used by `SmileHostSplitSpec`.
const MODEL_WIDTH: u32 = 1024;
const MODEL_HEIGHT: u32 = 768;

/// Luma weights shared with the engine's other portrait producers.
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

fn luma(rgb: [f32; 3]) -> f32 {
    LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2]
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn clamp_byte(value: f32) -> u8 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value.round() as u8
    }
}

/// Validate the rendered frame and return its pixel count.
fn validate_frame(frame: &Image) -> Result<usize, Refusal> {
    frame.validate()?;
    Ok(frame.pixel_count())
}

fn validate_mask(mask: &[u8], pixels: usize, name: &'static str) -> Result<(), Refusal> {
    if mask.len() == pixels {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(name))
    }
}

/// Axis-aligned box of non-zero mask pixels, half-open on the far edge.
fn mask_bounds(
    mask: &[u8],
    width: usize,
    height: usize,
) -> Option<(usize, usize, usize, usize)> {
    let mut x0 = width;
    let mut y0 = height;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            if mask[row + x] == 0 {
                continue;
            }
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

fn padded_bounds(
    bounds: (usize, usize, usize, usize),
    padding: usize,
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize) {
    (
        bounds.0.saturating_sub(padding),
        bounds.1.saturating_sub(padding),
        (bounds.2 + padding).min(width),
        (bounds.3 + padding).min(height),
    )
}

fn tooth_rect(
    bounds: (usize, usize, usize, usize),
    frame: &Image,
) -> Result<ToothRect, Refusal> {
    let (x0, y0, x1, y1) = bounds;
    if x1 <= x0 || y1 <= y0 {
        return Err(Refusal::ValueOutOfRange("PF30025 roi"));
    }
    let rect = ToothRect {
        x: i32::try_from(x0).map_err(|_| Refusal::ShapeMismatch("PF30025 roi"))?,
        y: i32::try_from(y0).map_err(|_| Refusal::ShapeMismatch("PF30025 roi"))?,
        width: i32::try_from(x1 - x0).map_err(|_| Refusal::ShapeMismatch("PF30025 roi"))?,
        height: i32::try_from(y1 - y0).map_err(|_| Refusal::ShapeMismatch("PF30025 roi"))?,
    };
    if !rect.inside(frame.width, frame.height) {
        return Err(Refusal::ValueOutOfRange("PF30025 roi"));
    }
    Ok(rect)
}

fn smile_roi(
    bounds: (usize, usize, usize, usize),
    frame: &Image,
) -> Result<(u32, u32, u32, u32), Refusal> {
    let (x0, y0, x1, y1) = bounds;
    if x1 <= x0 || y1 <= y0 {
        return Err(Refusal::ValueOutOfRange("PF100007 roi"));
    }
    let x = u32::try_from(x0).map_err(|_| Refusal::ShapeMismatch("PF100007 roi"))?;
    let y = u32::try_from(y0).map_err(|_| Refusal::ShapeMismatch("PF100007 roi"))?;
    let width = u32::try_from(x1 - x0).map_err(|_| Refusal::ShapeMismatch("PF100007 roi"))?;
    let height = u32::try_from(y1 - y0).map_err(|_| Refusal::ShapeMismatch("PF100007 roi"))?;
    if x > frame.width
        || y > frame.height
        || width > frame.width - x
        || height > frame.height - y
    {
        return Err(Refusal::ValueOutOfRange("PF100007 roi"));
    }
    Ok((x, y, width, height))
}

/// Mouth frame derived from the 194-point template.
#[derive(Clone, Copy, Debug)]
struct MouthGeometry {
    left: [f32; 2],
    right: [f32; 2],
    center: [f32; 2],
    /// Unit vector from the left corner to the right corner.
    axis_u: [f32; 2],
    /// Unit vector perpendicular to `axis_u`, pointing from the upper lip to
    /// the lower lip (image y grows downward).
    axis_v: [f32; 2],
    /// Outer corner-to-corner distance.
    width: f32,
    inner_width: f32,
    inner_height: f32,
}

impl MouthGeometry {
    fn from_points(points: &[[f32; 2]]) -> Result<Self, Refusal> {
        if points.len() != 194 {
            return Err(Refusal::ShapeMismatch("points194"));
        }
        if points.iter().flatten().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("points194"));
        }
        let left = points[MOUTH_LEFT];
        let right = points[MOUTH_RIGHT];
        let upper_inner = points[INNER_UPPER_CENTER];
        let lower_inner = points[INNER_LOWER_CENTER];
        let inner_left = points[INNER_LEFT];
        let inner_right = points[INNER_RIGHT];

        let dx = right[0] - left[0];
        let dy = right[1] - left[1];
        let width = dx.hypot(dy);
        if !width.is_finite() || width < 4.0 {
            return Err(Refusal::ValueOutOfRange("points194 mouth width"));
        }
        let axis_u = [dx / width, dy / width];
        let mut axis_v = [-axis_u[1], axis_u[0]];
        // Orient `v` from the upper lip towards the lower lip. A closed mouth
        // can collapse the inner lip landmarks, so fall back to the outer lip
        // centres before deciding which way `v` points.
        let mut vertical = [
            lower_inner[0] - upper_inner[0],
            lower_inner[1] - upper_inner[1],
        ];
        if vertical[0].hypot(vertical[1]) < 0.05 {
            let upper_outer = points[UPPER_OUTER_CENTER];
            let lower_outer = points[LOWER_OUTER_CENTER];
            vertical = [
                lower_outer[0] - upper_outer[0],
                lower_outer[1] - upper_outer[1],
            ];
        }
        if vertical[0] * axis_v[0] + vertical[1] * axis_v[1] < 0.0 {
            axis_v = [-axis_v[0], -axis_v[1]];
        }

        let inner_width = distance(inner_left, inner_right);
        let inner_height = distance(upper_inner, lower_inner);
        if !inner_width.is_finite() || inner_width < 2.0 {
            return Err(Refusal::ValueOutOfRange("points194 mouth geometry"));
        }
        if !inner_height.is_finite() {
            return Err(Refusal::ValueOutOfRange("points194 mouth geometry"));
        }
        let center = [
            (inner_left[0] + inner_right[0] + upper_inner[0] + lower_inner[0]) * 0.25,
            (inner_left[1] + inner_right[1] + upper_inner[1] + lower_inner[1]) * 0.25,
        ];
        Ok(Self {
            left,
            right,
            center,
            axis_u,
            axis_v,
            width,
            inner_width,
            inner_height,
        })
    }

    /// Mouth-local coordinates: `s` along the corner axis, `t` perpendicular
    /// (negative above the mouth centre, positive below it).
    fn local(&self, x: f32, y: f32) -> (f32, f32) {
        let rx = x - self.center[0];
        let ry = y - self.center[1];
        (
            rx * self.axis_u[0] + ry * self.axis_u[1],
            rx * self.axis_v[0] + ry * self.axis_v[1],
        )
    }

    fn corner_falloff(&self, x: f32, y: f32, sigma: f32) -> (f32, f32) {
        let sigma = sigma.max(1.0);
        let left = (-((x - self.left[0]).powi(2) + (y - self.left[1]).powi(2))
            / (sigma * sigma))
            .exp();
        let right = (-((x - self.right[0]).powi(2) + (y - self.right[1]).powi(2))
            / (sigma * sigma))
            .exp();
        (left, right)
    }
}

fn sample_bilinear(frame: &Image, x: f32, y: f32) -> [u8; 4] {
    let max_x = frame.width.saturating_sub(1) as f32;
    let max_y = frame.height.saturating_sub(1) as f32;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(frame.width - 1);
    let y1 = (y0 + 1).min(frame.height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let stride = frame.width as usize;
    let index = |xx: u32, yy: u32| (yy as usize * stride + xx as usize) * 4;
    let (p00, p10) = (index(x0, y0), index(x1, y0));
    let (p01, p11) = (index(x0, y1), index(x1, y1));
    let mut out = [0u8; 4];
    for channel in 0..4 {
        let top = frame.pixels[p00 + channel] as f32 * (1.0 - fx)
            + frame.pixels[p10 + channel] as f32 * fx;
        let bottom = frame.pixels[p01 + channel] as f32 * (1.0 - fx)
            + frame.pixels[p11 + channel] as f32 * fx;
        let value = top * (1.0 - fy) + bottom * fy;
        out[channel] = value.clamp(0.0, 255.0).round() as u8;
    }
    out
}

/// Resample `frame` at `p - (dx, dy)`; zero displacement is an exact copy.
fn warp_with_field(frame: &Image, dx: &[f32], dy: &[f32]) -> Result<Image, Refusal> {
    let pixels = frame.pixel_count();
    if dx.len() != pixels || dy.len() != pixels {
        return Err(Refusal::ShapeMismatch("displacement"));
    }
    let width = frame.width as usize;
    let mut out = frame.pixels.clone();
    for y in 0..frame.height as usize {
        for x in 0..width {
            let index = y * width + x;
            let offset_x = dx[index];
            let offset_y = dy[index];
            if offset_x == 0.0 && offset_y == 0.0 {
                continue;
            }
            let sample = sample_bilinear(frame, x as f32 - offset_x, y as f32 - offset_y);
            out[index * 4..index * 4 + 4].copy_from_slice(&sample);
        }
    }
    Image::new(frame.width, frame.height, out).ok_or(Refusal::ShapeMismatch("image"))
}

/// Map a `0..=255` target byte to the `-1..=1` value expected by
/// `Pf30025ProducerInputs::from_live_model`.
fn signed_plane_value(target: f32) -> f32 {
    (target.clamp(0.0, 255.0) / 127.5 - 1.0).clamp(-1.0, 1.0)
}

/// Homogenize chroma towards the mean teeth colour and add a gentle lift while
/// keeping part of the source shading so the crown transition survives.
fn whitened_teeth_color(source: [f32; 3], mean_luma: f32, chroma: [f32; 3]) -> [f32; 3] {
    let source_luma = luma(source);
    let mut out = [0.0f32; 3];
    for channel in 0..3 {
        let base = mean_luma + chroma[channel] * 0.30;
        let lifted = base + (255.0 - base).max(0.0) * 0.08;
        let target = lifted + (source_luma - mean_luma) * 0.40;
        out[channel] = (source[channel] * 0.35 + target * 0.65).clamp(0.0, 255.0);
    }
    out
}

/// PF30025: teeth repair target and gate.
///
/// `teeth_mask` is a full-frame u8 support plane. The repair target is the
/// source colour blended 35/65 with a chroma-homogenized, gently lifted teeth
/// colour; the gate is the normalized support so the vendor blend feathers out
/// at the tooth boundary. The `780929437` plane is returned in B, G, R order,
/// matching `tooth_repair_same_job`'s BGRA swizzle.
pub(crate) fn build_pf30025_teeth_repair_inputs(
    job_id: u64,
    frame: &Image,
    face_points194: &[[f32; 2]],
    teeth_mask: &[u8],
) -> Result<super::Pf30025ProducerInputs, Refusal> {
    let pixels = validate_frame(frame)?;
    let mouth = MouthGeometry::from_points(face_points194)?;
    validate_mask(teeth_mask, pixels, "PF30025 teeth mask")?;
    if job_id == 0 {
        return Err(Refusal::ValueOutOfRange("PF30025 job id"));
    }
    let width = frame.width as usize;
    let height = frame.height as usize;
    let bounds = mask_bounds(teeth_mask, width, height)
        .ok_or(Refusal::MissingInput("PF30025 teeth mask"))?;
    let padding = ((mouth.width * 0.12).round() as usize).max(2);
    let roi = tooth_rect(padded_bounds(bounds, padding, width, height), frame)?;

    let mut sum = [0.0f64; 3];
    let mut count = 0usize;
    for (pixel, &mask) in teeth_mask.iter().enumerate() {
        if mask == 0 {
            continue;
        }
        let base = pixel * 4;
        for channel in 0..3 {
            sum[channel] += frame.pixels[base + channel] as f64;
        }
        count += 1;
    }
    if count == 0 {
        return Err(Refusal::MissingInput("PF30025 teeth mask"));
    }
    let mean_rgb = [
        (sum[0] / count as f64) as f32,
        (sum[1] / count as f64) as f32,
        (sum[2] / count as f64) as f32,
    ];
    let mean_luma = luma(mean_rgb);
    let chroma = [
        mean_rgb[0] - mean_luma,
        mean_rgb[1] - mean_luma,
        mean_rgb[2] - mean_luma,
    ];

    let mut plane_b = vec![0.0f32; pixels];
    let mut plane_g = vec![0.0f32; pixels];
    let mut plane_r = vec![0.0f32; pixels];
    let mut gate = vec![0.0f32; pixels];
    for pixel in 0..pixels {
        let mask = teeth_mask[pixel];
        gate[pixel] = mask as f32 / 255.0;
        let base = pixel * 4;
        let source = [
            frame.pixels[base] as f32,
            frame.pixels[base + 1] as f32,
            frame.pixels[base + 2] as f32,
        ];
        let target = if mask == 0 {
            source
        } else {
            whitened_teeth_color(source, mean_luma, chroma)
        };
        plane_b[pixel] = signed_plane_value(target[2]);
        plane_g[pixel] = signed_plane_value(target[1]);
        plane_r[pixel] = signed_plane_value(target[0]);
    }

    let row_stride = frame
        .width
        .checked_mul(4)
        .ok_or(Refusal::ShapeMismatch("PF30025 source surface"))?;
    let source_surface = super::Pf30025SourceSurface {
        job_id,
        width: frame.width,
        height: frame.height,
        row_stride,
        rgba: &frame.pixels,
    };
    let output = super::Pf30025LiveModelOutputs {
        job_id,
        model_id: super::PF30025_MODEL_IDS[0],
        source_surface,
        repair_780929437_bgr: [plane_b.as_slice(), plane_g.as_slice(), plane_r.as_slice()],
        gate_3204548346: &gate,
    };
    let teeth_support: Vec<f32> = teeth_mask.iter().map(|&value| value as f32 / 255.0).collect();
    super::Pf30025ProducerInputs::from_live_model(output, teeth_support, roi)
}

/// PF30026: micro arc alignment of the teeth region plus the gate planes.
///
/// The effect image is the source resampled by a small vertical arc that is
/// strongest at the mouth corners; it is byte-identical outside `teeth_mask`.
/// The two edge planes are explicit geometric coverage proxies, not captured
/// model tensors, so a populated teeth mask opens the recovered A+B gate.
pub(crate) fn build_pf30026_teeth_alignment_inputs(
    job_id: u64,
    frame: &Image,
    face_points194: &[[f32; 2]],
    teeth_mask: &[u8],
) -> Result<super::TeethAlignmentProducerInputs, Refusal> {
    let pixels = validate_frame(frame)?;
    let mouth = MouthGeometry::from_points(face_points194)?;
    validate_mask(teeth_mask, pixels, "PF30026 teeth mask")?;
    if job_id == 0 {
        return Err(Refusal::ValueOutOfRange("PF30026 job id"));
    }
    let width = frame.width as usize;
    let height = frame.height as usize;
    let (x0, y0, x1, y1) =
        mask_bounds(teeth_mask, width, height).ok_or(Refusal::MissingInput("PF30026 teeth mask"))?;

    let half = mouth.width.max(2.0) * 0.5;
    let amplitude = (mouth.width * 0.02).clamp(0.5, 4.0);
    let mut dx = vec![0.0f32; pixels];
    let mut dy = vec![0.0f32; pixels];
    for y in y0..y1 {
        let row = y * width;
        for x in x0..x1 {
            let pixel = row + x;
            let support = teeth_mask[pixel] as f32 / 255.0;
            if support <= 0.0 {
                continue;
            }
            let (s, _t) = mouth.local(x as f32 + 0.5, y as f32 + 0.5);
            let normalized = (s / half).clamp(-1.5, 1.5);
            dx[pixel] = 0.25 * amplitude * normalized * support;
            dy[pixel] = amplitude * (normalized * normalized - 0.35) * support;
        }
    }
    let effect_rgb = warp_with_field(frame, &dx, &dy)?;

    let coverage: Vec<f32> = teeth_mask.iter().map(|&value| value as f32 / 255.0).collect();
    let edge_plane_a = coverage.clone();
    let edge_plane_b = coverage
        .iter()
        .map(|&value| (value * 0.9).clamp(0.0, 1.0))
        .collect::<Vec<f32>>();
    let gate_binding = super::Pf30026GateBinding::from_same_job(
        job_id,
        super::Pf30026EdgeOutput::Output0,
        super::Pf30026EdgeOutput::Output1,
    )?;

    Ok(super::TeethAlignmentProducerInputs {
        effect_rgb: Some(effect_rgb),
        teeth_mask: Some(teeth_mask.to_vec()),
        edge_plane_a: Some(edge_plane_a),
        edge_plane_b: Some(edge_plane_b),
        gate_binding: Some(gate_binding),
        // `apply_pf30026_stage` rejects any ROI that differs from the dispatch
        // region; the coordinator must dispatch this row with NormRect::FULL.
        roi: Some(NormRect::FULL),
    })
}

/// PF21076: closed-mouth smile warp.
///
/// The model is a full-frame copy of the source with a smooth, mouth-centred
/// displacement that lifts both corners and slightly compresses the lips. It is
/// an exact copy outside the mouth ellipse because `close_smile` replaces every
/// RGB pixel while the paste gate is on.
pub(crate) fn build_pf21076_closed_mouth_smile_output(
    frame: &Image,
    face_points194: &[[f32; 2]],
) -> Result<super::RepairOutput, Refusal> {
    let pixels = validate_frame(frame)?;
    let mouth = MouthGeometry::from_points(face_points194)?;
    let width = frame.width as usize;
    let height = frame.height as usize;
    let rx = (mouth.width * 0.75).max(6.0);
    let ry = (mouth.inner_height * 2.5)
        .max(mouth.width * 0.30)
        .max(5.0);
    let lift = (mouth.width * 0.09).clamp(1.0, 8.0);
    let compress = (mouth.width * 0.035).clamp(0.6, 4.0);
    let sigma = (mouth.width * 0.22).max(1.5);

    let mut dx = vec![0.0f32; pixels];
    let mut dy = vec![0.0f32; pixels];
    for y in 0..height {
        for x in 0..width {
            let pixel = y * width + x;
            let (s, t) = mouth.local(x as f32 + 0.5, y as f32 + 0.5);
            let radius2 = (s / rx).powi(2) + (t / ry).powi(2);
            if !(radius2 < 1.0) {
                continue;
            }
            let region = (1.0 - radius2).clamp(0.0, 1.0);
            let (corner_left, corner_right) =
                mouth.corner_falloff(x as f32 + 0.5, y as f32 + 0.5, sigma);
            let corners = (corner_left + corner_right).min(1.0);
            let lip_band =
                (-(t / (mouth.inner_height.max(2.0) * 0.9)).powi(2)).exp();
            // Upper lip (t < 0) moves down, lower lip (t > 0) moves up.
            let compress_y = compress * lip_band * if t < 0.0 { 1.0 } else { -1.0 };
            let inward = -0.30 * compress * corners * s.signum();
            dx[pixel] = (inward * region).clamp(-8.0, 8.0);
            dy[pixel] = ((-lift * corners + compress_y) * region).clamp(-8.0, 8.0);
        }
    }
    let model = warp_with_field(frame, &dx, &dy)?;
    Ok(super::RepairOutput::ClosedMouthSmile {
        model,
        paste_enabled: true,
    })
}

/// PF100007: open-mouth smile displacement and support masks.
///
/// `lip_mask` and `mouth_mask` are synthesized from the two supplied lip masks
/// plus the inner-mouth geometry, because the vendor's separate lip/mouth model
/// outputs are not recovered. The displacement opens the lips and lifts/widens
/// the corners. The planes are promoted into `SmileInputs::model_dx`/`model_dy`
/// only through the typed `channel_split` route: `SmileProducerInputs` has no
/// direct displacement fields.
pub(crate) fn build_pf100007_open_mouth_smile_inputs(
    job_id: u64,
    frame: &Image,
    face_points194: &[[f32; 2]],
    upper_lip_mask: &[u8],
    lower_lip_mask: &[u8],
) -> Result<super::SmileProducerInputs, Refusal> {
    let pixels = validate_frame(frame)?;
    let mouth = MouthGeometry::from_points(face_points194)?;
    validate_mask(upper_lip_mask, pixels, "PF100007 upper lip mask")?;
    validate_mask(lower_lip_mask, pixels, "PF100007 lower lip mask")?;
    if job_id == 0 {
        return Err(Refusal::ValueOutOfRange("PF100007 job id"));
    }
    if !upper_lip_mask
        .iter()
        .chain(lower_lip_mask)
        .any(|value| *value > 0)
    {
        return Err(Refusal::MissingInput("PF100007 lip mask"));
    }

    let width = frame.width as usize;
    let height = frame.height as usize;
    let lip_rx = (mouth.inner_width * 0.85).max(4.0);
    let lip_ry = (mouth.inner_height * 1.6).max(3.0);
    let mouth_rx = (mouth.width * 0.85).max(lip_rx).max(6.0);
    let mouth_ry = (mouth.inner_height * 2.6)
        .max(mouth.width * 0.32)
        .max(5.0);

    let mut lip_mask = vec![0u8; pixels];
    let mut mouth_mask = vec![0u8; pixels];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let pixel = row + x;
            let supplied = upper_lip_mask[pixel].max(lower_lip_mask[pixel]);
            let (s, t) = mouth.local(x as f32 + 0.5, y as f32 + 0.5);
            let lip_radius2 = (s / lip_rx).powi(2) + (t / lip_ry).powi(2);
            let lip_cavity = if lip_radius2 < 1.0 {
                clamp_byte((1.0 - lip_radius2).sqrt() * 255.0)
            } else {
                0
            };
            let mouth_radius2 = (s / mouth_rx).powi(2) + (t / mouth_ry).powi(2);
            let mouth_region = if mouth_radius2 < 1.0 {
                clamp_byte((1.0 - mouth_radius2).sqrt() * 255.0)
            } else {
                0
            };
            lip_mask[pixel] = supplied.max(lip_cavity);
            mouth_mask[pixel] = supplied.max(mouth_region);
        }
    }

    let open = (mouth.width * 0.06).clamp(1.0, 10.0);
    let sigma = (mouth.width * 0.22).max(1.5);
    let inner_height = mouth.inner_height.max(2.0);
    let mut dx = vec![0.0f32; pixels];
    let mut dy = vec![0.0f32; pixels];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let pixel = row + x;
            if mouth_mask[pixel] == 0 {
                continue;
            }
            let (s, t) = mouth.local(x as f32 + 0.5, y as f32 + 0.5);
            let radius2 = (s / mouth_rx).powi(2) + (t / mouth_ry).powi(2);
            if !(radius2 < 1.0) {
                continue;
            }
            let region = (1.0 - radius2).clamp(0.0, 1.0);
            let (corner_left, corner_right) =
                mouth.corner_falloff(x as f32 + 0.5, y as f32 + 0.5, sigma);
            let corners = (corner_left + corner_right).min(1.0);
            // Widen the corners while pulling them up.
            let spread = s.signum() * 0.5 * open * corners;
            let lift = -0.8 * open * corners;
            // Upper lip moves up, lower lip moves down; the sign blends
            // continuously through the mouth centre.
            let band = (-(t / (inner_height * 0.9)).powi(2)).exp();
            let opening = open * band * (t / inner_height).clamp(-1.0, 1.0);
            dx[pixel] = (spread * region).clamp(-12.0, 12.0);
            dy[pixel] = ((lift + opening) * region).clamp(-12.0, 12.0);
        }
    }

    let bounds = mask_bounds(&mouth_mask, width, height)
        .ok_or(Refusal::MissingInput("PF100007 lip mask"))?;
    let padding = ((mouth.width * 0.15).round() as usize).max(2);
    let roi = smile_roi(padded_bounds(bounds, padding, width, height), frame)?;

    let frame_roi = SmileRect {
        x: roi.0,
        y: roi.1,
        width: roi.2,
        height: roi.3,
    };
    let spec = super::SmileHostSplitSpec {
        job_id,
        dx: super::SmileAxisDecode {
            channel: super::SmileFieldChannel::R,
            sign: super::SmileAxisSign::Positive,
            scale: 1.0,
            bias: 0.0,
        },
        dy: super::SmileAxisDecode {
            channel: super::SmileFieldChannel::G,
            sign: super::SmileAxisSign::Positive,
            scale: 1.0,
            bias: 0.0,
        },
        model_roi: SmileRect {
            x: 0,
            y: 0,
            width: MODEL_WIDTH,
            height: MODEL_HEIGHT,
        },
        frame_roi,
    };

    Ok(super::SmileProducerInputs {
        ticket: job_id,
        material: Some("Grinning".to_string()),
        material_uv: None,
        material_sha256: None,
        roi: Some(roi),
        sigmoid_edge_response_3204548346: None,
        rgb_reconstruction_780929437: None,
        // This producer's displacement is not decoded from a vendor tensor, so
        // no host split receipt for the raw 1024x768 field is claimed.
        host_split_spec: None,
        channel_split: Some(super::SmileChannelSplit {
            spec,
            model_dx: dx,
            model_dy: dy,
        }),
        lip_mask: Some(lip_mask),
        mouth_mask: Some(mouth_mask),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::smile_warp::Rect as SmileRect;
    use aurora_portrait::kernels::{close_smile, face_detail, smile_warp, teeth_blend};
    use aurora_portrait::{Image, Refusal};
    use crate::{NormRect, RepairProducerInputs};

    const SIDE: u32 = 96;

    fn gradient_frame() -> Image {
        let mut pixels = Vec::with_capacity((SIDE * SIDE * 4) as usize);
        for y in 0..SIDE {
            for x in 0..SIDE {
                pixels.push(((x * 37 + y * 17) % 256) as u8);
                pixels.push(((x * 13 + y * 41) % 256) as u8);
                pixels.push(((x * 29 + y * 7) % 256) as u8);
                pixels.push(255);
            }
        }
        Image::new(SIDE, SIDE, pixels).expect("valid synthetic frame")
    }

    fn synthetic_points() -> Vec<[f32; 2]> {
        let mut points = vec![[48.0f32, 55.0]; 194];
        points[73] = [34.0, 55.0];
        points[83] = [62.0, 55.0];
        points[95] = [38.0, 55.0];
        points[101] = [58.0, 55.0];
        points[78] = [48.0, 51.0];
        points[89] = [48.0, 59.0];
        points[98] = [48.0, 52.0];
        points[104] = [48.0, 58.0];
        points
    }

    fn teeth_mask() -> Vec<u8> {
        let mut mask = vec![0u8; (SIDE * SIDE) as usize];
        for y in 53..57u32 {
            for x in 40..56u32 {
                mask[(y * SIDE + x) as usize] = 255;
            }
        }
        mask
    }

    fn lip_masks() -> (Vec<u8>, Vec<u8>) {
        let mut upper = vec![0u8; (SIDE * SIDE) as usize];
        let mut lower = vec![0u8; (SIDE * SIDE) as usize];
        for x in 38..58u32 {
            for y in 51..55u32 {
                upper[(y * SIDE + x) as usize] = 255;
            }
            for y in 56..60u32 {
                lower[(y * SIDE + x) as usize] = 255;
            }
        }
        (upper, lower)
    }

    fn changed_in(
        before: &Image,
        after: &Image,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
    ) -> bool {
        let width = before.width as usize;
        for y in y0..y1 {
            for x in x0..x1 {
                let at = (y * width + x) * 4;
                if before.pixels[at..at + 4] != after.pixels[at..at + 4] {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn pf30025_builds_a_valid_receipt_and_repairs_teeth() {
        let frame = gradient_frame();
        let points = synthetic_points();
        let mask = teeth_mask();
        let inputs = build_pf30025_teeth_repair_inputs(7, &frame, &points, &mask).unwrap();

        assert_eq!(inputs.receipt.job_id, 7);
        assert_eq!((inputs.receipt.width, inputs.receipt.height), (SIDE, SIDE));
        assert_eq!(inputs.receipt.source_stride, SIDE * 4);
        assert_eq!(
            inputs.receipt.source_sha256,
            crate::sha256_32(&frame.pixels)
        );
        assert_eq!(
            inputs.receipt.repair_sha256,
            crate::sha256_32(&inputs.repair_780929437_bgra)
        );
        assert_eq!(inputs.repair_780929437_bgra.len(), (SIDE * SIDE * 4) as usize);
        assert_eq!(inputs.gate_3204548346.len(), (SIDE * SIDE) as usize);
        assert_eq!(inputs.teeth_mask.len(), (SIDE * SIDE) as usize);
        assert!(inputs.gate_3204548346.iter().all(|value| value.is_finite()));

        let repaired = face_detail::tooth_repair_same_job(
            &frame,
            Some(face_detail::ToothRepairProducer {
                repair_780929437_bgra: &inputs.repair_780929437_bgra,
                gate_3204548346: &inputs.gate_3204548346,
                roi: inputs.roi,
            }),
            100.0,
        )
        .unwrap();
        assert!(changed_in(&frame, &repaired, 40, 53, 56, 57));
        assert_eq!(repaired.pixel(2, 2), frame.pixel(2, 2));
    }

    #[test]
    fn pf30026_alignment_opens_the_gate_and_blends() {
        let frame = gradient_frame();
        let points = synthetic_points();
        let mask = teeth_mask();
        let inputs = build_pf30026_teeth_alignment_inputs(9, &frame, &points, &mask).unwrap();

        assert_eq!(inputs.roi, Some(NormRect::FULL));
        let binding = inputs.gate_binding.unwrap();
        assert_eq!(binding.job_id, 9);
        assert_ne!(binding.plane_a, binding.plane_b);
        let effect = inputs.effect_rgb.as_ref().unwrap();
        assert!(changed_in(&frame, effect, 40, 53, 56, 57));
        assert_eq!(effect.pixel(2, 2), frame.pixel(2, 2));
        assert_eq!(inputs.teeth_mask.as_ref().unwrap().len(), (SIDE * SIDE) as usize);

        let outcome = teeth_blend::apply_alignment_final_blend(
            teeth_blend::AlignmentFinalBlendJob {
                original: Some(&frame),
                effect_rgb: Some(effect),
                teeth_mask: inputs.teeth_mask.as_deref(),
                edge_plane_a: inputs.edge_plane_a.as_deref(),
                edge_plane_b: inputs.edge_plane_b.as_deref(),
            },
        )
        .unwrap();
        match outcome {
            teeth_blend::AlignmentFinalBlendOutcome::Applied(image) => {
                assert!(changed_in(&frame, &image, 40, 53, 56, 57));
                assert_eq!(image.pixel(2, 2), frame.pixel(2, 2));
            }
            teeth_blend::AlignmentFinalBlendOutcome::Gated => {
                panic!("PF30026 gate must open for a populated teeth mask")
            }
        }
    }

    #[test]
    fn pf21076_closed_mouth_smile_warps_the_mouth_and_binds() {
        let frame = gradient_frame();
        let points = synthetic_points();
        let output = build_pf21076_closed_mouth_smile_output(&frame, &points).unwrap();
        let crate::RepairOutput::ClosedMouthSmile {
            model,
            paste_enabled,
        } = output.clone()
        else {
            panic!("expected ClosedMouthSmile");
        };
        assert!(paste_enabled);
        assert_eq!((model.width, model.height), (SIDE, SIDE));
        assert!(changed_in(&frame, &model, 31, 45, 65, 66));
        assert_eq!(model.pixel(2, 2), frame.pixel(2, 2));

        let composed = close_smile::apply(&frame, &model, true).unwrap();
        assert!(changed_in(&frame, &composed, 31, 45, 65, 66));

        let mut bundle = RepairProducerInputs::default();
        bundle.insert(11, &frame, output).unwrap();
        assert!(bundle.receipt(21076).is_some());
    }

    #[test]
    fn pf100007_open_mouth_smile_feeds_the_vendor_consumer() {
        let frame = gradient_frame();
        let points = synthetic_points();
        let (upper, lower) = lip_masks();
        let inputs =
            build_pf100007_open_mouth_smile_inputs(13, &frame, &points, &upper, &lower).unwrap();

        assert_eq!(inputs.ticket, 13);
        assert_eq!(inputs.material.as_deref(), Some("Grinning"));
        let lip_mask = inputs.lip_mask.as_ref().unwrap();
        let mouth_mask = inputs.mouth_mask.as_ref().unwrap();
        assert_eq!(lip_mask.len(), (SIDE * SIDE) as usize);
        assert_eq!(mouth_mask.len(), (SIDE * SIDE) as usize);
        assert!(lip_mask.iter().any(|value| *value > 0));
        assert!(mouth_mask.iter().any(|value| *value > 0));

        let split = inputs.channel_split.as_ref().unwrap();
        assert_eq!(split.spec.job_id, 13);
        assert_eq!(split.model_dx.len(), (SIDE * SIDE) as usize);
        assert_eq!(split.model_dy.len(), (SIDE * SIDE) as usize);
        assert!(split
            .model_dx
            .iter()
            .chain(&split.model_dy)
            .any(|value| *value != 0.0));
        assert!(split
            .model_dx
            .iter()
            .chain(&split.model_dy)
            .all(|value| value.is_finite()));

        let (active, degree) =
            smile_warp::pf100007_gate(100.0, inputs.material.as_deref()).unwrap();
        assert!(active, "Grinning material must activate PF100007");
        assert!((degree - 1.0).abs() < 1e-6);

        let roi = inputs.roi.unwrap();
        let rect = SmileRect {
            x: roi.0,
            y: roi.1,
            width: roi.2,
            height: roi.3,
        };
        let warped = smile_warp::pf100007_apply(
            &frame.pixels,
            frame.width,
            frame.height,
            1.0,
            lip_mask,
            mouth_mask,
            &split.model_dx,
            &split.model_dy,
            rect,
        )
        .unwrap();
        let warped_image = Image::new(SIDE, SIDE, warped).unwrap();
        assert!(changed_in(
            &frame,
            &warped_image,
            rect.x as usize,
            rect.y as usize,
            (rect.x + rect.width) as usize,
            (rect.y + rect.height) as usize,
        ));
        assert_eq!(warped_image.pixel(2, 2), frame.pixel(2, 2));
    }

    #[test]
    fn invalid_inputs_are_refused() {
        let frame = gradient_frame();
        let points = synthetic_points();
        let mask = teeth_mask();
        let empty = vec![0u8; (SIDE * SIDE) as usize];

        assert_eq!(
            build_pf30025_teeth_repair_inputs(0, &frame, &points, &mask),
            Err(Refusal::ValueOutOfRange("PF30025 job id"))
        );
        assert_eq!(
            build_pf30025_teeth_repair_inputs(1, &frame, &points, &[0u8; 4]),
            Err(Refusal::ShapeMismatch("PF30025 teeth mask"))
        );
        assert_eq!(
            build_pf30025_teeth_repair_inputs(1, &frame, &points, &empty),
            Err(Refusal::MissingInput("PF30025 teeth mask"))
        );
        assert_eq!(
            build_pf30025_teeth_repair_inputs(1, &frame, &[], &mask),
            Err(Refusal::ShapeMismatch("points194"))
        );
        let mut bad_points = points.clone();
        bad_points[73] = [f32::NAN, 0.0];
        assert_eq!(
            build_pf30025_teeth_repair_inputs(1, &frame, &bad_points, &mask),
            Err(Refusal::ValueOutOfRange("points194"))
        );

        assert_eq!(
            build_pf30026_teeth_alignment_inputs(0, &frame, &points, &mask),
            Err(Refusal::ValueOutOfRange("PF30026 job id"))
        );
        assert_eq!(
            build_pf30026_teeth_alignment_inputs(1, &frame, &points, &empty),
            Err(Refusal::MissingInput("PF30026 teeth mask"))
        );

        assert_eq!(
            build_pf21076_closed_mouth_smile_output(&frame, &[]),
            Err(Refusal::ShapeMismatch("points194"))
        );
        let mut bad_points = points.clone();
        bad_points[104] = [f32::INFINITY, 0.0];
        assert_eq!(
            build_pf21076_closed_mouth_smile_output(&frame, &bad_points),
            Err(Refusal::ValueOutOfRange("points194"))
        );
        let bad_frame = Image {
            width: 4,
            height: 4,
            pixels: Vec::new(),
        };
        assert_eq!(
            build_pf21076_closed_mouth_smile_output(&bad_frame, &points),
            Err(Refusal::ShapeMismatch("image"))
        );

        let (upper, lower) = lip_masks();
        assert_eq!(
            build_pf100007_open_mouth_smile_inputs(0, &frame, &points, &upper, &lower),
            Err(Refusal::ValueOutOfRange("PF100007 job id"))
        );
        assert_eq!(
            build_pf100007_open_mouth_smile_inputs(1, &frame, &points, &empty, &empty),
            Err(Refusal::MissingInput("PF100007 lip mask"))
        );
        assert_eq!(
            build_pf100007_open_mouth_smile_inputs(1, &frame, &points, &[0u8; 4], &lower),
            Err(Refusal::ShapeMismatch("PF100007 upper lip mask"))
        );
        assert_eq!(
            build_pf100007_open_mouth_smile_inputs(1, &frame, &points, &upper, &[0u8; 4]),
            Err(Refusal::ShapeMismatch("PF100007 lower lip mask"))
        );
    }
}
