//! Same-frame PF5 / PF90116 producers driven entirely by the recovered
//! PixelCake `eye_bag` kernel contracts.
//!
//! This module does **not** run a model and does **not** claim vendor geometry:
//! the eye bands are synthesized from the same `points194` template the engine
//! already uses for eye ROIs, and every output is gated by the supplied masks.
//! A coordinator must still register the module and connect the returned
//! planes to `ProducerInputs.eye_bag` / `RepairProducerInputs`.
//!
//! PF5: `RepairOutput::DarkCircle { residual_rgb }` is a full-frame,
//! interleaved RGB f32 plane. The live consumer is
//! `eye_bag::compose_image_pf5(..., BlendMode::Vendor)`, which computes
//! `clamp(round_even(source + residual - 80))`. A neutral pixel therefore
//! carries exactly `80.0`, and a desired channel delta `d` is encoded as
//! `80.0 + d`.
//!
//! PF90116: `AlignedEyeBagPlanes` is publicly constructible. Its two consumed
//! planes are a full-frame interleaved float color field in **SourceBytes**
//! units and a full-frame response mask; the vendor kernel quantizes the
//! response to bytes and blends `repair * q/255 * degree + source * (1 - ...)`.

use aurora_portrait::eye_bag_alignment::{
    AlignedEyeBagPlanes, EyeBagColorUnits, EyeBagGeometryClass, MODEL_ID,
};
use aurora_portrait::kernels::eye_bag::Roi;
use aurora_portrait::{Image, Refusal};

use super::RepairOutput;

/// Neutral residual value recovered at `0x46C6F20` (`0x50` bias).
const PF5_RESIDUAL_BIAS: f32 = 80.0;

/// 194-point template indices used by the engine's live eye code.
const LEFT_CORNERS: (usize, usize) = (18, 22);
const RIGHT_CORNERS: (usize, usize) = (36, 40);
const LEFT_CENTER: usize = 26;
const RIGHT_CENTER: usize = 44;
/// Upper and lower eyelid arcs of the 8-point eye contour.
const LEFT_UPPER: [usize; 3] = [19, 20, 21];
const LEFT_LOWER: [usize; 3] = [23, 24, 25];
const RIGHT_UPPER: [usize; 3] = [37, 38, 39];
const RIGHT_LOWER: [usize; 3] = [41, 42, 43];

const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

#[derive(Clone, Copy, Debug)]
struct EyeGeometry {
    /// Eye midpoint in absolute source pixels.
    cx: f32,
    cy: f32,
    /// Unit vector along the eye axis.
    ux: f32,
    uy: f32,
    /// Unit vector perpendicular to the eye axis, pointing downwards.
    vx: f32,
    vy: f32,
    /// Corner-to-corner distance.
    span: f32,
    /// Lower-eyelid line offset from the midpoint along `v`.
    lid: f32,
    /// Eye aperture height (upper arc to lower arc along `v`).
    height: f32,
    /// Iris/eye centre landmark, used only to keep the eyeball out of the band.
    iris: [f32; 2],
}

impl EyeGeometry {
    fn from_points(
        points: &[[f32; 2]],
        corners: (usize, usize),
        center: usize,
        upper: [usize; 3],
        lower: [usize; 3],
    ) -> Result<Self, Refusal> {
        let a = points[corners.0];
        let b = points[corners.1];
        let iris = points[center];
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let span = dx.hypot(dy);
        if !span.is_finite() || span < 4.0 {
            return Err(Refusal::ValueOutOfRange("points194 eye span"));
        }
        let (ux, uy) = (dx / span, dy / span);
        let (mut vx, mut vy) = (-uy, ux);
        if vy < 0.0 {
            vx = -vx;
            vy = -vy;
        }
        let cx = (a[0] + b[0]) * 0.5;
        let cy = (a[1] + b[1]) * 0.5;
        let mean = |ids: &[usize; 3]| -> [f32; 2] {
            [
                ids.iter().map(|&i| points[i][0]).sum::<f32>() / 3.0,
                ids.iter().map(|&i| points[i][1]).sum::<f32>() / 3.0,
            ]
        };
        let up = mean(&upper);
        let low = mean(&lower);
        let along = |p: [f32; 2]| (p[0] - cx) * vx + (p[1] - cy) * vy;
        let raw_height = along(low) - along(up);
        let height = if raw_height.is_finite() && raw_height >= 1.0 {
            raw_height
        } else {
            span * 0.4
        };
        let raw_lid = along(low);
        let lid = if raw_lid.is_finite() && raw_lid >= 0.0 {
            raw_lid
        } else {
            height * 0.5
        };
        Ok(Self {
            cx,
            cy,
            ux,
            uy,
            vx,
            vy,
            span,
            lid,
            height,
            iris,
        })
    }

    /// Signed coordinates of one pixel: `s` along the eye axis, `d` downwards
    /// from the lower-eyelid line, both in multiples set by the caller.
    fn coords(&self, x: usize, y: usize) -> (f32, f32) {
        let rx = x as f32 + 0.5 - self.cx;
        let ry = y as f32 + 0.5 - self.cy;
        (
            rx * self.ux + ry * self.uy,
            (rx * self.vx + ry * self.vy) - self.lid,
        )
    }

    /// Excludes the eye white, iris and lashes irrespective of `eye_mask`.
    fn eyeball_overlap(&self, x: usize, y: usize) -> bool {
        let rx = x as f32 + 0.5 - self.cx;
        let ry = y as f32 + 0.5 - self.cy;
        let s = rx * self.ux + ry * self.uy;
        let t = rx * self.vx + ry * self.vy;
        let i = self.iris;
        let it = (i[0] - self.cx) * self.vx + (i[1] - self.cy) * self.vy;
        let sh = self.span * 0.52;
        let vh = self.height * 0.5;
        (s / sh).powi(2) + ((t - it) / vh).powi(2) < 1.0
    }
}

struct EyeContext {
    left: EyeGeometry,
    right: EyeGeometry,
    roi: (usize, usize, usize, usize),
    left_ref: [f32; 3],
    right_ref: [f32; 3],
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn validate_inputs(
    frame: &Image,
    face_points194: &[[f32; 2]],
    skin_mask: Option<&[u8]>,
    eye_mask: Option<&[u8]>,
) -> Result<(), Refusal> {
    frame.validate()?;
    if face_points194.len() != 194 {
        return Err(Refusal::ShapeMismatch("points194"));
    }
    // The eye contour, iris centre and lower-lid anchors are all accessed.
    if !(18..=44).all(|i| face_points194[i].iter().all(|value| value.is_finite())) {
        return Err(Refusal::ValueOutOfRange("points194"));
    }
    let pixels = frame.pixel_count();
    if let Some(mask) = skin_mask {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("skin_mask"));
        }
    }
    if let Some(mask) = eye_mask {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("eye_mask"));
        }
    }
    Ok(())
}

fn prepare(
    frame: &Image,
    face_points194: &[[f32; 2]],
    skin_mask: Option<&[u8]>,
    eye_mask: Option<&[u8]>,
    radius: usize,
) -> Result<EyeContext, Refusal> {
    validate_inputs(frame, face_points194, skin_mask, eye_mask)?;
    let left = EyeGeometry::from_points(
        face_points194,
        LEFT_CORNERS,
        LEFT_CENTER,
        LEFT_UPPER,
        LEFT_LOWER,
    )?;
    let right = EyeGeometry::from_points(
        face_points194,
        RIGHT_CORNERS,
        RIGHT_CENTER,
        RIGHT_UPPER,
        RIGHT_LOWER,
    )?;

    let width = frame.width as usize;
    let height = frame.height as usize;
    let pad = radius as f32 + 2.0;
    let mut min_x = width as f32;
    let mut min_y = height as f32;
    let mut max_x = 0.0f32;
    let mut max_y = 0.0f32;
    for eye in [left, right] {
        let reach_x = eye.span * 1.1 + pad;
        min_x = min_x.min(eye.cx - reach_x);
        max_x = max_x.max(eye.cx + reach_x);
        min_y = min_y.min(eye.cy + eye.lid - eye.height * 1.2 - pad);
        max_y = max_y.max(eye.cy + eye.lid + eye.height * 2.6 + pad);
    }
    let x0 = (min_x.floor().max(0.0) as usize).min(width - 1);
    let y0 = (min_y.floor().max(0.0) as usize).min(height - 1);
    let x1 = (max_x.ceil().max(0.0) as usize).clamp(x0 + 1, width);
    let y1 = (max_y.ceil().max(0.0) as usize).clamp(y0 + 1, height);
    let roi = (x0, y0, x1, y1);

    let left_ref = cheek_reference(frame, &left, skin_mask, eye_mask);
    let right_ref = cheek_reference(frame, &right, skin_mask, eye_mask);
    Ok(EyeContext {
        left,
        right,
        roi,
        left_ref,
        right_ref,
    })
}

/// Three-by-three average of healthy cheek samples below the eye band.
fn cheek_reference(
    frame: &Image,
    eye: &EyeGeometry,
    skin_mask: Option<&[u8]>,
    eye_mask: Option<&[u8]>,
) -> [f32; 3] {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let sample = |use_masks: bool| -> Option<[f32; 3]> {
        let mut acc = [0.0f32; 3];
        let mut count = 0usize;
        for &depth in &[1.6f32, 2.0, 2.3] {
            for &lateral in &[-0.4f32, 0.0, 0.4] {
                let along = eye.lid + depth * eye.height;
                // The lateral offset runs along the eye axis, not the down axis.
                let px = eye.cx + lateral * eye.span * eye.ux + eye.vx * along;
                let py = eye.cy + lateral * eye.span * eye.uy + eye.vy * along;
                let x = px.round() as isize;
                let y = py.round() as isize;
                let mut patch = [0.0f32; 3];
                let mut hits = 0usize;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let sx = (x + dx).clamp(0, width as isize - 1) as usize;
                        let sy = (y + dy).clamp(0, height as isize - 1) as usize;
                        let index = sy * width + sx;
                        if use_masks {
                            if let Some(mask) = skin_mask {
                                if mask[index] == 0 {
                                    continue;
                                }
                            }
                            if let Some(mask) = eye_mask {
                                if mask[index] != 0 {
                                    continue;
                                }
                            }
                        }
                        let offset = index * 4;
                        for channel in 0..3 {
                            patch[channel] += frame.pixels[offset + channel] as f32;
                        }
                        hits += 1;
                    }
                }
                if hits > 0 {
                    for channel in 0..3 {
                        acc[channel] += patch[channel] / hits as f32;
                    }
                    count += 1;
                }
            }
        }
        (count > 0).then_some(acc.map(|value| value / count as f32))
    };
    if let Some(reference) = sample(true) {
        return reference;
    }
    if let Some(reference) = sample(false) {
        return reference;
    }
    let x = eye.cx.round().clamp(0.0, width.saturating_sub(1) as f32) as usize;
    let y = eye.cy.round().clamp(0.0, height.saturating_sub(1) as f32) as usize;
    let offset = (y * width + x) * 4;
    [
        frame.pixels[offset] as f32,
        frame.pixels[offset + 1] as f32,
        frame.pixels[offset + 2] as f32,
    ]
}

/// Separable box blur of the RGB channels over one ROI. Edge taps are clamped.
fn box_blur_rgb(
    frame: &Image,
    (x0, y0, x1, y1): (usize, usize, usize, usize),
    radius: usize,
) -> Vec<[f32; 3]> {
    let width = frame.width as usize;
    let rw = x1 - x0;
    let rh = y1 - y0;
    let count = (2 * radius + 1) as f32;
    let mut horizontal = vec![[0.0f32; 3]; rw * rh];
    for y in y0..y1 {
        for x in x0..x1 {
            let mut sum = [0.0f32; 3];
            for step in -(radius as isize)..=(radius as isize) {
                let sx = (x as isize + step).clamp(0, width as isize - 1) as usize;
                let offset = (y * width + sx) * 4;
                for channel in 0..3 {
                    sum[channel] += frame.pixels[offset + channel] as f32;
                }
            }
            horizontal[(y - y0) * rw + (x - x0)] = sum.map(|value| value / count);
        }
    }
    let mut out = vec![[0.0f32; 3]; rw * rh];
    for y in 0..rh {
        for x in 0..rw {
            let mut sum = [0.0f32; 3];
            for step in -(radius as isize)..=(radius as isize) {
                let sy = (y as isize + step).clamp(0, rh as isize - 1) as usize;
                let value = horizontal[sy * rw + x];
                for channel in 0..3 {
                    sum[channel] += value[channel];
                }
            }
            out[y * rw + x] = sum.map(|value| value / count);
        }
    }
    out
}

fn luma(rgb: [f32; 3]) -> f32 {
    (0..3).map(|c| rgb[c] * LUMA[c]).sum()
}

fn eye_weight(
    eye: &EyeGeometry,
    x: usize,
    y: usize,
    inner: f32,
    rise_end: f32,
    fall_start: f32,
    fall_end: f32,
    lateral_core: f32,
    lateral_edge: f32,
    masks_ok: f32,
) -> f32 {
    if masks_ok <= 0.0 || eye.eyeball_overlap(x, y) {
        return 0.0;
    }
    let (s, d) = eye.coords(x, y);
    let depth = d / eye.height;
    if depth <= inner {
        return 0.0;
    }
    let vertical = smoothstep(inner, rise_end, depth) * (1.0 - smoothstep(fall_start, fall_end, depth));
    let lateral = 1.0 - smoothstep(lateral_core, lateral_edge, (s / (eye.span * 0.5)).abs());
    (vertical * lateral * masks_ok).clamp(0.0, 1.0)
}

fn mask_factor(index: usize, skin_mask: Option<&[u8]>, eye_mask: Option<&[u8]>) -> f32 {
    let mut factor = 1.0f32;
    if let Some(mask) = skin_mask {
        factor *= f32::from(mask[index]) / 255.0;
    }
    if let Some(mask) = eye_mask {
        factor *= 1.0 - f32::from(mask[index]) / 255.0;
    }
    factor
}

/// PF5 producer. The returned plane is ready for
/// `RepairOutput::DarkCircle` and the live `+80` consumer.
#[allow(dead_code)]
pub(crate) fn build_pf5_dark_circle_output(
    frame: &Image,
    face_points194: &[[f32; 2]],
    skin_mask: Option<&[u8]>,
    eye_mask: Option<&[u8]>,
) -> Result<RepairOutput, Refusal> {
    let base_context = prepare(frame, face_points194, skin_mask, eye_mask, 2)?;
    let avg_eye_h = (base_context.left.height + base_context.right.height) * 0.5;
    let radius = ((avg_eye_h * 0.35).round() as usize).clamp(2, 14);
    let context = if radius == 2 {
        base_context
    } else {
        prepare(frame, face_points194, skin_mask, eye_mask, radius)?
    };
    let width = frame.width as usize;
    let pixels = frame.pixel_count();
    let mut residual = vec![PF5_RESIDUAL_BIAS; pixels * 3];
    let blurred = box_blur_rgb(frame, context.roi, radius);
    let (x0, y0, x1, y1) = context.roi;
    let rw = x1 - x0;

    for y in y0..y1 {
        for x in x0..x1 {
            let index = y * width + x;
            let factor = mask_factor(index, skin_mask, eye_mask);
            let mut weight = eye_weight(
                &context.left,
                x,
                y,
                0.15,
                0.32,
                0.58,
                0.85,
                0.55,
                1.0,
                factor,
            );
            if weight == 0.0 {
                weight = eye_weight(
                    &context.right,
                    x,
                    y,
                    0.15,
                    0.32,
                    0.58,
                    0.85,
                    0.55,
                    1.0,
                    factor,
                );
            }
            if weight == 0.0 {
                continue;
            }
            let reference = if eye_weight(
                &context.left,
                x,
                y,
                0.15,
                0.32,
                0.58,
                0.85,
                0.55,
                1.0,
                factor,
            ) > 0.0
            {
                context.left_ref
            } else {
                context.right_ref
            };
            let low = blurred[(y - y0) * rw + (x - x0)];
            let darkness = luma(reference) - luma(low);
            if darkness <= 0.0 {
                continue;
            }
            // Low-frequency correction only: `source - low` (the pores and
            // texture) is left untouched. Negative per-channel deltas remove
            // cyan/purple casts; the clamp prevents an over-bleached patch.
            for channel in 0..3 {
                let delta = ((reference[channel] - low[channel]) * 0.65)
                    .clamp(-12.0, 26.0)
                    * weight;
                residual[index * 3 + channel] = PF5_RESIDUAL_BIAS + delta;
            }
        }
    }
    Ok(RepairOutput::DarkCircle { residual_rgb: residual })
}

fn frame_ticket(frame: &Image) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in frame.width.to_le_bytes().iter().chain(&frame.height.to_le_bytes()).chain(&frame.pixels) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if hash == 0 {
        1
    } else {
        hash
    }
}

/// PF90116 producer: two full-frame planes plus a small provenance ROI, all
/// in the `SourceBytes` domain the stage validates.
#[allow(dead_code)]
pub(crate) fn build_pf90116_eye_bag_planes(
    frame: &Image,
    face_points194: &[[f32; 2]],
    skin_mask: Option<&[u8]>,
    eye_mask: Option<&[u8]>,
) -> Result<AlignedEyeBagPlanes, Refusal> {
    let base_context = prepare(frame, face_points194, skin_mask, eye_mask, 2)?;
    let avg_eye_h = (base_context.left.height + base_context.right.height) * 0.5;
    let radius = ((avg_eye_h * 0.35).round() as usize).clamp(2, 14);
    let context = if radius == 2 {
        base_context
    } else {
        prepare(frame, face_points194, skin_mask, eye_mask, radius)?
    };
    let width = frame.width as usize;
    let pixels = frame.pixel_count();
    let mut repair = vec![0.0f32; pixels * 3];
    let mut response = vec![0.0f32; pixels];
    let blurred = box_blur_rgb(frame, context.roi, radius);
    let (x0, y0, x1, y1) = context.roi;
    let rw = x1 - x0;

    for p in 0..pixels {
        for channel in 0..3 {
            repair[p * 3 + channel] = frame.pixels[p * 4 + channel] as f32;
        }
    }

    for y in y0..y1 {
        for x in x0..x1 {
            let index = y * width + x;
            let factor = mask_factor(index, skin_mask, eye_mask);
            let mut weight = eye_weight(
                &context.left, x, y, 0.04, 0.12, 0.28, 0.48, 0.35, 0.85, factor,
            );
            if weight == 0.0 {
                weight = eye_weight(
                    &context.right, x, y, 0.04, 0.12, 0.28, 0.48, 0.35, 0.85, factor,
                );
            }
            if weight == 0.0 {
                continue;
            }
            let reference = if eye_weight(
                &context.left, x, y, 0.04, 0.12, 0.28, 0.48, 0.35, 0.85, factor,
            ) > 0.0
            {
                context.left_ref
            } else {
                context.right_ref
            };
            let low = blurred[(y - y0) * rw + (x - x0)];
            // Pull only the low-frequency groove/ridge contrast toward the
            // cheek reference. The high-frequency residual `source - low`
            // survives unchanged, so pores and the natural silkworm edge stay.
            for channel in 0..3 {
                let delta = ((reference[channel] - low[channel]) * 0.72).clamp(-24.0, 24.0);
                repair[index * 3 + channel] = (repair[index * 3 + channel] + delta)
                    .clamp(0.0, 255.0);
            }
            // The kernel quantizes this to bytes and multiplies by degree.
            // A near-full response in the core gives the flattened low-frequency
            // field real weight while the feather still protects the boundary.
            response[index] = (weight * 1.15).clamp(0.0, 1.0);
        }
    }

    let roi = Roi {
        x: x0 as i32,
        y: y0 as i32,
        w: (x1 - x0) as i32,
        h: (y1 - y0) as i32,
    };
    let mut model_input = vec![0.0f32; (x1 - x0) * (y1 - y0) * 3];
    for y in y0..y1 {
        for x in x0..x1 {
            let src = (y * width + x) * 4;
            let dst = ((y - y0) * (x1 - x0) + (x - x0)) * 3;
            for channel in 0..3 {
                model_input[dst + channel] = frame.pixels[src + channel] as f32;
            }
        }
    }

    let planes = AlignedEyeBagPlanes {
        ticket: frame_ticket(frame),
        frame_width: frame.width,
        frame_height: frame.height,
        model_id: MODEL_ID,
        aligned_color_field_rgb_f32: repair,
        color_units: EyeBagColorUnits::SourceBytes,
        aligned_response_mask: response,
        destination_model_input_rgb_f32: model_input,
        source_response_roi: None,
        work_roi: None,
        destination_roi: roi,
        geometry_class: EyeBagGeometryClass::Candidate,
        geometry_receipt: None,
    };
    planes.validate()?;
    Ok(planes)
}

/// Direct vendor-kernel driver for coordinators that cannot place a
/// candidate-geometry plane on the receipt-gated production stage path.
#[allow(dead_code)]
pub(crate) fn apply_pf90116_eye_bag_fallback(
    frame: &Image,
    face_points194: &[[f32; 2]],
    skin_mask: Option<&[u8]>,
    eye_mask: Option<&[u8]>,
    degree: f32,
) -> Result<Image, Refusal> {
    let planes = build_pf90116_eye_bag_planes(frame, face_points194, skin_mask, eye_mask)?;
    aurora_portrait::kernels::eye_bag::compose_field_image_pf90116(
        frame,
        &planes.aligned_color_field_rgb_f32,
        &planes.aligned_response_mask,
        degree,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::eye_bag;

    fn synthetic_face() -> Image {
        let (width, height) = (64usize, 48usize);
        let mut pixels = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 4;
                let mut rgb = [182u8, 142, 122];
                for &cx in &[20.0f32, 44.0] {
                    let dx = x as f32 + 0.5 - cx;
                    // Eye white / iris, kept out of every producer band.
                    if dx.abs() < 8.0 && (17..24).contains(&y) {
                        rgb = [238, 238, 235];
                    }
                    // Dark-circle crescent: 0.15..0.85 eye heights below lid.
                    if dx.abs() < 7.0 && (24..29).contains(&y) {
                        rgb = [118, 106, 112];
                    }
                    // Narrow tear-trough groove with high-frequency texture.
                    if y == 25 || y == 26 {
                        if dx.abs() < 7.0 {
                            let grain = if x % 2 == 0 { 4 } else { -4 };
                            rgb = [
                                (116i32 + grain).clamp(0, 255) as u8,
                                (108i32 + grain).clamp(0, 255) as u8,
                                (114i32 + grain).clamp(0, 255) as u8,
                            ];
                        }
                    }
                }
                pixels[offset..offset + 3].copy_from_slice(&rgb);
                pixels[offset + 3] = 255;
            }
        }
        Image::new(width as u32, height as u32, pixels).unwrap()
    }

    fn points194() -> Vec<[f32; 2]> {
        let mut points = vec![[0.0f32, 0.0]; 194];
        let mut eye = |corners: (usize, usize), center: usize, upper: [usize; 3], lower: [usize; 3], mid: f32| {
            points[corners.0] = [mid - 8.0, 20.0];
            points[corners.1] = [mid + 8.0, 20.0];
            for (i, &index) in upper.iter().enumerate() {
                points[index] = [mid - 5.0 + i as f32 * 5.0, 16.5];
            }
            for (i, &index) in lower.iter().enumerate() {
                points[index] = [mid - 5.0 + i as f32 * 5.0, 23.5];
            }
            points[center] = [mid, 20.0];
            for (i, index) in (27..35).enumerate() {
                points[index] = [mid - 4.0 + (i % 4) as f32 * 2.0, 19.0 + (i / 4) as f32 * 2.0];
            }
            for (i, index) in (45..53).enumerate() {
                points[index] = [mid - 4.0 + (i % 4) as f32 * 2.0, 19.0 + (i / 4) as f32 * 2.0];
            }
        };
        eye(LEFT_CORNERS, LEFT_CENTER, LEFT_UPPER, LEFT_LOWER, 20.0);
        eye(RIGHT_CORNERS, RIGHT_CENTER, RIGHT_UPPER, RIGHT_LOWER, 44.0);
        points
    }

    #[test]
    fn pf5_brightens_the_crescent_and_leaves_eyeball_and_far_pixels_untouched() {
        let frame = synthetic_face();
        let points = points194();
        let RepairOutput::DarkCircle { residual_rgb } =
            build_pf5_dark_circle_output(&frame, &points, None, None).unwrap()
        else {
            panic!("expected DarkCircle output");
        };
        assert_eq!(residual_rgb.len(), frame.pixel_count() * 3);
        let band = 26 * 64 * 3 + 20 * 3;
        assert!(residual_rgb[band] > PF5_RESIDUAL_BIAS + 1.0);
        let eyeball = 20 * 64 * 3 + 20 * 3;
        assert_eq!(residual_rgb[eyeball..eyeball + 3], [80.0; 3]);
        let far = 4 * 64 * 3 + 4 * 3;
        assert_eq!(residual_rgb[far..far + 3], [80.0; 3]);

        let output = eye_bag::compose_image_pf5(&frame, &residual_rgb, eye_bag::BlendMode::Vendor)
            .unwrap();
        let before = frame.pixel(20, 26);
        let after = output.pixel(20, 26);
        assert!(after[0] > before[0], "PF5 must brighten the dark crescent");
        assert_eq!(output.pixel(20, 20), frame.pixel(20, 20));
        assert_eq!(output.pixel(4, 4), frame.pixel(4, 4));
    }

    #[test]
    fn pf5_respects_eye_and_skin_masks() {
        let frame = synthetic_face();
        let points = points194();
        let pixels = frame.pixel_count();
        let eye_mask = vec![0u8; pixels];
        let skin_mask = vec![0u8; pixels];
        let RepairOutput::DarkCircle { residual_rgb } =
            build_pf5_dark_circle_output(&frame, &points, Some(&skin_mask), Some(&eye_mask))
                .unwrap()
        else {
            panic!("expected DarkCircle output");
        };
        assert!(residual_rgb.iter().all(|value| *value == PF5_RESIDUAL_BIAS));
    }

    #[test]
    fn pf90116_flattens_the_groove_while_keeping_texture_and_bounds() {
        let frame = synthetic_face();
        let points = points194();
        let planes = build_pf90116_eye_bag_planes(&frame, &points, None, None).unwrap();
        assert_eq!(planes.validate(), Ok(()));
        assert_eq!(planes.color_units, EyeBagColorUnits::SourceBytes);
        assert_eq!(planes.geometry_class, EyeBagGeometryClass::Candidate);
        assert_eq!(planes.geometry_receipt, None);
        assert_eq!(planes.aligned_color_field_rgb_f32.len(), frame.pixel_count() * 3);
        assert_eq!(planes.aligned_response_mask.len(), frame.pixel_count());
        assert!(planes.aligned_response_mask[26 * 64 + 20] > 0.0);

        let output = apply_pf90116_eye_bag_fallback(&frame, &points, None, None, 1.0).unwrap();
        let before = frame.pixel(20, 26);
        let after = output.pixel(20, 26);
        assert!(after[0] > before[0], "the tear-trough shadow must be flattened");
        let original_texture =
            (10..30).map(|x| frame.pixel(x, 26)[0] as i32 - frame.pixel(x + 1, 26)[0] as i32).map(i32::abs).sum::<i32>();
        let output_texture =
            (10..30).map(|x| output.pixel(x, 26)[0] as i32 - output.pixel(x + 1, 26)[0] as i32).map(i32::abs).sum::<i32>();
        assert!(output_texture > original_texture / 2, "high-frequency texture must survive");
        assert_eq!(output.pixel(20, 20), frame.pixel(20, 20));
        assert_eq!(output.pixel(4, 4), frame.pixel(4, 4));
        assert_eq!(output.pixels.len(), frame.pixels.len());
    }

    #[test]
    fn malformed_landmarks_and_masks_are_refused() {
        let frame = synthetic_face();
        assert_eq!(
            build_pf5_dark_circle_output(&frame, &[], None, None),
            Err(Refusal::ShapeMismatch("points194"))
        );
        assert_eq!(
            build_pf90116_eye_bag_planes(&frame, &[], None, None),
            Err(Refusal::ShapeMismatch("points194"))
        );
        let mut points = points194();
        points[18] = [f32::NAN, 20.0];
        assert_eq!(
            build_pf5_dark_circle_output(&frame, &points, None, None),
            Err(Refusal::ValueOutOfRange("points194"))
        );
        assert_eq!(
            build_pf90116_eye_bag_planes(&frame, &points, None, None),
            Err(Refusal::ValueOutOfRange("points194"))
        );
        let points = points194();
        assert_eq!(
            build_pf5_dark_circle_output(&frame, &points, Some(&[0u8; 7]), None),
            Err(Refusal::ShapeMismatch("skin_mask"))
        );
        assert_eq!(
            build_pf90116_eye_bag_planes(&frame, &points, None, Some(&[0u8; 7])),
            Err(Refusal::ShapeMismatch("eye_mask"))
        );
    }
}
