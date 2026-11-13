//! PF30017 pixel law.
//!
//! Identity check first: UI key `poreEnhance` / PF 30017 is bound to the
//! `CollarBoneEnhance` producer (`0x7541`), not to a pore/noise filter. There
//! is **no shader, no noise texture and no frequency-domain consumer** in the
//! audited route: the pixel consumer is a Gaussian control-point warp plus a
//! masked model composite (R8 P01/P09 cores, RVA `0x8414390`).
//!
//! Ported from `CollarBoneEnhanceCore.Apply`. All model-owned surfaces stay
//! explicit: the model composite, the collarbone mask, the judge mask and the
//! control points. Nothing is generated here.
//!
//! Channels are RGBA per `Image`; the C# source is also RGBA here.

use crate::{Image, Refusal};

pub const PF: u32 = 30017;
pub const PF_KEY: u32 = 0x7541;
pub const UI_KEY: &str = "poreEnhance";
pub const EFFECT: &str = "CollarBoneEnhance";

pub const RVA_RENDER: u32 = 0x7DAD_FF0;
pub const RVA_PF_IMMEDIATE: u32 = 0x7DAE_07C;
pub const RVA_BIND: u32 = 0x4F1F_6B0;
pub const RVA_ACTIVE: u32 = 0x4F1F_7E0;
pub const RVA_MASK_CPU: u32 = 0x6056_AB0;
pub const RVA_APPLY: u32 = 0x842_0630;
pub const RVA_BUILD_MASK: u32 = 0x842_1B70;
pub const RVA_DISPATCH: u32 = 0x841_FEA0;
pub const RVA_CORE: u32 = 0x841_4390;
pub const RVA_STAGE2_AUG: u32 = 0x841_9840;
pub const RVA_STAGE3_ADD: u32 = 0x841_C820;

const EPSILON: f32 = 1e-8;
const WEIGHT_CUTOFF: f32 = 1e-5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Control {
    pub source: Point,
    pub target: Point,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// `clamp(ui/100, 0, 1)`; `active` is strictly above zero.
pub fn degree_from_ui(ui: f32) -> Result<f32, Refusal> {
    if !ui.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF30017_UI"));
    }
    Ok((ui / 100.0).clamp(0.0, 1.0))
}

fn check_image(image: &Image) -> Result<usize, Refusal> {
    if image.width == 0 || image.height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    let expected = (image.width as usize)
        .checked_mul(image.height as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or(Refusal::ShapeMismatch("image"))?;
    if image.pixels.len() != expected {
        return Err(Refusal::ShapeMismatch("image"));
    }
    Ok(expected / 4)
}

fn plane(values: &[u8], count: usize, name: &'static str) -> Result<(), Refusal> {
    if values.len() != count {
        Err(Refusal::ShapeMismatch(name))
    } else {
        Ok(())
    }
}

/// Bilinear gray sample at pixel coordinates, clamp-to-edge (C# `Sample`).
pub fn sample_gray(mask: &[u8], width: u32, height: u32, x: f32, y: f32) -> f32 {
    let max_x = (width - 1) as f32;
    let max_y = (height - 1) as f32;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let at = |px: u32, py: u32| mask[(py * width + px) as usize] as f32;
    let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
    let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
    (top + (bottom - top) * fy) / 255.0
}

fn sample_rgba(image: &Image, x: f32, y: f32) -> [f32; 4] {
    let max_x = (image.width - 1) as f32;
    let max_y = (image.height - 1) as f32;
    let x = x.clamp(0.0, max_x);
    let y = y.clamp(0.0, max_y);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width - 1);
    let y1 = (y0 + 1).min(image.height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0.0f32; 4];
    for channel in 0..4usize {
        let at = |px: u32, py: u32| {
            image.pixels[((py * image.width + px) * 4) as usize + channel] as f32
        };
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
        out[channel] = (top + (bottom - top) * fy) / 255.0;
    }
    out
}

/// `MathF.Round` default mode: half to even.
fn to_byte(value: f32) -> u8 {
    (value * 255.0).round_ties_even().clamp(0.0, 255.0) as u8
}

fn roi_valid(roi: Roi, width: u32, height: u32) -> bool {
    roi.width > 0
        && roi.height > 0
        && roi.x.checked_add(roi.width).map_or(false, |right| right <= width)
        && roi.y.checked_add(roi.height).map_or(false, |bottom| bottom <= height)
}

/// Gaussian point-pair warp + masked model composite.
///
/// `degree = clamp(ui/100,0,1)`; inactive when `degree == 0`, `!face_present`
/// or `!front_view` (the last two are live geometry gates).
pub fn apply(
    source: &Image,
    model: &Image,
    collarbone_mask: &[u8],
    judge_mask: &[u8],
    controls: &[Control],
    ui: f32,
    face_present: bool,
    front_view: bool,
    roi: Roi,
) -> Result<Image, Refusal> {
    let count = check_image(source)?;
    if source.width != model.width || source.height != model.height {
        return Err(Refusal::ShapeMismatch("PF30017_MODEL_COMPOSITE"));
    }
    check_image(model)?;
    plane(collarbone_mask, count, "PF30017_COLLARBONE_MASK")?;
    plane(judge_mask, count, "PF30017_JUDGE_MASK")?;
    if !roi_valid(roi, source.width, source.height) {
        return Err(Refusal::ShapeMismatch("PF30017_ROI"));
    }
    for control in controls {
        if !control.source.x.is_finite()
            || !control.source.y.is_finite()
            || !control.target.x.is_finite()
            || !control.target.y.is_finite()
            || !control.radius.is_finite()
            || control.radius < 0.0
        {
            return Err(Refusal::ValueOutOfRange("PF30017_CONTROL"));
        }
    }

    let degree = degree_from_ui(ui)?;
    let mut out = source.clone();
    if degree <= 0.0 || !face_present || !front_view {
        return Ok(out);
    }
    let active: Vec<&Control> = controls
        .iter()
        .filter(|control| control.radius > EPSILON)
        .collect();

    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let px = x as f32;
            let py = y as f32;
            let mut dx = 0.0f32;
            let mut dy = 0.0f32;
            let mut weight_sum = 0.0f32;
            for control in &active {
                let vx = px - control.source.x;
                let vy = py - control.source.y;
                let sigma2 = 2.0 * control.radius * control.radius;
                let weight = (-(vx * vx + vy * vy) / sigma2).exp();
                if weight < WEIGHT_CUTOFF {
                    continue;
                }
                dx += weight * (control.target.x - control.source.x);
                dy += weight * (control.target.y - control.source.y);
                weight_sum += weight;
            }

            let mask = sample_gray(collarbone_mask, source.width, source.height, px, py);
            let judge = sample_gray(judge_mask, source.width, source.height, px, py);
            if weight_sum > EPSILON {
                dx = degree * mask * dx / weight_sum;
                dy = degree * mask * dy / weight_sum;
            }

            let sampled = sample_rgba(source, px + dx, py + dy);
            let offset = (y * source.width + x) as usize * 4;
            let model_alpha = model.pixels[offset + 3] as f32 / 255.0;
            let blend_alpha = (degree * mask * judge * model_alpha).clamp(0.0, 1.0);

            for channel in 0..3 {
                let value = model.pixels[offset + channel] as f32 / 255.0;
                out.pixels[offset + channel] =
                    to_byte(sampled[channel] + (value - sampled[channel]) * blend_alpha);
            }
            out.pixels[offset + 3] = source.pixels[offset + 3];
        }
    }
    Ok(out)
}

/// Integer ROI from an inclusive pixel box, clamped to the frame.
pub fn roi_from_bounds(width: u32, height: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> Option<Roi> {
    if width == 0 || height == 0 || x1 >= width || y1 >= height || x1 < x0 || y1 < y0 {
        return None;
    }
    Some(Roi {
        x: x0,
        y: y0,
        width: x1 - x0 + 1,
        height: y1 - y0 + 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Image, Image, [u8; 4], [u8; 4]) {
        let source = Image::new(
            2,
            2,
            vec![
                10, 20, 30, 255, 110, 120, 130, 255, 210, 220, 230, 255, 40, 50, 60, 255,
            ],
        )
        .unwrap();
        let model = Image::new(
            2,
            2,
            vec![
                110, 120, 130, 255, 210, 220, 230, 255, 40, 50, 60, 255, 10, 20, 30, 255,
            ],
        )
        .unwrap();
        (source, model, [255, 0, 0, 0], [255; 4])
    }

    #[test]
    fn degree_is_clamped_unit() {
        assert_eq!(degree_from_ui(0.0).unwrap(), 0.0);
        assert_eq!(degree_from_ui(50.0).unwrap(), 0.5);
        assert_eq!(degree_from_ui(150.0).unwrap(), 1.0);
        assert_eq!(degree_from_ui(-5.0).unwrap(), 0.0);
        assert_eq!(
            degree_from_ui(f32::NAN),
            Err(Refusal::ValueOutOfRange("PF30017_UI"))
        );
    }

    #[test]
    fn golden_first_pixel_from_evidence() {
        let (source, model, mask, judge) = fixture();
        let out = apply(
            &source,
            &model,
            &mask,
            &judge,
            &[Control {
                source: Point { x: 0.0, y: 0.0 },
                target: Point { x: 1.0, y: 0.0 },
                radius: 4.0,
            }],
            50.0,
            true,
            true,
            Roi { x: 0, y: 0, width: 2, height: 2 },
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [85, 95, 105, 255]);
        // mask 0 -> identity for that pixel
        assert_eq!(out.pixel(1, 0), [110, 120, 130, 255]);
    }

    #[test]
    fn gates_face_present_and_front_view() {
        let (source, model, mask, judge) = fixture();
        let control = [Control {
            source: Point { x: 0.0, y: 0.0 },
            target: Point { x: 1.0, y: 0.0 },
            radius: 4.0,
        }];
        let roi = Roi { x: 0, y: 0, width: 2, height: 2 };
        assert_eq!(
            apply(&source, &model, &mask, &judge, &control, 50.0, false, true, roi).unwrap(),
            source
        );
        assert_eq!(
            apply(&source, &model, &mask, &judge, &control, 50.0, true, false, roi).unwrap(),
            source
        );
        assert_eq!(
            apply(&source, &model, &mask, &judge, &control, 0.0, true, true, roi).unwrap(),
            source
        );
    }

    #[test]
    fn zero_radius_control_is_ignored() {
        let (source, model, mask, judge) = fixture();
        let out = apply(
            &source,
            &model,
            &mask,
            &judge,
            &[Control {
                source: Point { x: 0.0, y: 0.0 },
                target: Point { x: 100.0, y: 100.0 },
                radius: 0.0,
            }],
            100.0,
            true,
            true,
            Roi { x: 0, y: 0, width: 2, height: 2 },
        )
        .unwrap();
        // no displacement; mask 255 makes the first pixel the model colour
        assert_eq!(out.pixel(0, 0), [110, 120, 130, 255]);
    }

    #[test]
    fn alpha_is_source_alpha() {
        let source = Image::solid(1, 1, [10, 20, 30, 42]);
        let model = Image::solid(1, 1, [200, 100, 50, 255]);
        let out = apply(
            &source,
            &model,
            &[255],
            &[255],
            &[],
            100.0,
            true,
            true,
            Roi { x: 0, y: 0, width: 1, height: 1 },
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0)[3], 42);
        // no active control -> no displacement, but the composite still applies
        assert_eq!(out.pixel(0, 0), [200, 100, 50, 42]);
    }

    #[test]
    fn roi_outside_frame_refused() {
        let (source, model, mask, judge) = fixture();
        assert_eq!(
            apply(
                &source,
                &model,
                &mask,
                &judge,
                &[],
                100.0,
                true,
                true,
                Roi { x: 1, y: 0, width: 2, height: 1 },
            ),
            Err(Refusal::ShapeMismatch("PF30017_ROI"))
        );
    }

    #[test]
    fn mask_shape_and_control_finiteness_refused() {
        let (source, model, _, judge) = fixture();
        let roi = Roi { x: 0, y: 0, width: 2, height: 2 };
        assert_eq!(
            apply(&source, &model, &[0u8; 3], &judge, &[], 100.0, true, true, roi),
            Err(Refusal::ShapeMismatch("PF30017_COLLARBONE_MASK"))
        );
        assert_eq!(
            apply(
                &source,
                &model,
                &[0u8; 4],
                &judge,
                &[Control { source: Point { x: f32::NAN, y: 0.0 }, target: Point { x: 0.0, y: 0.0 }, radius: 1.0 }],
                100.0,
                true,
                true,
                roi,
            ),
            Err(Refusal::ValueOutOfRange("PF30017_CONTROL"))
        );
    }
}
/// UI key says `poreEnhance`, but PF30017 is bound to the collarbone enhance
/// producer; the label records that correction.
pub const META: &[crate::EffectMeta] = &[crate::EffectMeta {
    pf: 30017,
    key: "poreEnhance",
    label: "Collar bone enhance (UI key poreEnhance)",
    access: crate::Access::Cpu,
    inputs: &["source", "model composite", "collarbone mask", "judge mask", "control points", "ui"],
}];
