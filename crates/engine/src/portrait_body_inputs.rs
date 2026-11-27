//! Same-frame body segmentation and per-instance pose evidence.
//!
//! This module is deliberately narrow. It uses the shipped subject-probability
//! model for segmentation and the installed `BodyPoseModel` for semantics; it
//! never treats a face/body-skin plane, the whole image, a detector box, or an
//! unknown layout as a person mask. The baseline model is an externally supplied
//! Capture One-compatible model discovered by [`BodyPoseModel::from_local`], not
//! a self-trained weight, and this module adds no third-party files or deps.
//!
//! Image size is bounded before model work: the source is area-downscaled to a
//! resolution-limited analysis frame, the subject model runs there, and every
//! component is mapped back to the full frame with the same center/scale. Pose
//! runs on the same downscaled crop, so anchors are local normalized coordinates
//! of the subject rect and the mask is a local `width * height` probability
//! plane. No model runs on a 24MP tensor and the segmentation plane stays at
//! analysis resolution; only a returned subject's local mask can be full-rect
//! sized.

use std::cmp::Reverse;

use aurora_portrait::kernels::body_pose_adapter::AnchorSet;
use aurora_portrait::{Image, Refusal};

use crate::aimask::body_pose::BodyPoseModel;

pub(crate) const BODY_SUBJECT_SOURCE: &str = "body subject source";
pub(crate) const BODY_SUBJECT_MODEL: &str = "body subject segmentation model";
pub(crate) const BODY_SUBJECT_EMPTY: &str = "body subject";
pub(crate) const BODY_POSE_MODEL: &str = "body pose model";
pub(crate) const BODY_POSE_INFERENCE: &str = "body pose inference";
pub(crate) const BODY_POSE_ANCHORS: &str = "body pose anchors";

/// Longest side used for the subject-probability and pose crops. The shipped
/// subject model is 320x320 and the pose model is much smaller than this; 1024
/// keeps small people separable without ever building a 24MP f32 plane.
const ANALYSIS_MAX_SIDE: u32 = 1024;
/// The subject model emits a probability in `0..=1`; this is a threshold, not a
/// substitute for the probability map.
const SUBJECT_THRESHOLD: f32 = 0.5;
const POSE_MIN_SCORE: f32 = 0.3;

/// One isolated person in full-frame coordinates.
#[derive(Clone, Debug)]
pub(crate) struct BodySubject {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Semantic anchors in this subject rect's local normalized coordinates.
    pub anchors: AnchorSet,
    /// Subject probability in this subject rect's local pixel coordinates,
    /// row-major, `width * height` values. It is not a full-frame plane.
    pub mask: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug)]
struct Component {
    root: u32,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    area: usize,
}

impl Component {
    fn width(&self) -> usize {
        self.max_x - self.min_x + 1
    }

    fn height(&self) -> usize {
        self.max_y - self.min_y + 1
    }
}

/// Segments this frame and returns one entry per verifiably separate person.
///
/// A missing subject model, a non-person frame, a missing pose model, or a pose
/// with no semantic anchors is a typed [`Refusal`]. Components whose boxes
/// overlap an already accepted subject are dropped conservatively; the masks
/// themselves are connected-component planes, never boxes.
pub(crate) fn infer_subjects(source: &Image) -> Result<Vec<BodySubject>, Refusal> {
    source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch(BODY_SUBJECT_SOURCE))?;
    let analysis = bounded_analysis(source)?;
    let probability = crate::aimask::subject_mask(
        &analysis.pixels,
        analysis.width,
        analysis.height,
    )
    .map_err(|_| Refusal::MissingInput(BODY_SUBJECT_MODEL))?;
    let expected = analysis.width as usize * analysis.height as usize;
    if probability.len() != expected {
        return Err(Refusal::ShapeMismatch(BODY_SUBJECT_MODEL));
    }
    let (components, labels) =
        connected_components(&probability, analysis.width, analysis.height);
    if components.is_empty() {
        return Err(Refusal::MissingInput(BODY_SUBJECT_EMPTY));
    }
    let components = select_non_overlapping(components);
    if components.is_empty() {
        return Err(Refusal::MissingInput(BODY_SUBJECT_EMPTY));
    }

    // Pose is loaded only after a person exists; a blank frame must report "no
    // person", not a missing pose model.
    let pose = BodyPoseModel::from_local().map_err(|_| Refusal::MissingInput(BODY_POSE_MODEL))?;
    let mut subjects = Vec::with_capacity(components.len());
    for component in components {
        let rect = map_rect(
            &component,
            analysis.width,
            analysis.height,
            source.width,
            source.height,
        )?;
        let mask = local_mask(
            &probability,
            &labels,
            &component,
            analysis.width,
            analysis.height,
            rect,
        )?;
        let anchors = pose_for_component(&pose, &analysis, &component)?;
        subjects.push(BodySubject {
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height,
            anchors,
            mask,
        });
    }
    Ok(subjects)
}

/// Area-averages the source to at most [`ANALYSIS_MAX_SIDE`]. The returned frame
/// and the source share the same center; `map_rect` uses that scale.
fn bounded_analysis(source: &Image) -> Result<Image, Refusal> {
    let max_side = source.width.max(source.height);
    if max_side <= ANALYSIS_MAX_SIDE {
        return Ok(source.clone());
    }
    let scale = ANALYSIS_MAX_SIDE as f32 / max_side as f32;
    let width = ((source.width as f32 * scale).round() as u32).max(1);
    let height = ((source.height as f32 * scale).round() as u32).max(1);
    let mut pixels = vec![0u8; width as usize * height as usize * 4];
    for y in 0..height {
        let sy0 = (y as u64 * source.height as u64 / height as u64) as u32;
        let sy1 = (((y as u64 + 1) * source.height as u64 / height as u64) as u32)
            .max(sy0 + 1)
            .min(source.height);
        for x in 0..width {
            let sx0 = (x as u64 * source.width as u64 / width as u64) as u32;
            let sx1 = (((x as u64 + 1) * source.width as u64 / width as u64) as u32)
                .max(sx0 + 1)
                .min(source.width);
            let mut sum = [0u64; 4];
            let mut count = 0u64;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let at = ((sy as usize * source.width as usize + sx as usize) * 4) as usize;
                    for channel in 0..4 {
                        sum[channel] += source.pixels[at + channel] as u64;
                    }
                    count += 1;
                }
            }
            let out = ((y as usize * width as usize + x as usize) * 4) as usize;
            for channel in 0..4 {
                pixels[out + channel] = (sum[channel] / count.max(1)) as u8;
            }
        }
    }
    Image::new(width, height, pixels).ok_or(Refusal::ShapeMismatch(BODY_SUBJECT_SOURCE))
}

fn connected_components(
    probability: &[f32],
    width: u32,
    height: u32,
) -> (Vec<Component>, Vec<u32>) {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 || probability.len() != width * height {
        return (Vec::new(), Vec::new());
    }
    let mut labels = vec![0u32; width * height];
    let mut parent = vec![0u32];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if probability[index] <= SUBJECT_THRESHOLD {
                continue;
            }
            let mut label = 0u32;
            if x > 0 && labels[index - 1] != 0 {
                label = find(&mut parent, labels[index - 1]);
            }
            if y > 0 && labels[index - width] != 0 {
                let up = find(&mut parent, labels[index - width]);
                if label == 0 {
                    label = up;
                } else {
                    union(&mut parent, label, up);
                    label = find(&mut parent, label);
                }
            }
            if label == 0 {
                label = parent.len() as u32;
                parent.push(label);
            }
            labels[index] = label;
        }
    }
    for label in &mut labels {
        if *label != 0 {
            *label = find(&mut parent, *label);
        }
    }
    let mut components: Vec<Component> = Vec::new();
    let mut slots = vec![u32::MAX; parent.len()];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let root = labels[index];
            if root == 0 {
                continue;
            }
            let slot = slots[root as usize];
            let component = if slot == u32::MAX {
                let slot = components.len() as u32;
                slots[root as usize] = slot;
                components.push(Component {
                    root,
                    min_x: x,
                    min_y: y,
                    max_x: x,
                    max_y: y,
                    area: 0,
                });
                &mut components[slot as usize]
            } else {
                &mut components[slot as usize]
            };
            component.min_x = component.min_x.min(x);
            component.min_y = component.min_y.min(y);
            component.max_x = component.max_x.max(x);
            component.max_y = component.max_y.max(y);
            component.area += 1;
        }
    }
    (components, labels)
}

fn find(parent: &mut [u32], mut index: u32) -> u32 {
    while parent[index as usize] != index {
        let grand = parent[parent[index as usize] as usize];
        parent[index as usize] = grand;
        index = grand;
    }
    index
}

fn union(parent: &mut [u32], left: u32, right: u32) {
    let left = find(parent, left);
    let right = find(parent, right);
    if left != right {
        parent[right as usize] = left;
    }
}

/// Keeps the largest component first and drops any component whose envelope
/// overlaps an accepted one. The component masks are still disjoint; this is the
/// conservative guard against treating two people as one blended subject.
fn select_non_overlapping(mut components: Vec<Component>) -> Vec<Component> {
    components.sort_by_key(|component| Reverse(component.area));
    let mut kept: Vec<Component> = Vec::new();
    for component in components {
        if kept.iter().all(|kept| !overlap(kept, &component)) {
            kept.push(component);
        }
    }
    kept.sort_by_key(|component| (component.min_y, component.min_x));
    kept
}

fn overlap(left: &Component, right: &Component) -> bool {
    left.min_x <= right.max_x
        && right.min_x <= left.max_x
        && left.min_y <= right.max_y
        && right.min_y <= left.max_y
}

fn map_rect(
    component: &Component,
    analysis_width: u32,
    analysis_height: u32,
    source_width: u32,
    source_height: u32,
) -> Result<Rect, Refusal> {
    if analysis_width == 0 || analysis_height == 0 || source_width == 0 || source_height == 0 {
        return Err(Refusal::ShapeMismatch(BODY_SUBJECT_SOURCE));
    }
    // Match bounded_analysis's half-open source cells exactly. A common
    // rounded edge stays common in two adjacent ROIs; ceil/floor on opposite
    // sides would falsely overlap those owners after an odd-ratio downscale.
    let edge = |i: usize, source: u32, analysis: u32| ((i as u64 * source as u64) / analysis as u64) as u32;
    let x0 = edge(component.min_x, source_width, analysis_width);
    let y0 = edge(component.min_y, source_height, analysis_height);
    let x1 = edge(component.max_x + 1, source_width, analysis_width);
    let y1 = edge(component.max_y + 1, source_height, analysis_height);
    Ok(Rect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

/// Upsamples only this component's connected pixels into the subject rect. The
/// local mask is a probability plane, not a filled box.
fn local_mask(
    probability: &[f32],
    labels: &[u32],
    component: &Component,
    analysis_width: u32,
    analysis_height: u32,
    rect: Rect,
) -> Result<Vec<f32>, Refusal> {
    if analysis_width == 0
        || analysis_height == 0
        || probability.len() != analysis_width as usize * analysis_height as usize
        || labels.len() != probability.len()
    {
        return Err(Refusal::ShapeMismatch(BODY_SUBJECT_MODEL));
    }
    let mut mask = vec![0.0f32; rect.width as usize * rect.height as usize];
    let scale_x = rect.width as f32 / component.width() as f32;
    let scale_y = rect.height as f32 / component.height() as f32;
    for sy in component.min_y..=component.max_y {
        for sx in component.min_x..=component.max_x {
            let index = sy * analysis_width as usize + sx;
            if probability[index] <= SUBJECT_THRESHOLD || labels[index] != component.root {
                continue;
            }
            let local_x = (sx - component.min_x) as f32;
            let local_y = (sy - component.min_y) as f32;
            let fx0 = ((local_x * scale_x).floor() as u32).min(rect.width - 1);
            let fx1 = (((local_x + 1.0) * scale_x).ceil() as u32)
                .max(fx0 + 1)
                .min(rect.width);
            let fy0 = ((local_y * scale_y).floor() as u32).min(rect.height - 1);
            let fy1 = (((local_y + 1.0) * scale_y).ceil() as u32)
                .max(fy0 + 1)
                .min(rect.height);
            for fy in fy0..fy1 {
                let row = fy as usize * rect.width as usize;
                for fx in fx0..fx1 {
                    mask[row + fx as usize] = probability[index];
                }
            }
        }
    }
    Ok(mask)
}

fn pose_for_component(
    pose: &BodyPoseModel,
    analysis: &Image,
    component: &Component,
) -> Result<AnchorSet, Refusal> {
    let rect = Rect {
        x: component.min_x as u32,
        y: component.min_y as u32,
        width: component.width() as u32,
        height: component.height() as u32,
    };
    let crop = crop_image(analysis, rect)?;
    let inference = pose
        .infer(&crop)
        .map_err(|_| Refusal::MissingInput(BODY_POSE_INFERENCE))?;
    let anchors = inference.semantic_anchors(POSE_MIN_SCORE)?;
    if anchors.present_count() == 0 {
        return Err(Refusal::MissingInput(BODY_POSE_ANCHORS));
    }
    Ok(anchors)
}

fn crop_image(source: &Image, rect: Rect) -> Result<Image, Refusal> {
    if rect.width == 0 || rect.height == 0 || rect.x + rect.width > source.width || rect.y + rect.height > source.height {
        return Err(Refusal::ShapeMismatch(BODY_SUBJECT_SOURCE));
    }
    let row_bytes = rect.width as usize * 4;
    let mut pixels = Vec::with_capacity(row_bytes * rect.height as usize);
    for y in rect.y..rect.y + rect.height {
        let start = ((y as usize * source.width as usize + rect.x as usize) * 4) as usize;
        pixels.extend_from_slice(&source.pixels[start..start + row_bytes]);
    }
    Image::new(rect.width, rect.height, pixels).ok_or(Refusal::ShapeMismatch(BODY_SUBJECT_SOURCE))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component(root: u32, min_x: usize, min_y: usize, max_x: usize, max_y: usize, area: usize) -> Component {
        Component { root, min_x, min_y, max_x, max_y, area }
    }

    #[test]
    fn overlapping_subject_boxes_are_dropped_conservatively() {
        let components = vec![
            component(1, 0, 0, 4, 4, 20),
            component(2, 3, 3, 8, 8, 12),
            component(3, 9, 0, 12, 3, 8),
        ];
        let kept = select_non_overlapping(components);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|component| component.root == 1));
        assert!(kept.iter().any(|component| component.root == 3));
        assert!(!kept.iter().any(|component| component.root == 2));
    }

    #[test]
    fn components_and_local_masks_are_disjoint_probability_planes() {
        let (width, height) = (8usize, 6usize);
        let mut probability = vec![0.0f32; width * height];
        for y in 1..4 {
            for x in 1..4 {
                probability[y * width + x] = 0.9;
            }
        }
        probability[2 * width + 6] = 0.8;
        let (components, labels) =
            connected_components(&probability, width as u32, height as u32);
        assert_eq!(components.len(), 2);
        let kept = select_non_overlapping(components);
        assert_eq!(kept.len(), 2);
        let rect = map_rect(&kept[0], width as u32, height as u32, 80, 60).expect("rect");
        let mask = local_mask(
            &probability,
            &labels,
            &kept[0],
            width as u32,
            height as u32,
            rect,
        )
        .expect("local mask");
        assert_eq!(mask.len(), rect.width as usize * rect.height as usize);
        assert!(mask.iter().any(|value| *value > 0.0));
        assert!(mask.iter().all(|value| *value <= 1.0));
    }

    #[test]
    fn invalid_source_is_typed_before_any_model_runs() {
        let source = Image {
            width: 2,
            height: 2,
            pixels: Vec::new(),
        };
        let error = infer_subjects(&source).unwrap_err();
        assert_eq!(error, Refusal::ShapeMismatch(BODY_SUBJECT_SOURCE));
    }
}
