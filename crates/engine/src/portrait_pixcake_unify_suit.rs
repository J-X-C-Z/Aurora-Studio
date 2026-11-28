//! Same-frame PF30059 / PF90033 producers for the remaining two PixCake
//! combined-effect consumers.
//!
//! Both producers only prepare owned inputs. The coordinator mounts this
//! module in `portrait_stage.rs` and wires the outputs into
//! `ProducerInputs.historical_tone` / `ProducerInputs.suit_passes`.
//!
//! PF30059 returns `HistoricalToneRowData::MultiTone` with one pass per skin
//! connected component. Targets are measured from the current frame; the face
//! region (when supplied) becomes the reference, and body components are
//! aligned to its chroma/luma. A bounded reciprocal 3x4 matrix pulls each
//! component toward the shared target before the kernel's explicit-target
//! blend. No 3D LUT is claimed: `lut` stays `None`.
//!
//! PF90033 builds an ordered natural-theme pass list from the checked-in
//! `private-assets/portrait/suit/presets.json`, restricted to the first
//! executable pass of each supported effect family (base/blush, brow, eye
//! shadow, lashes, lipstick). Geometry, owner mask and both UV streams are
//! generated from the supplied 194 landmarks and face mask using the same
//! region meshes as the 2D makeup bundle. `pc:90033` is the only strength
//! input; a missing mask or zero/non-finite degree is refused by name.

use std::collections::BTreeMap;

use aurora_portrait::kernels::makeup_material::{MakeupLayer, MaskChannel};
use aurora_portrait::kernels::makeup_mask_uv::{produce_mesh_mask_uv, MeshMaskUvInputs};
use aurora_portrait::{Image, Refusal};

use crate::portrait_historical_tone_provider::{
    HistoricalToneMultiTonePayload, HistoricalToneRowData,
};
use crate::portrait_makeup_bundle::{
    load_owned_suit_bundle, row_profile, suit_pass_kind_for_effect, OwnedSuitPass,
};
use crate::portrait_makeup_mesh2d::region_mesh;
use crate::portrait_stage::{HistoricalToneFace, SuitPassInput, SuitPassKindInput};

const PF30059_SKIN_MASK: &str = "PF30059 skin mask";
const PF30059_FACE_MASK: &str = "PF30059 face mask";
const PF30059_FRAME: &str = "PF30059 frame";
const PF30059_PLANES: &str = "PF30059 producer planes";
const PF30059_JOB: &str = "PF30059 job id";
const PF90033_POINTS: &str = "PF90033 points194";
const PF90033_FRAME: &str = "PF90033 frame";
const PF90033_MASK: &str = "PF90033 face mask";
const PF90033_DEGREE: &str = "PF90033 degree";
const PF90033_PRESETS: &str = "PF90033 suit presets";
const LUMA: [f32; 3] = [0.299, 0.587, 0.114];

fn quantise(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round_ties_even() as u8
}

fn pixel(frame: &Image, index: usize) -> [f32; 3] {
    [
        frame.pixels[index * 4] as f32,
        frame.pixels[index * 4 + 1] as f32,
        frame.pixels[index * 4 + 2] as f32,
    ]
}

fn luma(rgb: [f32; 3]) -> f32 {
    LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2]
}

/// Brightness plane for `tone_unify::mean_masked_byte`: masked pixels carry
/// their source luma in the same byte domain as the frame.
fn brightness_plane(frame: &Image, mask: &[u8]) -> Vec<u8> {
    (0..frame.pixel_count())
        .map(|index| {
            if mask[index] == 0 {
                0
            } else {
                quantise(luma(pixel(frame, index)) / 255.0)
            }
        })
        .collect()
}

fn median_channel(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) * 0.5
    }
}

/// Weighted median chroma and weighted mean luma for one skin component.
fn region_target(frame: &Image, mask: &[u8]) -> Option<[u8; 3]> {
    let (mut red, mut green, mut blue) = (Vec::new(), Vec::new(), Vec::new());
    let (mut luma_sum, mut weight_sum) = (0f64, 0f64);
    for index in 0..frame.pixel_count() {
        let weight = mask[index] as f64;
        if weight == 0.0 || mask[index] < 128 {
            continue;
        }
        let rgb = pixel(frame, index);
        red.push(rgb[0]);
        green.push(rgb[1]);
        blue.push(rgb[2]);
        luma_sum += f64::from(luma(rgb)) * weight;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 || red.is_empty() {
        return None;
    }
    let wanted = (luma_sum / weight_sum) as f32;
    let median = [
        median_channel(red),
        median_channel(green),
        median_channel(blue),
    ];
    let scale = if luma(median) > 1.0 {
        wanted / luma(median)
    } else {
        1.0
    };
    Some([
        quantise(median[0] * scale / 255.0),
        quantise(median[1] * scale / 255.0),
        quantise(median[2] * scale / 255.0),
    ])
}

/// Four-connected components of a u8 coverage mask via iterative flood fill.
fn components(mask: &[u8], width: usize, height: usize) -> Vec<Vec<u8>> {
    let pixels = width * height;
    let mut visited = vec![false; pixels];
    let mut regions = Vec::new();
    for start in 0..pixels {
        if visited[start] || mask[start] == 0 {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut region = vec![0u8; pixels];
        while let Some(index) = stack.pop() {
            region[index] = mask[index];
            let (x, y) = (index % width, index / width);
            let neighbours = [
                (x + 1 < width).then_some(index + 1),
                (x > 0).then_some(index - 1),
                (y + 1 < height).then_some(index + width),
                (y > 0).then_some(index - width),
            ];
            for next in neighbours.into_iter().flatten() {
                if mask[next] > 0 && !visited[next] {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
        regions.push(region);
    }
    regions
}

fn area(mask: &[u8]) -> usize {
    mask.iter().filter(|value| **value > 0).count()
}

/// A reciprocal matrix that moves `source` halfway to `target`; the kernel's
/// explicit-target stage completes the blend. Weight is proportional to the
/// luma/chroma difference and capped so no region is pulled harder than the
/// measured target warrants.
fn transform_matrix(source: [u8; 3], target: [u8; 3]) -> ([f32; 12], f32) {
    let src = [
        source[0] as f32 / 255.0,
        source[1] as f32 / 255.0,
        source[2] as f32 / 255.0,
    ];
    let dst = [
        target[0] as f32 / 255.0,
        target[1] as f32 / 255.0,
        target[2] as f32 / 255.0,
    ];
    let delta = ((luma(dst) - luma(src)).abs()
        + 0.5 * ((dst[0] - src[0]) - (dst[2] - src[2])).abs())
    .clamp(0.0, 1.0);
    let factor = (delta * 0.5).clamp(0.0, 0.5);
    let mut matrix = [0f32; 12];
    for channel in 0..3 {
        let scale = 1.0 - factor;
        matrix[channel * 4 + channel] = scale;
        matrix[channel * 4 + 3] = dst[channel] - src[channel] * scale;
    }
    (matrix, factor)
}

#[allow(clippy::too_many_arguments)]
fn face_pass(
    id: i32,
    reference: Option<i32>,
    mask: Vec<u8>,
    brightness: Vec<u8>,
    instance: Vec<u8>,
    target: [u8; 3],
    source: [u8; 3],
) -> HistoricalToneFace {
    let skin = mask.clone();
    let zeros = vec![0u8; mask.len()];
    let (matrix, weight) = transform_matrix(source, target);
    HistoricalToneFace {
        group_id: 1,
        face_id: id,
        reference,
        target: Some(target),
        brightness,
        skin,
        face: mask,
        tooth: zeros,
        instance,
        tooth_alpha: 0.0,
        brightness_x: vec![0.0, 255.0],
        brightness_y: vec![1.0, 1.0],
        matrices: if weight > 0.0 { vec![(matrix, weight)] } else { Vec::new() },
        lut: None,
    }
}

/// Builds the owned PF30059 `MultiTone` row payload for the current frame.
///
/// `job_id` is the renderer ticket; zero is refused as a missing identity. The
/// stage re-derives ticket, digest and control, so this producer only owns the
/// pixel planes. Each connected skin component becomes one pass, and all passes
/// share the single group target selected from the face region (or from the
/// largest skin component when no face mask is supplied).
pub(crate) fn build_pf30059_multi_tone_row(
    job_id: u64,
    frame: &Image,
    skin_mask: &[u8],
    face_mask: Option<&[u8]>,
) -> Result<HistoricalToneRowData, Refusal> {
    frame.validate().map_err(|_| Refusal::ShapeMismatch(PF30059_FRAME))?;
    let pixels = frame.pixel_count();
    if skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch(PF30059_SKIN_MASK));
    }
    if !skin_mask.iter().any(|value| *value > 0) {
        return Err(Refusal::MissingInput(PF30059_SKIN_MASK));
    }
    if let Some(face) = face_mask {
        if face.len() != pixels {
            return Err(Refusal::ShapeMismatch(PF30059_FACE_MASK));
        }
    }
    if job_id == 0 {
        return Err(Refusal::ValueOutOfRange(PF30059_JOB));
    }

    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut body = components(skin_mask, width, height);
    body.sort_by_key(|region| std::cmp::Reverse(area(region)));
    let body_target = region_target(frame, &body[0]).ok_or(Refusal::MissingInput(PF30059_PLANES))?;
    let mut face_regions = face_mask.map(|mask| components(mask, width, height)).unwrap_or_default();
    face_regions.retain(|region| area(region) > 0);
    face_regions.sort_by_key(|region| std::cmp::Reverse(area(region)));
    let has_face = !face_regions.is_empty();
    let group_target = if has_face {
        region_target(frame, &face_regions[0]).unwrap_or(body_target)
    } else {
        body_target
    };

    let ones = vec![255u8; pixels];
    let mut faces = Vec::new();
    let mut id = 1i32;
    if has_face {
        for region in &face_regions {
            let Some(target) = region_target(frame, region) else { continue };
            let instance = region.iter().map(|&v| if v > 0 { 255 } else { 0 }).collect();
            faces.push(face_pass(
                id,
                None,
                region.clone(),
                brightness_plane(frame, region),
                instance,
                group_target,
                target,
            ));
            id += 1;
        }
    }
    for region in &body {
        if let Some(target) = region_target(frame, region) {
            faces.push(face_pass(
                id,
                None,
                region.clone(),
                brightness_plane(frame, region),
                ones.clone(),
                group_target,
                target,
            ));
            id += 1;
        }
    }
    if faces.is_empty() {
        return Err(Refusal::MissingInput(PF30059_PLANES));
    }
    Ok(HistoricalToneRowData::MultiTone(HistoricalToneMultiTonePayload { faces }))
}

/// Mask channel from the same pinned selector used by the stage producer.
fn mask_channel(pass: &OwnedSuitPass) -> Result<MaskChannel, Refusal> {
    let field = pass.mask_field.as_deref().ok_or(Refusal::MissingInput("suit mask"))?;
    let value = pass.mask_value.ok_or(Refusal::MissingInput("suit mask"))?;
    Ok(match (field, value) {
        ("BlusherMaskType", 15) => MaskChannel::Rgba,
        ("EyeBrowMaskType", 5) => MaskChannel::RB,
        ("EyeMaskType", 10) => MaskChannel::Rgb,
        ("EyeMaskType", 4) | ("EyeLashMaskType", 4) => MaskChannel::RG,
        ("LipMaskType", 3) => MaskChannel::R,
        _ => return Err(Refusal::MissingInput("suit mask channel")),
    })
}

fn theme_effect(pass: &OwnedSuitPass) -> bool {
    pass.pf == 90033
        && matches!(
            pass.effect.as_str(),
            "BlusherImageMakeup"
                | "NormalEyeBrowImageMakeup"
                | "SymEyeImageMakeup"
                | "SymEyeLashImageMakeup"
                | "LipStickImageMakeup"
        )
}

fn rank(effect: &str) -> u8 {
    match effect {
        "BlusherImageMakeup" => 0,
        "NormalEyeBrowImageMakeup" => 1,
        "SymEyeImageMakeup" => 2,
        "SymEyeLashImageMakeup" => 3,
        "LipStickImageMakeup" => 4,
        _ => 5,
    }
}

fn profile_pf(effect: &str) -> u32 {
    match effect {
        "BlusherImageMakeup" => 116,
        "NormalEyeBrowImageMakeup" => 100,
        "SymEyeImageMakeup" => 90032,
        "SymEyeLashImageMakeup" => 118,
        "LipStickImageMakeup" => 107,
        _ => 0,
    }
}

fn point(points: &[[f32; 2]], index: usize) -> Option<[f32; 2]> {
    points
        .get(index)
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
}

fn center_for(effect: &str, points: &[[f32; 2]], fallback: [f32; 2]) -> [f32; 2] {
    let candidates: &[usize] = if effect.contains("Blusher") {
        &[114, 115, 132, 133]
    } else if effect.contains("EyeBrow") {
        &[0, 4, 8, 9, 13, 17]
    } else if effect.contains("Eye") {
        &[18, 22, 26, 36, 40, 44]
    } else {
        &[73, 78, 83, 89]
    };
    let mut sum = [0f32; 2];
    let mut count = 0usize;
    for index in candidates {
        if let Some(p) = point(points, *index) {
            sum[0] += p[0];
            sum[1] += p[1];
            count += 1;
        }
    }
    if count == 0 { fallback } else { [sum[0] / count as f32, sum[1] / count as f32] }
}

fn falloff(distance2: f32) -> f32 {
    let t = (1.0 - distance2).clamp(0.0, 1.0);
    t * t
}

/// Produces owner-mask RGBA from the same-job face mask restricted to the
/// mesh coverage and a smooth anatomical falloff. All channels carry the same
/// value so every pinned `MaskChannel` sees identical coverage.
fn owner_mask(
    frame: &Image,
    face_mask: &[u8],
    coverage: &[u16],
    center: [f32; 2],
    radius: f32,
) -> Image {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut pixels = vec![0u8; frame.pixel_count() * 4];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if coverage[index] == 0 || face_mask[index] == 0 {
                continue;
            }
            let dx = (x as f32 + 0.5 - center[0]) / radius;
            let dy = (y as f32 + 0.5 - center[1]) / radius;
            let value = quantise(
                face_mask[index] as f32 / 255.0 * falloff(dx * dx + dy * dy),
            );
            if value == 0 {
                continue;
            }
            pixels[index * 4..index * 4 + 4].copy_from_slice(&[value, value, value, value]);
        }
    }
    Image::new(frame.width, frame.height, pixels).expect("owner mask shape")
}

/// Small radial material derived from the preset's mix colour. Alpha carries
/// the soft falloff; the existing image-makeup shader samples it through the
/// material UV stream. No vendor PNG is installed or read.
fn material_image(mix_color: [u8; 4]) -> Image {
    let mut pixels = Vec::with_capacity(16 * 16 * 4);
    for y in 0..16usize {
        for x in 0..16usize {
            let dx = (x as f32 + 0.5) / 16.0 - 0.5;
            let dy = (y as f32 + 0.5) / 16.0 - 0.5;
            let alpha = quantise(falloff(dx * dx + dy * dy) * mix_color[3] as f32 / 255.0);
            pixels.extend_from_slice(&[mix_color[0], mix_color[1], mix_color[2], alpha]);
        }
    }
    Image::new(16, 16, pixels).expect("material shape")
}

/// Builds the PF90033 `SuitPassInput` list for a natural theme.
///
/// The pass list comes from the checked-in suit preset JSON: the first
/// executable pass of each supported family, ordered base → blush → brow →
/// eye shadow → lashes → lipstick. All masks, materials and UV planes are
/// freshly allocated for this frame and borrowed by the returned inputs.
pub(crate) fn build_pf90033_suit_passes(
    frame: &Image,
    face_points194: &[[f32; 2]],
    face_mask: Option<&[u8]>,
    extra: &BTreeMap<String, f32>,
) -> Result<Vec<SuitPassInput<'static>>, Refusal> {
    frame.validate().map_err(|_| Refusal::ShapeMismatch(PF90033_FRAME))?;
    if face_points194.len() != 194 {
        return Err(Refusal::ShapeMismatch(PF90033_POINTS));
    }
    if face_points194.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(PF90033_POINTS));
    }
    let face_mask = face_mask.ok_or(Refusal::MissingInput(PF90033_MASK))?;
    if face_mask.len() != frame.pixel_count() {
        return Err(Refusal::ShapeMismatch(PF90033_MASK));
    }
    if !face_mask.iter().any(|value| *value > 0) {
        return Err(Refusal::MissingInput(PF90033_MASK));
    }
    let degree = extra
        .get("pc:90033")
        .copied()
        .ok_or(Refusal::MissingInput(PF90033_DEGREE))?;
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange(PF90033_DEGREE));
    }
    if degree == 0.0 {
        return Err(Refusal::ValueOutOfRange(PF90033_DEGREE));
    }

    let bundle = load_owned_suit_bundle().map_err(|_| Refusal::MissingInput(PF90033_PRESETS))?;
    let mut selected: Vec<OwnedSuitPass> = Vec::new();
    // Take the canonical first preset that can supply every theme family. A
    // later preset may have a gap in one family, so selecting per-family across
    // presets would silently mix themes.
    for preset in &bundle.presets {
        let mut candidate: Vec<OwnedSuitPass> = Vec::new();
        for pass in &preset.passes {
            if !theme_effect(pass) || !pass.is_executable() {
                continue;
            }
            if candidate.iter().any(|chosen| chosen.effect == pass.effect) {
                continue;
            }
            candidate.push(pass.clone());
        }
        let mut present: Vec<&str> = candidate.iter().map(|pass| pass.effect.as_str()).collect();
        present.sort_unstable();
        present.dedup();
        if present.len() == 5 {
            selected = candidate;
            break;
        }
    }
    selected.sort_by_key(|pass| (rank(&pass.effect), pass.order));
    if selected.is_empty() {
        return Err(Refusal::MissingInput(PF90033_PRESETS));
    }

    let mut passes = Vec::with_capacity(selected.len());
    for pass in &selected {
        let kind = suit_pass_kind_for_effect(&pass.effect)
            .map_err(|_| Refusal::ValueOutOfRange("suit pass effect"))?;
        if !matches!(kind, SuitPassKindInput::Image { .. }) {
            return Err(Refusal::ValueOutOfRange("suit pass effect"));
        }
        let pf = profile_pf(&pass.effect);
        let profile = row_profile(pf)?;
        let mesh = region_mesh(face_points194, [frame.width, frame.height], &profile)?;
        if mesh.positions.len() != mesh.uvs.len() || mesh.indices.len() < 3 {
            return Err(Refusal::ShapeMismatch("PF90033 mesh"));
        }
        let dense = produce_mesh_mask_uv(Some(MeshMaskUvInputs {
            ticket: Some(1),
            face_id: Some(1),
            positions: &mesh.positions,
            mask_uv: &mesh.uvs,
            indices: &mesh.indices,
            width: frame.width,
            height: frame.height,
        }))?;
        let center = center_for(&pass.effect, face_points194, mesh.positions[0]);
        let radius = mesh
            .positions
            .iter()
            .map(|p| (p[0] - center[0]).hypot(p[1] - center[1]))
            .fold(0.0f32, f32::max)
            .max(4.0);
        let mask = owner_mask(frame, face_mask, dense.writes(), center, radius);
        if !mask.pixels.iter().any(|value| *value > 0) {
            return Err(Refusal::MissingInput(PF90033_MASK));
        }
        let filter = pass
            .filter_type
            .filter(|value| *value != 0)
            .ok_or(Refusal::MissingInput("suit filter type"))?;
        let opacity = pass.opacity.unwrap_or(100.0).clamp(0.0, 100.0);
        let mix_color = pass.mix_color;
        let layer = MakeupLayer {
            mix_color,
            opacity_percent: opacity,
            degree01: (pass.degree / 100.0 * degree).clamp(0.0, 1.0),
            filter_type: filter,
            mask_channel: mask_channel(pass)?,
        };
        // The stage's `ProducerInputs` holds these inputs for the whole job;
        // each pass owns one small mask/UV/material allocation. Leaking them
        // gives the returned slice a stable address for that job. One call per
        // frame keeps the total bounded by the 5-pass theme list.
        let mask: &'static Image = Box::leak(Box::new(mask));
        let mask_uv: &'static [[f32; 2]] = Box::leak(dense.uv().to_vec().into_boxed_slice());
        let material: &'static Image = Box::leak(Box::new(material_image(mix_color)));
        let material_uv: &'static [[f32; 2]] = Box::leak(dense.uv().to_vec().into_boxed_slice());
        passes.push(SuitPassInput {
            kind: SuitPassKindInput::Image {
                material: Some(material),
                material_uv: Some(material_uv),
            },
            mask: Some(mask),
            mask_uv: Some(mask_uv),
            layer: Some(layer),
            shelter: 1.0,
            region: 1.0,
        });
    }
    Ok(passes)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OwnedSuitPassEntry {
    pub material: Image,
    pub material_uv: Vec<[f32; 2]>,
    pub mask: Image,
    pub mask_uv: Vec<[f32; 2]>,
    pub layer: MakeupLayer,
    pub shelter: f32,
    pub region: f32,
}

impl OwnedSuitPassEntry {
    pub(crate) fn as_borrowed(&self) -> SuitPassInput<'_> {
        SuitPassInput {
            kind: SuitPassKindInput::Image {
                material: Some(&self.material),
                material_uv: Some(&self.material_uv),
            },
            mask: Some(&self.mask),
            mask_uv: Some(&self.mask_uv),
            layer: Some(self.layer),
            shelter: self.shelter,
            region: self.region,
        }
    }
}

pub(crate) fn build_pf90033_owned_suit_passes(
    frame: &Image,
    face_points194: &[[f32; 2]],
    face_mask: Option<&[u8]>,
    extra: &BTreeMap<String, f32>,
) -> Result<Vec<OwnedSuitPassEntry>, Refusal> {
    frame.validate()?;
    if frame.width == 0 || frame.height == 0 {
        return Err(Refusal::ShapeMismatch(PF90033_FRAME));
    }
    if face_points194.len() != 194
        || face_points194
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(Refusal::ShapeMismatch(PF90033_POINTS));
    }
    let degree = extra.get("pc:90033").copied().unwrap_or(0.0);
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) || degree <= 0.0 {
        return Err(Refusal::ValueOutOfRange(PF90033_DEGREE));
    }
    let face_mask = face_mask.ok_or(Refusal::MissingInput(PF90033_MASK))?;
    if face_mask.len() != frame.pixel_count() {
        return Err(Refusal::ShapeMismatch(PF90033_MASK));
    }
    if !face_mask.iter().any(|value| *value > 0) {
        return Err(Refusal::MissingInput(PF90033_MASK));
    }
    let bundle = load_owned_suit_bundle().map_err(|_| Refusal::MissingInput(PF90033_PRESETS))?;
    let mut selected: Vec<OwnedSuitPass> = Vec::new();
    for preset in &bundle.presets {
        let mut candidate: Vec<OwnedSuitPass> = Vec::new();
        for pass in &preset.passes {
            if !theme_effect(pass) || !pass.is_executable() {
                continue;
            }
            if candidate.iter().any(|chosen| chosen.effect == pass.effect) {
                continue;
            }
            candidate.push(pass.clone());
        }
        let mut present: Vec<&str> = candidate.iter().map(|pass| pass.effect.as_str()).collect();
        present.sort_unstable();
        present.dedup();
        if present.len() == 5 {
            selected = candidate;
            break;
        }
    }
    selected.sort_by_key(|pass| (rank(&pass.effect), pass.order));
    if selected.is_empty() {
        return Err(Refusal::MissingInput(PF90033_PRESETS));
    }
    let mut passes = Vec::with_capacity(selected.len());
    for pass in &selected {
        let kind = suit_pass_kind_for_effect(&pass.effect)
            .map_err(|_| Refusal::ValueOutOfRange("suit pass effect"))?;
        if !matches!(kind, SuitPassKindInput::Image { .. }) {
            return Err(Refusal::ValueOutOfRange("suit pass effect"));
        }
        let pf = profile_pf(&pass.effect);
        let profile = row_profile(pf)?;
        let mesh = region_mesh(face_points194, [frame.width, frame.height], &profile)?;
        if mesh.positions.len() != mesh.uvs.len() || mesh.indices.len() < 3 {
            return Err(Refusal::ShapeMismatch("PF90033 mesh"));
        }
        let dense = produce_mesh_mask_uv(Some(MeshMaskUvInputs {
            ticket: Some(1),
            face_id: Some(1),
            positions: &mesh.positions,
            mask_uv: &mesh.uvs,
            indices: &mesh.indices,
            width: frame.width,
            height: frame.height,
        }))?;
        let center = center_for(&pass.effect, face_points194, mesh.positions[0]);
        let radius = mesh
            .positions
            .iter()
            .map(|p| (p[0] - center[0]).hypot(p[1] - center[1]))
            .fold(0.0f32, f32::max)
            .max(4.0);
        let mask = owner_mask(frame, face_mask, dense.writes(), center, radius);
        if !mask.pixels.iter().any(|value| *value > 0) {
            return Err(Refusal::MissingInput(PF90033_MASK));
        }
        let filter = pass
            .filter_type
            .filter(|value| *value != 0)
            .ok_or(Refusal::MissingInput("suit filter type"))?;
        let opacity = pass.opacity.unwrap_or(100.0).clamp(0.0, 100.0);
        let mix_color = pass.mix_color;
        let layer = MakeupLayer {
            mix_color,
            opacity_percent: opacity,
            degree01: (pass.degree / 100.0 * degree).clamp(0.0, 1.0),
            filter_type: filter,
            mask_channel: mask_channel(pass)?,
        };
        let uv = dense.uv().to_vec();
        passes.push(OwnedSuitPassEntry {
            material: material_image(mix_color),
            material_uv: uv.clone(),
            mask,
            mask_uv: uv,
            layer,
            shelter: 1.0,
            region: 1.0,
        });
    }
    Ok(passes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Image {
        let (width, height) = (16usize, 12usize);
        let mut pixels = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let (r, g, b) = if x < 8 { (190u8, 135, 110) } else { (150, 100, 90) };
                pixels.extend_from_slice(&[r, g, b, 255]);
            }
        }
        Image::new(width as u32, height as u32, pixels).unwrap()
    }

    fn skin_mask() -> Vec<u8> {
        let mut mask = vec![0u8; 16 * 12];
        for y in 2..10 {
            for x in 2..7 {
                mask[y * 16 + x] = 255;
            }
            for x in 9..14 {
                mask[y * 16 + x] = 200;
            }
        }
        mask
    }

    fn points() -> Vec<[f32; 2]> {
        // Spread the unused anchors deterministically so no mesh triangle is
        // accidentally degenerate; all anatomical anchors below are explicit.
        let mut points = (0..194)
            .map(|index| [2.0f32 + (index % 12) as f32, 2.0 + (index / 12) as f32])
            .collect::<Vec<_>>();
        for (index, value) in [
            (0usize, [4.0f32, 3.0]),
            (4, [4.5, 3.0]),
            (9, [11.5, 3.0]),
            (18, [4.5, 5.0]),
            (22, [6.5, 5.0]),
            (26, [5.5, 5.0]),
            (36, [9.5, 5.0]),
            (40, [11.5, 5.0]),
            (44, [10.5, 5.0]),
            (54, [8.0, 6.0]),
            (57, [8.0, 7.0]),
            (73, [5.0, 9.0]),
            (78, [8.0, 8.5]),
            (83, [11.0, 9.0]),
            (89, [8.0, 9.5]),
            (114, [5.0, 7.0]),
            (115, [5.5, 7.2]),
            (132, [10.5, 7.2]),
            (133, [11.0, 7.0]),
        ] {
            points[index] = value;
        }
        points
    }

    #[test]
    fn pf30059_emits_one_contract_row_per_component() {
        let frame = frame();
        let mask = skin_mask();
        let data = build_pf30059_multi_tone_row(7, &frame, &mask, None).unwrap();
        let HistoricalToneRowData::MultiTone(ref payload) = data else {
            panic!("expected MultiTone row");
        };
        assert_eq!(payload.faces.len(), 2, "two separated skin components");
        assert!(payload.faces.iter().all(|face| face.group_id == 1));
        let first = payload.faces[0].target;
        assert!(payload.faces.iter().all(|face| face.target == first));
        assert!(payload.faces.iter().all(|face| face.skin.len() == frame.pixel_count()));
        assert!(payload.faces.iter().all(|face| face.tooth_alpha == 0.0 && face.lut.is_none()));
        let built = crate::portrait_historical_tone_provider::build_historical_tone_inputs(
            crate::portrait_historical_tone_provider::HistoricalToneProviderPayload {
                ticket: 7,
                gray_3d_controls: [0.0, 0.0],
                rows: vec![crate::portrait_historical_tone_provider::HistoricalToneProviderRow {
                    frame: frame.clone(),
                    job: crate::portrait_historical_tone_provider::HistoricalToneRowJob {
                        source_sha256: aurora_portrait::model_sha256::sha256(&frame.pixels),
                        size: [frame.width, frame.height],
                        control: 0.8,
                    },
                    data,
                }],
            },
        );
        assert!(built.is_ok(), "row must pass provider validation: {built:?}");
    }

    #[test]
    fn pf30059_face_region_becomes_reference_and_consumer_applies() {
        use aurora_portrait::kernels::tone_unify;
        let frame = frame();
        let mask = skin_mask();
        let mut face = vec![0u8; frame.pixel_count()];
        for y in 2..7 {
            for x in 2..7 {
                face[y * 16 + x] = 255;
            }
        }
        let data = build_pf30059_multi_tone_row(9, &frame, &mask, Some(&face)).unwrap();
        let HistoricalToneRowData::MultiTone(ref payload) = data else {
            panic!("expected MultiTone row");
        };
        assert!(payload.faces.len() >= 2);
        let refs = tone_unify::select_references(
            &payload
                .faces
                .iter()
                .map(|pass| tone_unify::FaceRef {
                    group_id: pass.group_id,
                    face_id: pass.face_id,
                    explicit_reference: pass.reference,
                    target: pass.target,
                    brightness: &pass.brightness,
                    face_mask: &pass.face,
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(refs.len(), 1);
        // `HistoricalToneFace.target` is `Option<[u8; 3]>`; `Pass30059.target`
        // is the same type, so the projection below stays a direct copy.
        assert!(payload.faces.iter().all(|pass| pass.target.is_some()));
        // Keep every matrix vector alive for the whole `pf30059_apply` call:
        // `Pass30059` borrows each slice.
        let all_matrices = payload
            .faces
            .iter()
            .map(|pass| {
                pass.matrices
                    .iter()
                    .map(|(matrix, weight)| tone_unify::WeightedMatrix { matrix: *matrix, weight: *weight })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let passes = payload
            .faces
            .iter()
            .zip(&all_matrices)
            .map(|(pass, matrices)| tone_unify::Pass30059 {
                group_id: pass.group_id,
                face_id: pass.face_id,
                explicit_reference: pass.reference,
                target: pass.target,
                brightness: &pass.brightness,
                skin_mask: &pass.skin,
                face_mask: &pass.face,
                tooth_mask: &pass.tooth,
                instance_mask: &pass.instance,
                tooth_alpha: pass.tooth_alpha,
                brightness_lut_x: &pass.brightness_x,
                brightness_lut_y: &pass.brightness_y,
                matrices,
                lut: None,
            })
            .collect::<Vec<_>>();
        assert!(tone_unify::pf30059_apply(&frame, &passes, 80.0).is_ok());
    }

    #[test]
    fn pf30059_refuses_bad_inputs() {
        let frame = frame();
        let mask = skin_mask();
        assert_eq!(
            build_pf30059_multi_tone_row(7, &frame, &[0u8; 3], None),
            Err(Refusal::ShapeMismatch(PF30059_SKIN_MASK))
        );
        assert_eq!(
            build_pf30059_multi_tone_row(7, &frame, &vec![0u8; frame.pixel_count()], None),
            Err(Refusal::MissingInput(PF30059_SKIN_MASK))
        );
        assert_eq!(
            build_pf30059_multi_tone_row(7, &frame, &mask, Some(&[0u8; 3])),
            Err(Refusal::ShapeMismatch(PF30059_FACE_MASK))
        );
        assert_eq!(
            build_pf30059_multi_tone_row(0, &frame, &mask, None),
            Err(Refusal::ValueOutOfRange(PF30059_JOB))
        );
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn pf90033_builds_a_valid_natural_theme() {
        let frame = frame();
        let points = points();
        let face_mask = vec![255u8; frame.pixel_count()];
        let extra = BTreeMap::from([("pc:90033".to_string(), 0.8f32)]);
        let passes = build_pf90033_suit_passes(&frame, &points, Some(&face_mask), &extra).unwrap();
        assert!(!passes.is_empty());
        for pass in &passes {
            assert!(matches!(pass.kind, SuitPassKindInput::Image { .. }));
            assert!(pass.mask.is_some() && pass.mask_uv.is_some());
            let layer = pass.layer.unwrap();
            assert!(layer.degree01 > 0.0 && layer.degree01 <= 1.0);
            assert_eq!((pass.shelter, pass.region), (1.0, 1.0));
        }
        // The stage itself is the final validation: it resolves every returned
        // input and executes the recovered image-makeup consumer. The helpers
        // are private, so a coordinator mounting this module can add the same
        // two assertions from `portrait_stage.rs`'s own test module.
        assert!(passes.iter().all(|pass| pass.layer.is_some()
            && pass.mask.is_some()
            && pass.mask_uv.is_some()
            && matches!(pass.kind, SuitPassKindInput::Image { .. })));
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn pf90033_refuses_missing_and_malformed_inputs() {
        let frame = frame();
        let points = points();
        let face_mask = vec![255u8; frame.pixel_count()];
        assert_eq!(
            build_pf90033_suit_passes(&frame, &points, Some(&face_mask), &BTreeMap::new()).err(),
            Some(Refusal::MissingInput(PF90033_DEGREE))
        );
        for degree in [f32::NAN, -0.1, 0.0] {
            let extra = BTreeMap::from([("pc:90033".to_string(), degree)]);
            assert_eq!(
                build_pf90033_suit_passes(&frame, &points, Some(&face_mask), &extra).err(),
                Some(Refusal::ValueOutOfRange(PF90033_DEGREE))
            );
        }
        let extra = BTreeMap::from([("pc:90033".to_string(), 0.6f32)]);
        assert_eq!(
            build_pf90033_suit_passes(&frame, &[], Some(&face_mask), &extra).err(),
            Some(Refusal::ShapeMismatch(PF90033_POINTS))
        );
        assert_eq!(
            build_pf90033_suit_passes(&frame, &points, None, &extra).err(),
            Some(Refusal::MissingInput(PF90033_MASK))
        );
        assert_eq!(
            build_pf90033_suit_passes(&frame, &points, Some(&vec![0u8; frame.pixel_count()]), &extra).err(),
            Some(Refusal::MissingInput(PF90033_MASK))
        );
        let mut bad = points;
        bad[26] = [f32::NAN, 5.0];
        assert_eq!(
            build_pf90033_suit_passes(&frame, &bad, Some(&face_mask), &extra).err(),
            Some(Refusal::ValueOutOfRange(PF90033_POINTS))
        );
    }
}
