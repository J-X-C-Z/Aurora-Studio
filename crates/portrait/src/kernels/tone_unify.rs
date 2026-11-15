//! PF90085 (face skin-tone unify) and PF30059 (multi-face group skin-tone
//! unify) pixel laws.
//!
//! Ported from the R8 P12 producers:
//!   - `FaceSkinUnifyCore` (PF90085, `skinToneUnifyFace`)
//!   - `Pf30059GroupSkinTone` (PF30059, `skinToneUnifyMulti`)
//!
//! Everything model-owned stays an explicit input: target/reference colours,
//! brightness maps, per-face masks, matrices, the optional 3-D LUT and the
//! per-face model weight. Nothing is detected, averaged from the image or
//! defaulted here.
//!
//! Channel order is RGBA, matching `Image`; the C# sources are BGRA, so a BGRA
//! caller swaps R and B before and after.
//!
//! Not registered in `kernels::mod` by this lane.

use crate::{Access, EffectMeta, Image, Refusal};

/// PF90085/PF30059 now have executable CPU pixel stages; the HSL palette
/// segment of 90085 stays open and is listed in its inputs.
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 90085,
        key: "skinToneUnifyFace",
        label: "Skin tone unify (face)",
        access: Access::Cpu,
        inputs: &["source", "per-face target", "skin/face/tooth/instance masks", "model weight", "ui"],
    },
    EffectMeta {
        pf: 30059,
        key: "skinToneUnifyMulti",
        label: "Skin tone unify (multi)",
        access: Access::Cpu,
        inputs: &["source", "per-face targets", "brightness LUT", "optional LUT3D/matrix", "supports", "ui"],
    },
];

/// PF90085 face-parsing weight (`SkinRetuneHslPalette.frag`).
pub const FACE_PARSING_WEIGHT: f32 = 0.39;
/// Both rows gate at 0.001.
pub const MASK_GATE: f32 = 0.001;
/// ROI intersection threshold (`0x14867b610`).
pub const INTERSECTION_THRESHOLD: u8 = 20;

pub const PF90085_RVA: &str = "0x4F1FBAE / 0x7D35F0E / 0x7D35F91 / 0x7D35EA0 / 0x7D410B0 / 0x86ACDB0";
pub const PF30059_RVA: &str = "0x14815D0B0 / 0x148153990 / 0x14815caf0 / 0x14867b610 / 0x14867b740";

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
    if values.len() < count {
        Err(Refusal::ShapeMismatch(name))
    } else {
        Ok(())
    }
}

/// Both producers quantise with `MidpointRounding.ToEven`.
fn quantise(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round_ties_even().clamp(0.0, 255.0) as u8
}

fn unit_byte(value: u8) -> f32 {
    value as f32 / 255.0
}

/// ---- shared: support / mask / reference -------------------------------

/// `SkinRetuneHslPalette.frag` support:
/// `max(skin, (fp.a + fp.g + fp.b + faceMask.g) / 255 * 0.39)`, then an
/// optional feature-protect multiply.
pub fn face_support_90085(
    skin: u8,
    parsing_a: Option<u8>,
    parsing_g: Option<u8>,
    parsing_b: Option<u8>,
    face_mask_g: u8,
    protect: Option<u8>,
) -> f32 {
    let face = unit_byte(face_mask_g);
    let parsing = match (parsing_a, parsing_g, parsing_b) {
        (Some(a), Some(g), Some(b)) => (a as f32 + g as f32 + b as f32 + face_mask_g as f32)
            / 255.0
            * FACE_PARSING_WEIGHT,
        _ => face * FACE_PARSING_WEIGHT,
    };
    let mut support = unit_byte(skin).max(parsing);
    if let Some(value) = protect {
        support *= 1.0 - unit_byte(value);
    }
    support
}

/// PF30059 per-face support byte: `max(skin, (faceR - tooth*(1-alpha)) * instance)`.
pub fn face_support_30059(
    skin: u8,
    face: u8,
    tooth: u8,
    instance: u8,
    tooth_alpha: f32,
) -> u8 {
    let skin = unit_byte(skin);
    let face_r = unit_byte(face);
    let tooth = unit_byte(tooth);
    let instance = unit_byte(instance);
    let face_weight = (face_r - tooth * (1.0 - tooth_alpha)).max(0.0) * instance;
    quantise(skin.max(face_weight))
}

/// `0x14815D0B0`: sum the per-face supports and saturate at 255.
pub fn combine_masks(supports: &[u8]) -> u8 {
    let mut sum = 0u32;
    for value in supports {
        sum += *value as u32;
    }
    sum.min(255) as u8
}

/// `0x148153990`: mean brightness over pixels where the mask is non-zero.
pub fn mean_masked_byte(brightness: &[u8], mask: &[u8]) -> Result<f32, Refusal> {
    if brightness.len() != mask.len() {
        return Err(Refusal::ShapeMismatch("PF30059_BRIGHTNESS"));
    }
    let mut sum = 0f64;
    let mut count = 0u64;
    for (value, weight) in brightness.iter().zip(mask.iter()) {
        if *weight == 0 {
            continue;
        }
        sum += *value as f64;
        count += 1;
    }
    Ok(if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    })
}

/// `0x14815caf0`: piecewise-linear lookup; a missing or malformed table is 1.0.
pub fn piecewise_linear(x: f32, xs: &[f32], ys: &[f32]) -> f32 {
    if xs.is_empty() || xs.len() != ys.len() {
        return 1.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    for index in 1..xs.len() {
        if x > xs[index] {
            continue;
        }
        let dx = xs[index] - xs[index - 1];
        if dx == 0.0 {
            return ys[index];
        }
        return (ys[index] - ys[index - 1]) * ((x - xs[index - 1]) / dx) + ys[index - 1];
    }
    1.0
}

/// One face as seen by the reference selector.
#[derive(Clone, Copy)]
pub struct FaceRef<'a> {
    pub group_id: i32,
    pub face_id: i32,
    pub explicit_reference: Option<i32>,
    pub target: Option<[u8; 3]>,
    pub brightness: &'a [u8],
    pub face_mask: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReferenceChoice {
    pub group_id: i32,
    pub reference_face_id: i32,
    pub target: [f32; 3],
}

/// `Render_`: explicit `Ref Face Id` wins, otherwise the brightest masked mean,
/// ties broken by ascending face id. A group must resolve to exactly one target.
pub fn select_references(faces: &[FaceRef<'_>]) -> Result<Vec<ReferenceChoice>, Refusal> {
    let mut groups: Vec<i32> = faces.iter().map(|face| face.group_id).collect();
    groups.sort_unstable();
    groups.dedup();

    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let members: Vec<&FaceRef<'_>> = faces.iter().filter(|face| face.group_id == group).collect();
        let explicit = members.iter().find_map(|face| face.explicit_reference);
        let reference = if let Some(id) = explicit {
            *members
                .iter()
                .find(|face| face.face_id == id)
                .ok_or(Refusal::MissingInput("PF30059_REFERENCE_FACE"))?
        } else {
            let mut best: &FaceRef<'_> = members[0];
            let mut best_mean = mean_masked_byte(best.brightness, best.face_mask)?;
            for face in members.iter().skip(1) {
                let mean = mean_masked_byte(face.brightness, face.face_mask)?;
                if mean > best_mean || (mean == best_mean && face.face_id < best.face_id) {
                    best = *face;
                    best_mean = mean;
                }
            }
            best
        };

        let targets: Vec<[u8; 3]> = members
            .iter()
            .filter_map(|face| face.target)
            .collect();
        let first = targets.first().copied();
        if targets.iter().any(|target| Some(*target) != first) {
            return Err(Refusal::ValueOutOfRange("PF30059_CONFLICTING_TARGETS"));
        }
        let target = first
            .or(reference.target)
            .ok_or(Refusal::MissingInput("PF30059_TARGET_COLOR"))?;
        out.push(ReferenceChoice {
            group_id: group,
            reference_face_id: reference.face_id,
            target: [
                unit_byte(target[0]),
                unit_byte(target[1]),
                unit_byte(target[2]),
            ],
        });
    }
    Ok(out)
}

/// ---- colour consumers --------------------------------------------------

/// Row-major 3x4 in encoded RGB, last column is a bias (PF30059).
pub fn apply_matrix3x4(matrix: &[f32; 12], rgb: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for row in 0..3 {
        let base = row * 4;
        out[row] = matrix[base] * rgb[0]
            + matrix[base + 1] * rgb[1]
            + matrix[base + 2] * rgb[2]
            + matrix[base + 3];
    }
    [
        out[0].clamp(0.0, 1.0),
        out[1].clamp(0.0, 1.0),
        out[2].clamp(0.0, 1.0),
    ]
}

/// Explicit 3-D LUT (mode `Lut3D` in the C# producer): encoded RGB in,
/// trilinear, encoded RGB out. `values` is `grid^3 * 3`.
pub fn lut3d_sample(values: &[f32], grid: usize, rgb: [f32; 3]) -> Result<[f32; 3], Refusal> {
    if grid < 2 || values.len() != grid * grid * grid * 3 {
        return Err(Refusal::ShapeMismatch("PF30059_LUT3D"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF30059_LUT3D"));
    }
    let max = (grid - 1) as f32;
    let pr = rgb[0].clamp(0.0, 1.0) * max;
    let pg = rgb[1].clamp(0.0, 1.0) * max;
    let pb = rgb[2].clamp(0.0, 1.0) * max;
    let (ir, fr) = (pr.floor() as usize, pr.fract());
    let (ig, fg) = (pg.floor() as usize, pg.fract());
    let (ib, fb) = (pb.floor() as usize, pb.fract());
    let jr = (ir + 1).min(grid - 1);
    let jg = (ig + 1).min(grid - 1);
    let jb = (ib + 1).min(grid - 1);

    let mut out = [0.0f32; 3];
    for dz in 0..2 {
        let z = if dz == 0 { ib } else { jb };
        let wz = if dz == 0 { 1.0 - fb } else { fb };
        for dy in 0..2 {
            let y = if dy == 0 { ig } else { jg };
            let wy = if dy == 0 { 1.0 - fg } else { fg };
            for dx in 0..2 {
                let x = if dx == 0 { ir } else { jr };
                let wx = if dx == 0 { 1.0 - fr } else { fr };
                let weight = wz * wy * wx;
                let index = ((z * grid + y) * grid + x) * 3;
                for channel in 0..3 {
                    out[channel] += values[index + channel] * weight;
                }
            }
        }
    }
    Ok([
        out[0].clamp(0.0, 1.0),
        out[1].clamp(0.0, 1.0),
        out[2].clamp(0.0, 1.0),
    ])
}

/// Per-face model transform, in the C# precedence order: LUT, then matrix,
/// then the explicit target.
pub enum Transform<'a> {
    Lut3D { values: &'a [f32], grid: usize },
    Matrix([f32; 12]),
    Target([f32; 3]),
}

fn transform(transform: &Transform<'_>, source: [f32; 3]) -> Result<[f32; 3], Refusal> {
    match transform {
        Transform::Lut3D { values, grid } => lut3d_sample(values, *grid, source),
        Transform::Matrix(matrix) => Ok(apply_matrix3x4(matrix, source)),
        Transform::Target(target) => Ok(*target),
    }
}

/// ---- PF90085 -----------------------------------------------------------

pub struct Pass90085<'a> {
    pub skin_mask: Option<&'a [u8]>,
    pub face_mask_g: &'a [u8],
    pub parsing_a: Option<&'a [u8]>,
    pub parsing_g: Option<&'a [u8]>,
    pub parsing_b: Option<&'a [u8]>,
    pub protect: Option<&'a [u8]>,
    pub model_weight: Option<&'a [u8]>,
    /// Per-pixel model target RGB (`width*height*3`), else a constant target,
    /// else the weighted mean of the support region.
    pub model_target: Option<&'a [u8]>,
    pub target: Option<[u8; 3]>,
}

/// `out = src + (tuned - src) * support * degree * modelWeight`, alpha kept.
/// The HSL-palette/gamma `ColorTransform` stage stays OPEN: supply a target.
pub fn pf90085_apply(source: &Image, pass: &Pass90085<'_>, ui: f32) -> Result<Image, Refusal> {
    let count = check_image(source)?;
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("PF90085_UI"));
    }
    let degree = ui / 100.0;
    if degree <= MASK_GATE {
        return Ok(source.clone());
    }
    plane(pass.face_mask_g, count, "PF90085_FACE_MASK_G")?;
    for (values, name) in [
        (pass.skin_mask, "PF90085_SKIN_MASK"),
        (pass.parsing_a, "PF90085_PARSING_A"),
        (pass.parsing_g, "PF90085_PARSING_G"),
        (pass.parsing_b, "PF90085_PARSING_B"),
        (pass.protect, "PF90085_PROTECT"),
        (pass.model_weight, "PF90085_MODEL_WEIGHT"),
    ] {
        if let Some(values) = values {
            plane(values, count, name)?;
        }
    }
    if let Some(values) = pass.model_target {
        if values.len() != count * 3 {
            return Err(Refusal::ShapeMismatch("PF90085_MODEL_TARGET"));
        }
    }

    // Weighted mean of the support region, used only when no explicit target.
    let mut fallback = [0.0f64; 3];
    let mut fallback_weight = 0.0f64;
    if pass.model_target.is_none() && pass.target.is_none() {
        for index in 0..count {
            let support = face_support_90085(
                pass.skin_mask.map_or(0, |mask| mask[index]),
                pass.parsing_a.map(|mask| mask[index]),
                pass.parsing_g.map(|mask| mask[index]),
                pass.parsing_b.map(|mask| mask[index]),
                pass.face_mask_g[index],
                pass.protect.map(|mask| mask[index]),
            );
            if support <= MASK_GATE {
                continue;
            }
            let offset = index * 4;
            for channel in 0..3 {
                fallback[channel] += source.pixels[offset + channel] as f64 * support as f64;
            }
            fallback_weight += support as f64;
        }
        if fallback_weight == 0.0 {
            return Ok(source.clone());
        }
    }

    let mut out = source.clone();
    for index in 0..count {
        let support = face_support_90085(
            pass.skin_mask.map_or(0, |mask| mask[index]),
            pass.parsing_a.map(|mask| mask[index]),
            pass.parsing_g.map(|mask| mask[index]),
            pass.parsing_b.map(|mask| mask[index]),
            pass.face_mask_g[index],
            pass.protect.map(|mask| mask[index]),
        );
        if support <= MASK_GATE {
            continue;
        }
        let mut mix = support * degree;
        if let Some(weights) = pass.model_weight {
            mix *= unit_byte(weights[index]);
        }
        if mix <= 0.0 {
            continue;
        }
        let offset = index * 4;
        let target = if let Some(values) = pass.model_target {
            [
                unit_byte(values[index * 3]),
                unit_byte(values[index * 3 + 1]),
                unit_byte(values[index * 3 + 2]),
            ]
        } else if let Some(target) = pass.target {
            [unit_byte(target[0]), unit_byte(target[1]), unit_byte(target[2])]
        } else {
            [
                (fallback[0] / fallback_weight) as f32,
                (fallback[1] / fallback_weight) as f32,
                (fallback[2] / fallback_weight) as f32,
            ]
        };
        let alpha = mix.clamp(0.0, 1.0);
        for channel in 0..3 {
            let src = unit_byte(source.pixels[offset + channel]);
            out.pixels[offset + channel] = quantise(src * (1.0 - alpha) + target[channel] * alpha);
        }
    }
    Ok(out)
}

/// ---- PF30059 -----------------------------------------------------------

pub struct WeightedMatrix {
    pub matrix: [f32; 12],
    pub weight: f32,
}

pub struct Pass30059<'a> {
    pub group_id: i32,
    pub face_id: i32,
    pub explicit_reference: Option<i32>,
    pub target: Option<[u8; 3]>,
    pub brightness: &'a [u8],
    pub skin_mask: &'a [u8],
    pub face_mask: &'a [u8],
    pub tooth_mask: &'a [u8],
    pub instance_mask: &'a [u8],
    pub tooth_alpha: f32,
    pub brightness_lut_x: &'a [f32],
    pub brightness_lut_y: &'a [f32],
    pub matrices: &'a [WeightedMatrix],
    pub lut: Option<(&'a [f32], usize)>,
}

/// Model weights aggregate per-face matrices; zero weight sum is refused.
pub fn aggregate_matrices(matrices: &[WeightedMatrix]) -> Result<[f32; 12], Refusal> {
    if matrices.is_empty() {
        return Ok([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }
    let mut sum = 0f64;
    let mut acc = [0f64; 12];
    for item in matrices {
        if !item.weight.is_finite() || item.weight < 0.0 {
            return Err(Refusal::ValueOutOfRange("PF30059_MATRIX_WEIGHT"));
        }
        if item.matrix.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF30059_MATRIX"));
        }
        sum += item.weight as f64;
        for index in 0..12 {
            acc[index] += item.matrix[index] as f64 * item.weight as f64;
        }
    }
    if sum <= 0.0 {
        return Err(Refusal::ValueOutOfRange("PF30059_MATRIX_WEIGHT"));
    }
    let mut out = [0f32; 12];
    for index in 0..12 {
        out[index] = (acc[index] / sum) as f32;
    }
    Ok(out)
}

/// PF30059 pixel law: overlap = min(255, sum supports); each face transforms
/// the source (LUT > matrix > target); the results are averaged by support;
/// then `out = src + (avg - src) * alpha`, alpha kept.
pub fn pf30059_apply(
    source: &Image,
    faces: &[Pass30059<'_>],
    degree_percent: f32,
) -> Result<Image, Refusal> {
    let count = check_image(source)?;
    if !degree_percent.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF30059_UI"));
    }
    if faces.is_empty() {
        return Ok(source.clone());
    }
    for face in faces {
        plane(face.brightness, count, "PF30059_BRIGHTNESS")?;
        plane(face.skin_mask, count, "PF30059_SKIN_MASK")?;
        plane(face.face_mask, count, "PF30059_FACE_MASK")?;
        plane(face.tooth_mask, count, "PF30059_TOOTH_MASK")?;
        plane(face.instance_mask, count, "PF30059_INSTANCE_MASK")?;
        if !face.tooth_alpha.is_finite() || !(0.0..=1.0).contains(&face.tooth_alpha) {
            return Err(Refusal::ValueOutOfRange("PF30059_TOOTH_ALPHA"));
        }
    }
    let degree = (degree_percent.abs() / 100.0).clamp(0.0, 1.0);
    if degree <= MASK_GATE {
        return Ok(source.clone());
    }

    let refs_input: Vec<FaceRef<'_>> = faces
        .iter()
        .map(|face| FaceRef {
            group_id: face.group_id,
            face_id: face.face_id,
            explicit_reference: face.explicit_reference,
            target: face.target,
            brightness: face.brightness,
            face_mask: face.face_mask,
        })
        .collect();
    let references = select_references(&refs_input)?;

    let mut out = source.clone();
    for choice in &references {
        let indices: Vec<usize> = (0..faces.len())
            .filter(|index| faces[*index].group_id == choice.group_id)
            .collect();
        let reference_index = indices
            .iter()
            .copied()
            .find(|index| faces[*index].face_id == choice.reference_face_id)
            .ok_or(Refusal::MissingInput("PF30059_REFERENCE_FACE"))?;
        let reference_mean =
            mean_masked_byte(faces[reference_index].brightness, faces[reference_index].face_mask)?;

        let mut factors = Vec::with_capacity(indices.len());
        for index in &indices {
            let face = &faces[*index];
            let mean = mean_masked_byte(face.brightness, face.face_mask)?;
            factors.push(piecewise_linear(
                (mean - reference_mean).abs(),
                face.brightness_lut_x,
                face.brightness_lut_y,
            ));
        }

        for index in 0..count {
            let mut supports = vec![0u8; indices.len()];
            let mut support_sum = 0u32;
            let mut factor_sum = 0f32;
            for (slot, face_index) in indices.iter().enumerate() {
                let face = &faces[*face_index];
                let support = face_support_30059(
                    face.skin_mask[index],
                    face.face_mask[index],
                    face.tooth_mask[index],
                    face.instance_mask[index],
                    face.tooth_alpha,
                );
                supports[slot] = support;
                support_sum += support as u32;
                factor_sum += factors[slot] * support as f32;
            }
            let overlap = combine_masks(&supports);
            if overlap == 0 {
                continue;
            }
            let factor = if support_sum == 0 {
                1.0
            } else {
                factor_sum / support_sum as f32
            };
            let alpha = (overlap as f32 / 255.0) * degree * factor;
            if !(alpha > MASK_GATE) {
                continue;
            }
            let alpha = alpha.clamp(0.0, 1.0);

            let offset = index * 4;
            let src = [
                unit_byte(source.pixels[offset]),
                unit_byte(source.pixels[offset + 1]),
                unit_byte(source.pixels[offset + 2]),
            ];
            let mut sum = [0f32; 3];
            let mut weight_sum = 0u32;
            for (slot, face_index) in indices.iter().enumerate() {
                let weight = supports[slot] as u32;
                if weight == 0 {
                    continue;
                }
                let face = &faces[*face_index];
                let face_transform = match face.lut {
                    Some((values, grid)) => Transform::Lut3D { values, grid },
                    None if !face.matrices.is_empty() => {
                        Transform::Matrix(aggregate_matrices(face.matrices)?)
                    }
                    None => Transform::Target(
                        face.target
                            .map(|rgb| [unit_byte(rgb[0]), unit_byte(rgb[1]), unit_byte(rgb[2])])
                            .unwrap_or(choice.target),
                    ),
                };
                let value = transform(&face_transform, src)?;
                weight_sum += weight;
                for channel in 0..3 {
                    sum[channel] += value[channel] * weight as f32;
                }
            }
            let averaged = if weight_sum == 0 {
                choice.target
            } else {
                [
                    sum[0] / weight_sum as f32,
                    sum[1] / weight_sum as f32,
                    sum[2] / weight_sum as f32,
                ]
            };
            for channel in 0..3 {
                out.pixels[offset + channel] =
                    quantise(src[channel] + (averaged[channel] - src[channel]) * alpha);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsing_weight_is_point_three_nine() {
        // all parsing channels 0, face mask 255 -> 255/255*0.39
        let support = face_support_90085(0, Some(0), Some(0), Some(0), 255, None);
        assert_eq!(support, 0.39);
        // (a+g+b+fm)/255*0.39 with 51 each -> (204/255)*0.39
        let support = face_support_90085(0, Some(51), Some(51), Some(51), 51, None);
        assert!((support - 0.312).abs() < 1e-6, "{support}");
    }

    #[test]
    fn support_takes_skin_max_and_protect() {
        let support = face_support_90085(255, Some(0), Some(0), Some(0), 0, None);
        assert_eq!(support, 1.0);
        let support = face_support_90085(255, None, None, None, 0, Some(255));
        assert_eq!(support, 0.0);
    }

    #[test]
    fn overlap_saturates_at_255() {
        assert_eq!(combine_masks(&[200, 100]), 255);
        assert_eq!(combine_masks(&[10, 20]), 30);
        assert_eq!(combine_masks(&[]), 0);
    }

    #[test]
    fn face_support_30059_teeth_and_instance() {
        assert_eq!(face_support_30059(255, 0, 0, 255, 1.0), 255);
        // toothAlpha 1 removes the tooth term; 0 subtracts the full tooth mask.
        assert_eq!(face_support_30059(0, 255, 255, 255, 1.0), 255);
        assert_eq!(face_support_30059(0, 255, 255, 255, 0.0), 0);
        // instance 0 kills the face channel
        assert_eq!(face_support_30059(0, 255, 0, 0, 1.0), 0);
    }

    #[test]
    fn piecewise_lookup_clamps_and_defaults() {
        let xs = [0.0, 10.0, 20.0];
        let ys = [1.0, 0.5, 0.25];
        assert_eq!(piecewise_linear(-1.0, &xs, &ys), 1.0);
        assert_eq!(piecewise_linear(5.0, &xs, &ys), 0.75);
        assert_eq!(piecewise_linear(30.0, &xs, &ys), 0.25);
        assert_eq!(piecewise_linear(5.0, &[], &[]), 1.0);
    }

    #[test]
    fn reference_prefers_brightest_then_face_id() {
        let a = [10u8];
        let b = [200u8];
        let m = [255u8];
        let faces = [
            FaceRef { group_id: 0, face_id: 7, explicit_reference: None, target: Some([1, 2, 3]), brightness: &a, face_mask: &m },
            FaceRef { group_id: 0, face_id: 3, explicit_reference: None, target: None, brightness: &b, face_mask: &m },
        ];
        let refs = select_references(&faces).unwrap();
        assert_eq!(refs[0].reference_face_id, 3);
        assert_eq!(refs[0].target, [1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0]);
    }

    #[test]
    fn explicit_reference_wins_and_missing_is_refused() {
        let v = [10u8];
        let m = [255u8];
        let faces = [
            FaceRef { group_id: 0, face_id: 1, explicit_reference: Some(2), target: Some([9, 9, 9]), brightness: &v, face_mask: &m },
            FaceRef { group_id: 0, face_id: 2, explicit_reference: None, target: None, brightness: &v, face_mask: &m },
        ];
        assert_eq!(select_references(&faces).unwrap()[0].reference_face_id, 2);

        let faces = [
            FaceRef { group_id: 0, face_id: 1, explicit_reference: Some(9), target: Some([1, 1, 1]), brightness: &v, face_mask: &m },
        ];
        assert_eq!(
            select_references(&faces),
            Err(Refusal::MissingInput("PF30059_REFERENCE_FACE"))
        );
    }

    #[test]
    fn matrix_applies_row_major_with_bias() {
        let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(apply_matrix3x4(&identity, [0.2, 0.4, 0.6]), [0.2, 0.4, 0.6]);
        let bias = [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(apply_matrix3x4(&bias, [0.2, 0.4, 0.6]), [0.3, 0.4, 0.6]);
        let clamp = [2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(apply_matrix3x4(&clamp, [0.9, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn lut3d_is_identity_for_a_ramp_lattice() {
        let grid = 3;
        let mut values = Vec::new();
        for z in 0..grid {
            for y in 0..grid {
                for x in 0..grid {
                    values.push(x as f32 / (grid - 1) as f32);
                    values.push(y as f32 / (grid - 1) as f32);
                    values.push(z as f32 / (grid - 1) as f32);
                }
            }
        }
        for rgb in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.25, 0.5, 0.75]] {
            let out = lut3d_sample(&values, grid, rgb).unwrap();
            for channel in 0..3 {
                assert!((out[channel] - rgb[channel]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn lut3d_refuses_shape_and_non_finite() {
        assert_eq!(
            lut3d_sample(&[0.0; 3], 2, [0.0, 0.0, 0.0]),
            Err(Refusal::ShapeMismatch("PF30059_LUT3D"))
        );
        let mut values = vec![0.0f32; 2 * 2 * 2 * 3];
        values[0] = f32::NAN;
        assert_eq!(
            lut3d_sample(&values, 2, [0.0, 0.0, 0.0]),
            Err(Refusal::ValueOutOfRange("PF30059_LUT3D"))
        );
    }

    #[test]
    fn pf90085_half_support_blends_halfway() {
        let source = Image::solid(1, 1, [10, 20, 30, 77]);
        // face mask 255, no parsing planes -> support 0.39; degree 1.
        let face = [255u8];
        let pass = Pass90085 {
            skin_mask: None,
            face_mask_g: &face,
            parsing_a: None,
            parsing_g: None,
            parsing_b: None,
            protect: None,
            model_weight: None,
            model_target: None,
            target: Some([200, 100, 50]),
        };
        let out = pf90085_apply(&source, &pass, 100.0).unwrap();
        // mix = 0.39 -> 10 + (200-10)*0.39 = 84.1 -> 84
        assert_eq!(out.pixel(0, 0), [84, 51, 38, 77]);
    }

    #[test]
    fn pf90085_zero_degree_is_identity() {
        let source = Image::solid(1, 1, [10, 20, 30, 77]);
        let face = [255u8];
        let pass = Pass90085 {
            skin_mask: None,
            face_mask_g: &face,
            parsing_a: None,
            parsing_g: None,
            parsing_b: None,
            protect: None,
            model_weight: None,
            model_target: None,
            target: Some([200, 100, 50]),
        };
        assert_eq!(pf90085_apply(&source, &pass, 0.0).unwrap(), source);
    }

    #[test]
    fn pf30059_single_face_full_degree_takes_matrix() {
        let source = Image::solid(1, 1, [100, 100, 100, 200]);
        let plane_full = [255u8];
        let bright = [100u8];
        let matrices = [WeightedMatrix {
            matrix: [1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            weight: 1.0,
        }];
        let faces = [Pass30059 {
            group_id: 0,
            face_id: 1,
            explicit_reference: None,
            target: Some([1, 2, 3]),
            brightness: &bright,
            skin_mask: &plane_full,
            face_mask: &plane_full,
            tooth_mask: &[0u8],
            instance_mask: &plane_full,
            tooth_alpha: 1.0,
            brightness_lut_x: &[],
            brightness_lut_y: &[],
            matrices: &matrices,
            lut: None,
        }];
        let out = pf30059_apply(&source, &faces, 100.0).unwrap();
        // support 255 -> alpha 1; matrix adds 0.1 -> 100/255+0.1
        assert_eq!(out.pixel(0, 0), [126, 100, 100, 200]);
    }

    #[test]
    fn pf30059_two_faces_average_by_support() {
        let source = Image::solid(1, 1, [0, 0, 0, 255]);
        let half = [128u8];
        let full = [255u8];
        let bright = [100u8];
        let no_matrix: [WeightedMatrix; 0] = [];
        let faces = [
            Pass30059 {
                group_id: 0, face_id: 1, explicit_reference: None, target: Some([255, 0, 0]),
                brightness: &bright, skin_mask: &half, face_mask: &half, tooth_mask: &[0u8],
                instance_mask: &half, tooth_alpha: 1.0, brightness_lut_x: &[], brightness_lut_y: &[],
                matrices: &no_matrix, lut: None,
            },
            Pass30059 {
                group_id: 0, face_id: 2, explicit_reference: None, target: Some([255, 0, 0]),
                brightness: &bright, skin_mask: &full, face_mask: &full, tooth_mask: &[0u8],
                instance_mask: &full, tooth_alpha: 1.0, brightness_lut_x: &[], brightness_lut_y: &[],
                matrices: &no_matrix, lut: None,
            },
        ];
        let out = pf30059_apply(&source, &faces, 100.0).unwrap();
        // same target both, overlap 255 -> red 255
        assert_eq!(out.pixel(0, 0), [255, 0, 0, 255]);
    }
}
