use std::collections::BTreeMap;

use aurora_portrait::kernels::body_geometry::Vec2;
use aurora_portrait::{Image, Refusal};

use crate::aimask::body_pose::BodyPoseInference;
use crate::body_shader_abi::{BodyKeypointSet, BODY_LANDMARK_COUNT};

use super::portrait_body_stage::{BodyGeometryInput, BodyGeometryJob, SixPointWarp};
use super::{PortraitJobReceipt, ProducerInputs};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BodyLandmarkDomain {
    Face,
    Body,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyLandmarkRef {
    pub domain: BodyLandmarkDomain,
    pub index: usize,
}

impl BodyLandmarkRef {
    pub const fn face(index: usize) -> Self {
        Self {
            domain: BodyLandmarkDomain::Face,
            index,
        }
    }

    pub const fn body(index: usize) -> Self {
        Self {
            domain: BodyLandmarkDomain::Body,
            index,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyLandmarkFrame {
    pub face_source: Vec<Vec2>,
    pub face_target: Vec<Vec2>,
    pub body_source: Vec<Vec2>,
    pub body_target: Vec<Vec2>,
}

impl BodyLandmarkFrame {
    pub fn from_pose(pose: &BodyPoseInference) -> Result<Self, Refusal> {
        let points = BodyKeypointSet::from_inference(pose)?;
        Ok(Self {
            body_source: points.landmarks().to_vec(),
            ..Self::default()
        })
    }

    fn source(&self, landmark: BodyLandmarkRef) -> Result<Vec2, Refusal> {
        self.point(landmark, false)
    }

    fn target(&self, landmark: BodyLandmarkRef) -> Result<Vec2, Refusal> {
        self.point(landmark, true)
    }

    fn point(&self, landmark: BodyLandmarkRef, target: bool) -> Result<Vec2, Refusal> {
        let (points, name) = match (landmark.domain, target) {
            (BodyLandmarkDomain::Face, false) => (&self.face_source, "body face source landmark"),
            (BodyLandmarkDomain::Face, true) => (&self.face_target, "body face target landmark"),
            (BodyLandmarkDomain::Body, false) => (&self.body_source, "body pose source landmark"),
            (BodyLandmarkDomain::Body, true) => (&self.body_target, "body pose target landmark"),
        };
        points
            .get(landmark.index)
            .copied()
            .ok_or(Refusal::MissingInput(name))
    }

    fn validate(&self) -> Result<(), Refusal> {
        for (source, target, name) in [
            (&self.face_source, &self.face_target, "body face landmarks"),
            (&self.body_source, &self.body_target, "body pose landmarks"),
        ] {
            if name == "body pose landmarks"
                && [source.len(), target.len()]
                    .into_iter()
                    .any(|len| len != 0 && len != BODY_LANDMARK_COUNT)
            {
                return Err(Refusal::ShapeMismatch(name));
            }
            if !source.is_empty() && !target.is_empty() && source.len() != target.len() {
                return Err(Refusal::ShapeMismatch(name));
            }
            if source
                .iter()
                .chain(target)
                .any(|point| !point.x.is_finite() || !point.y.is_finite())
            {
                return Err(Refusal::ValueOutOfRange(name));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BodyLandmarkEffect {
    SixPoint {
        source: [BodyLandmarkRef; 6],
        destination: [BodyLandmarkRef; 6],
        degrees: [f32; 6],
        amount: f32,
        inverse_aspect: f32,
        valid_point_count: usize,
    },
    SameJob(BodyGeometryInput),
}

/// Same-frame producer for all twelve body geometry rows. Pose inference is
/// evidence only; dense masks, meshes, UV fields and shift maps still arrive as
/// explicit `SameJob` payloads and are never synthesized here.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyGeometryProducerInputs {
    pub job_id: u64,
    pub frame_size: (u32, u32),
    pub source_sha256: [u8; 32],
    pub pose: Option<BodyPoseInference>,
    pub landmarks: BodyLandmarkFrame,
    pub effects: BTreeMap<u32, BodyLandmarkEffect>,
}

impl BodyGeometryProducerInputs {
    pub fn from_current_frame(
        job_id: u64,
        source: &Image,
        pose: Option<BodyPoseInference>,
        mut landmarks: BodyLandmarkFrame,
        effects: BTreeMap<u32, BodyLandmarkEffect>,
    ) -> Result<Self, Refusal> {
        source
            .validate()
            .map_err(|_| Refusal::ShapeMismatch("body geometry source"))?;
        if let Some(pose) = &pose {
            if pose.frame_size != (source.width, source.height)
                || pose.source_sha256 != crate::sha256_32(&source.pixels)
            {
                return Err(Refusal::ValueOutOfRange("body pose same-frame receipt"));
            }
            if landmarks.body_source.is_empty() {
                landmarks.body_source = BodyKeypointSet::from_inference(pose)?.landmarks().to_vec();
            }
        }
        let input = Self {
            job_id,
            frame_size: (source.width, source.height),
            source_sha256: crate::sha256_32(&source.pixels),
            pose,
            landmarks,
            effects,
        };
        input.validate()?;
        Ok(input)
    }

    pub fn produce(self) -> Result<BodyGeometryJob, Refusal> {
        self.validate()?;
        let mut effects = BTreeMap::new();
        for (pf, effect) in self.effects {
            let input = match effect {
                BodyLandmarkEffect::SixPoint {
                    source,
                    destination,
                    degrees,
                    amount,
                    inverse_aspect,
                    valid_point_count,
                } => {
                    if !matches!(pf, 400 | 404) {
                        return Err(Refusal::ShapeMismatch("body six-point PF"));
                    }
                    let source_points = source
                        .iter()
                        .map(|&point| self.landmarks.source(point))
                        .collect::<Result<Vec<_>, _>>()?
                        .try_into()
                        .map_err(|_| Refusal::ShapeMismatch("body six-point source"))?;
                    let destination_points = destination
                        .iter()
                        .map(|&point| self.landmarks.target(point))
                        .collect::<Result<Vec<_>, _>>()?
                        .try_into()
                        .map_err(|_| Refusal::ShapeMismatch("body six-point destination"))?;
                    BodyGeometryInput::SixPoint(SixPointWarp {
                        source_points,
                        destination_points,
                        degrees,
                        amount,
                        inverse_aspect,
                        valid_point_count,
                    })
                }
                BodyLandmarkEffect::SameJob(input) => input,
            };
            if !effect_matches_pf(pf, &input) {
                return Err(Refusal::ShapeMismatch("body geometry effect kind"));
            }
            if effects.insert(pf, input).is_some() {
                return Err(Refusal::ValueOutOfRange("duplicate body geometry PF"));
            }
        }
        Ok(BodyGeometryJob {
            job_id: self.job_id,
            frame_size: self.frame_size,
            source_sha256: self.source_sha256,
            body_params: super::portrait_body_stage::BodyVendorParams::for_effects(&effects),
            effects,
        })
    }

    pub fn attach(
        self,
        producer: &mut ProducerInputs,
        receipt: &PortraitJobReceipt,
    ) -> Result<(), Refusal> {
        let job = self.produce()?;
        producer.write_body_geometry(job, receipt)
    }

    fn validate(&self) -> Result<(), Refusal> {
        if self.job_id == 0 {
            return Err(Refusal::ValueOutOfRange("body geometry job id"));
        }
        if self.source_sha256.iter().all(|byte| *byte == 0) {
            return Err(Refusal::MissingInput("body geometry frame SHA-256"));
        }
        if self.effects.is_empty() {
            return Err(Refusal::MissingInput("body geometry effects"));
        }
        if self.effects.keys().any(|pf| {
            !super::portrait_body_stage::supports(*pf)
        }) {
            return Err(Refusal::MissingInput("body geometry effects"));
        }
        self.landmarks.validate()
    }
}

fn effect_matches_pf(pf: u32, effect: &BodyGeometryInput) -> bool {
    super::portrait_body_stage::accepts(pf, effect)
}

use aurora_portrait::kernels::{body_geometry_params as curves, body_pose_adapter::AnchorSet};
use super::portrait_body_inputs::{self, BodySubject};
use super::portrait_body_stage::{self, BodyVendorParams};
use super::portrait_body_warp::{self, Tile};
use super::{PortraitBackend, PortraitEffectRun, PortraitSkipReason};

fn body_pf(key: &str) -> Option<u32> {
    let number = key.strip_prefix("pc:")?;
    if number.starts_with(['0', '+']) { return None; }
    let pf = number.parse().ok()?;
    portrait_body_stage::supports(pf).then_some(pf)
}

pub(crate) fn has_live_body_request(extra: &BTreeMap<String, f32>) -> bool {
    extra.iter().any(|(key, &value)| body_pf(key).is_some_and(|pf| {
        value != if pf == 90041 { 0.5 } else { 0.0 }
    }))
}

/// Product storage is 0..1 with zero neutral; the bipolar AI slider is the
/// exception (0.5 neutral). Exact integer percent is only the old external
/// PF90041 shader ABI, not the UI storage domain.
fn mapped_strength(pf: u32, ui: f32) -> Result<f32, Refusal> {
    let signed = matches!(pf, 400 | 403 | 405 | 406 | 410 | 415 | 416 | 417 | 418 | 30074 | 30075);
    if !((if signed { -1.0 } else { 0.0 })..=1.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("body slider"));
    }
    if pf == 90041 { return Ok(ui * 2.0 - 1.0); }
    if matches!(pf, 417 | 418) {
        let pair = if ui < 0.0 { curves::SHOULDER_WIDTH_NARROW } else { curves::SHOULDER_WIDTH_WIDEN };
        return if ui < 0.0 { pair.narrow } else { pair.widen }.remap((ui + 1.0)*0.5);
    }
    let curve = curves::control_curve(pf).ok_or(Refusal::UnknownEffect(pf))?;
    // The two segment-length tables give absolute internal scales; our saved
    // amount is a displacement relative to the unmoved frame.
    let neutral = if matches!(pf, 30094 | 30095) { 0.0 } else { curve.default_param };
    let upper = curve.params[curve.nodes - 1];
    let target = if ui >= 0.0 { upper } else { curve.params[0] };
    let range = target - neutral;
    Ok(curve.remap(neutral + range * ui.abs())? - curve.remap(neutral)?)
}

fn requested(extra: &BTreeMap<String, f32>) -> BTreeMap<u32, Result<f32, Refusal>> {
    extra.iter().filter_map(|(k, &v)| {
        let pf = body_pf(k)?;
        match mapped_strength(pf, v) {
            Ok(n) if n.abs() < 1e-6 => None,
            other => Some((pf, other)),
        }
    }).collect()
}

pub(super) fn generate_body_geometry(
    pf: u32, degree: f32, width: u32, height: u32, anchors: &AnchorSet, mask: &[f32],
) -> Result<BodyGeometryInput, Refusal> {
    if width < 3 || height < 3 || (width as usize).checked_mul(height as usize) != Some(mask.len()) {
        return Err(Refusal::ShapeMismatch("body subject tile"));
    }
    if !degree.is_finite() || mask.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        return Err(Refusal::ValueOutOfRange("body degree or mask"));
    }
    if !mask.iter().any(|&v| v > 0.0) {
        return Err(Refusal::MissingInput("body subject mask"));
    }
    let payload = match pf {
        401 | 403 | 406 | 410..=415 | 417 | 418 | 30074 | 30075 =>
            super::portrait_body_mesh::generate(pf, degree, width, height, anchors, mask)?,
        400 | 404 | 405 | 407 | 408 | 416 | 38556 | 90041 | 30094 | 30095 =>
            super::portrait_body_fields::generate(pf, degree, width, height, anchors, mask)?,
        _ => return Err(Refusal::UnknownEffect(pf)),
    };
    if !portrait_body_stage::accepts(pf, &payload) {
        return Err(Refusal::ShapeMismatch("body geometry effect kind"));
    }
    Ok(payload)
}

fn effect_run(pf: u32, status: Result<(), Refusal>) -> PortraitEffectRun {
    let refusal = status.err();
    let executed = refusal.is_none();
    PortraitEffectRun {
        pf, backend: if executed { PortraitBackend::Cpu } else { PortraitBackend::Skipped },
        dispatches: Vec::new(), metal_dispatches: Vec::new(),
        fallback_reason: refusal.as_ref().map(|e| format!("{e:?}")),
        skipped: !executed, executed,
        skip_reason: refusal.clone().map(PortraitSkipReason::InputRefused), refusal,
    }
}

fn body_cpu_allowed() -> bool {
    if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() { return false; }
    let request = |key| std::env::var(key).ok().map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty() && s != "auto");
    request("AURORA_PORTRAIT_RENDER_BACKEND").or_else(|| request("AURORA_GPU_BACKEND"))
        .is_none_or(|s| s == "cpu")
}

/// Real renderer entry; absence never injects a full-image mask or a template
/// pose. Body geometry is the last image-space transform, after color effects.
pub(crate) fn apply_live_body(
    rgba: &mut [u8], wide: Option<&mut [u16]>, size: (u32, u32), job_id: u64,
    extra: &BTreeMap<String, f32>,
) -> Vec<PortraitEffectRun> {
    let plan = requested(extra);
    if plan.is_empty() { return Vec::new(); }
    let inputs = if !body_cpu_allowed() {
        Err(Refusal::MissingInput("requested backend has no live body geometry"))
    } else if !plan.values().any(Result::is_ok) {
        // Invalid saved sliders are reported below; a model run cannot fix them.
        Ok(Vec::new())
    } else {
        Image::new(size.0, size.1, rgba.to_vec()).ok_or(Refusal::ShapeMismatch("body stage frame"))
            .and_then(|source| portrait_body_inputs::infer_subjects(&source))
    };
    match inputs {
        Ok(subjects) => apply_prepared(rgba, wide, size, job_id, plan, &subjects),
        Err(reason) => plan.into_iter().map(|(pf, strength)|
            effect_run(pf, strength.and(Err(reason.clone())))).collect(),
    }
}

/// Same consumer as `apply_live_body`, with an explicit immutable subject set.
/// The only alternate caller is the generated seam fixture, not a live fallback.
#[cfg(test)]
pub(super) fn apply_body_subjects(
    rgba: &mut [u8], wide: Option<&mut [u16]>, size: (u32, u32), job_id: u64,
    extra: &BTreeMap<String, f32>, subjects: &[BodySubject],
) -> Vec<PortraitEffectRun> {
    apply_prepared(rgba, wide, size, job_id, requested(extra), subjects)
}

fn apply_prepared(
    rgba: &mut [u8], mut wide: Option<&mut [u16]>, size: (u32, u32), job_id: u64,
    plan: BTreeMap<u32, Result<f32, Refusal>>, subjects: &[BodySubject],
) -> Vec<PortraitEffectRun> {
    let bounds = subject_geometry(size, rgba.len(), wide.as_ref().map(|x| x.len()), subjects);
    plan.into_iter().map(|(pf, strength)| {
        let result = strength.and_then(|degree| {
            bounds.clone()?;
            // Build all subject outputs for this row before writing any of them;
            // a stale/invalid pose must not leave a half-applied row in a group.
            let tiles: Vec<_> = subjects.iter().map(|subject| {
                let input = generate_body_geometry(pf, degree, subject.width, subject.height,
                    &subject.anchors, &subject.mask)?;
                let mut tile = Tile {
                    narrow: read_tile(rgba, size.0, subject),
                    wide: wide.as_deref().map(|p| read_tile(p, size.0, subject)),
                };
                let effects = BTreeMap::from([(pf, input)]);
                let job = BodyGeometryJob { job_id, frame_size: (subject.width, subject.height),
                    source_sha256: crate::sha256_32(&tile.narrow),
                    body_params: BodyVendorParams::for_effects(&effects), effects };
                portrait_body_warp::apply(&mut tile, pf, job_id, &job, &subject.mask)?;
                Ok(tile)
            }).collect::<Result<_, Refusal>>()?;
            for (subject, tile) in subjects.iter().zip(tiles) {
                write_tile(rgba, size.0, subject, &tile.narrow);
                if let (Some(dst), Some(src)) = (wide.as_deref_mut(), tile.wide.as_deref()) {
                    write_tile(dst, size.0, subject, src);
                }
            }
            Ok(())
        });
        effect_run(pf, result)
    }).collect()
}

fn subject_geometry(size: (u32, u32), len: usize, wide: Option<usize>, subjects: &[BodySubject]) -> Result<(), Refusal> {
    if (size.0 as usize).checked_mul(size.1 as usize).and_then(|n| n.checked_mul(4)) != Some(len)
        || wide.is_some_and(|n| n != len)
    { return Err(Refusal::ShapeMismatch("body frame")); }
    if subjects.is_empty() { return Err(Refusal::MissingInput("body subject")); }
    for (i, s) in subjects.iter().enumerate() {
        let right = s.x.checked_add(s.width).filter(|&x| x <= size.0)
            .ok_or(Refusal::ShapeMismatch("body ROI"))?;
        let bottom = s.y.checked_add(s.height).filter(|&y| y <= size.1)
            .ok_or(Refusal::ShapeMismatch("body ROI"))?;
        if s.width < 3 || s.height < 3
            || (s.width as usize).checked_mul(s.height as usize) != Some(s.mask.len())
        { return Err(Refusal::ShapeMismatch("body ROI")); }
        if subjects[..i].iter().any(|a| s.x < a.x+a.width && a.x < right
            && s.y < a.y+a.height && a.y < bottom)
        { return Err(Refusal::ValueOutOfRange("overlapping body ROIs")); }
    }
    Ok(())
}

fn tile_rows(stride: u32, subject: &BodySubject) -> impl Iterator<Item=std::ops::Range<usize>> {
    (subject.y..subject.y+subject.height).map(move |y| {
        let lo = (y as usize * stride as usize + subject.x as usize) * 4;
        lo..lo + subject.width as usize * 4
    })
}
fn read_tile<T: Copy>(src: &[T], stride: u32, subject: &BodySubject) -> Vec<T> {
    let mut tile = Vec::with_capacity(subject.width as usize * subject.height as usize * 4);
    for range in tile_rows(stride, subject) { tile.extend_from_slice(&src[range]); }
    tile
}
fn write_tile<T: Copy>(dst: &mut [T], stride: u32, subject: &BodySubject, src: &[T]) {
    for (range, row) in tile_rows(stride, subject).zip(src.chunks_exact(subject.width as usize * 4)) {
        dst[range].copy_from_slice(row);
    }
}
