use std::collections::BTreeMap;

use aurora_portrait::kernels::body_geometry::{Mesh, Vec2};
use aurora_portrait::kernels::body_pose_adapter::{
    adopt, six_point_anchors, AnchorSet, ModelLayout, ObservedKeypoint,
};
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

/// Caller-supplied six-point destination and per-point degrees.
///
/// The source points come from a real body-pose model through
/// [`six_point_warp_from_pose`]. This struct never derives the destination or
/// the degrees from the slider value: the vendor host law that maps a PF
/// degree onto control-point displacement is not recovered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SixPointTargets {
    pub destination: [Vec2; 6],
    pub degrees: [f32; 6],
    /// Native strength for PF400 (native scale for PF404).
    pub amount: f32,
    pub inverse_aspect: f32,
}

/// Maps one pose inference onto the named semantic anchors.
///
/// `layout` is the model's keypoint system (COCO-17 / Halpe-26 / ...). Points
/// below `min_score`, non-finite, or outside `[0,1]` are recorded as missing by
/// `adopt`; nothing is interpolated.
pub fn anchor_set_from_pose(
    pose: &BodyPoseInference,
    layout: ModelLayout,
    min_score: f32,
) -> Result<AnchorSet, Refusal> {
    let observed: Vec<ObservedKeypoint> = pose
        .keypoints
        .iter()
        .map(|point| ObservedKeypoint {
            x: point.x,
            y: point.y,
            score: point.score,
            present: true,
        })
        .collect();
    adopt(layout, &observed, min_score)
}

/// Builds PF400's six-point warp from a pose inference.
///
/// Source points are the model's left/right shoulder, elbow and wrist anchors
/// (`six_point_anchors(400)`). PF404 is refused by name: its buttock rim points
/// are not keypoints, so there is no honest source for them here.
pub fn six_point_warp_from_pose(
    pf: u32,
    pose: &BodyPoseInference,
    layout: ModelLayout,
    min_score: f32,
    targets: SixPointTargets,
) -> Result<SixPointWarp, Refusal> {
    let Some(anchors) = six_point_anchors(pf) else {
        return Err(Refusal::MissingInput("PF404 six-point anchors"));
    };
    let set = anchor_set_from_pose(pose, layout, min_score)?;
    let mut source_points = [Vec2::new(0.0, 0.0); 6];
    for (slot, anchor) in anchors.iter().enumerate() {
        let point = set.require(*anchor)?;
        source_points[slot] = Vec2::new(point.x, point.y);
    }
    if targets
        .destination
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
        || targets.degrees.iter().any(|value| !value.is_finite())
        || !targets.amount.is_finite()
        || !targets.inverse_aspect.is_finite()
        || targets.inverse_aspect <= 0.0
    {
        return Err(Refusal::ValueOutOfRange("body six-point targets"));
    }
    Ok(SixPointWarp {
        source_points,
        destination_points: targets.destination,
        degrees: targets.degrees,
        amount: targets.amount,
        inverse_aspect: targets.inverse_aspect,
        valid_point_count: 6,
    })
}

/// Builds a same-job body geometry job whose six-point rows take their source
/// points from `pose`.
///
/// `targets` must name every row explicitly; an empty list is a named refusal
/// rather than an empty job. Rows without a target entry are simply not part of
/// the job, so the stage reports its own missing-payload refusal for them.
pub fn six_point_job_from_pose(
    job_id: u64,
    source: &Image,
    pose: &BodyPoseInference,
    layout: ModelLayout,
    min_score: f32,
    targets: &[(u32, SixPointTargets)],
) -> Result<BodyGeometryJob, Refusal> {
    if targets.is_empty() {
        return Err(Refusal::MissingInput("body six-point targets"));
    }
    // The pose is consumed into named anchors here; `from_current_frame` would
    // otherwise require the full 69-point ABI that this path does not use.
    if pose.frame_size != (source.width, source.height)
        || pose.source_sha256 != crate::sha256_32(&source.pixels)
    {
        return Err(Refusal::ValueOutOfRange("body pose same-frame receipt"));
    }
    let mut effects = BTreeMap::new();
    for (pf, target) in targets {
        let warp = six_point_warp_from_pose(*pf, pose, layout, min_score, *target)?;
        if effects
            .insert(*pf, BodyLandmarkEffect::SameJob(BodyGeometryInput::SixPoint(warp)))
            .is_some()
        {
            return Err(Refusal::ValueOutOfRange("duplicate body geometry PF"));
        }
    }
    BodyGeometryProducerInputs::from_current_frame(
        job_id,
        source,
        None,
        BodyLandmarkFrame::default(),
        effects,
    )?
    .produce()
}

/// JSON file holding the six-point target payload used by
/// [`attach_pose_six_point`]. Six destination points, six degrees, strength and
/// inverse aspect. Never derived from the slider value.
pub const SIX_POINT_TARGETS_ENV: &str = "AURORA_BODY_SIX_POINT_TARGETS";
/// Keypoint layout of the configured body pose model.
pub const BODY_POSE_LAYOUT_ENV: &str = "AURORA_BODY_POSE_LAYOUT";
/// Optional confidence floor for anchor adoption (0..1).
pub const BODY_POSE_MIN_SCORE_ENV: &str = "AURORA_BODY_POSE_MIN_SCORE";

#[derive(serde::Deserialize)]
struct SixPointTargetsFile {
    destination: [[f32; 2]; 6],
    degrees: [f32; 6],
    amount: f32,
    inverse_aspect: f32,
}

fn parse_layout(value: &str) -> Option<ModelLayout> {
    match value.to_ascii_lowercase().as_str() {
        "coco17" | "coco-17" => Some(ModelLayout::Coco17),
        "halpe26" | "halpe-26" => Some(ModelLayout::Halpe26),
        "openpose25" | "openpose-25" | "body25" => Some(ModelLayout::OpenPose25),
        "mediapipe33" | "mediapipe-33" => Some(ModelLayout::MediaPipe33),
        "wholebody133" | "wholebody-133" | "rtmw133" => Some(ModelLayout::WholeBody133Body),
        _ => None,
    }
}

fn load_targets_from_env() -> Result<Option<SixPointTargets>, Refusal> {
    let Some(path) = std::env::var_os(SIX_POINT_TARGETS_ENV) else {
        return Ok(None);
    };
    let bytes = std::fs::read(&path).map_err(|_| Refusal::MissingInput(SIX_POINT_TARGETS_ENV))?;
    let file: SixPointTargetsFile =
        serde_json::from_slice(&bytes).map_err(|_| Refusal::ValueOutOfRange(SIX_POINT_TARGETS_ENV))?;
    Ok(Some(SixPointTargets {
        destination: file.destination.map(|point| Vec2::new(point[0], point[1])),
        degrees: file.degrees,
        amount: file.amount,
        inverse_aspect: file.inverse_aspect,
    }))
}

/// Attaches PF400's six-point job using a locally configured body-pose model.
///
/// Opt-in and fail-closed:
/// - no active `pc:400` → `Ok(false)`, nothing loaded;
/// - `AURORA_BODY_POSE_MODEL` / its `.sha256` and `AURORA_BODY_POSE_LAYOUT`
///   must be present, otherwise a named `MissingInput`;
/// - `AURORA_BODY_SIX_POINT_TARGETS` must point at a JSON target payload; the
///   source points come from the pose model, the destinations never do.
///
/// Other body PFs are left to their own same-job payloads; they keep refusing
/// by name rather than receiving synthesized geometry.
pub fn attach_pose_six_point(
    source: &Image,
    extra: &BTreeMap<String, f32>,
    receipt: &PortraitJobReceipt,
    producer: &mut ProducerInputs,
) -> Result<bool, Refusal> {
    let degree = extra.get("pc:400").copied().unwrap_or(0.0);
    if !degree.is_finite() || degree == 0.0 {
        return Ok(false);
    }
    let layout_name =
        std::env::var(BODY_POSE_LAYOUT_ENV).map_err(|_| Refusal::MissingInput(BODY_POSE_LAYOUT_ENV))?;
    let layout =
        parse_layout(&layout_name).ok_or(Refusal::ValueOutOfRange(BODY_POSE_LAYOUT_ENV))?;
    let min_score = match std::env::var(BODY_POSE_MIN_SCORE_ENV) {
        Ok(value) => value
            .parse::<f32>()
            .map_err(|_| Refusal::ValueOutOfRange(BODY_POSE_MIN_SCORE_ENV))?,
        Err(_) => aurora_portrait::kernels::body_pose_adapter::MIN_SCORE_DEFAULT,
    };
    let targets = load_targets_from_env()?
        .ok_or(Refusal::MissingInput(SIX_POINT_TARGETS_ENV))?;
    let model = crate::aimask::body_pose::BodyPoseModel::from_local()
        .map_err(|_| Refusal::MissingInput(crate::aimask::body_pose::BODY_POSE_ACQUISITION_REFUSAL))?;
    let pose = model.infer(source).map_err(|error| {
        log::warn!("body pose inference failed: {error:#}");
        Refusal::MissingInput("body pose inference")
    })?;
    let job = six_point_job_from_pose(
        receipt.job_id(),
        source,
        &pose,
        layout,
        min_score,
        &[(400, targets)],
    )?;
    producer.write_body_geometry(job, receipt)?;
    Ok(true)
}

fn effect_matches_pf(pf: u32, effect: &BodyGeometryInput) -> bool {
    super::portrait_body_stage::accepts(pf, effect)
}

use aurora_portrait::kernels::body_geometry_params as curves;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aimask::body_pose::{
        BodyKeypoint, BodyPoseOutputLayout, BodyPoseTensorContract,
    };

    fn frame() -> Image {
        Image::new(64, 48, vec![32; 64 * 48 * 4]).expect("frame")
    }

    fn synthetic_pose(source: &Image, override_score: Option<(usize, f32)>) -> BodyPoseInference {
        let uv = [
            (0.50, 0.30), // 0 nose
            (0.45, 0.28), // 1 left eye
            (0.55, 0.28), // 2 right eye
            (0.42, 0.30), // 3 left ear
            (0.58, 0.30), // 4 right ear
            (0.35, 0.45), // 5 left shoulder
            (0.65, 0.45), // 6 right shoulder
            (0.30, 0.58), // 7 left elbow
            (0.70, 0.58), // 8 right elbow
            (0.25, 0.72), // 9 left wrist
            (0.75, 0.72), // 10 right wrist
            (0.40, 0.75), // 11 left hip
            (0.60, 0.75), // 12 right hip
            (0.40, 0.88), // 13 left knee
            (0.60, 0.88), // 14 right knee
            (0.40, 0.97), // 15 left ankle
            (0.60, 0.97), // 16 right ankle
        ];
        let keypoints = uv
            .iter()
            .enumerate()
            .map(|(index, (x, y))| {
                let score = match override_score {
                    Some((target, score)) if target == index => score,
                    _ => 0.9,
                };
                BodyKeypoint {
                    x: *x,
                    y: *y,
                    score,
                }
            })
            .collect();
        BodyPoseInference {
            model_sha256: [7; 32],
            provider: "test".to_string(),
            frame_size: (source.width, source.height),
            source_sha256: crate::sha256_32(&source.pixels),
            input_size: (256, 192),
            tensor_contract: BodyPoseTensorContract {
                input_name: "input".to_string(),
                input_dims: [1, 3, 256, 192],
                input_element_type: 1,
                output_name: "output".to_string(),
                output_dims: vec![1, 17, 3],
                output_element_type: 1,
                output_layout: BodyPoseOutputLayout::KeypointXyz,
                keypoints: 17,
            },
            keypoints,
            channels: crate::aimask::body_pose::PoseChannels {
                third: crate::aimask::body_pose::PoseThirdChannel::Score,
                visibility: None,
                presence: None,
            },
        }
    }

    fn targets() -> SixPointTargets {
        SixPointTargets {
            destination: [
                Vec2::new(0.34, 0.45),
                Vec2::new(0.66, 0.45),
                Vec2::new(0.29, 0.58),
                Vec2::new(0.71, 0.58),
                Vec2::new(0.24, 0.72),
                Vec2::new(0.76, 0.72),
            ],
            degrees: [1.0; 6],
            amount: 0.8,
            inverse_aspect: 1.0,
        }
    }

    #[test]
    fn pf400_source_points_come_from_pose_anchors() {
        let image = frame();
        let pose = synthetic_pose(&image, None);
        let warp = six_point_warp_from_pose(400, &pose, ModelLayout::Coco17, 0.3, targets())
            .expect("six point warp");
        assert_eq!(warp.source_points[0], Vec2::new(0.35, 0.45));
        assert_eq!(warp.source_points[5], Vec2::new(0.75, 0.72));
        assert_eq!(warp.valid_point_count, 6);
        assert_eq!(warp.destination_points[0], Vec2::new(0.34, 0.45));
    }

    #[test]
    fn low_score_anchor_is_named_not_interpolated() {
        let image = frame();
        let pose = synthetic_pose(&image, Some((9, 0.1)));
        let error = six_point_warp_from_pose(400, &pose, ModelLayout::Coco17, 0.3, targets())
            .expect_err("missing left wrist must be refused");
        assert_eq!(error, Refusal::MissingInput("body pose anchor A5"));
    }

    #[test]
    fn pf404_has_no_keypoint_source() {
        let image = frame();
        let pose = synthetic_pose(&image, None);
        let error = six_point_warp_from_pose(404, &pose, ModelLayout::Coco17, 0.3, targets())
            .expect_err("PF404 buttock rim is not a keypoint");
        assert_eq!(error, Refusal::MissingInput("PF404 six-point anchors"));
    }

    #[test]
    fn job_without_targets_is_a_named_refusal() {
        let image = frame();
        let pose = synthetic_pose(&image, None);
        let error = six_point_job_from_pose(7, &image, &pose, ModelLayout::Coco17, 0.3, &[])
            .expect_err("empty targets");
        assert_eq!(error, Refusal::MissingInput("body six-point targets"));
    }

    #[test]
    fn job_from_pose_carries_the_pf400_six_point_warp() {
        let image = frame();
        let pose = synthetic_pose(&image, None);
        let job = six_point_job_from_pose(
            7,
            &image,
            &pose,
            ModelLayout::Coco17,
            0.3,
            &[(400, targets())],
        )
        .expect("job");
        assert_eq!(job.job_id, 7);
        assert!(matches!(
            job.effects.get(&400),
            Some(BodyGeometryInput::SixPoint(_))
        ));
    }
}
