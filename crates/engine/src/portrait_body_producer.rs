use std::collections::BTreeMap;

use aurora_portrait::kernels::body_geometry::{Mesh, Vec2};
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
            !matches!(*pf, 400 | 401 | 404..=408 | 410 | 415 | 416 | 38556 | 90041)
        }) {
            return Err(Refusal::MissingInput("body geometry effects"));
        }
        self.landmarks.validate()
    }
}

fn effect_matches_pf(pf: u32, effect: &BodyGeometryInput) -> bool {
    matches!(
        (pf, effect),
        (400 | 404, BodyGeometryInput::SixPoint(_))
            | (
                401 | 406 | 410 | 415,
                BodyGeometryInput::Mesh {
                    mesh: Mesh { .. },
                    ..
                }
            )
            | (405, BodyGeometryInput::BodySlim { .. })
            | (407, BodyGeometryInput::Leg { .. })
            | (408, BodyGeometryInput::Height { .. })
            | (416, BodyGeometryInput::Waist { .. })
            | (38556, BodyGeometryInput::BreastLift { .. })
            | (90041, BodyGeometryInput::Shift(_))
    )
}
