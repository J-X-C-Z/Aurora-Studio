//! Stable body-pose to body-shader ABI.
//!
//! The native model contract is exactly 69 ordered keypoints. Slot order is
//! the model output order normalized by `BodyPoseOutputLayout`; this module
//! does not assign semantic names to unknown slots or synthesize missing ones.
//!
//! A model whose `K` is not 69 has **no** route into this set: its keypoints go
//! through `BodyPoseInference::semantic_anchors`, which returns anchored
//! semantics. Writing that semantic order into these 69 slots would invent a
//! vendor slot table, so `BodyKeypointSet::from_inference` keeps refusing any
//! contract other than [`BODY_LANDMARK_COUNT`].

use aurora_portrait::kernels::body_geometry::{Mesh, Vec2};
use aurora_portrait::Refusal;

use crate::aimask::body_pose::{
    BodyKeypoint, BodyPoseInference, BODY_POSE_KEYPOINT_COUNT,
};
use crate::portrait_stage::{BodyGeometryInput, SixPointWarp};

pub const BODY_LANDMARK_COUNT: usize = BODY_POSE_KEYPOINT_COUNT;
pub const BODY_SIX_POINT_SLOTS: usize = 6;
pub const BODY_VBO_STRIDE_BYTES: usize = 16;
pub const BODY_VBO_DESTINATION_XY_OFFSET: usize = 0;
pub const BODY_VBO_SOURCE_UV_OFFSET: usize = 8;
pub const BODY_EBO_INDEX_BYTES: usize = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct BodyKeypointSet {
    points: [BodyKeypoint; BODY_LANDMARK_COUNT],
}

impl BodyKeypointSet {
    pub fn from_inference(inference: &BodyPoseInference) -> Result<Self, Refusal> {
        if inference.tensor_contract.keypoints != BODY_LANDMARK_COUNT
            || inference.keypoints.len() != BODY_LANDMARK_COUNT
        {
            return Err(Refusal::ShapeMismatch("body pose 69-point ABI"));
        }
        let points = inference
            .keypoints
            .as_slice()
            .try_into()
            .map_err(|_| Refusal::ShapeMismatch("body pose 69-point ABI"))?;
        let set = Self { points };
        set.validate()?;
        Ok(set)
    }

    pub fn from_ordered(points: [BodyKeypoint; BODY_LANDMARK_COUNT]) -> Result<Self, Refusal> {
        let set = Self { points };
        set.validate()?;
        Ok(set)
    }

    pub fn points(&self) -> &[BodyKeypoint; BODY_LANDMARK_COUNT] {
        &self.points
    }

    pub fn landmarks(&self) -> [Vec2; BODY_LANDMARK_COUNT] {
        self.points.map(|point| Vec2::new(point.x, point.y))
    }

    fn validate(&self) -> Result<(), Refusal> {
        if self.points.iter().any(|point| {
            !point.x.is_finite()
                || !point.y.is_finite()
                || !point.score.is_finite()
                || !(0.0..=1.0).contains(&point.x)
                || !(0.0..=1.0).contains(&point.y)
        }) {
            return Err(Refusal::ValueOutOfRange("body pose 69-point ABI"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BodySixPointShaderInput {
    pub source_points: [Vec2; BODY_SIX_POINT_SLOTS],
    pub destination_points: [Vec2; BODY_SIX_POINT_SLOTS],
    pub degrees: [f32; BODY_SIX_POINT_SLOTS],
    pub amount: f32,
    pub inverse_aspect: f32,
    pub valid_point_count: u32,
}

impl BodySixPointShaderInput {
    pub fn bind_ordered(
        keypoints: &BodyKeypointSet,
        slots: [usize; BODY_SIX_POINT_SLOTS],
    ) -> Result<[Vec2; BODY_SIX_POINT_SLOTS], Refusal> {
        let landmarks = keypoints.landmarks();
        let points = slots
            .iter()
            .map(|&slot| {
                landmarks
                    .get(slot)
                    .copied()
                    .ok_or(Refusal::ValueOutOfRange("body landmark slot"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        points
            .try_into()
            .map_err(|_| Refusal::ShapeMismatch("body six-point slots"))
    }

    pub fn into_geometry_input(self) -> Result<BodyGeometryInput, Refusal> {
        if self.valid_point_count == 0
            || self.valid_point_count as usize > BODY_SIX_POINT_SLOTS
            || !self.amount.is_finite()
            || !self.inverse_aspect.is_finite()
            || self.source_points.iter().any(|point| !finite(*point))
            || self
                .destination_points
                .iter()
                .any(|point| !finite(*point))
            || self.degrees.iter().any(|degree| !degree.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("body six-point shader ABI"));
        }
        Ok(BodyGeometryInput::SixPoint(SixPointWarp {
            source_points: self.source_points,
            destination_points: self.destination_points,
            degrees: self.degrees,
            amount: self.amount,
            inverse_aspect: self.inverse_aspect,
            valid_point_count: self.valid_point_count as usize,
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedBodyGeometry {
    pub vbo: Vec<u8>,
    pub ebo: Vec<u8>,
}

/// Encode the native mesh stream as LE f32 destination.xy followed by source UV.
pub fn encode_body_geometry(mesh: &Mesh) -> Result<EncodedBodyGeometry, Refusal> {
    let mesh = Mesh::new(
        mesh.source_uv.clone(),
        mesh.destination.clone(),
        mesh.indices.clone(),
    )?;
    let mut vbo = Vec::with_capacity(mesh.source_uv.len() * BODY_VBO_STRIDE_BYTES);
    for (&source_uv, &destination) in mesh.source_uv.iter().zip(&mesh.destination) {
        for value in [destination.x, destination.y, source_uv.x, source_uv.y] {
            vbo.extend_from_slice(&value.to_le_bytes());
        }
    }
    let mut ebo = Vec::with_capacity(mesh.indices.len() * BODY_EBO_INDEX_BYTES);
    for index in &mesh.indices {
        ebo.extend_from_slice(&index.to_le_bytes());
    }
    Ok(EncodedBodyGeometry { vbo, ebo })
}

pub fn decode_body_geometry(vbo: &[u8], ebo: &[u8]) -> Result<Mesh, Refusal> {
    if vbo.is_empty() {
        return Err(Refusal::MissingInput("body geometry VBO"));
    }
    if vbo.len() % BODY_VBO_STRIDE_BYTES != 0 {
        return Err(Refusal::ShapeMismatch("body geometry VBO"));
    }
    if ebo.is_empty() {
        return Err(Refusal::MissingInput("body geometry EBO"));
    }
    if ebo.len() % BODY_EBO_INDEX_BYTES != 0 {
        return Err(Refusal::ShapeMismatch("body geometry EBO"));
    }

    let mut destination = Vec::with_capacity(vbo.len() / BODY_VBO_STRIDE_BYTES);
    let mut source_uv = Vec::with_capacity(vbo.len() / BODY_VBO_STRIDE_BYTES);
    for vertex in vbo.chunks_exact(BODY_VBO_STRIDE_BYTES) {
        let value = |offset: usize| {
            f32::from_le_bytes(
                vertex[offset..offset + 4]
                    .try_into()
                    .expect("fixed VBO field"),
            )
        };
        let destination_xy = [
            value(BODY_VBO_DESTINATION_XY_OFFSET),
            value(BODY_VBO_DESTINATION_XY_OFFSET + 4),
        ];
        let source_xy = [
            value(BODY_VBO_SOURCE_UV_OFFSET),
            value(BODY_VBO_SOURCE_UV_OFFSET + 4),
        ];
        if destination_xy
            .iter()
            .chain(&source_xy)
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("body geometry VBO"));
        }
        destination.push(Vec2::new(destination_xy[0], destination_xy[1]));
        source_uv.push(Vec2::new(source_xy[0], source_xy[1]));
    }
    let indices = ebo
        .chunks_exact(BODY_EBO_INDEX_BYTES)
        .map(|index| u16::from_le_bytes([index[0], index[1]]))
        .collect();
    Mesh::new(source_uv, destination, indices)
        .map_err(|_| Refusal::ValueOutOfRange("body geometry VBO/EBO"))
}

fn finite(point: Vec2) -> bool {
    point.x.is_finite() && point.y.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aimask::body_pose::{BodyPoseOutputLayout, BodyPoseTensorContract, PoseChannels};

    fn inference(count: usize) -> BodyPoseInference {
        BodyPoseInference {
            model_sha256: [1; 32],
            provider: "synthetic-test".into(),
            frame_size: (100, 100),
            source_sha256: [2; 32],
            input_size: (256, 256),
            tensor_contract: BodyPoseTensorContract {
                input_name: "image".into(),
                input_dims: [1, 3, 256, 256],
                input_element_type: 1,
                output_name: "pose".into(),
                output_dims: vec![1, count as i64, 3],
                output_element_type: 1,
                output_layout: BodyPoseOutputLayout::KeypointXyz,
                keypoints: count,
            },
            keypoints: (0..count)
                .map(|index| BodyKeypoint {
                    x: (index as f32 + 0.5) / BODY_LANDMARK_COUNT as f32,
                    y: (index as f32 + 0.5) / BODY_LANDMARK_COUNT as f32,
                    score: 1.0,
                })
                .collect(),
            // 69-slot ABI 不读语义通道；这里只是构造一个合法的推断记录。
            channels: PoseChannels::default(),
        }
    }

    #[test]
    fn model_order_and_six_slots_are_stable() {
        let pose = inference(BODY_LANDMARK_COUNT);
        let points = BodyKeypointSet::from_inference(&pose).unwrap();
        let slots = BodySixPointShaderInput::bind_ordered(&points, [0, 1, 2, 3, 4, 68])
            .unwrap();
        assert_eq!(slots[0], points.landmarks()[0]);
        assert_eq!(slots[5], points.landmarks()[68]);
        let input = BodySixPointShaderInput {
            source_points: slots,
            destination_points: slots,
            degrees: [1.0; BODY_SIX_POINT_SLOTS],
            amount: 1.0,
            inverse_aspect: 1.0,
            valid_point_count: BODY_SIX_POINT_SLOTS as u32,
        };
        assert!(matches!(
            input.into_geometry_input().unwrap(),
            BodyGeometryInput::SixPoint(_)
        ));
        assert!(BodyKeypointSet::from_inference(&inference(BODY_LANDMARK_COUNT - 1)).is_err());
    }

    #[test]
    fn vbo_destination_then_source_uv_round_trips() {
        let mesh = Mesh::new(
            vec![
                Vec2::new(0.1, 0.2),
                Vec2::new(0.3, 0.4),
                Vec2::new(0.5, 0.6),
            ],
            vec![
                Vec2::new(0.7, 0.8),
                Vec2::new(0.9, 0.1),
                Vec2::new(0.2, 0.3),
            ],
            vec![0, 1, 2],
        )
        .unwrap();
        let encoded = encode_body_geometry(&mesh).unwrap();
        assert_eq!(encoded.vbo.len(), 3 * BODY_VBO_STRIDE_BYTES);
        assert_eq!(
            f32::from_le_bytes(encoded.vbo[0..4].try_into().unwrap()),
            0.7
        );
        assert_eq!(
            f32::from_le_bytes(encoded.vbo[8..12].try_into().unwrap()),
            0.1
        );
        assert_eq!(
            decode_body_geometry(&encoded.vbo, &encoded.ebo).unwrap(),
            mesh
        );
    }
}
