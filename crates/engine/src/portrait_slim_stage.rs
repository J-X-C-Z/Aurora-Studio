use std::hash::{Hash, Hasher};

use aurora_portrait::cuda_v2fuseoffset::{self, PipelineInput};
use aurora_portrait::cuda_v2warp::Mesh;
use aurora_portrait::raster::Rules;
use aurora_portrait::{Image, Refusal};

use super::sha256_32;
use crate::NormRect;

/// One producer's V2 inputs for the full RGBA8 frame at the slim stage.
#[derive(Clone, Debug, PartialEq)]
pub struct SlimProducerInputs {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    /// SHA-256 of the exact RGBA8 bytes immediately before this stage.
    pub source_sha256: [u8; 32],
    pub grid: Vec<u8>,
    pub base: Vec<[f32; 4]>,
    pub points194: Vec<[f32; 2]>,
    pub framebuffer_positions_w1: Vec<[f32; 2]>,
    pub texcoord: Vec<[f32; 2]>,
    pub offset_texcoord: Vec<[f32; 2]>,
    pub face_scales: Vec<[f32; 2]>,
    pub ebo: Vec<u16>,
    pub rules: Rules,
    pub mat2: [f32; 4],
    pub f_o: [f32; 2],
}

impl SlimProducerInputs {
    /// Builds the same-frame face-width producer input from real landmarks.
    ///
    /// PF8025/PF8026 require their own recovered V2 mesh payload and therefore
    /// stay empty here and fail closed.
    pub fn from_landmarks(
        job_id: u64,
        source: &Image,
        points194: Vec<[f32; 2]>,
    ) -> Result<Self, Refusal> {
        source.validate()?;
        if job_id == 0 {
            return Err(Refusal::ValueOutOfRange("face landmarks job ticket"));
        }
        if points194.len() != 194 {
            return Err(Refusal::ShapeMismatch("face landmarks 194"));
        }
        if points194.iter().flatten().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("face landmarks"));
        }
        Ok(Self {
            job_id,
            width: source.width,
            height: source.height,
            source_sha256: sha256_32(&source.pixels),
            grid: Vec::new(),
            base: Vec::new(),
            points194,
            framebuffer_positions_w1: Vec::new(),
            texcoord: Vec::new(),
            offset_texcoord: Vec::new(),
            face_scales: Vec::new(),
            ebo: Vec::new(),
            rules: Rules {
                winding: aurora_portrait::raster::Winding::PositiveAreaIsFront,
                cull: aurora_portrait::raster::Cull::None,
                edge: aurora_portrait::raster::Edge::Inclusive,
                overlap: aurora_portrait::raster::Overlap::Last,
            },
            mat2: [1.0, 0.0, 0.0, 1.0],
            f_o: [0.0, 0.0],
        })
    }

    pub(super) fn fingerprint(&self) -> u64 {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        self.job_id.hash(&mut hash);
        self.width.hash(&mut hash);
        self.height.hash(&mut hash);
        self.source_sha256.hash(&mut hash);
        self.grid.hash(&mut hash);
        self.ebo.hash(&mut hash);
        let rules = self.rules;
        (
            rules.winding as u8,
            rules.cull as u8,
            rules.edge as u8,
            rules.overlap as u8,
        )
            .hash(&mut hash);
        for plane in [
            &self.points194,
            &self.framebuffer_positions_w1,
            &self.texcoord,
            &self.offset_texcoord,
            &self.face_scales,
        ] {
            plane.len().hash(&mut hash);
            for point in plane {
                point.map(f32::to_bits).hash(&mut hash);
            }
        }
        self.base.len().hash(&mut hash);
        for texel in &self.base {
            texel.map(f32::to_bits).hash(&mut hash);
        }
        self.mat2.map(f32::to_bits).hash(&mut hash);
        self.f_o.map(f32::to_bits).hash(&mut hash);
        hash.finish()
    }

    pub(super) fn validate_ticket(
        &self,
        source: &Image,
        region: NormRect,
        job_id: Option<u64>,
    ) -> Result<(), Refusal> {
        if self.job_id == 0 || Some(self.job_id) != job_id {
            return Err(Refusal::ValueOutOfRange("PF8025/PF8026 job ticket"));
        }
        if (self.width, self.height) != (source.width, source.height) {
            return Err(Refusal::ShapeMismatch("PF8025/PF8026 frame/region"));
        }
        Ok(())
    }

    pub(super) fn mesh(&self) -> Mesh<'_> {
        Mesh {
            framebuffer_positions_w1: &self.framebuffer_positions_w1,
            texcoord: &self.texcoord,
            offset_texcoord: &self.offset_texcoord,
            face_scales: &self.face_scales,
            ebo: &self.ebo,
        }
    }
}

pub(super) fn apply(
    source: &Image,
    region: NormRect,
    job_id: Option<u64>,
    input: Option<&SlimProducerInputs>,
    controls: [f32; 2],
) -> Result<Option<Image>, Refusal> {
    if controls.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err(Refusal::ValueOutOfRange("PF8025/PF8026 UI"));
    }
    if controls == [0.0, 0.0] {
        return Ok(None);
    }
    let input = input.ok_or(Refusal::MissingInput("PF8025/PF8026 same-frame V2 inputs"))?;
    if input.job_id == 0 || Some(input.job_id) != job_id {
        return Err(Refusal::ValueOutOfRange("PF8025/PF8026 job ticket"));
    }
    if (input.width, input.height) != (source.width, source.height) {
        return Err(Refusal::ShapeMismatch("PF8025/PF8026 frame/region"));
    }
    if input.source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("PF8025/PF8026 frame digest"));
    }

    let mesh = input.mesh();
    let mut has_area = false;
    for triangle in mesh.ebo.chunks_exact(3) {
        let point = |index: u16| mesh.framebuffer_positions_w1.get(index as usize);
        let [Some(a), Some(b), Some(c)] =
            [point(triangle[0]), point(triangle[1]), point(triangle[2])]
        else {
            return Err(Refusal::ValueOutOfRange("PF8025/PF8026 mesh index"));
        };
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if !area.is_finite() {
            return Err(Refusal::ValueOutOfRange("PF8025/PF8026 triangle area"));
        }
        has_area |= area.abs() > 1e-12;
    }
    if !has_area {
        return Err(Refusal::ValueOutOfRange("PF8025/PF8026 degenerate mesh"));
    }
    let [left_degree, right_degree] = controls.map(|value| value * 1.5);
    let output = cuda_v2fuseoffset::apply_cpu(&PipelineInput {
        ticket: input.job_id,
        source,
        grid: &input.grid,
        base: &input.base,
        left_degree,
        right_degree,
        points194: &input.points194,
        mesh,
        rules: input.rules,
        mat2: input.mat2,
        f_o: input.f_o,
    })?;
    Ok(Some(output.image))
}

#[cfg(test)]
#[path = "portrait_slim_stage_tests.rs"]
pub(super) mod tests;
