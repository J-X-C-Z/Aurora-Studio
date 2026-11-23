//! Same-job mesh handoff for forehead (PF30080..PF30083) and face width
//! (PF30092/PF30093).
//!
//! The caller supplies only draws produced for the exact frame. This module
//! validates the recovered part/PF identity and topology, but it does not
//! synthesize points, EBOs, textures, degrees or a producer ticket.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[cfg(windows)]
use crate::cuda::{CudaError, Dispatch};
#[cfg(windows)]
use crate::cuda_mesh::MeshRaster;
use crate::kernels::{eye_geometry, face_detail, face_geometry, forehead_common};
use crate::{Image, Refusal, raster, raster_fragment};

pub const FACE_WIDTH_PARTS: [u8; 4] = [25, 26, 27, 28];
pub const FOREHEAD_WIDTH_PARTS: [u8; 2] = [35, 36];
pub const FOREHEAD_HEIGHT_PARTS: [u8; 2] = [37, 38];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Type1011,
    Type1012,
    Type1016,
}

pub fn family_for_part(part: u8) -> Option<Family> {
    match part {
        6 | 7 => Some(Family::Type1011),
        35..=38 => Some(Family::Type1012),
        25..=28 => Some(Family::Type1016),
        _ => None,
    }
}

pub fn pf_for_part(part: u8) -> Option<u32> {
    match part {
        6 => Some(30073),
        7 => Some(30079),
        35 => Some(30080),
        36 => Some(30081),
        37 => Some(30082),
        38 => Some(30083),
        25 | 27 => Some(30092),
        26 | 28 => Some(30093),
        _ => None,
    }
}

/// One already-produced vendor draw.
#[derive(Clone, Debug, PartialEq)]
pub struct Draw {
    pub part: u8,
    pub pf: u32,
    pub controls: [f32; 2],
    pub source_sha256: [u8; 32],
    pub positions: Vec<[f32; 2]>,
    pub source_uv: Vec<[f32; 2]>,
    pub indices: Vec<u16>,
}

impl Draw {
    pub fn new(
        part: u8,
        pf: u32,
        controls: [f32; 2],
        source_sha256: [u8; 32],
        positions: Vec<[f32; 2]>,
        source_uv: Vec<[f32; 2]>,
        indices: Vec<u16>,
    ) -> Result<Self, Refusal> {
        let family = family_for_part(part).ok_or(Refusal::ValueOutOfRange("face mesh part"))?;
        if pf_for_part(part) != Some(pf) {
            return Err(Refusal::ValueOutOfRange("face mesh part/PF"));
        }
        if controls
            .iter()
            .any(|value| !value.is_finite() || !(-100.0..=100.0).contains(value))
        {
            return Err(Refusal::ValueOutOfRange("face mesh controls"));
        }
        if positions.len() != source_uv.len() {
            return Err(Refusal::ShapeMismatch("face mesh vertices"));
        }
        if positions
            .iter()
            .chain(&source_uv)
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("face mesh vertex"));
        }
        if indices.is_empty() || indices.len() % 3 != 0 {
            return Err(Refusal::ShapeMismatch("face mesh indices"));
        }
        if indices
            .iter()
            .any(|&index| index as usize >= positions.len())
        {
            return Err(Refusal::ValueOutOfRange("face mesh index"));
        }

        match family {
            Family::Type1011 => {
                if positions.len() != 100 {
                    return Err(Refusal::ShapeMismatch("Type1011 prepared100 topology"));
                }
                let prepared: Vec<eye_geometry::Point> = positions
                    .iter()
                    .map(|point| eye_geometry::Point::new(point[0], point[1]))
                    .collect();
                let generated = eye_geometry::type1011_indices(&prepared)
                    .map_err(|_| Refusal::ShapeMismatch("Type1011 prepared100 EBO"))?;
                if indices.as_slice() != generated.as_slice() {
                    return Err(Refusal::ValueOutOfRange("Type1011 prepared100 EBO"));
                }
            }
            Family::Type1012 => {
                if positions.len() != 136 || indices.len() != 705 {
                    return Err(Refusal::ShapeMismatch("Type1012 136/705 topology"));
                }
                if indices.as_slice() != face_detail::TYPE1012_INDICES.as_slice() {
                    return Err(Refusal::ValueOutOfRange("Type1012 fixed EBO"));
                }
            }
            Family::Type1016 => {
                if positions.len() != 27 {
                    return Err(Refusal::ShapeMismatch("Type1016 27-vertex topology"));
                }
            }
        }

        Ok(Self {
            part,
            pf,
            controls,
            source_sha256,
            positions,
            source_uv,
            indices,
        })
    }

    /// Consumes the prepared100 source points and the already-deformed target
    /// points from the Type1011 producer. The EBO must be generated from the
    /// prepared points, never from the deformed target.
    pub fn from_type1011(
        part: u8,
        pf: u32,
        controls: [f32; 2],
        source_sha256: [u8; 32],
        prepared: &[[f32; 2]],
        positions: Vec<[f32; 2]>,
        source_uv: Vec<[f32; 2]>,
        indices: Vec<u16>,
    ) -> Result<Self, Refusal> {
        if prepared.len() != 100 || positions.len() != 100 || source_uv.len() != 100 {
            return Err(Refusal::ShapeMismatch("Type1011 prepared100 topology"));
        }
        if positions
            .iter()
            .chain(&source_uv)
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("face mesh vertex"));
        }
        let source_positions = prepared;
        let prepared: Vec<eye_geometry::Point> = prepared
            .iter()
            .map(|point| eye_geometry::Point::new(point[0], point[1]))
            .collect();
        let generated = eye_geometry::type1011_indices(&prepared)
            .map_err(|_| Refusal::ShapeMismatch("Type1011 prepared100 EBO"))?;
        if indices.as_slice() != generated.as_slice() {
            return Err(Refusal::ValueOutOfRange("Type1011 prepared100 EBO"));
        }
        let mut draw = Self::new(
            part,
            pf,
            controls,
            source_sha256,
            source_positions.to_vec(),
            source_uv,
            indices,
        )?;
        draw.positions = positions;
        Ok(draw)
    }

    /// Consumes a Type1016 part already produced by `face_geometry::face_width`.
    pub fn from_face_geometry(
        controls: [f32; 2],
        source_sha256: [u8; 32],
        part: &face_geometry::PartGeometry,
    ) -> Result<Self, Refusal> {
        let draw = part
            .vendor_draw
            .as_ref()
            .ok_or(Refusal::MissingInput("face width vendor draw"))?;
        Self::new(
            part.part,
            part.pf,
            controls,
            source_sha256,
            draw.positions
                .iter()
                .map(|point| [point.x, point.y])
                .collect(),
            draw.source_uv
                .iter()
                .map(|point| [point.x, point.y])
                .collect(),
            draw.indices.clone(),
        )
    }

    /// Consumes one current common-mesh Type1012 result.
    pub fn from_forehead_common(
        part: u8,
        pf: u32,
        controls: [f32; 2],
        source_sha256: [u8; 32],
        mesh: &forehead_common::Mesh,
        width: u32,
        height: u32,
    ) -> Result<Self, Refusal> {
        if width == 0 || height == 0 {
            return Err(Refusal::ShapeMismatch("face mesh frame"));
        }
        let source_uv = mesh
            .a
            .iter()
            .map(|point| [point.x / width as f32, point.y / height as f32])
            .collect();
        Self::new(
            part,
            pf,
            controls,
            source_sha256,
            mesh.b.iter().map(|point| [point.x, point.y]).collect(),
            source_uv,
            mesh.initial_indices.to_vec(),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub draws: Vec<Draw>,
}

pub type FaceMeshInputs = Inputs;

impl Inputs {
    pub fn new(job_id: u64, width: u32, height: u32) -> Self {
        Self {
            job_id,
            width,
            height,
            draws: Vec::new(),
        }
    }

    pub fn push(&mut self, draw: Draw) {
        self.draws.push(draw);
    }

    pub fn fingerprint(&self) -> u64 {
        let mut hash = DefaultHasher::new();
        (self.job_id, self.width, self.height).hash(&mut hash);
        self.draws.len().hash(&mut hash);
        for draw in &self.draws {
            (
                draw.part,
                draw.pf,
                draw.controls.map(f32::to_bits),
                draw.source_sha256,
            )
                .hash(&mut hash);
            for point in &draw.positions {
                point.map(f32::to_bits).hash(&mut hash);
            }
            for point in &draw.source_uv {
                point.map(f32::to_bits).hash(&mut hash);
            }
            draw.indices.hash(&mut hash);
        }
        hash.finish()
    }
}

const RESHAPE_RULES: raster::Rules = raster::Rules {
    winding: raster::Winding::PositiveAreaIsFront,
    cull: raster::Cull::None,
    edge: raster::Edge::Inclusive,
    overlap: raster::Overlap::Last,
};
const RESHAPE_SAMPLER: raster::Sampler = raster::Sampler {
    wrap: raster::Wrap::ClampToEdge,
};
const PASS1: raster_fragment::Defines = raster_fragment::Defines {
    clamp_edge: true,
    uv_map: false,
    uv_map_ba: false,
    catmull_rom: true,
};
const PASS2: raster_fragment::Defines = raster_fragment::Defines {
    clamp_edge: true,
    uv_map: true,
    uv_map_ba: false,
    catmull_rom: false,
};
pub const RESHAPE_PASSES: [raster_fragment::Defines; 2] = [PASS1, PASS2];

#[cfg(windows)]
#[derive(Clone, Debug, PartialEq)]
pub struct CudaOutput {
    pub image: Image,
    pub dispatches: Vec<Dispatch>,
}

/// Executes the recovered Type1016 two-pass fragment chain in producer order.
///
/// This is the existing CPU `ReshapeNormal` raster contract. On Windows,
/// [`rasterize_cuda_with`] executes the same passes through the fragment emit
/// kernel and returns the actual launch receipts with the image.
pub fn rasterize(source: &Image, draws: &[&Draw]) -> Result<Image, Refusal> {
    validate_draws(source, draws)?;

    let mut current = source.clone();
    for draw in draws {
        let mesh = raster::Mesh {
            dest: &draw.positions,
            source_uv: &draw.source_uv,
            ebo: &draw.indices,
        };
        for defines in RESHAPE_PASSES {
            current = raster::rasterize_with_fragment(
                &current,
                mesh,
                RESHAPE_RULES,
                RESHAPE_SAMPLER,
                raster_fragment::Edge::ReshapeNormal,
                defines,
                raster_fragment::Clamp::disabled(),
                raster::Tolerance::RESHAPE,
            )?;
        }
    }
    Ok(current)
}

fn validate_draws(source: &Image, draws: &[&Draw]) -> Result<(), Refusal> {
    source.validate()?;
    if draws.is_empty() {
        return Err(Refusal::MissingInput("face mesh draws"));
    }
    if draws.iter().any(|draw| draw.family() == Family::Type1012) {
        return Err(Refusal::MissingInput(
            "PF30080..PF30083 Type1012 texture/V2 handoff",
        ));
    }
    Ok(())
}

#[cfg(windows)]
pub fn rasterize_cuda(source: &Image, draws: &[&Draw]) -> Result<CudaOutput, CudaError> {
    let gpu = MeshRaster::open()?;
    rasterize_cuda_with(&gpu, source, draws)
}

/// Runs the producer-ordered two-pass chain on the NVIDIA fragment kernel.
///
/// Every returned dispatch is a launch that actually ran. Callers must keep the
/// rows on `Cpu` when this returns an error and they fall back.
#[cfg(windows)]
pub fn rasterize_cuda_with(
    gpu: &MeshRaster,
    source: &Image,
    draws: &[&Draw],
) -> Result<CudaOutput, CudaError> {
    validate_draws(source, draws).map_err(CudaError::Refused)?;
    let mut current = source.clone();
    let mut dispatches = Vec::with_capacity(draws.len() * RESHAPE_PASSES.len() * 2);
    for draw in draws {
        let mesh = raster::Mesh {
            dest: &draw.positions,
            source_uv: &draw.source_uv,
            ebo: &draw.indices,
        };
        for defines in RESHAPE_PASSES {
            let output = gpu.rasterize_fragment(
                &current,
                mesh,
                RESHAPE_RULES,
                RESHAPE_SAMPLER,
                raster_fragment::Edge::ReshapeNormal,
                defines,
                raster_fragment::Clamp::disabled(),
                raster::Tolerance::RESHAPE,
            )?;
            dispatches.push(output.select_dispatch);
            dispatches.push(output.emit_dispatch);
            current = output.image;
        }
    }
    Ok(CudaOutput {
        image: current,
        dispatches,
    })
}

impl Draw {
    pub fn family(&self) -> Family {
        family_for_part(self.part).expect("Draw::new validated the part")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Image {
        Image::new(
            8,
            8,
            (0..8 * 8 * 4)
                .map(|index| ((index * 37 + 11) % 256) as u8)
                .collect(),
        )
        .unwrap()
    }

    fn face_width_draw() -> Draw {
        let mut positions = vec![[4.0, 4.0]; 27];
        positions[0] = [1.0, 1.0];
        positions[1] = [7.0, 1.0];
        positions[2] = [1.0, 7.0];
        let mut source_uv = vec![[0.5, 0.5]; 27];
        source_uv[0] = [0.25, 0.125];
        source_uv[1] = [0.9375, 0.125];
        source_uv[2] = [0.25, 0.875];
        Draw::new(
            25,
            30092,
            [50.0, 50.0],
            [0; 32],
            positions,
            source_uv,
            vec![0, 1, 2],
        )
        .unwrap()
    }

    fn type1011_draw(part: u8, pf: u32) -> Draw {
        let prepared: Vec<[f32; 2]> = (0..100)
            .map(|index| {
                [
                    0.4 + (index % 10) as f32 * 0.75,
                    0.4 + (index / 10) as f32 * 0.75,
                ]
            })
            .collect();
        let points: Vec<eye_geometry::Point> = prepared
            .iter()
            .map(|point| eye_geometry::Point::new(point[0], point[1]))
            .collect();
        let indices = eye_geometry::type1011_indices(&points).unwrap();
        let mut target = prepared.clone();
        target[14][0] += 1.0;
        target[63][0] -= 1.0;
        let source_uv = prepared
            .iter()
            .map(|point| [point[0] / 8.0, point[1] / 8.0])
            .collect();
        Draw::from_type1011(
            part,
            pf,
            [0.5, 0.5],
            [0; 32],
            &prepared,
            target,
            source_uv,
            indices,
        )
        .unwrap()
    }

    #[test]
    fn face_width_raster_changes_a_non_neutral_frame() {
        let source = source();
        let draw = face_width_draw();
        let output = rasterize(&source, &[&draw]).unwrap();
        assert_ne!(output, source);
    }

    #[test]
    fn type1011_prepared_ebo_rasterizes_both_eyelids() {
        for (part, pf) in [(6, 30073), (7, 30079)] {
            let draw = type1011_draw(part, pf);
            assert_eq!(draw.positions.len(), 100);
            assert_eq!(draw.family(), Family::Type1011);
            assert_ne!(rasterize(&source(), &[&draw]).unwrap(), source());

            let mut wrong = draw.indices.clone();
            wrong[0] = 99;
            assert_eq!(
                Draw::from_type1011(
                    part,
                    pf,
                    draw.controls,
                    draw.source_sha256,
                    &draw.positions,
                    draw.positions.clone(),
                    draw.source_uv.clone(),
                    wrong,
                ),
                Err(Refusal::ValueOutOfRange("Type1011 prepared100 EBO"))
            );
        }
    }

    #[test]
    fn type1012_is_validated_but_not_rasterized_with_type63() {
        let positions = (0..136)
            .map(|index| [index as f32 % 8.0 + 1.0, index as f32 / 18.0 + 1.0])
            .collect();
        let source_uv = (0..136)
            .map(|index| [index as f32 % 8.0 / 8.0, index as f32 / 18.0 / 8.0])
            .collect();
        let draw = Draw::new(
            35,
            30080,
            [50.0, 50.0],
            [0; 32],
            positions,
            source_uv,
            face_detail::TYPE1012_INDICES.to_vec(),
        )
        .unwrap();
        assert_eq!(
            rasterize(&source(), &[&draw]),
            Err(Refusal::MissingInput(
                "PF30080..PF30083 Type1012 texture/V2 handoff"
            ))
        );
    }

    #[test]
    fn constructor_rejects_wrong_part_pf_and_topology() {
        assert_eq!(
            Draw::new(
                25,
                30093,
                [0.0, 0.0],
                [0; 32],
                vec![[0.0, 0.0]; 27],
                vec![[0.0, 0.0]; 27],
                vec![0, 1, 2],
            )
            .unwrap_err(),
            Refusal::ValueOutOfRange("face mesh part/PF")
        );
        assert_eq!(
            Draw::new(
                25,
                30092,
                [0.0, 0.0],
                [0; 32],
                vec![[0.0, 0.0]; 26],
                vec![[0.0, 0.0]; 26],
                vec![0, 1, 2],
            )
            .unwrap_err(),
            Refusal::ShapeMismatch("Type1016 27-vertex topology")
        );
    }
}
