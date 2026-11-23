use std::collections::{BTreeMap, BTreeSet};

use aurora_portrait::cuda_vface;
use aurora_portrait::kernels::eye_geometry;
use aurora_portrait::Refusal;

use super::portrait_face_stage::{
    hash_f32, FaceGeometry, FaceGeometryDraw, FaceGeometryInputs, SmileHostSplitReceipt,
};
use super::{sha256_32, SmileChannelSplit, SmileRgbReconstructionField};

/// Every PF this provider knows about, including rows that can only be
/// represented by a named refusal.
pub const PROVIDER_ROUTED_PFS: &[u32] = &[
    242, 243, 247, 248, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012,
    8013, 8014, 8015, 8016, 8017, 8018, 8019, 8020, 8021, 8022, 8023, 8024, 8025, 8026, 8027, 8028,
    8029, 8030, 8031, 8032, 8033, 30073, 30079, 30080, 30081, 30082, 30083, 30084, 30085, 30086,
    30087, 30088, 30089, 30090, 30091, 30092, 30093, 41922, 44124, 48019, 49221,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceGeometryProviderRoute {
    V2Positions,
    VFace,
    Type1011,
    Eye,
    Smile,
    SlimContract,
    FaceMeshContract,
    Refused(&'static str),
}

pub fn provider_route(pf: u32) -> Option<FaceGeometryProviderRoute> {
    Some(match pf {
        242 | 243 | 8023 | 8024 | 8027 | 8028 | 8029 | 8030 => {
            FaceGeometryProviderRoute::V2Positions
        }
        247 | 248 => FaceGeometryProviderRoute::VFace,
        8001
        | 8002
        | 8005
        | 8006
        | 8009
        | 8010
        | 8011..=8022
        | 41922
        | 44124
        | 48019
        | 49221 => FaceGeometryProviderRoute::Eye,
        30073 | 30079 => FaceGeometryProviderRoute::Type1011,
        8033 => FaceGeometryProviderRoute::Smile,
        8025 | 8026 => FaceGeometryProviderRoute::SlimContract,
        30092 | 30093 => FaceGeometryProviderRoute::FaceMeshContract,
        8003 | 8004 | 8007 | 8008 => FaceGeometryProviderRoute::Refused(
            "PF8003/PF8004/PF8007/PF8008 FacialRefine mesh producer is open",
        ),
        30080..=30083 => FaceGeometryProviderRoute::Refused(
            "PF30080..PF30083 Type1012 forehead texture/V2 handoff is not an eye-quad draw",
        ),
        8031 | 8032 => FaceGeometryProviderRoute::Refused(
            "PF8031/PF8032 neck slim has no same-job face-geometry producer",
        ),
        30084..=30091 => FaceGeometryProviderRoute::Refused(
            "PF30084..PF30091 have no routed face-geometry producer",
        ),
        _ => return None,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceGeometryProviderMeshReceipt {
    pub mesh_id: u32,
    pub job_id: u64,
    pub vertex_count: usize,
    pub index_count: usize,
    pub positions_sha256: [u8; 32],
    pub vbo_sha256: [u8; 32],
    pub ebo_sha256: [u8; 32],
    pub uv_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceGeometryProviderReceipt {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub source_sha256: [u8; 32],
    pub points142_sha256: [u8; 32],
    pub points194_sha256: [u8; 32],
    pub meshes: Vec<FaceGeometryProviderMeshReceipt>,
}

/// One same-job VBO/EBO/UV binding. `vbo` is the exact packed byte sequence
/// `positions[0..n]` followed by `uv[0..n]`, each float2 little-endian.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometryProviderMesh {
    pub mesh_id: u32,
    pub job_id: u64,
    pub positions: Vec<[f32; 2]>,
    pub vbo: Vec<u8>,
    pub ebo: Vec<u16>,
    pub uv: Vec<[f32; 2]>,
}

/// A producer result that has not yet been bound to the renderer frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometryProviderRow {
    pub pf: u32,
    pub controls: [f32; 2],
    /// Required for V2/V-face/eye rows. PF8033 may use no mesh because its
    /// complete displacement/mask planes are the producer output.
    pub mesh_id: Option<u32>,
    pub geometry: FaceGeometry,
}

/// All same-job inputs needed to emit one `FaceGeometryInputs` bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometryProviderInputs {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub source_sha256: [u8; 32],
    pub points142: Vec<[f32; 2]>,
    pub points194: Vec<[f32; 2]>,
    pub meshes: Vec<FaceGeometryProviderMesh>,
    pub receipt: FaceGeometryProviderReceipt,
    pub rows: Vec<FaceGeometryProviderRow>,
}

/// Bind real `780929437` host-split output to the PF8033 forward warp.
pub fn build_pf8033_smile_row(
    job_id: u64,
    controls: [f32; 2],
    model_output: &SmileRgbReconstructionField,
    split: &SmileChannelSplit,
    mask: Vec<u8>,
) -> Result<FaceGeometryProviderRow, Refusal> {
    if split.spec.job_id != job_id {
        return Err(Refusal::ValueOutOfRange("PF8033 host split ticket"));
    }
    validate_controls(controls, true)?;
    let receipt = SmileHostSplitReceipt::from_model_output(
        job_id,
        model_output,
        &split.model_dx,
        &split.model_dy,
        &mask,
    )?;
    Ok(FaceGeometryProviderRow {
        pf: 8033,
        controls,
        mesh_id: None,
        geometry: FaceGeometry::Smile {
            dx: split.model_dx.clone(),
            dy: split.model_dy.clone(),
            mask,
            host_split_receipt: Some(receipt),
        },
    })
}

/// Bind producer rows to one frame and one receipt. No geometry is accepted
/// from another job, another frame, or a mesh whose bytes differ from receipt.
pub fn build_face_geometry_inputs(
    input: &FaceGeometryProviderInputs,
) -> Result<FaceGeometryInputs, Refusal> {
    validate_common(input)?;
    let meshes = validate_meshes(input)?;
    if input.rows.is_empty() {
        return Err(Refusal::MissingInput("face geometry provider rows"));
    }

    let mut seen = BTreeSet::new();
    let mut draws = Vec::with_capacity(input.rows.len());
    for row in &input.rows {
        if !seen.insert(row.pf) {
            return Err(Refusal::ValueOutOfRange(
                "duplicate face geometry provider PF",
            ));
        }
        let route = provider_route(row.pf).ok_or(Refusal::NotImplemented(row.pf))?;
        match route {
            FaceGeometryProviderRoute::V2Positions => {
                validate_v2_positions(row, meshes.get(&required_mesh(row)?))?;
            }
            FaceGeometryProviderRoute::VFace => {
                validate_vface(row, meshes.get(&required_mesh(row)?))?;
            }
            FaceGeometryProviderRoute::Type1011 => {
                validate_type1011(row)?;
            }
            FaceGeometryProviderRoute::Eye => {
                validate_eye(row, meshes.get(&required_mesh(row)?))?;
            }
            FaceGeometryProviderRoute::Smile => {
                validate_smile(row, input.width, input.height, input.job_id)?;
                if let Some(mesh_id) = row.mesh_id {
                    if !meshes.contains_key(&mesh_id) {
                        return Err(Refusal::MissingInput("face geometry provider row mesh"));
                    }
                }
            }
            FaceGeometryProviderRoute::SlimContract => {
                return Err(Refusal::MissingInput(
                    "PF8025/PF8026 use SlimProducerInputs",
                ));
            }
            FaceGeometryProviderRoute::FaceMeshContract => {
                return Err(Refusal::MissingInput(
                    "PF30092/PF30093 use the face-mesh group contract",
                ));
            }
            FaceGeometryProviderRoute::Refused(reason) => {
                return Err(Refusal::MissingInput(reason));
            }
        }
        draws.push(FaceGeometryDraw {
            pf: row.pf,
            controls: row.controls,
            source_sha256: input.source_sha256,
            geometry: row.geometry.clone(),
        });
    }

    Ok(FaceGeometryInputs {
        job_id: input.job_id,
        width: input.width,
        height: input.height,
        draws,
    })
}

fn validate_common(input: &FaceGeometryProviderInputs) -> Result<(), Refusal> {
    if input.job_id == 0 {
        return Err(Refusal::ValueOutOfRange(
            "face geometry provider job ticket",
        ));
    }
    if input.width == 0 || input.height == 0 {
        return Err(Refusal::ShapeMismatch("face geometry provider frame size"));
    }
    if input.receipt.job_id != input.job_id {
        return Err(Refusal::ValueOutOfRange(
            "face geometry provider receipt job",
        ));
    }
    if (input.receipt.width, input.receipt.height) != (input.width, input.height) {
        return Err(Refusal::ShapeMismatch(
            "face geometry provider receipt frame",
        ));
    }
    if input.receipt.source_sha256 != input.source_sha256 {
        return Err(Refusal::ValueOutOfRange(
            "face geometry provider receipt source digest",
        ));
    }
    if input.points142.len() != 142 {
        return Err(Refusal::MissingInput(
            "face geometry provider same-job 142 points",
        ));
    }
    if input.points194.len() != 194 {
        return Err(Refusal::MissingInput(
            "face geometry provider same-job 194 points",
        ));
    }
    if !finite_points(&input.points142) || !finite_points(&input.points194) {
        return Err(Refusal::ValueOutOfRange("face geometry provider landmarks"));
    }
    if input.receipt.points142_sha256 != hash_points(&input.points142) {
        return Err(Refusal::ValueOutOfRange(
            "face geometry provider receipt 142 points",
        ));
    }
    if input.receipt.points194_sha256 != hash_points(&input.points194) {
        return Err(Refusal::ValueOutOfRange(
            "face geometry provider receipt 194 points",
        ));
    }
    if input.meshes.is_empty() {
        return Err(Refusal::MissingInput(
            "face geometry provider same-job mesh VBO/EBO/UV",
        ));
    }
    Ok(())
}

fn validate_meshes<'a>(
    input: &'a FaceGeometryProviderInputs,
) -> Result<BTreeMap<u32, &'a FaceGeometryProviderMesh>, Refusal> {
    if input.receipt.meshes.len() != input.meshes.len() {
        return Err(Refusal::ShapeMismatch(
            "face geometry provider mesh receipts",
        ));
    }
    let mut receipts = BTreeMap::new();
    for receipt in &input.receipt.meshes {
        if receipts.insert(receipt.mesh_id, receipt).is_some() {
            return Err(Refusal::ValueOutOfRange(
                "duplicate face geometry mesh receipt",
            ));
        }
    }

    let mut meshes = BTreeMap::new();
    for mesh in &input.meshes {
        if meshes.insert(mesh.mesh_id, mesh).is_some() {
            return Err(Refusal::ValueOutOfRange(
                "duplicate face geometry provider mesh",
            ));
        }
        if mesh.job_id != input.job_id {
            return Err(Refusal::ValueOutOfRange("face geometry provider mesh job"));
        }
        if mesh.positions.is_empty() || mesh.positions.len() != mesh.uv.len() {
            return Err(Refusal::ShapeMismatch(
                "face geometry provider mesh vertices",
            ));
        }
        if !finite_points(&mesh.positions) || !finite_points(&mesh.uv) {
            return Err(Refusal::ValueOutOfRange(
                "face geometry provider mesh vertex",
            ));
        }
        if mesh.ebo.is_empty() || mesh.ebo.len() % 3 != 0 {
            return Err(Refusal::ShapeMismatch("face geometry provider mesh EBO"));
        }
        if mesh
            .ebo
            .iter()
            .any(|index| usize::from(*index) >= mesh.positions.len())
        {
            return Err(Refusal::ValueOutOfRange(
                "face geometry provider mesh index",
            ));
        }
        let packed = pack_mesh_vbo(&mesh.positions, &mesh.uv);
        if mesh.vbo != packed {
            return Err(Refusal::ShapeMismatch(
                "face geometry provider mesh VBO bytes",
            ));
        }
        let receipt = receipts
            .get(&mesh.mesh_id)
            .ok_or(Refusal::MissingInput("face geometry provider mesh receipt"))?;
        if receipt.job_id != input.job_id
            || receipt.vertex_count != mesh.positions.len()
            || receipt.index_count != mesh.ebo.len()
            || receipt.positions_sha256 != hash_points(&mesh.positions)
            || receipt.uv_sha256 != hash_points(&mesh.uv)
            || receipt.vbo_sha256 != sha256_32(&mesh.vbo)
            || receipt.ebo_sha256 != hash_indices(&mesh.ebo)
        {
            return Err(Refusal::ValueOutOfRange(
                "face geometry provider receipt mesh binding",
            ));
        }
    }
    Ok(meshes)
}

fn required_mesh(row: &FaceGeometryProviderRow) -> Result<u32, Refusal> {
    row.mesh_id
        .ok_or(Refusal::MissingInput("face geometry provider row mesh"))
}

fn validate_v2_positions(
    row: &FaceGeometryProviderRow,
    mesh: Option<&&FaceGeometryProviderMesh>,
) -> Result<(), Refusal> {
    let mesh = mesh.ok_or(Refusal::MissingInput("face geometry provider V2 mesh"))?;
    validate_controls(row.controls, false)?;
    match &row.geometry {
        FaceGeometry::V2Positions(points) => {
            if points.is_empty() {
                return Err(Refusal::MissingInput("face geometry provider V2 positions"));
            }
            for &(index, point) in points {
                if usize::from(index) >= mesh.positions.len() {
                    return Err(Refusal::ValueOutOfRange(
                        "face geometry provider V2 vertex index",
                    ));
                }
                if !finite_point(&point) {
                    return Err(Refusal::ValueOutOfRange(
                        "face geometry provider V2 position",
                    ));
                }
            }
            Ok(())
        }
        _ => Err(Refusal::ValueOutOfRange(
            "face geometry provider PF/geometry kind",
        )),
    }
}

fn validate_type1011(row: &FaceGeometryProviderRow) -> Result<(), Refusal> {
    validate_controls(row.controls, false)?;
    let FaceGeometry::Type1011 {
        prepared,
        param_list,
        center,
    } = &row.geometry
    else {
        return Err(Refusal::ValueOutOfRange(
            "face geometry provider PF/geometry kind",
        ));
    };
    if prepared.len() != 100 {
        return Err(Refusal::ShapeMismatch(
            "face geometry provider Type1011 prepared100",
        ));
    }
    if param_list.len() != 6 {
        return Err(Refusal::MissingInput(
            "face geometry provider Type1011 six weights",
        ));
    }
    if !finite_points(prepared)
        || center.iter().any(|value| !value.is_finite())
        || param_list.iter().any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange(
            "face geometry provider Type1011 values",
        ));
    }
    let prepared: Vec<eye_geometry::Point> = prepared
        .iter()
        .map(|point| eye_geometry::Point::new(point[0], point[1]))
        .collect();
    eye_geometry::type1011_indices(&prepared)
        .map(|_| ())
        .map_err(|_| Refusal::ShapeMismatch("Type1011 prepared100 EBO"))
}

fn validate_vface(
    row: &FaceGeometryProviderRow,
    mesh: Option<&&FaceGeometryProviderMesh>,
) -> Result<(), Refusal> {
    if mesh.is_none() {
        return Err(Refusal::MissingInput("face geometry provider V-face mesh"));
    }
    validate_controls(row.controls, false)?;
    match &row.geometry {
        FaceGeometry::VFace {
            rows,
            encoded_offset,
            atlases,
        } => {
            let edge_pixels = usize::try_from(cuda_vface::OFFSET_EDGE * cuda_vface::OFFSET_EDGE)
                .map_err(|_| Refusal::ShapeMismatch("face geometry provider V-face atlas"))?;
            for atlas in atlases {
                atlas.validate()?;
                if !rows.contains(&atlas.row) {
                    return Err(Refusal::ValueOutOfRange(
                        "face geometry provider V-face atlas row",
                    ));
                }
            }
            if atlases.is_empty() && encoded_offset.len() != edge_pixels {
                return Err(Refusal::ShapeMismatch(
                    "face geometry provider V-face atlas",
                ));
            }
            if rows.is_empty() && row.controls != [0.0; 2] {
                return Err(Refusal::MissingInput("face geometry provider V-face rows"));
            }
            if encoded_offset
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
            {
                return Err(Refusal::ValueOutOfRange(
                    "face geometry provider V-face offset",
                ));
            }
            Ok(())
        }
        _ => Err(Refusal::ValueOutOfRange(
            "face geometry provider PF/geometry kind",
        )),
    }
}

fn validate_eye(
    row: &FaceGeometryProviderRow,
    mesh: Option<&&FaceGeometryProviderMesh>,
) -> Result<(), Refusal> {
    let mesh = mesh.ok_or(Refusal::MissingInput("face geometry provider eye mesh"))?;
    validate_controls(row.controls, false)?;
    match &row.geometry {
        FaceGeometry::Eye {
            dest,
            uv,
            ebo,
            defines,
            clamp,
            ..
        } => {
            if dest.is_empty() || dest.len() != mesh.positions.len() {
                return Err(Refusal::ShapeMismatch("face geometry provider eye VBO"));
            }
            if uv != &mesh.uv || ebo != &mesh.ebo {
                return Err(Refusal::ValueOutOfRange(
                    "face geometry provider eye UV/EBO binding",
                ));
            }
            if !finite_points(dest) || !finite_points(uv) {
                return Err(Refusal::ValueOutOfRange(
                    "face geometry provider eye vertex",
                ));
            }
            if defines.uv_map || defines.uv_map_ba || !clamp.factor.is_finite() {
                return Err(Refusal::ValueOutOfRange(
                    "face geometry provider eye fragment state",
                ));
            }
            Ok(())
        }
        _ => Err(Refusal::ValueOutOfRange(
            "face geometry provider PF/geometry kind",
        )),
    }
}

fn validate_smile(
    row: &FaceGeometryProviderRow,
    width: u32,
    height: u32,
    job_id: u64,
) -> Result<(), Refusal> {
    validate_controls(row.controls, true)?;
    let pixels = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(Refusal::ShapeMismatch(
            "face geometry provider PF8033 frame",
        ))?;
    match &row.geometry {
        FaceGeometry::Smile {
            dx,
            dy,
            mask,
            host_split_receipt,
        } => {
            if dx.len() != pixels || dy.len() != pixels || mask.len() != pixels {
                return Err(Refusal::ShapeMismatch(
                    "face geometry provider PF8033 planes",
                ));
            }
            if dx.iter().chain(dy).any(|value| !value.is_finite()) {
                return Err(Refusal::ValueOutOfRange(
                    "face geometry provider PF8033 displacement",
                ));
            }
            let receipt = host_split_receipt
                .as_ref()
                .ok_or(Refusal::MissingInput("PF8033 host split receipt"))?;
            if receipt.job_id != job_id
                || receipt.model_output_id != SmileRgbReconstructionField::OUTPUT_ID
                || receipt.dx_sha256 != hash_f32(dx)
                || receipt.dy_sha256 != hash_f32(dy)
                || receipt.mask_sha256 != sha256_32(mask)
            {
                return Err(Refusal::ValueOutOfRange("PF8033 host split receipt"));
            }
            Ok(())
        }
        _ => Err(Refusal::ValueOutOfRange(
            "face geometry provider PF/geometry kind",
        )),
    }
}

fn validate_controls(controls: [f32; 2], zero_to_one: bool) -> Result<(), Refusal> {
    let minimum = if zero_to_one { 0.0 } else { -1.0 };
    if controls
        .iter()
        .any(|value| !value.is_finite() || !(minimum..=1.0).contains(value))
    {
        return Err(Refusal::ValueOutOfRange("face geometry provider controls"));
    }
    Ok(())
}

fn finite_point(point: &[f32; 2]) -> bool {
    point.iter().all(|value| value.is_finite())
}

fn finite_points(points: &[[f32; 2]]) -> bool {
    points.iter().all(finite_point)
}

fn hash_points(points: &[[f32; 2]]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(points.len() * 8);
    for point in points {
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
    }
    sha256_32(&bytes)
}

fn hash_indices(indices: &[u16]) -> [u8; 32] {
    let bytes: Vec<u8> = indices
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();
    sha256_32(&bytes)
}

fn pack_mesh_vbo(positions: &[[f32; 2]], uv: &[[f32; 2]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((positions.len() + uv.len()) * 8);
    for point in positions.iter().chain(uv) {
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
    }
    bytes
}

pub fn mesh_receipt(mesh: &FaceGeometryProviderMesh) -> FaceGeometryProviderMeshReceipt {
    FaceGeometryProviderMeshReceipt {
        mesh_id: mesh.mesh_id,
        job_id: mesh.job_id,
        vertex_count: mesh.positions.len(),
        index_count: mesh.ebo.len(),
        positions_sha256: hash_points(&mesh.positions),
        vbo_sha256: sha256_32(&mesh.vbo),
        ebo_sha256: hash_indices(&mesh.ebo),
        uv_sha256: hash_points(&mesh.uv),
    }
}

pub fn landmark_receipt_hashes(
    points142: &[[f32; 2]],
    points194: &[[f32; 2]],
) -> ([u8; 32], [u8; 32]) {
    (hash_points(points142), hash_points(points194))
}

#[cfg(test)]
#[path = "portrait_face_geometry_provider_tests.rs"]
mod tests;
