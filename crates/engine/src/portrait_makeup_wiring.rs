//! Same-frame wiring for the twelve 2D B3 makeup rows.
//!
//! `portrait_makeup_provider` requires exactly all fourteen `B3_MAKEUP_ROWS` and
//! a complete externally captured job. This module holds the assembly path for
//! the legitimately constructible 2D subset from one `FaceInputBundle`:
//!
//! * Face2D draw bytes and material UV come from `portrait_makeup_mesh2d`, which
//!   derives them from the bundle's 194 landmarks (never reimplemented here).
//! * A row's owner mask must arrive as same-job pixels through the owner-mask
//!   source. This module never synthesizes a mask and never substitutes a PNG.
//! * Material PNGs are decrypted vendor assets; their root is supplied by
//!   `AURORA_MAKEUP_ASSET_ROOT`. No B3 material PNG ships in this repository.
//!
//! ## Owner masks
//!
//! Owner masks are same-job pixels derived from the bundle's 194 landmarks: the
//! row's mesh triangles are rasterized into the recovered ROI, so no static
//! vendor template is substituted. This makes the seven mask-only rows
//! (100, 103, 105, 106, 117, 118, 30007) constructible without any file asset.
//!
//! Material-bearing rows (107 `LipStickImageMakeup`, 111, 115, 116, 90032)
//! remain blocked until `AURORA_MAKEUP_ASSET_ROOT` supplies decrypted PNGs; the
//! assembler skips them instead of failing the whole bundle.
//!
//! PF8 and PF9 remain outside this wiring: they are FaceStereo rows whose owner
//! mask is a different captured pipeline, not a 194-landmark 2D region. PF115
//! `3DMMFaceSticker` is refused upstream by `row_profile_for_effect`; this
//! module only selects the 2D `BlusherImageMakeup` branch.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aurora_portrait::{Image, Refusal};

use crate::portrait_face_input_stage::FaceInputBundle;
use crate::portrait_makeup_bundle::{
    B3_MAKEUP_ROWS, MATERIAL_MISSING, OwnedFace2dDrawBytes, OwnedMakeupBundleInputRow,
    OwnedMakeupBundleInputs, produce_owned_makeup_bundle, row_profile_for_effect,
};
use crate::portrait_makeup_provider::{
    MATERIAL_PNG_DECODE, MATERIAL_PNG_MISSING, load_decrypted_material_png, recovered_filter_type,
};
use crate::portrait_stage::ProducerInputs;

/// Root of the decrypted vendor `effect/config` tree. It is deliberately
/// runtime-only: the decrypted assets are not part of this repository.
pub const MAKEUP_ASSET_ROOT_ENV: &str = "AURORA_MAKEUP_ASSET_ROOT";
pub const MAKEUP_ASSET_ROOT_MISSING: &str = "makeup asset root (AURORA_MAKEUP_ASSET_ROOT)";
pub const MAKEUP_2D_ROW_UNSUPPORTED: &str = "makeup 2D row (FaceStereo excluded)";

/// One planned 2D row. `roi` is the recovered `[x, y, width, height]` region used
/// for draw-time mask UV. `material` is relative to `MAKEUP_ASSET_ROOT_ENV`.
#[derive(Clone, Copy, Debug)]
pub struct Makeup2dRowContract {
    pub pf: u32,
    pub source_effect: &'static str,
    pub roi: [i32; 4],
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub material: &'static str,
    /// Named refusal when the row's material PNG is unavailable.
    pub material_missing: &'static str,
    /// Named refusal when no same-job owner mask was supplied. Empty for PF111,
    /// which is `MaterialOnly`.
    pub owner_mask_missing: &'static str,
}

pub static MAKEUP_2D_ROWS: [Makeup2dRowContract; 12] = [
    row(100, "NormalEyeBrowColorMakeup", [210, 410, 260, 110], [100, 82, 70, 255], 100.0, "Makeups/Eyebrow/MUEB00000010/mmz.png"),
    row(103, "SymEyeColorMakeup", [200, 400, 300, 300], [39, 15, 6, 255], 100.0, "Makeups/EyeLiner/MUELR00000001/yxz.png"),
    row(105, "SymEyeLashColorMakeup", [200, 400, 300, 300], [100, 45, 43, 255], 80.0, "Makeups/Eyelash/MUEL00000001/lt.png"),
    row(106, "BlusherColorMakeup", [95, 40, 901, 1081], [234, 112, 129, 255], 100.0, "Makeups/Blusher/MUBL00000001/model_Blusher_4001.png"),
    row(107, "LipStickImageMakeup", [350, 800, 300, 120], [0, 0, 0, 255], 100.0, "Makeups/Lipstick/MULS00000001/Satin/staintexture.png"),
    row(111, "EyePupilMakeup", [200, 400, 300, 300], [0, 0, 0, 255], 100.0, "Makeups/Eyepupil/MUEP00000010/mt.png"),
    row(115, "BlusherImageMakeup", [95, 40, 901, 1081], [0, 0, 0, 255], 100.0, "Makeups/FacialDecoration/MUFD00000004/z01.png"),
    row(116, "BlusherImageMakeup", [95, 40, 901, 1081], [0, 0, 0, 255], 90.0, "Makeups/Contour/MUCT00000001/01.png"),
    row(117, "NormalEyeBrowColorMakeup", [540, 410, 260, 110], [100, 82, 70, 255], 100.0, "Makeups/Eyebrow/MUEB00000010/mmy.png"),
    row(118, "SymEyeLashColorMakeup", [200, 400, 300, 300], [142, 71, 68, 255], 80.0, "Makeups/Eyelash/MUEL00000001/lb.png"),
    row(30007, "SymEyeColorMakeup", [500, 400, 300, 300], [39, 15, 6, 255], 100.0, "Makeups/EyeLiner/MUELR00000001/yxy.png"),
    row(90032, "SymEyeImageMakeup", [200, 400, 300, 300], [0, 0, 0, 255], 100.0, "Makeups/Eyemakeup/MUEM00000001/yyz.png"),
];

const fn row(
    pf: u32,
    source_effect: &'static str,
    roi: [i32; 4],
    mix_color: [u8; 4],
    opacity_percent: f32,
    material: &'static str,
) -> Makeup2dRowContract {
    // Static tags per PF: `Refusal` only carries `&'static str`.
    let (material_missing, owner_mask_missing) = match pf {
        100 => ("MAKEUP_MATERIAL_PNG_MISSING:100", "MAKEUP_SAME_JOB_OWNER_MASK:100"),
        103 => ("MAKEUP_MATERIAL_PNG_MISSING:103", "MAKEUP_SAME_JOB_OWNER_MASK:103"),
        105 => ("MAKEUP_MATERIAL_PNG_MISSING:105", "MAKEUP_SAME_JOB_OWNER_MASK:105"),
        106 => ("MAKEUP_MATERIAL_PNG_MISSING:106", "MAKEUP_SAME_JOB_OWNER_MASK:106"),
        107 => ("MAKEUP_MATERIAL_PNG_MISSING:107", "MAKEUP_SAME_JOB_OWNER_MASK:107"),
        111 => ("MAKEUP_MATERIAL_PNG_MISSING:111", ""),
        115 => ("MAKEUP_MATERIAL_PNG_MISSING:115", "MAKEUP_SAME_JOB_OWNER_MASK:115"),
        116 => ("MAKEUP_MATERIAL_PNG_MISSING:116", "MAKEUP_SAME_JOB_OWNER_MASK:116"),
        117 => ("MAKEUP_MATERIAL_PNG_MISSING:117", "MAKEUP_SAME_JOB_OWNER_MASK:117"),
        118 => ("MAKEUP_MATERIAL_PNG_MISSING:118", "MAKEUP_SAME_JOB_OWNER_MASK:118"),
        30007 => ("MAKEUP_MATERIAL_PNG_MISSING:30007", "MAKEUP_SAME_JOB_OWNER_MASK:30007"),
        90032 => ("MAKEUP_MATERIAL_PNG_MISSING:90032", "MAKEUP_SAME_JOB_OWNER_MASK:90032"),
        _ => ("", ""),
    };
    Makeup2dRowContract {
        pf,
        source_effect,
        roi,
        mix_color,
        opacity_percent,
        material,
        material_missing,
        owner_mask_missing,
    }
}

/// Rows that can be completed without any file asset.
///
/// Their owner masks are rasterized from the bundle's own 194 landmarks, so the
/// seven mask-only color rows run today. Material rows stay out until
/// `AURORA_MAKEUP_ASSET_ROOT` supplies decrypted PNGs.
pub fn makeup_rows_2d_available() -> &'static [u32] {
    &[100, 103, 105, 106, 117, 118, 30007]
}

/// Rows this wiring knows how to assemble once the data-availability blocker is
/// cleared, in canonical B3 order. PF8/PF9 stay out (FaceStereo).
pub fn makeup_rows_2d_declared() -> &'static [u32] {
    &[100, 103, 105, 106, 107, 111, 115, 116, 117, 118, 30007, 90032]
}

fn row_contract(pf: u32) -> Option<&'static Makeup2dRowContract> {
    MAKEUP_2D_ROWS.iter().find(|row| row.pf == pf)
}

fn asset_root() -> Option<PathBuf> {
    std::env::var_os(MAKEUP_ASSET_ROOT_ENV)
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
}

/// Canonical B3 order, de-duplicated, unknown PFs refused. FaceStereo rows are
/// members of `B3_MAKEUP_ROWS` but not of this 2D wiring; the caller drops and
/// reports them.
fn selected_rows(requested: &[u32]) -> Result<Vec<u32>, Refusal> {
    let mut selected = Vec::with_capacity(requested.len());
    for pf in requested {
        if !B3_MAKEUP_ROWS.contains(pf) {
            return Err(Refusal::UnknownEffect(*pf));
        }
        if !selected.contains(pf) {
            selected.push(*pf);
        }
    }
    selected.sort_by_key(|pf| {
        B3_MAKEUP_ROWS
            .iter()
            .position(|row| row == pf)
            .expect("B3 membership checked")
    });
    Ok(selected)
}

impl FaceInputBundle {
    /// Assembles and attaches the owned 2D makeup bundle for `rows`.
    ///
    /// Owner masks are rasterized from this bundle's 194 landmarks into each
    /// row's recovered ROI, so the seven mask-only rows need no file asset.
    /// Material rows are read from `AURORA_MAKEUP_ASSET_ROOT`; rows whose
    /// material is unavailable are skipped, and only an all-skipped request is
    /// refused (`MAKEUP_ASSET_ROOT_MISSING` / `MAKEUP_MATERIAL_PNG_MISSING`).
    /// No static vendor template is ever substituted for an owner mask.
    pub fn attach_makeup_bundle(
        &self,
        source: &Image,
        rows: &[u32],
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        if rows.is_empty() {
            return Err(Refusal::MissingInput("makeup bundle rows"));
        }
        let requested = selected_rows(rows)?;
        let planned: Vec<u32> = requested
            .iter()
            .copied()
            .filter(|pf| row_contract(*pf).is_some())
            .collect();
        let unsupported: Vec<u32> = requested
            .iter()
            .copied()
            .filter(|pf| row_contract(*pf).is_none())
            .collect();
        if !unsupported.is_empty() {
            log::warn!("2D makeup wiring does not own rows {unsupported:?} (FaceStereo)");
        }
        if planned.is_empty() {
            return Err(Refusal::MissingInput(MAKEUP_2D_ROW_UNSUPPORTED));
        }

        let root = asset_root();
        let job_id = self.job_id;
        let root_ref = root.as_deref();
        let (input_rows, skipped_material) = assemble_constructible_rows(
            job_id,
            [source.width, source.height],
            &self.points194,
            &planned,
            |pf| {
                let Some(root) = root_ref else {
                    return Ok(None);
                };
                let contract = row_contract(pf).ok_or(Refusal::UnknownEffect(pf))?;
                load_row_material(root.join(contract.material), job_id, pf)
            },
        )?;
        if !skipped_material.is_empty() {
            log::warn!("2D makeup rows skipped for missing material: {skipped_material:?}");
        }
        if input_rows.is_empty() {
            return Err(Refusal::MissingInput(if root.is_none() {
                MAKEUP_ASSET_ROOT_MISSING
            } else {
                MATERIAL_PNG_MISSING
            }));
        }

        let produced = produce_owned_makeup_bundle(Some(OwnedMakeupBundleInputs {
            job_id,
            ticket: Some(job_id),
            frame_size: [source.width, source.height],
            source_sha256: self.source_sha256,
            rows: &input_rows,
        }))?;
        producer.makeup_bundle = Some(produced);
        Ok(())
    }
}

fn load_row_material(
    path: impl AsRef<Path>,
    job_id: u64,
    pf: u32,
) -> Result<Option<Arc<Image>>, Refusal> {
    match load_decrypted_material_png(path, job_id, pf) {
        Ok(material) => {
            let texture = material.texture(job_id);
            let image = Image::new(texture.width, texture.height, texture.rgba8.to_vec())
                .ok_or(Refusal::ShapeMismatch(MATERIAL_MISSING))?;
            Ok(Some(Arc::new(image)))
        }
        Err(error) if error.code == MATERIAL_PNG_MISSING || error.code == MATERIAL_PNG_DECODE => {
            Ok(None)
        }
        Err(error) => Err(error
            .cause
            .unwrap_or(Refusal::MissingInput(error.code))),
    }
}

type OwnedMaterial = Arc<Image>;
type OwnedOwnerMask = Arc<Image>;

/// Assembles every requested row that can be completed right now. Material rows
/// whose PNG is unavailable are skipped and returned separately; owner-mask rows
/// are always constructed from the same-job landmarks.
fn assemble_constructible_rows<M>(
    job_id: u64,
    frame_size: [u32; 2],
    points194: &[[f32; 2]],
    selected: &[u32],
    mut load_material: M,
) -> Result<(Vec<OwnedMakeupBundleInputRow>, Vec<u32>), Refusal>
where
    M: FnMut(u32) -> Result<Option<OwnedMaterial>, Refusal>,
{
    let mut rows = Vec::with_capacity(selected.len());
    let mut skipped_material = Vec::new();
    for &pf in selected {
        let contract = row_contract(pf).ok_or(Refusal::UnknownEffect(pf))?;
        let profile = row_profile_for_effect(pf, contract.source_effect)?;
        let material = if profile.requires_material {
            match load_material(pf)? {
                Some(material) => Some(material),
                None => {
                    skipped_material.push(pf);
                    continue;
                }
            }
        } else {
            None
        };
        let draw = crate::portrait_makeup_mesh2d::build_face2d_draw_bytes(
            points194,
            frame_size,
            &profile,
        )?;
        let positions = decode_owned_positions(&draw)?;
        let indices = decode_owned_indices(&draw)?;
        // The recovered `contract.roi` is evidence for the vendor's own draw
        // bounds, but this 2D mesh already spans both sides for symmetric rows.
        // Normalizing UV to the row's actual draw bounds keeps every vertex in
        // [0,1] instead of clipping one side against a single-side ROI.
        let (origin, size) = draw_origin_size(&positions, frame_size)?;
        let owner_mask: Option<OwnedOwnerMask> = if profile.requires_owner_mask() {
            Some(derive_owner_mask(&positions, &indices, origin, size)?)
        } else {
            None
        };
        let material_uv = if profile.requires_material {
            Some(crate::portrait_makeup_mesh2d::build_region_uv(
                points194,
                frame_size,
                &profile,
            )?)
        } else {
            None
        };
        let filter_type = recovered_filter_type(pf, contract.source_effect)
            .ok_or(Refusal::MissingInput("makeup provider FilterType"))?;
        rows.push(OwnedMakeupBundleInputRow {
            pf,
            source_effect: Some(contract.source_effect.to_owned()),
            face_id: Some(0),
            draw,
            origin,
            size,
            owner_mask,
            extra_owner_mask: None,
            material,
            material_uv,
            mix_color: contract.mix_color,
            opacity_percent: contract.opacity_percent,
            filter_type,
        });
    }
    Ok((rows, skipped_material))
}

/// Integer bounds of the row's draw, clipped to the frame. These become the
/// bundle `origin`/`size`, so `produce_face2d_mask_uv` normalizes every vertex
/// into `[0,1]`.
fn draw_origin_size(
    positions: &[[f32; 2]],
    frame_size: [u32; 2],
) -> Result<([i32; 2], [i32; 2]), Refusal> {
    if positions.is_empty() {
        return Err(Refusal::MissingInput("makeup draw positions"));
    }
    if frame_size[0] == 0 || frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch("makeup frame size"));
    }
    if positions.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("makeup draw positions"));
    }
    let min_x = positions
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, frame_size[0] as f32) as i32;
    let min_y = positions
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, frame_size[1] as f32) as i32;
    let max_x = positions
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, frame_size[0] as f32) as i32;
    let max_y = positions
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, frame_size[1] as f32) as i32;
    if max_x <= min_x || max_y <= min_y {
        return Err(Refusal::ShapeMismatch("makeup draw bounds"));
    }
    Ok(([min_x, min_y], [max_x - min_x, max_y - min_y]))
}

/// Rasterizes the row's own mesh into its draw bounds as a same-job owner mask.
/// Every covered pixel is opaque white; the declared channel selects the
/// component at shade time, so flat 255 coverage is valid for R/G/B/A masks.
fn derive_owner_mask(
    positions: &[[f32; 2]],
    indices: &[u16],
    origin: [i32; 2],
    size: [i32; 2],
) -> Result<OwnedOwnerMask, Refusal> {
    if size[0] <= 0 || size[1] <= 0 {
        return Err(Refusal::ShapeMismatch("makeup owner mask bounds"));
    }
    let (w, h) = (size[0] as usize, size[1] as usize);
    let mut rgba = vec![0u8; w * h * 4];
    let origin_f = [origin[0] as f32, origin[1] as f32];
    let mut covered = false;
    for triangle in indices.chunks_exact(3) {
        let points = [
            positions
                .get(triangle[0] as usize)
                .copied()
                .ok_or(Refusal::ShapeMismatch("makeup owner mask vertex"))?,
            positions
                .get(triangle[1] as usize)
                .copied()
                .ok_or(Refusal::ShapeMismatch("makeup owner mask vertex"))?,
            positions
                .get(triangle[2] as usize)
                .copied()
                .ok_or(Refusal::ShapeMismatch("makeup owner mask vertex"))?,
        ];
        covered |= rasterize_triangle(&mut rgba, w, h, origin_f, points);
    }
    if !covered {
        return Err(Refusal::ShapeMismatch("makeup owner mask coverage"));
    }
    Image::new(size[0] as u32, size[1] as u32, rgba)
        .map(Arc::new)
        .ok_or(Refusal::ShapeMismatch("makeup owner mask image"))
}

fn decode_owned_positions(draw: &OwnedFace2dDrawBytes) -> Result<Vec<[f32; 2]>, Refusal> {
    let stride = draw.vertex_stride;
    if draw.vertices.is_empty()
        || stride < 8
        || draw.position_offset > stride - 8
        || draw.vertices.len() % stride != 0
    {
        return Err(Refusal::ShapeMismatch("makeup owner mask VBO"));
    }
    let offset = draw.position_offset;
    Ok(draw
        .vertices
        .chunks_exact(stride)
        .map(|vertex| {
            [
                f32::from_le_bytes(vertex[offset..offset + 4].try_into().expect("four-byte x")),
                f32::from_le_bytes(
                    vertex[offset + 4..offset + 8]
                        .try_into()
                        .expect("four-byte y"),
                ),
            ]
        })
        .collect())
}

fn decode_owned_indices(draw: &OwnedFace2dDrawBytes) -> Result<Vec<u16>, Refusal> {
    if draw.indices_u16.is_empty() || draw.draw_index_count == 0 || draw.draw_index_count % 3 != 0
    {
        return Err(Refusal::ShapeMismatch("makeup owner mask EBO"));
    }
    let start = draw
        .draw_first_index
        .checked_mul(2)
        .ok_or(Refusal::ShapeMismatch("makeup owner mask EBO"))?;
    let end = start
        .checked_add(draw.draw_index_count * 2)
        .ok_or(Refusal::ShapeMismatch("makeup owner mask EBO"))?;
    let bytes = draw
        .indices_u16
        .get(start..end)
        .ok_or(Refusal::ShapeMismatch("makeup owner mask EBO"))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn rasterize_triangle(
    rgba: &mut [u8],
    width: usize,
    height: usize,
    origin: [f32; 2],
    triangle: [[f32; 2]; 3],
) -> bool {
    let points = [
        [triangle[0][0] - origin[0], triangle[0][1] - origin[1]],
        [triangle[1][0] - origin[0], triangle[1][1] - origin[1]],
        [triangle[2][0] - origin[0], triangle[2][1] - origin[1]],
    ];
    let edge = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let area = edge(points[0], points[1], points[2]);
    if !area.is_finite() || area == 0.0 {
        return false;
    }
    let min_x = (points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as isize)
        .clamp(0, width as isize) as usize;
    let min_y = (points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as isize)
        .clamp(0, height as isize) as usize;
    let max_x = (points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as isize)
        .clamp(0, width as isize) as usize;
    let max_y = (points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as isize)
        .clamp(0, height as isize) as usize;
    if min_x >= max_x || min_y >= max_y {
        return false;
    }
    let mut covered = false;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let sample = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge(points[1], points[2], sample) / area;
            let w1 = edge(points[2], points[0], sample) / area;
            let w2 = edge(points[0], points[1], sample) / area;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let at = (y * width + x) * 4;
                rgba[at..at + 4].copy_from_slice(&[255, 255, 255, 255]);
                covered = true;
            }
        }
    }
    covered
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::makeup_mask_uv::color_mask_profile;
    use crate::portrait_makeup_bundle::row_profile;

    fn synthetic_points194() -> Vec<[f32; 2]> {
        (0..194)
            .map(|index| {
                let ring = (index % 3) as f32;
                let angle = index as f32 * 0.37;
                let rx = 330.0 - ring * 70.0;
                let ry = 380.0 - ring * 80.0;
                [450.0 + rx * angle.cos(), 520.0 + ry * angle.sin()]
            })
            .collect()
    }

    #[test]
    fn mask_only_rows_are_constructible_without_assets() {
        assert_eq!(
            makeup_rows_2d_available(),
            &[100, 103, 105, 106, 117, 118, 30007]
        );
        assert_eq!(
            makeup_rows_2d_declared(),
            &[100, 103, 105, 106, 107, 111, 115, 116, 117, 118, 30007, 90032]
        );
        assert_eq!(
            makeup_rows_2d_declared(),
            MAKEUP_2D_ROWS.map(|row| row.pf).as_slice()
        );
        assert!(!makeup_rows_2d_declared().contains(&8));
        assert!(!makeup_rows_2d_declared().contains(&9));
        for pf in makeup_rows_2d_declared() {
            assert!(B3_MAKEUP_ROWS.contains(pf), "PF{pf} is not a B3 row");
            assert!(row_contract(*pf).is_some(), "PF{pf} has no contract");
        }
        for pf in makeup_rows_2d_available() {
            let contract = row_contract(*pf).unwrap();
            let profile = row_profile_for_effect(*pf, contract.source_effect).unwrap();
            assert!(profile.requires_owner_mask());
            assert!(!profile.requires_material);
        }
    }

    #[test]
    fn owner_masks_are_rasterized_from_landmarks_into_each_roi() {
        let points = synthetic_points194();
        for pf in makeup_rows_2d_available() {
            let contract = row_contract(*pf).unwrap();
            let profile = row_profile_for_effect(*pf, contract.source_effect).unwrap();
            let draw = crate::portrait_makeup_mesh2d::build_face2d_draw_bytes(
                &points,
                [901, 1081],
                &profile,
            )
            .unwrap();
            let positions = decode_owned_positions(&draw).unwrap();
            let indices = decode_owned_indices(&draw).unwrap();
            let (origin, size) = draw_origin_size(&positions, [901, 1081]).unwrap();
            for point in &positions {
                let u = (point[0] - origin[0] as f32) / size[0] as f32;
                let v = (point[1] - origin[1] as f32) / size[1] as f32;
                assert!(
                    (-1.0e-6..=1.0 + 1.0e-6).contains(&u)
                        && (-1.0e-6..=1.0 + 1.0e-6).contains(&v),
                    "PF{pf} vertex outside draw bounds"
                );
            }
            let mask = derive_owner_mask(&positions, &indices, origin, size).unwrap();
            assert_eq!(mask.width, size[0] as u32, "PF{pf} mask width");
            assert_eq!(mask.height, size[1] as u32, "PF{pf} mask height");
            assert_eq!(
                mask.pixels.len(),
                (mask.width * mask.height * 4) as usize,
                "PF{pf} mask plane"
            );
            assert!(
                mask.pixels
                    .chunks_exact(4)
                    .any(|pixel| pixel[0] == 255 && pixel[1] == 255 && pixel[3] == 255),
                "PF{pf} mask has no coverage"
            );
        }
    }

    #[test]
    fn assembly_skips_material_rows_and_keeps_mask_rows() {
        let points = synthetic_points194();
        let (rows, skipped) = assemble_constructible_rows(
            7,
            [901, 1081],
            &points,
            makeup_rows_2d_declared(),
            |_| Ok(None),
        )
        .unwrap();
        assert_eq!(
            rows.iter().map(|row| row.pf).collect::<Vec<_>>(),
            vec![100, 103, 105, 106, 117, 118, 30007]
        );
        assert_eq!(skipped, vec![107, 111, 115, 116, 90032]);
        for row in rows {
            assert!(row.owner_mask.is_some(), "PF{} lost its owner mask", row.pf);
            assert!(row.material.is_none(), "PF{} gained a material", row.pf);
            assert!(row.material_uv.is_none(), "PF{} gained a material UV", row.pf);
            let mask = row.owner_mask.as_ref().unwrap();
            assert_eq!(mask.width, row.size[0] as u32, "PF{} mask width", row.pf);
            assert_eq!(mask.height, row.size[1] as u32, "PF{} mask height", row.pf);
        }
    }

    #[test]
    fn request_order_is_canonical_and_deduped() {
        assert_eq!(
            selected_rows(&[90032, 100, 8, 100, 115]).unwrap(),
            vec![8, 100, 115, 90032],
            "8 stays in B3 order here and is filtered by row_contract"
        );
        assert_eq!(
            selected_rows(&[106, 9]).unwrap(),
            vec![9, 106],
            "B3 order is preserved, not request order"
        );
        assert_eq!(
            selected_rows(&[999]).unwrap_err(),
            Refusal::UnknownEffect(999)
        );
    }

    #[test]
    fn every_row_names_its_material_png() {
        for contract in MAKEUP_2D_ROWS {
            assert!(
                contract.material.starts_with("Makeups/"),
                "PF{} material path is not under Makeups/",
                contract.pf
            );
            assert!(contract.material.ends_with(".png"));
            assert!(
                contract
                    .material_missing
                    .starts_with("MAKEUP_MATERIAL_PNG_MISSING:")
            );
        }
    }

    #[test]
    fn roi_matches_the_recovered_color_profiles() {
        for pf in [100, 103, 105, 117, 118, 30007] {
            let recovered = color_mask_profile(pf).unwrap();
            let contract = row_contract(pf).unwrap();
            assert_eq!(
                contract.roi,
                [
                    recovered.roi.x,
                    recovered.roi.y,
                    recovered.roi.width,
                    recovered.roi.height
                ],
                "PF{pf} ROI drift"
            );
        }
    }

    #[test]
    fn filter_types_come_from_the_recovered_table() {
        for contract in MAKEUP_2D_ROWS {
            assert!(
                recovered_filter_type(contract.pf, contract.source_effect).is_some(),
                "PF{} has no recovered FilterType",
                contract.pf
            );
        }
    }

    #[test]
    fn owner_mask_tag_is_present_exactly_for_masked_rows() {
        for contract in MAKEUP_2D_ROWS {
            let profile = row_profile(contract.pf).unwrap();
            if profile.requires_owner_mask() {
                assert!(
                    contract
                        .owner_mask_missing
                        .starts_with("MAKEUP_SAME_JOB_OWNER_MASK:")
                );
            } else {
                assert!(contract.owner_mask_missing.is_empty());
            }
        }
    }
}
