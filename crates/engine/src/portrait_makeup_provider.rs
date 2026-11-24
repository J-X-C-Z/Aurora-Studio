//! External producer boundary for the fourteen B3 makeup rows.
//!
//! Static preset data never supplies owner-mask pixels, Face2D draw bytes, or
//! draw-time UVs. This module accepts those runtime values from one external
//! job, validates their job identity, copies them into owned buffers, and hands
//! the exact payload to `produce_owned_makeup_bundle`.

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use aurora_portrait::cuda_makeup::MakeupRoute;
use aurora_portrait::kernels::makeup_material::{
    FACE_STEREO_EFFECT, FACE_STEREO_FILTER_TYPE, FACE_STEREO_MIX_COLOR, FACE_STEREO_OPACITY_PERCENT,
    face_stereo_material_missing, face_stereo_mask_missing, face_stereo_mix_alpha,
    lipstick_blend_mode,
};
use aurora_portrait::model_sha256::sha256 as sha256_32;
use aurora_portrait::{Image, Refusal};

use crate::portrait_makeup_bundle::{
    B3_MAKEUP_ROWS, EBO_BYTES_MISSING, MATERIAL_MISSING, MATERIAL_UV_MISSING, MakeupRowProfile,
    OWNER_MASK_MISSING, OwnedFace2dDrawBytes, OwnedFace2dUvBytes, OwnedMakeupBundle,
    OwnedMakeupBundleInputRow, OwnedMakeupBundleInputs, SOURCE_SHA_MISSING, VBO_BYTES_MISSING,
    facestereo_material, facestereo_preset_row, produce_owned_makeup_bundle,
    row_profile_for_effect,
};

pub const PROVIDER_ROWS: &str = "makeup provider rows";
pub const PROVIDER_TICKET: &str = "makeup provider ticket";
pub const PROVIDER_SOURCE_SHA: &str = "makeup provider source SHA";
pub const PROVIDER_FILTER_TYPE: &str = "makeup provider FilterType";
pub const PROVIDER_OWNER_MASK_SHAPE: &str = "makeup provider owner mask shape";
pub const PROVIDER_MATERIAL_SHAPE: &str = "makeup provider material shape";
pub const PROVIDER_MATERIAL_UV_SHAPE: &str = "makeup provider material UV shape";
pub const PROVIDER_VBO_SHAPE: &str = "makeup provider VBO shape";
pub const PROVIDER_EBO_SHAPE: &str = "makeup provider EBO shape";
pub const PROVIDER_OWNER_MASK_UNEXPECTED: &str = "makeup provider owner mask unexpected";
pub const MATERIAL_PNG_MISSING: &str = "makeup decrypted material PNG";
pub const MATERIAL_PNG_DECODE: &str = "makeup decrypted material PNG decode";

/// Recovered `FilterType` per B3 row. `FilterType` is the blend macro (40
/// Normal, 41 Multiply, 42 SoftLight), not a routing tag, so a wrong value
/// silently changes pixels instead of failing. These values are the observed
/// preset values for the fourteen rows and match both the vendor preset fixture
/// in `tests/fixtures/makeup_owner_sources/static-source-map.json` and the
/// base-tree bundle helper table. PF107 is the `LipStickImageMakeup` preset;
/// the `LipColor` variant has no recovered value and stays caller-owned.
pub const RECOVERED_FILTER_TYPES: [(u32, u32); 14] = [
    (8, 42),
    (9, 42),
    (100, 11),
    (103, 11),
    (105, 11),
    (106, 11),
    (107, 42),
    (111, 40),
    (115, 41),
    (116, 42),
    (117, 11),
    (118, 11),
    (30007, 11),
    (90032, 41),
];

/// The recovered `FilterType` for a row, or `None` when the selected variant has
/// no recovered preset value. Only PF107 `LipColor` is currently in that
/// position.
pub fn recovered_filter_type(pf: u32, source_effect: &str) -> Option<u32> {
    if pf == 107 && source_effect == "LipColor" {
        return None;
    }
    RECOVERED_FILTER_TYPES
        .iter()
        .find(|(row_pf, _)| *row_pf == pf)
        .map(|(_, filter_type)| *filter_type)
}

/// Preset rows whose `FilterType` is not fixed for the PF. A vendor PF can
/// ship several preset rows that select different blend macros, so the macro is
/// per preset, not per PF. PF90032 eye makeup presets use Multiply 41, SoftLight
/// 42 and Screen 44 (`cfg:Makeups/Eyemakeup/...`).
pub const PRESET_FILTER_TYPE_SETS: [(u32, &str, &[u32]); 1] =
    [(90032, "SymEyeImageMakeup", &[41, 42, 44])];

/// Accepted `FilterType` values for one `(pf, selected preset variant)`.
///
/// When the caller supplies a preset macro this is the set it must belong to,
/// which keeps a foreign macro from silently changing pixels. Rows without an
/// explicit multi-value entry accept the single recovered value. `None` means
/// the variant has no recovered preset macro; a route-valid caller-supplied one
/// is then required (PF107 `LipColor`).
pub fn preset_filter_types(pf: u32, source_effect: &str) -> Option<&'static [u32]> {
    for (row_pf, effect, values) in PRESET_FILTER_TYPE_SETS {
        if row_pf == pf && effect == source_effect {
            return Some(values);
        }
    }
    const ONLY_11: &[u32] = &[11];
    const ONLY_40: &[u32] = &[40];
    const ONLY_41: &[u32] = &[41];
    const ONLY_42: &[u32] = &[42];
    let single: &'static [u32] = match recovered_filter_type(pf, source_effect)? {
        11 => ONLY_11,
        40 => ONLY_40,
        41 => ONLY_41,
        42 => ONLY_42,
        _ => return None,
    };
    Some(single)
}

/// A named rejection from the external provider boundary. `code` is stable for
/// host/UI reporting; `job_id` and `pf` identify the rejected row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderRefusal {
    pub code: &'static str,
    pub job_id: u64,
    pub ticket: Option<u64>,
    pub pf: Option<u32>,
    pub cause: Option<Refusal>,
}

impl ProviderRefusal {
    fn named(
        code: &'static str,
        job_id: u64,
        pf: Option<u32>,
        ticket: Option<u64>,
        cause: Option<Refusal>,
    ) -> Self {
        Self {
            code,
            job_id,
            ticket,
            pf,
            cause,
        }
    }

    fn from_bundle_refusal(job_id: u64, pf: Option<u32>, refusal: Refusal) -> Self {
        let code = refusal_code(&refusal);
        Self::named(code, job_id, pf, Some(job_id), Some(refusal))
    }
}

impl fmt::Display for ProviderRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (job={}", self.code, self.job_id)?;
        if let Some(pf) = self.pf {
            write!(f, ", pf={pf}")?;
        }
        if let Some(ticket) = self.ticket {
            write!(f, ", ticket={ticket}")?;
        }
        f.write_str(")")
    }
}

impl std::error::Error for ProviderRefusal {}

/// RGBA8 pixels plus the ticket that produced them. The provider copies the
/// pixels, so the external command buffer may be released after `build`.
#[derive(Clone, Copy, Debug)]
pub struct ExternalTexture<'a> {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub rgba8: &'a [u8],
}

/// Face2D position VBO and triangle EBO from one draw command.
#[derive(Clone, Copy, Debug)]
pub struct ExternalFace2dDraw<'a> {
    pub ticket: u64,
    pub vbo: &'a [u8],
    pub vertex_stride: usize,
    pub position_offset: usize,
    pub ebo_u16: &'a [u8],
    pub first_index: usize,
    pub index_count: usize,
}

/// A distinct Face2D attribute VBO used for material texcoords.
#[derive(Clone, Copy, Debug)]
pub struct ExternalFace2dUv<'a> {
    pub ticket: u64,
    pub vbo: &'a [u8],
    pub vertex_stride: usize,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ExternalMakeupRow<'a> {
    pub pf: u32,
    pub source_effect: &'a str,
    pub face_id: u32,
    pub ticket: u64,
    pub source_sha256: [u8; 32],
    pub draw: ExternalFace2dDraw<'a>,
    pub origin: [i32; 2],
    pub size: [i32; 2],
    pub owner_mask: Option<ExternalTexture<'a>>,
    pub extra_owner_mask: Option<ExternalTexture<'a>>,
    pub material: Option<ExternalTexture<'a>>,
    pub material_uv: Option<ExternalFace2dUv<'a>>,
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub filter_type: u32,
}

/// One `FaceStereo/effect.json` preset selection from an external producer.
/// PF8/PF9 are already first-class `ExternalMakeupRow`s; this selection only
/// carries the vendor `MeshType`/degree that the generic row does not model.
#[derive(Clone, Copy, Debug)]
pub struct ExternalFaceStereoRow<'a> {
    pub pf: u32,
    pub ticket: u64,
    /// Vendor `MeshType` (2 front, 4 SideFace).
    pub mesh_type: u32,
    /// `DefaultDegree` over `DegreeRange=[0,100]`.
    pub degree_percent: f32,
    /// Same-job capture name for the owner mask. No producer exists in this
    /// tree; a missing or unknown source is refused by name.
    pub owner_mask_source: Option<&'a str>,
}

#[derive(Clone, Copy, Debug)]
pub struct ExternalMakeupJob<'a> {
    pub job_id: u64,
    pub ticket: u64,
    pub frame_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub rows: &'a [ExternalMakeupRow<'a>],
    /// Optional FaceStereo preset selections. `None` means the caller does not
    /// supply the capture-side metadata; the generic rows still carry the
    /// preset ROI/FilterType.
    pub facestereo: Option<&'a [ExternalFaceStereoRow<'a>]>,
}

/// A decrypted PNG material owned by the provider. The asset remains external;
/// only decoded RGBA pixels and their SHA-256 cross the job boundary.
pub struct ExternalMakeupMaterial {
    image: Arc<Image>,
    sha256: [u8; 32],
}

impl ExternalMakeupMaterial {
    pub fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    pub fn texture(&self, ticket: u64) -> ExternalTexture<'_> {
        ExternalTexture {
            ticket,
            width: self.image.width,
            height: self.image.height,
            rgba8: &self.image.pixels,
        }
    }
}

/// Decodes a material that was already decrypted from the vendor container.
/// Missing, malformed, or empty PNGs stay named provider refusals.
pub fn load_decrypted_material_png(
    path: impl AsRef<Path>,
    job_id: u64,
    pf: u32,
) -> Result<ExternalMakeupMaterial, ProviderRefusal> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|_| {
        ProviderRefusal::named(
            MATERIAL_PNG_MISSING,
            job_id,
            Some(pf),
            None,
            Some(Refusal::MissingInput(MATERIAL_PNG_MISSING)),
        )
    })?;
    let decoded = image::load_from_memory(&bytes).map_err(|_| {
        ProviderRefusal::named(
            MATERIAL_PNG_DECODE,
            job_id,
            Some(pf),
            None,
            Some(Refusal::ShapeMismatch(MATERIAL_PNG_DECODE)),
        )
    })?;
    let rgba = decoded.into_rgba8();
    let (width, height) = rgba.dimensions();
    let image = Image::new(width, height, rgba.into_raw()).ok_or_else(|| {
        ProviderRefusal::named(
            MATERIAL_PNG_DECODE,
            job_id,
            Some(pf),
            None,
            Some(Refusal::ShapeMismatch(MATERIAL_PNG_DECODE)),
        )
    })?;
    Ok(ExternalMakeupMaterial {
        image: Arc::new(image),
        sha256: sha256_32(&bytes),
    })
}

pub struct ExternalMakeupProvider;

impl ExternalMakeupProvider {
    /// Builds the owned bundle only when every supplied resource belongs to the
    /// current job. Missing owner-mask pixels are rejected by name for the row
    /// that needs them.
    pub fn build(job: Option<ExternalMakeupJob<'_>>) -> Result<OwnedMakeupBundle, ProviderRefusal> {
        produce_external_owned_makeup_bundle(job)
    }
}

pub fn produce_external_owned_makeup_bundle(
    job: Option<ExternalMakeupJob<'_>>,
) -> Result<OwnedMakeupBundle, ProviderRefusal> {
    let job = job.ok_or_else(|| {
        ProviderRefusal::named(
            PROVIDER_ROWS,
            0,
            None,
            None,
            Some(Refusal::MissingInput(PROVIDER_ROWS)),
        )
    })?;

    check_job_identity(&job)?;
    check_row_coverage(job.job_id, job.rows)?;
    check_facestereo_presets(&job)?;

    let mut owned_rows = Vec::with_capacity(job.rows.len());
    for row in job.rows {
        owned_rows.push(own_row(&job, row)?);
    }

    produce_owned_makeup_bundle(Some(OwnedMakeupBundleInputs {
        job_id: job.job_id,
        ticket: Some(job.ticket),
        frame_size: job.frame_size,
        source_sha256: job.source_sha256,
        rows: &owned_rows,
    }))
    .map_err(|refusal| ProviderRefusal::from_bundle_refusal(job.job_id, None, refusal))
}

fn check_job_identity(job: &ExternalMakeupJob<'_>) -> Result<(), ProviderRefusal> {
    if job.job_id == 0 || job.ticket != job.job_id {
        return Err(ProviderRefusal::named(
            PROVIDER_TICKET,
            job.job_id,
            None,
            Some(job.ticket),
            Some(Refusal::ValueOutOfRange(PROVIDER_TICKET)),
        ));
    }
    if job.frame_size[0] == 0 || job.frame_size[1] == 0 {
        return Err(ProviderRefusal::named(
            "makeup provider frame",
            job.job_id,
            None,
            Some(job.ticket),
            Some(Refusal::ShapeMismatch("makeup provider frame")),
        ));
    }
    if job.source_sha256.iter().all(|byte| *byte == 0) {
        return Err(ProviderRefusal::named(
            SOURCE_SHA_MISSING,
            job.job_id,
            None,
            Some(job.ticket),
            Some(Refusal::MissingInput(SOURCE_SHA_MISSING)),
        ));
    }
    Ok(())
}

fn check_row_coverage(job_id: u64, rows: &[ExternalMakeupRow<'_>]) -> Result<(), ProviderRefusal> {
    if rows.len() != B3_MAKEUP_ROWS.len() {
        return Err(ProviderRefusal::named(
            PROVIDER_ROWS,
            job_id,
            None,
            None,
            Some(Refusal::ShapeMismatch(PROVIDER_ROWS)),
        ));
    }
    for (index, row) in rows.iter().enumerate() {
        if row.pf != B3_MAKEUP_ROWS[index] {
            return Err(ProviderRefusal::named(
                PROVIDER_ROWS,
                job_id,
                Some(row.pf),
                Some(row.ticket),
                Some(Refusal::ValueOutOfRange(PROVIDER_ROWS)),
            ));
        }
    }
    Ok(())
}

/// Validates PF8/PF9 preset selections against the static FaceStereo table.
/// The material name is resolved here so an unknown `MeshType` is a named
/// refusal. Owner-mask pixels and their producer are still caller-owned: a
/// missing capture source is refused by name, never synthesized.
fn check_facestereo_presets(job: &ExternalMakeupJob<'_>) -> Result<(), ProviderRefusal> {
    let Some(selections) = job.facestereo else {
        return Ok(());
    };
    let refuse = |code: &'static str, pf: u32, cause: Refusal| {
        ProviderRefusal::named(code, job.job_id, Some(pf), Some(job.ticket), Some(cause))
    };
    let mut seen = [false; 2];
    for selection in selections {
        let index = match selection.pf {
            8 => 0,
            9 => 1,
            other => return Err(refuse(PROVIDER_ROWS, other, Refusal::UnknownEffect(other))),
        };
        if seen[index] {
            return Err(refuse(PROVIDER_ROWS, selection.pf, Refusal::ShapeMismatch(PROVIDER_ROWS)));
        }
        seen[index] = true;
        if selection.ticket != job.job_id {
            return Err(refuse(PROVIDER_TICKET, selection.pf, Refusal::ValueOutOfRange(PROVIDER_TICKET)));
        }
        let preset = facestereo_preset_row(selection.pf).map_err(|cause| {
            refuse(PROVIDER_ROWS, selection.pf, cause)
        })?;
        if preset.mix_color != FACE_STEREO_MIX_COLOR
            || preset.opacity_percent as f32 != FACE_STEREO_OPACITY_PERCENT
        {
            return Err(refuse(PROVIDER_ROWS, selection.pf, Refusal::ValueOutOfRange(PROVIDER_ROWS)));
        }

        if !selection.degree_percent.is_finite() || !(0.0..=100.0).contains(&selection.degree_percent) {
            return Err(refuse(PROVIDER_ROWS, selection.pf, Refusal::ValueOutOfRange("FaceStereo DefaultDegree")));
        }
        let _material = facestereo_material(selection.pf, selection.mesh_type).map_err(|cause| {
            let name = face_stereo_material_missing(selection.pf).unwrap_or(MATERIAL_MISSING);
            refuse(name, selection.pf, cause)
        })?;
        let _mix_alpha = face_stereo_mix_alpha(
            preset.mix_color,
            preset.opacity_percent as f32,
            selection.degree_percent / 100.0,
        )
        .map_err(|cause| refuse(PROVIDER_FILTER_TYPE, selection.pf, cause))?;

        let missing_name = face_stereo_mask_missing(selection.pf).unwrap_or(OWNER_MASK_MISSING);
        let source = selection.owner_mask_source.filter(|name| !name.is_empty()).ok_or_else(|| {
            refuse(missing_name, selection.pf, Refusal::MissingInput(missing_name))
        })?;
        if !source.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '/' || c == ':') {
            return Err(refuse(missing_name, selection.pf, Refusal::ValueOutOfRange(missing_name)));
        }
        let row = job
            .rows
            .iter()
            .find(|row| row.pf == selection.pf)
            .ok_or_else(|| refuse(PROVIDER_ROWS, selection.pf, Refusal::MissingInput(PROVIDER_ROWS)))?;
        if row.source_effect != FACE_STEREO_EFFECT
            || row.filter_type != FACE_STEREO_FILTER_TYPE
            || row.owner_mask.is_none()
            || row.material.is_none()
        {
            return Err(refuse(PROVIDER_ROWS, selection.pf, Refusal::ValueOutOfRange(PROVIDER_ROWS)));
        }
    }
    Ok(())
}

fn own_row(
    job: &ExternalMakeupJob<'_>,
    row: &ExternalMakeupRow<'_>,
) -> Result<OwnedMakeupBundleInputRow, ProviderRefusal> {
    if row.ticket != job.job_id
        || row.draw.ticket != job.job_id
        || row
            .owner_mask
            .is_some_and(|texture| texture.ticket != job.job_id)
        || row
            .extra_owner_mask
            .is_some_and(|texture| texture.ticket != job.job_id)
        || row
            .material
            .is_some_and(|texture| texture.ticket != job.job_id)
        || row.material_uv.is_some_and(|uv| uv.ticket != job.job_id)
    {
        return Err(ProviderRefusal::named(
            PROVIDER_TICKET,
            job.job_id,
            Some(row.pf),
            Some(row.ticket),
            Some(Refusal::ValueOutOfRange(PROVIDER_TICKET)),
        ));
    }
    if row.source_sha256 != job.source_sha256 {
        return Err(ProviderRefusal::named(
            PROVIDER_SOURCE_SHA,
            job.job_id,
            Some(row.pf),
            Some(row.ticket),
            Some(Refusal::ValueOutOfRange(PROVIDER_SOURCE_SHA)),
        ));
    }

    let profile = row_profile_for_effect(row.pf, row.source_effect).map_err(|refusal| {
        ProviderRefusal::named(
            PROVIDER_ROWS,
            job.job_id,
            Some(row.pf),
            Some(row.ticket),
            Some(refusal),
        )
    })?;

    if profile.requires_owner_mask() && row.owner_mask.is_none() {
        return Err(ProviderRefusal::named(
            OWNER_MASK_MISSING,
            job.job_id,
            Some(row.pf),
            Some(row.ticket),
            Some(Refusal::MissingInput(OWNER_MASK_MISSING)),
        ));
    }
    if !profile.requires_owner_mask() && row.owner_mask.is_some() {
        return Err(ProviderRefusal::named(
            PROVIDER_OWNER_MASK_UNEXPECTED,
            job.job_id,
            Some(row.pf),
            Some(row.ticket),
            Some(Refusal::ValueOutOfRange(PROVIDER_OWNER_MASK_UNEXPECTED)),
        ));
    }
    if profile.requires_material && row.material.is_none() {
        return Err(ProviderRefusal::named(
            MATERIAL_MISSING,
            job.job_id,
            Some(row.pf),
            Some(row.ticket),
            Some(Refusal::MissingInput(MATERIAL_MISSING)),
        ));
    }
    if profile.requires_material && row.material_uv.is_none() {
        return Err(ProviderRefusal::named(
            MATERIAL_UV_MISSING,
            job.job_id,
            Some(row.pf),
            Some(row.ticket),
            Some(Refusal::MissingInput(MATERIAL_UV_MISSING)),
        ));
    }

    let filter_type = check_filter_type(job.job_id, row, &profile)?;

    Ok(OwnedMakeupBundleInputRow {
        pf: row.pf,
        source_effect: Some(row.source_effect.to_owned()),
        face_id: Some(row.face_id),
        draw: own_draw(job.job_id, row.pf, row.draw)?,
        origin: row.origin,
        size: row.size,
        owner_mask: row
            .owner_mask
            .map(|texture| own_texture(job.job_id, row.pf, texture, PROVIDER_OWNER_MASK_SHAPE))
            .transpose()?,
        extra_owner_mask: row
            .extra_owner_mask
            .map(|texture| own_texture(job.job_id, row.pf, texture, PROVIDER_OWNER_MASK_SHAPE))
            .transpose()?,
        material: row
            .material
            .map(|texture| own_texture(job.job_id, row.pf, texture, PROVIDER_MATERIAL_SHAPE))
            .transpose()?,
        material_uv: row
            .material_uv
            .map(|uv| own_uv(job.job_id, row.pf, uv))
            .transpose()?,
        mix_color: row.mix_color,
        opacity_percent: row.opacity_percent,
        filter_type,
    })
}

/// Validates the row `FilterType` against the same predicate the portrait stage
/// applies, and injects the recovered preset value when the caller leaves the
/// field at zero.
///
/// `FilterType` is the blend macro (`lipstick_blend_mode`), so a wrong value
/// silently changes pixels instead of erroring. Rather than keep a second
/// routing table here, this delegates to `MakeupRoute::resolve_same_job`, which
/// is exactly what `resolve_makeup_route` calls at consumption time, so any job
/// that passes this boundary is guaranteed to resolve downstream. The recovered
/// preset table then pins the blend macro for the rows where the route ignores
/// the supplied value.
fn check_filter_type(
    job_id: u64,
    row: &ExternalMakeupRow<'_>,
    profile: &MakeupRowProfile,
) -> Result<u32, ProviderRefusal> {
    let refuse = |cause: Refusal| {
        ProviderRefusal::named(
            PROVIDER_FILTER_TYPE,
            job_id,
            Some(row.pf),
            Some(row.ticket),
            Some(cause),
        )
    };

    let preset_filter_type = preset_filter_types(row.pf, row.source_effect);

    // Zero is the "caller did not select a preset macro" sentinel. The route
    // rejects it outright for preset-driven rows, so it is only ever usable as
    // an injection request. Fall back to the recovered single-value table only
    // when there is no preset macro to read.
    let effective = if row.filter_type == 0 {
        if let Some(preset) = preset_filter_type {
            preset[0]
        } else {
            recovered_filter_type(row.pf, row.source_effect)
                .ok_or_else(|| refuse(Refusal::MissingInput(PROVIDER_FILTER_TYPE)))?
        }
    } else {
        row.filter_type
    };

    let has_material = row.material.is_some() && profile.requires_material;
    // Keep the route's own refusal as the cause so the underlying variant and
    // name survive into the host report.
    let route =
        MakeupRoute::resolve_same_job(row.pf, effective, has_material).map_err(|refusal| {
            ProviderRefusal::named(
                refusal_code(&refusal),
                job_id,
                Some(row.pf),
                Some(row.ticket),
                Some(refusal),
            )
        })?;
    if effective != route.filter_type() {
        return Err(refuse(Refusal::ValueOutOfRange(PROVIDER_FILTER_TYPE)));
    }

    // The preset's own macro is authoritative when it is available; the
    // single-value table is the fallback for rows without per-preset data.
    // A macro outside the accepted set gets a named refusal instead of being
    // rewritten, so a wrong value can never silently change pixels.
    if let Some(accepted) = preset_filter_type {
        if !accepted.contains(&effective) {
            return Err(refuse(Refusal::ValueOutOfRange(PROVIDER_FILTER_TYPE)));
        }
    }

    // Reject any macro that has no blend implementation before it can reach the
    // shader, where an unmapped value would otherwise surface as a refusal.
    lipstick_blend_mode(effective)
        .map_err(|_| refuse(Refusal::MissingInput(PROVIDER_FILTER_TYPE)))?;
    Ok(effective)
}

/// Extracts the stable name carried by a route refusal so the provider can
/// report a single code while keeping the underlying refusal as the cause.
fn refusal_code(refusal: &Refusal) -> &'static str {
    match refusal {
        Refusal::MissingInput(name) | Refusal::ShapeMismatch(name) => name,
        Refusal::ValueOutOfRange(name) => name,
        Refusal::UnknownEffect(_) | Refusal::NotImplemented(_) => PROVIDER_ROWS,
    }
}

fn own_draw(
    job_id: u64,
    pf: u32,
    draw: ExternalFace2dDraw<'_>,
) -> Result<OwnedFace2dDrawBytes, ProviderRefusal> {
    if draw.vbo.is_empty() {
        return Err(ProviderRefusal::named(
            VBO_BYTES_MISSING,
            job_id,
            Some(pf),
            Some(draw.ticket),
            Some(Refusal::MissingInput(VBO_BYTES_MISSING)),
        ));
    }
    if draw.vertex_stride < 8
        || draw.position_offset > draw.vertex_stride.saturating_sub(8)
        || draw.vbo.len() % draw.vertex_stride != 0
    {
        return Err(ProviderRefusal::named(
            PROVIDER_VBO_SHAPE,
            job_id,
            Some(pf),
            Some(draw.ticket),
            Some(Refusal::ShapeMismatch(PROVIDER_VBO_SHAPE)),
        ));
    }
    if draw.ebo_u16.is_empty() {
        return Err(ProviderRefusal::named(
            EBO_BYTES_MISSING,
            job_id,
            Some(pf),
            Some(draw.ticket),
            Some(Refusal::MissingInput(EBO_BYTES_MISSING)),
        ));
    }
    if draw.ebo_u16.len() % 2 != 0
        || draw.index_count == 0
        || draw.index_count % 3 != 0
        || draw
            .first_index
            .checked_add(draw.index_count)
            .and_then(|end| end.checked_mul(2))
            .is_none_or(|end| end > draw.ebo_u16.len())
    {
        return Err(ProviderRefusal::named(
            PROVIDER_EBO_SHAPE,
            job_id,
            Some(pf),
            Some(draw.ticket),
            Some(Refusal::ShapeMismatch(PROVIDER_EBO_SHAPE)),
        ));
    }
    Ok(OwnedFace2dDrawBytes {
        vertices: Arc::from(draw.vbo),
        vertex_stride: draw.vertex_stride,
        position_offset: draw.position_offset,
        indices_u16: Arc::from(draw.ebo_u16),
        draw_first_index: draw.first_index,
        draw_index_count: draw.index_count,
    })
}

fn own_texture(
    job_id: u64,
    pf: u32,
    texture: ExternalTexture<'_>,
    shape_code: &'static str,
) -> Result<Arc<Image>, ProviderRefusal> {
    let expected = (texture.width as usize)
        .checked_mul(texture.height as usize)
        .and_then(|pixels| pixels.checked_mul(4));
    if texture.width == 0 || texture.height == 0 || expected != Some(texture.rgba8.len()) {
        return Err(ProviderRefusal::named(
            shape_code,
            job_id,
            Some(pf),
            Some(texture.ticket),
            Some(Refusal::ShapeMismatch(shape_code)),
        ));
    }
    Ok(Arc::new(Image {
        width: texture.width,
        height: texture.height,
        pixels: texture.rgba8.to_vec(),
    }))
}

fn own_uv(
    job_id: u64,
    pf: u32,
    uv: ExternalFace2dUv<'_>,
) -> Result<OwnedFace2dUvBytes, ProviderRefusal> {
    if uv.vbo.is_empty() {
        return Err(ProviderRefusal::named(
            MATERIAL_UV_MISSING,
            job_id,
            Some(pf),
            Some(uv.ticket),
            Some(Refusal::MissingInput(MATERIAL_UV_MISSING)),
        ));
    }
    if uv.vertex_stride < 8
        || uv.offset > uv.vertex_stride.saturating_sub(8)
        || uv.vbo.len() % uv.vertex_stride != 0
    {
        return Err(ProviderRefusal::named(
            PROVIDER_MATERIAL_UV_SHAPE,
            job_id,
            Some(pf),
            Some(uv.ticket),
            Some(Refusal::ShapeMismatch(PROVIDER_MATERIAL_UV_SHAPE)),
        ));
    }
    Ok(OwnedFace2dUvBytes {
        vertices: Arc::from(uv.vbo),
        vertex_stride: uv.vertex_stride,
        offset: uv.offset,
    })
}

/// Producer-side adapter for the external B3 makeup bundle.
///
/// It adds no policy of its own: it calls `ExternalMakeupProvider::build`, so every
/// same-job identity and shape refusal keeps its original named code, and only a
/// complete fourteen-row bundle ever becomes producer output.
pub mod adapter {
    use super::{ExternalMakeupJob, ExternalMakeupProvider, ProviderRefusal};
    use crate::portrait_stage::ProducerInputs;

    /// Builds the owned bundle and wraps it as producer output. A missing or partial
    /// job is refused before any `ProducerInputs` exists, so a refusal cannot leave a
    /// half-filled bundle behind for the stage to consume.
    pub fn makeup_bundle_from_external(
        job: Option<ExternalMakeupJob<'_>>,
    ) -> Result<ProducerInputs, ProviderRefusal> {
        let bundle = ExternalMakeupProvider::build(job)?;
        Ok(ProducerInputs {
            makeup_bundle: Some(bundle),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portrait_makeup_bundle::allowed_source_effects;
    use crate::portrait_stage::{
        find_makeup_input_for_contract_test, PortraitJobInputs, PortraitMasks,
    };

    const JOB: u64 = 77;
    const SHA: [u8; 32] = [0x5a; 32];
    const PIXEL: [u8; 4] = [1, 2, 3, 255];
    const VBO: [u8; 24] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63,
    ];
    const EBO: [u8; 6] = [0, 0, 1, 0, 2, 0];

    fn texture() -> ExternalTexture<'static> {
        ExternalTexture {
            ticket: JOB,
            width: 1,
            height: 1,
            rgba8: &PIXEL,
        }
    }

    fn draw() -> ExternalFace2dDraw<'static> {
        ExternalFace2dDraw {
            ticket: JOB,
            vbo: &VBO,
            vertex_stride: 8,
            position_offset: 0,
            ebo_u16: &EBO,
            first_index: 0,
            index_count: 3,
        }
    }

    fn uv() -> ExternalFace2dUv<'static> {
        ExternalFace2dUv {
            ticket: JOB,
            vbo: &VBO,
            vertex_stride: 8,
            offset: 0,
        }
    }

    pub(super) fn complete_rows() -> Vec<ExternalMakeupRow<'static>> {
        B3_MAKEUP_ROWS
            .into_iter()
            .map(|pf| {
                let effect = if pf == 107 {
                    "LipStickImageMakeup"
                } else {
                    allowed_source_effects(pf)
                        .unwrap()
                        .iter()
                        .find(|effect| row_profile_for_effect(pf, effect).is_ok())
                        .copied()
                        .unwrap()
                };
                let profile = row_profile_for_effect(pf, effect).unwrap();
                ExternalMakeupRow {
                    pf,
                    source_effect: effect,
                    face_id: 4,
                    ticket: JOB,
                    source_sha256: SHA,
                    draw: draw(),
                    origin: match crate::portrait_makeup_bundle::facestereo_preset_row(pf) {
                        Ok(preset) => [preset.roi[0], preset.roi[1]],
                        Err(_) => [0, 0],
                    },
                    size: match crate::portrait_makeup_bundle::facestereo_preset_row(pf) {
                        Ok(preset) => [preset.roi[2], preset.roi[3]],
                        Err(_) => [1, 1],
                    },
                    owner_mask: profile.requires_owner_mask().then(texture),
                    extra_owner_mask: None,
                    material: profile.requires_material.then(texture),
                    material_uv: profile.requires_material.then(uv),
                    mix_color: match crate::portrait_makeup_bundle::facestereo_preset_row(pf) {
                        Ok(preset) => preset.mix_color,
                        Err(_) => [0; 4],
                    },
                    opacity_percent: 100.0,
                    // The recovered per-row macro, not one blanket value: PF111
                    // is 40 and the image rows are 41/42.
                    filter_type: recovered_filter_type(pf, effect).unwrap(),
                }
            })
            .collect()
    }

    fn job<'a>(rows: &'a [ExternalMakeupRow<'a>]) -> ExternalMakeupJob<'a> {
        ExternalMakeupJob {
            job_id: JOB,
            ticket: JOB,
            frame_size: [4, 4],
            source_sha256: SHA,
            rows,
            facestereo: None,
        }
    }

    pub(super) fn facestereo<'a>(rows: &'a [ExternalMakeupRow<'a>]) -> ExternalMakeupJob<'a> {
        ExternalMakeupJob {
            job_id: JOB,
            ticket: JOB,
            frame_size: [4, 4],
            source_sha256: SHA,
            rows,
            facestereo: Some(&[
                ExternalFaceStereoRow {
                    pf: 8,
                    ticket: JOB,
                    mesh_type: 2,
                    degree_percent: 100.0,
                    owner_mask_source: Some("capture:faceparsing_rgba"),
                },
                ExternalFaceStereoRow {
                    pf: 9,
                    ticket: JOB,
                    mesh_type: 4,
                    degree_percent: 100.0,
                    owner_mask_source: Some("capture:blusher_rgba"),
                },
            ]),
        }
    }

    #[test]
    fn missing_owner_mask_is_named_with_pf_and_job() {
        let mut rows = complete_rows();
        let row = rows.iter_mut().find(|row| row.pf == 8).expect("PF8 row");
        row.owner_mask = None;

        let error = ExternalMakeupProvider::build(Some(job(&rows))).unwrap_err();
        assert_eq!(error.code, OWNER_MASK_MISSING);
        assert_eq!(error.pf, Some(8));
        assert_eq!(error.job_id, JOB);
        assert_eq!(error.cause, Some(Refusal::MissingInput(OWNER_MASK_MISSING)));
    }

    #[test]
    fn complete_job_builds_fourteen_same_job_rows() {
        let rows = complete_rows();
        let bundle = ExternalMakeupProvider::build(Some(job(&rows))).unwrap();

        assert_eq!(bundle.job_id, JOB);
        assert_eq!(bundle.ticket, JOB);
        assert_eq!(bundle.source_sha256, SHA);
        assert_eq!(bundle.rows.len(), B3_MAKEUP_ROWS.len());

        let owner_masks = bundle
            .rows
            .iter()
            .filter(|row| row.owner_mask.is_some())
            .count();
        let materials = bundle
            .rows
            .iter()
            .filter(|row| row.material.is_some())
            .count();
        assert_eq!(owner_masks, 13);
        assert_eq!(materials, 7);

        for row in &bundle.rows {
            if let Some(mask_uv) = &row.mask_uv {
                assert_eq!(mask_uv.ticket(), Some(JOB), "PF{} mask UV", row.profile.pf);
            }
            if let Some(material_uv) = &row.material_uv {
                assert_eq!(
                    material_uv.ticket(),
                    Some(JOB),
                    "PF{} material UV",
                    row.profile.pf
                );
            }
        }
    }

    #[test]
    fn cross_job_draw_is_rejected() {
        let mut rows = complete_rows();
        rows[0].draw.ticket = JOB + 1;

        let error = ExternalMakeupProvider::build(Some(job(&rows))).unwrap_err();
        assert_eq!(error.code, PROVIDER_TICKET);
        assert_eq!(error.pf, Some(8));
    }

    #[test]
    fn all_fourteen_rows_carry_their_recovered_filter_type() {
        let rows = complete_rows();
        for row in &rows {
            assert_eq!(
                row.filter_type,
                recovered_filter_type(row.pf, row.source_effect).unwrap(),
                "PF{}",
                row.pf
            );
        }
        // The blanket-11 mistake this replaces: PF111 is never 11.
        let pf111 = rows.iter().find(|row| row.pf == 111).expect("PF111 row");
        assert_eq!(pf111.filter_type, 40);
        ExternalMakeupProvider::build(Some(job(&rows))).expect("table-driven job builds");
    }

    #[test]
    fn pf111_rejects_a_foreign_filter_type() {
        let mut rows = complete_rows();
        let row = rows
            .iter_mut()
            .find(|row| row.pf == 111)
            .expect("PF111 row");
        row.filter_type = 11;

        let error = ExternalMakeupProvider::build(Some(job(&rows))).unwrap_err();
        assert_eq!(error.code, PROVIDER_FILTER_TYPE);
        assert_eq!(error.pf, Some(111));
        assert_eq!(error.job_id, JOB);
    }

    #[test]
    fn zero_filter_type_injects_the_recovered_macro() {
        let mut rows = complete_rows();
        for row in &mut rows {
            row.filter_type = 0;
        }

        let bundle = ExternalMakeupProvider::build(Some(job(&rows))).unwrap();
        for row in &bundle.rows {
            assert_eq!(
                row.filter_type,
                recovered_filter_type(row.profile.pf, row.source_effect.as_str()).unwrap(),
                "PF{}",
                row.profile.pf
            );
        }
    }

    #[test]
    fn lip_color_variant_has_no_recovered_macro_and_must_be_supplied() {
        assert_eq!(recovered_filter_type(107, "LipColor"), None);
        assert_eq!(recovered_filter_type(107, "LipStickImageMakeup"), Some(42));
        assert_eq!(recovered_filter_type(111, "EyePupilMakeup"), Some(40));

        // Zero with no recovered value is a named refusal, not a silent 0.
        let mut rows = complete_rows();
        let row = rows
            .iter_mut()
            .find(|row| row.pf == 107)
            .expect("PF107 row");
        row.source_effect = "LipColor";
        row.material = None;
        row.material_uv = None;
        row.filter_type = 0;

        let error = ExternalMakeupProvider::build(Some(job(&rows))).unwrap_err();
        assert_eq!(error.code, PROVIDER_FILTER_TYPE);
        assert_eq!(error.pf, Some(107));
    }

    #[test]
    fn lip_color_variant_accepts_a_route_valid_supplied_macro() {
        let mut rows = complete_rows();
        let row = rows
            .iter_mut()
            .find(|row| row.pf == 107)
            .expect("PF107 row");
        row.source_effect = "LipColor";
        row.material = None;
        row.material_uv = None;
        // LipColor resolves to the Color route, whose filter type is caller-chosen.
        row.filter_type = 11;

        let bundle = ExternalMakeupProvider::build(Some(job(&rows))).unwrap();
        let material_rows = bundle
            .rows
            .iter()
            .filter(|row| row.material.is_some())
            .count();
        // Six, not seven: the LipColor variant drops PF107's material requirement.
        assert_eq!(material_rows, 6);
    }

    #[test]
    fn complete_job_feeds_the_same_bundle_to_producer_inputs() {
        let rows = complete_rows();
        let producer = adapter::makeup_bundle_from_external(Some(job(&rows))).unwrap();
        let bundle = producer.makeup_bundle.as_ref().expect("adapter bundle");
        assert_eq!(bundle.rows.len(), B3_MAKEUP_ROWS.len());

        let masks = PortraitMasks::solid(1.0);
        let mut inputs = PortraitJobInputs::from_renderer_job(JOB, &masks);
        inputs.fill_from_producer_inputs(&producer);
        assert_eq!(inputs.job_id(), Some(JOB));

        for pf in B3_MAKEUP_ROWS.into_iter() {
            let stage_input = find_makeup_input_for_contract_test(pf, &inputs)
                .unwrap_or_else(|refusal| panic!("PF{pf} resolved to {refusal:?}"));
            let bundle_input = bundle.makeup_input(pf).unwrap();
            // The stage must resolve the row this very bundle owns, not a second
            // payload assembled from another job.
            assert_eq!(stage_input.pf, bundle_input.pf, "PF{pf}");
            assert_eq!(stage_input.filter_type, bundle_input.filter_type, "PF{pf}");
            assert_eq!(stage_input.mix_color, bundle_input.mix_color, "PF{pf}");
            assert_eq!(
                stage_input.opacity_percent, bundle_input.opacity_percent,
                "PF{pf}"
            );
            assert_eq!(
                stage_input.mask.map(|image| image.width),
                bundle_input.mask.map(|image| image.width),
                "PF{pf} mask"
            );
            assert_eq!(
                stage_input.material.map(|image| image.width),
                bundle_input.material.map(|image| image.width),
                "PF{pf} material"
            );
            assert_eq!(
                stage_input.mask_uv.is_some(),
                bundle_input.mask_uv.is_some(),
                "PF{pf} mask UV"
            );
            assert_eq!(
                stage_input.material_uv.is_some(),
                bundle_input.material_uv.is_some(),
                "PF{pf} material UV"
            );
        }

        // PF111 is the one row the stage must see with material and no mask stream.
        let pf111 = find_makeup_input_for_contract_test(111, &inputs).unwrap();
        assert!(pf111.mask.is_none(), "PF111 must carry no owner mask");
        assert!(pf111.mask_uv.is_none(), "PF111 must carry no mask UV");
        assert!(pf111.material.is_some(), "PF111 must carry material");
        assert!(pf111.material_uv.is_some(), "PF111 must carry material UV");
    }

    #[test]
    fn adapter_refuses_missing_owner_mask() {
        let mut rows = complete_rows();
        rows.iter_mut()
            .find(|row| row.pf == 8)
            .expect("PF8 row")
            .owner_mask = None;
        let error = adapter::makeup_bundle_from_external(Some(job(&rows)))
            .err()
            .expect("a missing owner mask must be refused");
        assert_eq!(error.code, OWNER_MASK_MISSING);
        assert_eq!(error.pf, Some(8));
        assert_eq!(error.job_id, JOB);
        assert_eq!(error.cause, Some(Refusal::MissingInput(OWNER_MASK_MISSING)));
    }

    #[test]
    fn adapter_refuses_missing_material() {
        let mut rows = complete_rows();
        rows.iter_mut()
            .find(|row| row.pf == 8)
            .expect("PF8 row")
            .material = None;
        let error = adapter::makeup_bundle_from_external(Some(job(&rows)))
            .err()
            .expect("a missing material must be refused");
        assert_eq!(error.code, MATERIAL_MISSING);
        assert_eq!(error.pf, Some(8));
        assert_eq!(error.job_id, JOB);
        assert_eq!(error.cause, Some(Refusal::MissingInput(MATERIAL_MISSING)));
    }

    #[test]
    fn missing_decrypted_material_png_is_named() {
        let missing = std::env::temp_dir().join("aurora-r25-makeup-no-such-material.png");
        let error = load_decrypted_material_png(missing, JOB, 116)
            .err()
            .expect("missing decrypted PNG must be refused");
        assert_eq!(error.code, MATERIAL_PNG_MISSING);
        assert_eq!(error.job_id, JOB);
        assert_eq!(error.pf, Some(116));
        assert_eq!(
            error.cause,
            Some(Refusal::MissingInput(MATERIAL_PNG_MISSING))
        );
    }

    // --- per-preset FilterType (loader wiring) -------------------------------

    /// Sets the PF90032 row in a complete 14-row job to a preset macro.
    fn with_pf90032_filter_type(rows: &mut [ExternalMakeupRow<'static>], filter_type: u32) {
        let row = rows
            .iter_mut()
            .find(|row| row.pf == 90032)
            .expect("PF90032 row");
        row.source_effect = "SymEyeImageMakeup";
        row.filter_type = filter_type;
    }

    #[test]
    fn pf90032_accepts_vendor_preset_filter_types_42_and_44() {
        for filter_type in [42, 44] {
            let mut rows = complete_rows();
            with_pf90032_filter_type(&mut rows, filter_type);
            let bundle = ExternalMakeupProvider::build(Some(job(&rows)))
                .unwrap_or_else(|error| panic!("preset FilterType {filter_type} refused: {error}"));
            let row = bundle
                .rows
                .iter()
                .find(|row| row.profile.pf == 90032)
                .expect("PF90032 row");
            assert_eq!(row.filter_type, filter_type);
        }
        // The old single-value table did not contain either value.
        assert_eq!(recovered_filter_type(90032, "SymEyeImageMakeup"), Some(41));
    }

    #[test]
    fn pf90032_unknown_preset_filter_type_is_named_refused() {
        let mut rows = complete_rows();
        with_pf90032_filter_type(&mut rows, 99);
        let error = ExternalMakeupProvider::build(Some(job(&rows))).unwrap_err();
        assert_eq!(error.code, PROVIDER_FILTER_TYPE);
        assert_eq!(error.pf, Some(90032));
        assert_eq!(error.job_id, JOB);
        assert_eq!(error.cause, Some(Refusal::ValueOutOfRange(PROVIDER_FILTER_TYPE)));
    }

    #[test]
    fn rows_without_preset_data_fall_back_to_the_single_value_table() {
        // PF111 has one recovered macro and no per-preset set: a foreign value
        // must still be refused, and zero still injects 40.
        assert_eq!(preset_filter_types(111, "EyePupilMakeup"), Some(&[40][..]));
        let rows = complete_rows();
        let bundle = ExternalMakeupProvider::build(Some(job(&rows))).unwrap();
        let pf111 = bundle
            .rows
            .iter()
            .find(|row| row.profile.pf == 111)
            .expect("PF111 row");
        assert_eq!(pf111.filter_type, 40);
        let mut rows = complete_rows();
        rows.iter_mut()
            .find(|row| row.pf == 111)
            .expect("PF111 row")
            .filter_type = 11;
        assert_eq!(
            ExternalMakeupProvider::build(Some(job(&rows))).unwrap_err().code,
            PROVIDER_FILTER_TYPE
        );
    }

    #[test]
    fn decoded_loader_value_feeds_preset_filter_lookup() {
        // Inline decoded effect.json row: the loader owns parsing, the provider
        // owns acceptance. A PF90032 row carrying FilterType=44 must pass.
        const EFFECT_JSON: &str = r#"{
          "Effects": [
            {"EffectName":"SymEyeImageMakeup","ParamFlag":90032,"FilterType":44,
             "Paths":["yyz.png"],"ROIRect":[200,400,300,300],
             "Opacity":80,"DefaultDegree":70}
          ]
        }"#;
        let effects = crate::portrait_vendor_config::parse_config(
            "Makeups/Eyemakeup/MUEM00000003/effect.json",
            EFFECT_JSON,
        )
        .expect("inline effect.json");
        let index = crate::portrait_vendor_config::VendorConfigIndex::from_effects(effects);
        let effect = index
            .for_pf(90032)
            .next()
            .expect("PF90032 entry from decoded config");
        let filter_type = effect.filter_type.expect("preset FilterType") as u32;
        assert_eq!(filter_type, 44);
        assert_eq!(effect.effect_name, "SymEyeImageMakeup");
        let mut rows = complete_rows();
        with_pf90032_filter_type(&mut rows, filter_type);
        let bundle = ExternalMakeupProvider::build(Some(job(&rows))).expect("loader-driven macro");
        let row = bundle
            .rows
            .iter()
            .find(|row| row.profile.pf == 90032)
            .expect("PF90032 row");
        assert_eq!(row.filter_type, 44);
    }
}

#[cfg(test)]
mod facestereo_provider_tests {
    use super::*;
    use super::tests::{complete_rows, facestereo};
    use crate::portrait_makeup_bundle::facestereo_preset_row;
    use crate::portrait_makeup_wiring::MAKEUP_2D_ROWS;

    fn pf_row<'a>(rows: &'a [ExternalMakeupRow<'a>], pf: u32) -> &'a ExternalMakeupRow<'a> {
        rows.iter().find(|row| row.pf == pf).expect("row")
    }

    #[test]
    fn facestereo_rows_resolve_with_recovered_preset_values() {
        let rows = complete_rows();
        for pf in [8, 9] {
            let preset = facestereo_preset_row(pf).unwrap();
            let row = pf_row(&rows, pf);
            assert_eq!(row.source_effect, preset.source_effect);
            assert_eq!(row.filter_type, preset.filter_type);
            assert_eq!(row.origin, [preset.roi[0], preset.roi[1]]);
            assert_eq!(row.size, [preset.roi[2], preset.roi[3]]);
            assert_eq!(row.mix_color, preset.mix_color);
            assert_eq!(row.opacity_percent, preset.opacity_percent as f32);
        }
        // The first-class static rows are separate from the 12-row 2D wiring:
        // FaceStereo mask capture cannot be synthesized from landmarks.
        for pf in [8, 9] {
            assert!(!MAKEUP_2D_ROWS.iter().any(|row| row.pf == pf));
        }
        let bundle = ExternalMakeupProvider::build(Some(facestereo(&rows))).unwrap();
        for pf in [8, 9] {
            let resolved = bundle
                .rows
                .iter()
                .find(|row| row.profile.pf == pf)
                .expect("FaceStereo row survives the provider");
            assert_eq!(resolved.filter_type, 42);
            assert!(resolved.owner_mask.is_some());
            assert!(resolved.material.is_some());
        }
    }

    #[test]
    fn missing_face_stereo_mask_source_is_named_per_pf() {
        let rows = complete_rows();
        for (pf, expected) in [(8, "PF8_FACE_STEREO_OWNER_MASK"), (9, "PF9_FACE_STEREO_OWNER_MASK")] {
            let mut job = facestereo(&rows);
            let mut selections = job.facestereo.unwrap().to_vec();
            let selection = selections.iter_mut().find(|selection| selection.pf == pf).unwrap();
            selection.owner_mask_source = None;
            job.facestereo = Some(&selections);
            let error = ExternalMakeupProvider::build(Some(job)).unwrap_err();
            assert_eq!(error.code, expected);
            assert_eq!(error.pf, Some(pf));
            assert_eq!(error.cause, Some(Refusal::MissingInput(expected)));
        }
    }

    #[test]
    fn unknown_mesh_type_and_bad_degree_are_named_refusals() {
        let rows = complete_rows();
        let mut job = facestereo(&rows);
        let mut selections = job.facestereo.unwrap().to_vec();
        selections[0].mesh_type = 3;
        job.facestereo = Some(&selections);
        let error = ExternalMakeupProvider::build(Some(job)).unwrap_err();
        assert_eq!(error.code, "PF8_FACE_STEREO_MATERIAL");
        assert_eq!(error.pf, Some(8));

        let mut job = facestereo(&rows);
        let mut selections = job.facestereo.unwrap().to_vec();
        selections[1].degree_percent = 101.0;
        job.facestereo = Some(&selections);
        let error = ExternalMakeupProvider::build(Some(job)).unwrap_err();
        assert_eq!(error.code, PROVIDER_ROWS);
        assert_eq!(error.pf, Some(9));
    }

    #[test]
    fn facestereo_job_preserves_recovered_scalar_factors() {
        assert_eq!(
            face_stereo_mix_alpha(FACE_STEREO_MIX_COLOR, FACE_STEREO_OPACITY_PERCENT, 1.0),
            Ok(1.0)
        );
        assert_eq!(
            face_stereo_mix_alpha(FACE_STEREO_MIX_COLOR, FACE_STEREO_OPACITY_PERCENT, 0.5),
            Ok(0.5)
        );
        // Both the preset and the provider agree on the one-factor law.
        assert_eq!(FACE_STEREO_FILTER_TYPE, 42);
        assert_eq!(lipstick_blend_mode(FACE_STEREO_FILTER_TYPE).is_ok(), true);
    }
}
