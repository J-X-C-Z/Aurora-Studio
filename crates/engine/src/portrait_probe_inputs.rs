//! Same-frame input evidence for the per-photo portrait capability probe.
//!
//! [`crate::portrait_runtime_probe`] answers "can *this* photo run row X on
//! *this* machine". Its first version had to answer `MissingInput` for every
//! teeth and eye row forever, because the planes those rows need are only
//! produced on the render path and the probe deliberately does not render. This
//! module supplies the missing half of the answer without rendering anything.
//!
//! It is split in two tiers on purpose, because a 24 MP frame makes the naive
//! shape of this wrong:
//!
//! | tier | entry point | what it costs | when to call |
//! | --- | --- | --- | --- |
//! | cheap | [`prove_pf_inputs`] / [`prove_pf_inputs_with_face_mask`] | no owner-mask rasterization, no UV/writes arrays, no per-row bundle; one teeth attach (one `frame_pixels` mask clone) and `stat`-level material checks | every capability snapshot |
//! | per row | [`produce_row`] | one row's owner mask at ROI size (RGBA, `roi_w * roi_h * 4`) plus that row's material; dropped before returning | only when the panel asks for that row, or on an explicit refresh |
//! | per row | [`prove_pf101`] | one full-frame `u8` face mask (`frame_pixels` bytes) plus the silkworm material | only when the panel asks for PF101 |
//!
//! No tier ever accumulates one plane per PF. Building all fourteen 2D rows for
//! one frame would hold fourteen ROI planes and fourteen UV/write arrays at
//! once; [`produce_row`] builds exactly one and returns facts, not pixels.
//!
//! ## What each tier actually proves
//!
//! Cheap tier, per 2D makeup row:
//!
//! * the row's draw assembles from *this* bundle's 194 landmarks
//!   (`portrait_makeup_mesh2d::build_face2d_draw_bytes`, which itself refuses
//!   out-of-frame or degenerate geometry),
//! * the recovered [`MakeupRowProfile`] says whether the row needs an owner mask
//!   and/or a material,
//! * the material the row would really select — the `pc:<pf>` style preset when
//!   the document sets one, otherwise the row contract's own PNG — exists on
//!   disk at the resolver's root, reported by name/size only.
//!
//! It does **not** claim the row is proven: the owner mask has not been
//! rasterized yet, so the verdict is [`PfInputVerdict::ReadyNotProduced`].
//!
//! Production tier ([`produce_row`]) runs the renderer's own
//! [`FaceInputBundle::attach_makeup_bundle_with_options`] for that single row and
//! then the consumer's own gate, [`OwnedMakeupBundle::makeup_input`]. On top of
//! that it samples the material texel under every material UV and the mask texel
//! under every mask UV, so "proven" means the UVs actually land on real,
//! non-transparent pixels — a file that merely exists, or a call that merely
//! returned `Ok`, is not enough.
//!
//! ## Per-PF notes the caller should keep
//!
//! * `108`, `109`, `110` are one capability. `pc:109`/`pc:110` are children of
//!   `pc:108` (`portrait_stage::pf108_ui_percents`), so they receive the same
//!   evidence; a capability table that grows a separate `109`/`110` row must not
//!   demand separate support for them.
//! * `8`/`9` are the FaceStereo 2D rows: their owner mask is the 194-point
//!   silhouette hull, optionally gated by the renderer's same-frame `Face` plane
//!   passed through [`produce_row`].
//! * `90032` is the `EyeMaskType=10` eye image makeup row; its proof includes the
//!   recovered mask semantic and channel from [`MakeupRowProfile`], which is the
//!   device-independent part of that row.
//! * `101` is not a 2D row at all: it needs a material, a landmark-derived face
//!   mask and a band UV pair, so it has its own entry point rather than being
//!   silently folded into the makeup rows.
//! * Mask-only rows (`100`, `103`, `105`, `106`, `117`, `118`, `30007`) need no
//!   file asset and can reach `Proven` with nothing installed. Material rows
//!   cannot: for them a missing material is [`PfInputVerdict::MissingAsset`],
//!   never `Proven`.
//!
//! ## Digests
//!
//! Nothing here hashes the frame: the ticket check reuses
//! [`FaceInputBundle::validate_ticket`], which already reuses the bundle's bound
//! digest, and the decoded-source cache already keeps the frame-level LRU.
//!
//! The two 786 KB teeth LUTs and the ID2G mask are hashed in full on every call,
//! with `model_sha256::sha256`. There is deliberately **no second-layer cache**:
//! a "same length, same first and last 4 KiB" fingerprint is not a content
//! identity, so a payload whose middle was changed would still read as verified.
//! Re-hashing 1.5 MB of LUT plus one plane per capability refresh is cheaper than
//! an evidence path that can lie.

use std::collections::BTreeMap;

use aurora_portrait::kernels::makeup_material::{MaskChannel, mask_weight, sample_rgba};
use aurora_portrait::{Image, Refusal, model_sha256};

use crate::portrait_eye_bag_inputs::{PF101_MATERIAL, pf101_default_material_candidates};
use crate::portrait_face_input_stage::FaceInputBundle;
use crate::portrait_makeup_bundle::{
    MATERIAL_MISSING, MakeupRowProfile, OwnedMakeupBundle, row_profile_for_effect,
};
use crate::portrait_makeup_provider::MATERIAL_PNG_MISSING;
use crate::portrait_makeup_wiring::{
    MAKEUP_2D_ROWS, MAKEUP_ASSET_ROOT_MISSING, makeup_presets_for_pf, makeup_rows_2d_declared,
};
use crate::portrait_stage::{
    MakeupInput, PF108_BRIGHT_LUT_IDENTITY, PF108_WHITEN_LUT_IDENTITY, Pf101ProducerInputs,
    ProducerInputs, TeethLutIdentity, TeethMaskFormat, TeethMaskSource, TeethProducerInputs,
    pf108_ui_percents,
};

/// PF108's two children. They are not rows of their own.
pub const TEETH_FAMILY_PFS: [u32; 3] = [108, 109, 110];

/// PF101's style presets; every one of them must resolve before the row's style
/// picker means anything.
pub const PF101_STYLE_PROBE_LIMIT: usize = 24;

/// A mask plane whose peak is zero carries no target at all.
const MASK_FLOOR: u8 = 1;

/// Every PF this module can speak about: the teeth family, then the 2D makeup
/// rows in their canonical order, then PF101.
pub fn probed_pfs() -> Vec<u32> {
    let mut pfs = TEETH_FAMILY_PFS.to_vec();
    for pf in makeup_rows_2d_declared() {
        if !pfs.contains(pf) {
            pfs.push(*pf);
        }
    }
    if !pfs.contains(&101) {
        pfs.push(101);
    }
    pfs
}

/// Same-frame ticket verdict, carried once per call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TicketVerdict {
    pub ok: bool,
    /// The refusal code when `ok` is false; every row then reports it too.
    pub code: Option<&'static str>,
}

/// Facts behind a `Proven` verdict. Sizes, digests and counts only: never pixel
/// data, never a filesystem path.
#[derive(Clone, Debug, PartialEq)]
pub enum PfInputProof {
    /// PF108 family: the real ID2G plane plus both LUTs, content-checked.
    Teeth {
        mask_source: &'static str,
        mask_format: &'static str,
        mask_pixels: u32,
        mask_peak: u8,
        mask_sha256: [u8; 32],
        bright_percent: f32,
        deyellow_percent: f32,
        job_id: u64,
        whiten_lut_ok: bool,
        bright_lut_ok: bool,
    },
    /// A 2D makeup row produced through the real wiring and accepted by
    /// `OwnedMakeupBundle::makeup_input`, with its UVs sampled.
    Makeup {
        mask_semantic: &'static str,
        mask_channel: Option<String>,
        owner_mask_size: Option<[u32; 2]>,
        owner_mask_peak: u8,
        material_size: Option<[u32; 2]>,
        filter_type: u32,
        opacity_percent: f32,
        mask_uv_vertices: usize,
        material_uv_vertices: usize,
        /// Mask UV vertices that landed on a non-zero owner-mask texel.
        mask_uv_hits: usize,
        /// Material UV vertices that landed on a non-transparent material texel.
        material_uv_hits: usize,
    },
    /// PF101 silkworm: material, landmark face mask and band UVs were all
    /// accepted by `Pf101ProducerInputs::from_same_job`.
    ///
    /// `Pf101ProducerInputs` keeps its fields private, so this variant records
    /// *what was validated* rather than copying sizes out of a type that does not
    /// expose them: no number here is guessed.
    Pf101 {
        style_index: usize,
        presets_on_disk: usize,
        /// The producing gate that accepted the row, for the panel's evidence
        /// trail.
        accepted_by: &'static str,
        material_accepted: bool,
        face_mask_accepted: bool,
        band_uv_accepted: bool,
    },
}

/// What the cheap tier can say about a row it has not produced yet.
///
/// This is a *pending measurement*, not a verdict: the owner mask has not been
/// rasterized, so nothing here says the row will produce pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct RowReadiness {
    pub mask_semantic: &'static str,
    pub mask_channel: Option<String>,
    pub requires_owner_mask: bool,
    pub requires_material: bool,
    /// The row's draw assembled from this bundle's 194 landmarks.
    pub geometry_ok: bool,
    /// The material this row would really select, when it needs one.
    pub material: Option<MaterialFile>,
}

/// One material file, described without leaking where it lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterialFile {
    /// File name only (the last path component).
    pub file_name: String,
    pub bytes: u64,
    /// `true` when the file came from a `pc:<pf>` style preset rather than the
    /// row contract's own PNG.
    pub from_style_preset: bool,
}

/// One PF's answer.
#[derive(Clone, Debug, PartialEq)]
pub enum PfInputVerdict {
    /// Every same-frame input the row declares was produced and passed its gate.
    Proven(PfInputProof),
    /// The row's inputs were not produced. The code is the refusal the real
    /// producer returned (or a `PORTRAIT_PROBE_`-prefixed check of this module).
    MissingInput(&'static str),
    /// The same-frame inputs exist, but a file asset the row needs does not.
    MissingAsset(&'static str),
    /// Cheap tier: everything checkable without producing the row holds, but the
    /// same-frame owner mask has not been rasterized yet. [`produce_row`]
    /// upgrades this to `Proven` / `MissingInput` / `MissingAsset`.
    ///
    /// Callers must render this as *pending measurement* and must not unlock on
    /// it: only [`PfInputVerdict::Proven`] is evidence that a row has its inputs.
    ReadyNotProduced(RowReadiness),
}

impl PfInputVerdict {
    pub fn is_proven(&self) -> bool {
        matches!(self, Self::Proven(_))
    }

    /// `true` for [`Self::ReadyNotProduced`]: the cheap tier found nothing
    /// missing, but the row has not been produced.
    ///
    /// This is **not** an availability claim. A caller must render it as *pending
    /// measurement* and only unlock on [`Self::is_proven`]; the name says
    /// "pending" so it cannot be mistaken for green.
    pub fn is_pending_measurement(&self) -> bool {
        matches!(self, Self::ReadyNotProduced(_))
    }

    pub fn code(&self) -> Option<&'static str> {
        match self {
            Self::Proven(_) | Self::ReadyNotProduced(_) => None,
            Self::MissingInput(code) | Self::MissingAsset(code) => Some(code),
        }
    }
}

/// One row of evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct PfInputRow {
    pub pf: u32,
    pub verdict: PfInputVerdict,
    /// Why this row is judged the way it is, when that is not obvious.
    pub note: &'static str,
}

/// The whole answer for one photo.
#[derive(Clone, Debug, PartialEq)]
pub struct PfInputEvidence {
    pub frame: [u32; 2],
    pub ticket: TicketVerdict,
    pub rows: Vec<PfInputRow>,
}

impl PfInputEvidence {
    pub fn row(&self, pf: u32) -> Option<&PfInputRow> {
        self.rows.iter().find(|row| row.pf == pf)
    }

    pub fn proven_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.verdict.is_proven())
            .count()
    }

    /// Rows the cheap tier found nothing missing for, but which are still
    /// unproduced. The panel must show these as *pending measurement*.
    pub fn pending_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|row| row.verdict.is_pending_measurement())
            .count()
    }
}

/// Cheap-tier evidence for one photo. Safe to call on every capability refresh.
///
/// `source` must be the decoded frame the bundle was produced from: the ticket,
/// size and digest check rejects anything else, which is what keeps evidence
/// from leaking between photos. `extra` is the document's live
/// `AdjustmentSet::extra` when the caller has one; it drives the PF108
/// bright/de-yellow percentages and the per-row style preset choice, and
/// `None` measures the rows that need no document at all.
pub fn prove_pf_inputs(
    bundle: &FaceInputBundle,
    source: &Image,
    extra: Option<&BTreeMap<String, f32>>,
) -> PfInputEvidence {
    prove_pf_inputs_with_face_mask(bundle, source, None, extra)
}

/// As [`prove_pf_inputs`], recording the same-frame face plane for the rows whose
/// note refers to it.
///
/// The plane is *not* consumed in this tier: gating an owner mask happens inside
/// the wiring, which only runs in [`produce_row`]. It is accepted here so the
/// cheap tier and the production tier describe the same photo with the same
/// inputs.
pub fn prove_pf_inputs_with_face_mask(
    bundle: &FaceInputBundle,
    source: &Image,
    face_mask: Option<&[u8]>,
    extra: Option<&BTreeMap<String, f32>>,
) -> PfInputEvidence {
    let frame = [source.width, source.height];
    let ticket = match bundle.validate_ticket(source, Some(bundle.job_id)) {
        Ok(()) => TicketVerdict {
            ok: true,
            code: None,
        },
        Err(refusal) => TicketVerdict {
            ok: false,
            code: Some(refusal_code(&refusal)),
        },
    };
    if let Some(code) = ticket.code {
        // Without a valid ticket nothing below is this photo's evidence, so no
        // row may be reported as unblocked.
        return PfInputEvidence {
            frame,
            ticket,
            rows: probed_pfs()
                .into_iter()
                .map(|pf| PfInputRow {
                    pf,
                    verdict: PfInputVerdict::MissingInput(code),
                    note: "same-frame ticket rejected",
                })
                .collect(),
        };
    }

    let face_plane = face_mask
        .filter(|plane| frame[0] as usize * frame[1] as usize == plane.len() && !plane.is_empty());
    let face_plane_note = match (face_mask, face_plane) {
        (None, _) => "no face plane supplied; owner mask is the landmark hull alone",
        (Some(_), None) => "supplied face plane ignored: length is not frame pixels",
        (Some(_), Some(_)) => "owner mask will be gated by the supplied same-frame face plane",
    };

    let mut rows = teeth_family_rows(bundle, source, extra);
    for pf in makeup_rows_2d_declared() {
        rows.push(makeup_row_readiness(
            bundle,
            source,
            *pf,
            extra,
            face_plane_note,
        ));
    }
    rows.push(pf101_readiness(bundle, extra));
    rows.sort_by_key(|row| (TEETH_FAMILY_PFS.binary_search(&row.pf).is_err(), row.pf));
    PfInputEvidence {
        frame,
        ticket,
        rows,
    }
}

/// Production tier: builds exactly one 2D makeup row through the renderer's own
/// wiring and returns facts, not pixels.
///
/// Call this only when the panel actually asks about `pf`, or on an explicit
/// refresh: it rasterizes that row's owner mask at ROI size and drops everything
/// before returning. `face_mask` is the renderer's same-frame
/// `masks.mask_u8(MaskKind::Face, ..)` plane; a plane of the wrong length is
/// ignored rather than trusted.
pub fn produce_row(
    bundle: &FaceInputBundle,
    source: &Image,
    pf: u32,
    face_mask: Option<&[u8]>,
    extra: Option<&BTreeMap<String, f32>>,
) -> PfInputRow {
    if TEETH_FAMILY_PFS.contains(&pf) {
        return PfInputRow {
            pf,
            verdict: PfInputVerdict::MissingInput("PORTRAIT_PROBE_NOT_A_2D_ROW"),
            note: "use prove_pf_inputs for the PF108 family",
        };
    }
    if pf == 101 {
        return prove_pf101_from_extra(bundle, source, extra);
    }
    if bundle.validate_ticket(source, Some(bundle.job_id)).is_err() {
        return PfInputRow {
            pf,
            verdict: PfInputVerdict::MissingInput("face input job ticket"),
            note: "same-frame ticket rejected",
        };
    }
    let frame = [source.width, source.height];
    let face_plane = face_mask
        .filter(|plane| frame[0] as usize * frame[1] as usize == plane.len() && !plane.is_empty());
    let note = match pf {
        8 | 9 => "FaceStereo hull row",
        90032 => "EyeMaskType=10 eye image makeup",
        111 => "MaterialOnly row: material with its own UV, never an owner mask",
        _ => "",
    };
    let mut producer = ProducerInputs::default();
    match bundle.attach_makeup_bundle_with_options(source, &[pf], face_plane, extra, &mut producer)
    {
        Err(refusal) => PfInputRow {
            pf,
            verdict: classify_refusal(refusal_code(&refusal)),
            note,
        },
        Ok(()) => match producer.makeup_bundle.as_ref() {
            None => PfInputRow {
                pf,
                verdict: PfInputVerdict::MissingInput("makeup bundle"),
                note,
            },
            Some(produced) => PfInputRow {
                pf,
                verdict: match prove_makeup_row(produced, pf) {
                    Ok(proof) => PfInputVerdict::Proven(proof),
                    Err(verdict) => verdict,
                },
                note,
            },
        },
    }
}

/// Production tier for PF101 (`卧蚕`, silkworm): material, landmark face mask and
/// band UVs, through the renderer's own
/// `attach_pf101_from_adjustments`.
///
/// Costs one full-frame `u8` face mask plus the silkworm material; that is why
/// it is not part of the cheap tier.
pub fn prove_pf101(bundle: &FaceInputBundle, source: &Image, style_index: usize) -> PfInputRow {
    let mut extra = BTreeMap::new();
    if style_index > 0 {
        extra.insert("pc:101:style".to_string(), style_index as f32);
    }
    prove_pf101_from_extra(bundle, source, Some(&extra))
}

/// Detail tier: the real attachment path reads the document's exact
/// `pc:101:style` and `pc:101:rgb`, then both textures are sampled on the same
/// screen pixels with the shader's bilinear half-texel rule. A row whose mask
/// and material each hit somewhere but never at the same pixel is not a usable
/// input and is refused.
fn prove_pf101_from_extra(
    bundle: &FaceInputBundle,
    source: &Image,
    extra: Option<&BTreeMap<String, f32>>,
) -> PfInputRow {
    let note =
        "PF101 needs material + landmark face mask + band UVs; joint same-pixel coverage required";
    if bundle.validate_ticket(source, Some(bundle.job_id)).is_err() {
        return PfInputRow {
            pf: 101,
            verdict: PfInputVerdict::MissingInput("face input job ticket"),
            note,
        };
    }
    let empty = BTreeMap::new();
    let extra = extra.unwrap_or(&empty);
    let mut producer = ProducerInputs::default();
    match bundle.attach_pf101_from_adjustments(source, 1.0, extra, &mut producer) {
        Err(refusal) => PfInputRow {
            pf: 101,
            verdict: classify_refusal(refusal_code(&refusal)),
            note,
        },
        Ok(()) => {
            let Some(pf101) = producer.pf101.as_ref() else {
                return PfInputRow {
                    pf: 101,
                    verdict: PfInputVerdict::MissingInput("PF101 same-job inputs"),
                    note,
                };
            };
            if pf101_joint_hits(pf101) == 0 {
                return PfInputRow {
                    pf: 101,
                    verdict: PfInputVerdict::MissingInput("PORTRAIT_PROBE_PF101_UV_DISJOINT"),
                    note,
                };
            }
            let presets = makeup_presets_for_pf(101);
            let presets_on_disk = presets
                .iter()
                .filter(|preset| preset.path.is_file())
                .count();
            let style_index = pf101_style_from_extra(Some(extra)).unwrap_or(0);
            PfInputRow {
                pf: 101,
                verdict: PfInputVerdict::Proven(PfInputProof::Pf101 {
                    style_index,
                    presets_on_disk,
                    accepted_by: "Pf101ProducerInputs::from_same_job",
                    material_accepted: true,
                    face_mask_accepted: true,
                    band_uv_accepted: true,
                }),
                note,
            }
        }
    }
}

fn pf101_style_from_extra(extra: Option<&BTreeMap<String, f32>>) -> Result<usize, Refusal> {
    let Some(&raw) = extra.and_then(|map| map.get("pc:101:style")) else {
        return Ok(0);
    };
    if !raw.is_finite() || raw.fract() != 0.0 || !(0.0..=16_777_215.0).contains(&raw) {
        return Err(Refusal::ValueOutOfRange("PF101 style ordinal"));
    }
    Ok(raw as usize)
}

fn pf101_joint_hits(input: &Pf101ProducerInputs) -> usize {
    let (mask, mask_uv, material, material_uv) = input.coverage_inputs();
    let pixels = mask_uv.len().min(material_uv.len());
    let mut joint = 0usize;
    for index in 0..pixels {
        let mask_point = mask_uv[index];
        let material_point = material_uv[index];
        if mask_point == [0.0, 0.0] && material_point == [0.0, 0.0] {
            continue;
        }
        let mask_sample = sample_rgba(mask, mask_point[0], mask_point[1]);
        let material_sample = sample_rgba(material, material_point[0], material_point[1]);
        if mask_weight(mask_sample, MaskChannel::R) > 0.0 && material_sample[3] > 0.0 {
            joint += 1;
        }
    }
    joint
}
/// `108` / `109` / `110`: one attachment, three verdicts.
fn teeth_family_rows(
    bundle: &FaceInputBundle,
    source: &Image,
    extra: Option<&BTreeMap<String, f32>>,
) -> Vec<PfInputRow> {
    let (bright_percent, deyellow_percent) = teeth_percents(extra);
    let mut producer = ProducerInputs::default();
    let prepared =
        bundle.attach_pf108_inputs(source, bright_percent, deyellow_percent, &mut producer);
    let verdict = match prepared {
        Err(refusal) => PfInputVerdict::MissingInput(refusal_code(&refusal)),
        Ok(()) => match producer.teeth.as_ref() {
            None => PfInputVerdict::MissingInput("PF108 teeth mask"),
            Some(teeth) => match validate_teeth(
                teeth,
                [source.width, source.height],
                &PF108_WHITEN_LUT_IDENTITY,
                &PF108_BRIGHT_LUT_IDENTITY,
            ) {
                Ok(facts) => PfInputVerdict::Proven(PfInputProof::Teeth {
                    mask_source: mask_source_label(teeth.mask_source),
                    mask_format: mask_format_label(teeth.mask_format),
                    mask_pixels: facts.mask_pixels,
                    mask_peak: facts.mask_peak,
                    mask_sha256: teeth.mask_sha256,
                    bright_percent: teeth.bright_ui,
                    deyellow_percent: teeth.deyellow_ui,
                    job_id: teeth.job_id,
                    whiten_lut_ok: facts.whiten_lut_ok,
                    bright_lut_ok: facts.bright_lut_ok,
                }),
                Err(code) => PfInputVerdict::MissingInput(code),
            },
        },
    };
    // Drop the cloned 24 MP plane as soon as the facts are out.
    drop(producer);
    TEETH_FAMILY_PFS
        .iter()
        .map(|pf| PfInputRow {
            pf: *pf,
            verdict: verdict.clone(),
            note: if *pf == 108 {
                "PF108 parent: pc:109/pc:110 are children of pc:108, not separate rows"
            } else {
                "child control of PF108: same capability, no separate support entry"
            },
        })
        .collect()
}

/// Cheap tier for one 2D makeup row: geometry, profile and the material that
/// would really be selected. No owner mask is rasterized here.
fn makeup_row_readiness(
    bundle: &FaceInputBundle,
    source: &Image,
    pf: u32,
    extra: Option<&BTreeMap<String, f32>>,
    face_plane_note: &'static str,
) -> PfInputRow {
    let note = match pf {
        8 | 9 => face_plane_note,
        90032 => {
            "EyeMaskType=10 eye image makeup: profile semantic is the device-independent input"
        }
        111 => "MaterialOnly row: material with its own UV, never an owner mask",
        _ => "",
    };
    let Some(contract) = MAKEUP_2D_ROWS.iter().find(|row| row.pf == pf) else {
        return PfInputRow {
            pf,
            verdict: PfInputVerdict::MissingInput("makeup 2D row contract"),
            note,
        };
    };
    let profile = match row_profile_for_effect(pf, contract.source_effect) {
        Ok(profile) => profile,
        Err(refusal) => {
            return PfInputRow {
                pf,
                verdict: PfInputVerdict::MissingInput(refusal_code(&refusal)),
                note,
            };
        }
    };
    let geometry_ok = crate::portrait_makeup_mesh2d::build_face2d_draw_bytes(
        &bundle.points194,
        [source.width, source.height],
        &profile,
    )
    .is_ok();
    if !geometry_ok {
        return PfInputRow {
            pf,
            verdict: PfInputVerdict::MissingInput("makeup same-frame geometry"),
            note,
        };
    }
    let material = match select_material(pf, extra) {
        Err(verdict) => return PfInputRow { pf, verdict, note },
        Ok(None) if profile.requires_material => {
            return PfInputRow {
                pf,
                verdict: PfInputVerdict::MissingAsset(contract.material_missing),
                note,
            };
        }
        Ok(file) => file,
    };
    PfInputRow {
        pf,
        verdict: PfInputVerdict::ReadyNotProduced(RowReadiness {
            mask_semantic: profile.mask_semantic,
            mask_channel: profile.declared_mask_channel.map(|c| format!("{c:?}")),
            requires_owner_mask: profile.requires_owner_mask(),
            requires_material: profile.requires_material,
            geometry_ok,
            material,
        }),
        note,
    }
}

/// The material this row would really load: the `pc:<pf>` style preset when the
/// document picks one, otherwise the contract's own PNG under the resolver root.
///
/// Only the file name, its size and its origin are reported: no absolute path
/// leaves this module.
fn select_material(
    pf: u32,
    extra: Option<&BTreeMap<String, f32>>,
) -> Result<Option<MaterialFile>, PfInputVerdict> {
    use crate::portrait_makeup_wiring::{makeup_asset_root, row_material_selection};
    let selected = row_material_selection(makeup_asset_root().as_deref(), pf, extra)
        .map_err(|refusal| classify_refusal(refusal_code(&refusal)))?;
    let Some(selected) = selected else {
        return Ok(None);
    };
    let meta = std::fs::metadata(&selected.path)
        .map_err(|_| PfInputVerdict::MissingAsset(MATERIAL_PNG_MISSING))?;
    if meta.len() == 0 || !meta.is_file() {
        return Err(PfInputVerdict::MissingAsset(MATERIAL_PNG_MISSING));
    }
    Ok(Some(MaterialFile {
        file_name: file_name_of(&selected.path),
        bytes: meta.len(),
        from_style_preset: selected.explicit,
    }))
}

fn file_name_of(path: &std::path::Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Cheap tier for PF101: does the preset list resolve, and is there a material to
/// read at all? The full proof needs one full-frame mask, so it lives in
/// [`prove_pf101`].
fn pf101_readiness(bundle: &FaceInputBundle, extra: Option<&BTreeMap<String, f32>>) -> PfInputRow {
    let note = "PF101: cheap tier checks the selected material and landmarks; call prove_pf101 for joint coverage";
    if bundle.points194.len() != 194 {
        return PfInputRow {
            pf: 101,
            verdict: PfInputVerdict::MissingInput("face input landmarks"),
            note,
        };
    }
    let style = match pf101_style_from_extra(extra) {
        Ok(style) => style,
        Err(refusal) => {
            return PfInputRow {
                pf: 101,
                verdict: PfInputVerdict::MissingInput(refusal_code(&refusal)),
                note,
            };
        }
    };
    let selected = if style > 0 {
        match makeup_presets_for_pf(101).get(style - 1) {
            Some(preset) => Some(preset.path.clone()),
            None => {
                return PfInputRow {
                    pf: 101,
                    verdict: PfInputVerdict::MissingInput("PF101 selected material"),
                    note,
                };
            }
        }
    } else {
        pf101_default_material_candidates()
            .into_iter()
            .find(|path| path.is_file())
    };
    let Some(path) = selected else {
        return PfInputRow {
            pf: 101,
            verdict: PfInputVerdict::MissingAsset(PF101_MATERIAL),
            note,
        };
    };
    let meta = match std::fs::metadata(&path) {
        Ok(meta) if meta.is_file() && meta.len() > 0 => meta,
        _ => {
            return PfInputRow {
                pf: 101,
                verdict: PfInputVerdict::MissingAsset(PF101_MATERIAL),
                note,
            };
        }
    };
    PfInputRow {
        pf: 101,
        verdict: PfInputVerdict::ReadyNotProduced(RowReadiness {
            mask_semantic: "not a 2D makeup row: landmark face mask + silkworm band UVs",
            mask_channel: None,
            requires_owner_mask: true,
            requires_material: true,
            geometry_ok: true,
            material: Some(MaterialFile {
                file_name: file_name_of(&path),
                bytes: meta.len(),
                from_style_preset: style > 0,
            }),
        }),
        note,
    }
}
/// The consumer's gate for a produced row, plus the UV-hit check.
///
/// `OwnedMakeupBundle::makeup_input` is what the provider actually consumes: it
/// refuses a row whose owner mask, mask UV, material or material UV is missing
/// for its [`MakeupRowProfile`]. Passing it is the strongest statement available
/// without running the kernel — and it is still not enough on its own, so every
/// UV vertex is sampled here.
pub fn prove_makeup_row(
    bundle: &OwnedMakeupBundle,
    pf: u32,
) -> Result<PfInputProof, PfInputVerdict> {
    let profile = bundle
        .rows
        .iter()
        .find(|row| row.profile.pf == pf)
        .map(|row| row.profile)
        .ok_or(PfInputVerdict::MissingInput(
            "PORTRAIT_PROBE_MAKEUP_ROW_ABSENT",
        ))?;
    let input: MakeupInput<'_> = bundle.makeup_input(pf).map_err(|refusal| {
        let code = refusal_code(&refusal);
        if code == MATERIAL_MISSING {
            PfInputVerdict::MissingAsset(MATERIAL_PNG_MISSING)
        } else {
            PfInputVerdict::MissingInput(code)
        }
    })?;
    if pf == 90032 && !is_eye_mask_ten(&profile) {
        return Err(PfInputVerdict::MissingInput(
            "PORTRAIT_PROBE_EYE_MASK_SEMANTIC",
        ));
    }

    // The owner mask is read through the channel the row's profile declares: a
    // brow mask carries its coverage in R and B, an eye mask in R and G, the
    // EyeMaskType=10 row in RGB. Scanning "any of RGB" would count a plane the
    // row never consumes.
    let owner_channel = profile.declared_mask_channel;
    let stats = uv_hit_stats(&input, owner_channel);
    let owner_mask_size = input.mask.map(|mask| [mask.width, mask.height]);
    let owner_mask_peak = match input.mask {
        None => 0,
        Some(mask) => {
            let channel = owner_channel.unwrap_or(MaskChannel::Rgba);
            let (peak, _covered) = plane_signal_channel(&mask.pixels, channel);
            if peak < MASK_FLOOR {
                return Err(PfInputVerdict::MissingInput(
                    "PORTRAIT_PROBE_MAKEUP_MASK_EMPTY",
                ));
            }
            peak
        }
    };
    let material_size = input
        .material
        .map(|material| [material.width, material.height]);

    // Sparse is legal — a lash or a sticker may paint a handful of texels and a
    // hull UV may legitimately read mostly transparent material. What is not
    // legal is a row whose UVs never sample anything at all, which would mean
    // the draw exists on paper and produces nothing.
    if input.mask.is_some() {
        if stats.mask_vertices == 0 {
            return Err(PfInputVerdict::MissingInput(
                "PORTRAIT_PROBE_MASK_UV_NO_WRITES",
            ));
        }
        if stats.mask_hits == 0 {
            return Err(PfInputVerdict::MissingInput(
                "PORTRAIT_PROBE_MASK_UV_OFF_TARGET",
            ));
        }
    }
    if input.material.is_some() {
        if stats.material_vertices == 0 {
            return Err(PfInputVerdict::MissingInput(
                "PORTRAIT_PROBE_MATERIAL_UV_NO_WRITES",
            ));
        }
        if stats.material_hits == 0 {
            return Err(PfInputVerdict::MissingInput(
                "PORTRAIT_PROBE_MATERIAL_UV_OFF_TARGET",
            ));
        }
    }
    require_joint_uv_hits(&stats, input.mask.is_some(), input.material.is_some())?;
    Ok(PfInputProof::Makeup {
        mask_semantic: profile.mask_semantic,
        mask_channel: profile.declared_mask_channel.map(|c| format!("{c:?}")),
        owner_mask_size,
        owner_mask_peak,
        material_size,
        filter_type: input.filter_type,
        opacity_percent: input.opacity_percent,
        mask_uv_vertices: stats.mask_vertices,
        material_uv_vertices: stats.material_vertices,
        mask_uv_hits: stats.mask_hits,
        material_uv_hits: stats.material_hits,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct UvHitStats {
    mask_vertices: usize,
    material_vertices: usize,
    mask_hits: usize,
    material_hits: usize,
    joint_hits: usize,
}

/// Samples the mask and material at the *same* pixel index, honouring the
/// per-stream write masks. A mask hit on pixel A and a material hit on pixel B
/// are not a usable pair; only a joint hit on the same index is.
fn uv_hit_stats(input: &MakeupInput<'_>, channel: Option<MaskChannel>) -> UvHitStats {
    let owner_channel = channel.unwrap_or(MaskChannel::Rgba);
    let mask = input.mask;
    let mask_uv = input.mask_uv;
    let mask_writes = input.mask_uv_writes;
    let material = input.material;
    let material_uv = input.material_uv;
    let material_writes = input.material_uv_writes;
    let pixels = mask_uv
        .map_or(0, |uv| uv.len())
        .max(material_uv.map_or(0, |uv| uv.len()));
    let mut stats = UvHitStats::default();
    for index in 0..pixels {
        let mask_active = mask.is_some()
            && mask_uv.is_some_and(|uv| index < uv.len())
            && mask_writes.map_or(true, |writes| writes.get(index).copied().unwrap_or(0) > 0);
        let material_active = material.is_some()
            && material_uv.is_some_and(|uv| index < uv.len())
            && material_writes.map_or(true, |writes| writes.get(index).copied().unwrap_or(0) > 0);
        let mask_sample = match (mask, mask_uv) {
            (Some(image), Some(uv)) if mask_active => {
                let point = uv[index];
                Some(sample_rgba(image, point[0], point[1]))
            }
            _ => None,
        };
        let material_sample = match (material, material_uv) {
            (Some(image), Some(uv)) if material_active => {
                let point = uv[index];
                Some(sample_rgba(image, point[0], point[1]))
            }
            _ => None,
        };
        if let Some(sample) = mask_sample {
            stats.mask_vertices += 1;
            if mask_weight(sample, owner_channel) > 0.0 {
                stats.mask_hits += 1;
            }
        }
        if let Some(sample) = material_sample {
            stats.material_vertices += 1;
            if sample[3] > 0.0 {
                stats.material_hits += 1;
            }
        }
        if let (Some(mask_sample), Some(material_sample)) = (mask_sample, material_sample) {
            if mask_weight(mask_sample, owner_channel) > 0.0 && material_sample[3] > 0.0 {
                stats.joint_hits += 1;
            }
        }
    }
    stats
}

fn require_joint_uv_hits(
    stats: &UvHitStats,
    has_mask: bool,
    has_material: bool,
) -> Result<(), PfInputVerdict> {
    if has_mask && has_material && stats.joint_hits == 0 {
        Err(PfInputVerdict::MissingInput("PORTRAIT_PROBE_UV_DISJOINT"))
    } else {
        Ok(())
    }
}
/// Which RGBA bytes a recovered [`MaskChannel`] occupies.
///
/// The owner-mask hit test reads only these bytes, never "any of RGB".
pub fn mask_channel_bytes(channel: MaskChannel) -> &'static [u8] {
    match channel {
        MaskChannel::R => &[0],
        MaskChannel::G => &[1],
        MaskChannel::B => &[2],
        MaskChannel::A => &[3],
        MaskChannel::RG => &[0, 1],
        MaskChannel::RB => &[0, 2],
        MaskChannel::RA => &[0, 3],
        MaskChannel::GB => &[1, 2],
        MaskChannel::GA => &[1, 3],
        MaskChannel::BA => &[2, 3],
        MaskChannel::Rgb => &[0, 1, 2],
        MaskChannel::Rga => &[0, 1, 3],
        MaskChannel::Rba => &[0, 2, 3],
        MaskChannel::Gba => &[1, 2, 3],
        MaskChannel::Rgba => &[0, 1, 2, 3],
    }
}

/// `(peak, covered pixels)` over the bytes a row's declared mask channel uses.
fn plane_signal_channel(pixels: &[u8], channel: MaskChannel) -> (u8, u32) {
    let bytes = mask_channel_bytes(channel);
    let mut peak = 0u8;
    let mut covered = 0u32;
    for chunk in pixels.chunks_exact(4) {
        let value = bytes
            .iter()
            .fold(0u8, |acc, &byte| acc.max(chunk[byte as usize]));
        if value > peak {
            peak = value;
        }
        if value > 0 {
            covered += 1;
        }
    }
    (peak, covered)
}

/// How many `uv` vertices sample a texel whose declared mask channel is non-zero.
///
/// Nearest-neighbour at `(u * width, v * height)`, clamped to the image.
pub fn count_uv_hits_channel(image: &Image, uv: &[[f32; 2]], channel: MaskChannel) -> usize {
    if image.width == 0 || image.height == 0 || image.pixels.len() != image.pixel_count() * 4 {
        return 0;
    }
    let bytes = mask_channel_bytes(channel);
    let mut hits = 0;
    for point in uv {
        let Some(at) = texel_offset(image, *point) else {
            continue;
        };
        if bytes
            .iter()
            .any(|&byte| image.pixels[at + byte as usize] > 0)
        {
            hits += 1;
        }
    }
    hits
}

/// How many `uv` vertices sample a material texel with non-zero alpha.
///
/// Coverage of a material is its alpha: a texel that paints opaque black is
/// content, so keying on colour would report every dark material as missing.
pub fn count_uv_hits_alpha(image: &Image, uv: &[[f32; 2]]) -> usize {
    if image.width == 0 || image.height == 0 || image.pixels.len() != image.pixel_count() * 4 {
        return 0;
    }
    let mut hits = 0;
    for point in uv {
        let Some(at) = texel_offset(image, *point) else {
            continue;
        };
        if image.pixels[at + 3] > 0 {
            hits += 1;
        }
    }
    hits
}

fn texel_offset(image: &Image, point: [f32; 2]) -> Option<usize> {
    if !point[0].is_finite() || !point[1].is_finite() {
        return None;
    }
    let x = ((point[0].clamp(0.0, 1.0) * image.width as f32) as u32).min(image.width - 1);
    let y = ((point[1].clamp(0.0, 1.0) * image.height as f32) as u32).min(image.height - 1);
    Some(((y * image.width + x) * 4) as usize)
}

/// `EyeMaskType=10` is the recovered semantic of PF90032's `SymEyeImageMakeup`.
fn is_eye_mask_ten(profile: &MakeupRowProfile) -> bool {
    profile.mask_semantic == "EyeMaskType=10"
        && matches!(profile.declared_mask_channel, Some(MaskChannel::Rgb))
}

/// PF108 percentages for one document, or zero when there is no document. Never
/// a fabricated per-child value: `pf108_ui_percents` is the one derivation the
/// dispatcher and the renderer share.
pub fn teeth_percents(extra: Option<&BTreeMap<String, f32>>) -> (f32, f32) {
    let empty = BTreeMap::new();
    let map = extra.unwrap_or(&empty);
    match pf108_ui_percents(map) {
        Ok((bright, deyellow)) => (bright, deyellow),
        // An out-of-range document is a document problem, not a lost input; the
        // attachment below refuses it with the real code and the rows report it.
        Err(_) => (f32::NAN, f32::NAN),
    }
}

/// Facts about a teeth attachment that passed every check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TeethFacts {
    pub mask_pixels: u32,
    pub mask_peak: u8,
    pub whiten_lut_ok: bool,
    pub bright_lut_ok: bool,
}

/// The PF108 gate, in the order `portrait_stage` applies it, over an already
/// attached [`TeethProducerInputs`].
///
/// Kept pure and identity-parameterised so a test can prove the *content* check
/// without needing the shipped LUT bytes.
pub fn validate_teeth(
    teeth: &TeethProducerInputs,
    frame: [u32; 2],
    whiten_identity: &TeethLutIdentity,
    bright_identity: &TeethLutIdentity,
) -> Result<TeethFacts, &'static str> {
    if teeth.job_id == 0 {
        return Err("PORTRAIT_PROBE_TEETH_JOB");
    }
    if teeth.mask_source != TeethMaskSource::FaceParsingV2Id2G
        || teeth.mask_format != TeethMaskFormat::U8
    {
        return Err("PORTRAIT_PROBE_TEETH_MASK_SOURCE");
    }
    if teeth.mask_width != frame[0] || teeth.mask_height != frame[1] {
        return Err("PORTRAIT_PROBE_TEETH_MASK_SHAPE");
    }
    let expected = frame[0] as usize * frame[1] as usize;
    if teeth.mask_g.len() != expected {
        return Err("PORTRAIT_PROBE_TEETH_MASK_SHAPE");
    }
    if teeth.mask_sha256 != model_sha256::sha256(&teeth.mask_g) {
        return Err("PORTRAIT_PROBE_TEETH_MASK_DIGEST");
    }
    let (mask_peak, mask_pixels) = plane_signal_u8(&teeth.mask_g);
    if mask_peak < MASK_FLOOR {
        return Err("PORTRAIT_PROBE_TEETH_MASK_EMPTY");
    }
    if !teeth.bright_ui.is_finite()
        || !teeth.deyellow_ui.is_finite()
        || !(0.0..=100.0).contains(&teeth.bright_ui)
        || !(0.0..=100.0).contains(&teeth.deyellow_ui)
    {
        return Err("PORTRAIT_PROBE_TEETH_UI_RANGE");
    }
    let whiten_lut_ok = lut_matches(whiten_identity, &teeth.whiten_lut)?;
    let bright_lut_ok = lut_matches(bright_identity, &teeth.bright_lut)?;
    Ok(TeethFacts {
        mask_pixels,
        mask_peak,
        whiten_lut_ok,
        bright_lut_ok,
    })
}

/// Length *and* full SHA-256 of the LUT payload against the recorded identity.
///
/// The length check alone would accept any 512x512 RGB8 image, so the digest is
/// what makes "the LUT this row expects" a real statement. It is always computed
/// over the whole payload: a payload that only matches on length and on its two
/// ends is not the same content, and a sampling cache would report a modified
/// middle as verified.
fn lut_matches(identity: &TeethLutIdentity, bytes: &[u8]) -> Result<bool, &'static str> {
    if bytes.len() != identity.rgb8_bytes {
        return Err("PORTRAIT_PROBE_TEETH_LUT_BYTES");
    }
    if model_sha256::sha256(bytes) == identity.rgb8_sha256 {
        Ok(true)
    } else {
        Err("PORTRAIT_PROBE_TEETH_LUT_DIGEST")
    }
}

/// `(peak, covered pixels)` over a one-byte-per-pixel plane: the ID2G mask.
fn plane_signal_u8(plane: &[u8]) -> (u8, u32) {
    let mut peak = 0u8;
    let mut covered = 0u32;
    for &value in plane {
        if value > peak {
            peak = value;
        }
        if value > 0 {
            covered += 1;
        }
    }
    (peak, covered)
}

fn mask_source_label(source: TeethMaskSource) -> &'static str {
    match source {
        TeethMaskSource::FaceParsingV2Id2G => "FaceParsingV2Id2G",
    }
}

fn mask_format_label(format: TeethMaskFormat) -> &'static str {
    match format {
        TeethMaskFormat::U8 => "U8",
    }
}

/// A refusal from any of the attachment functions, as the code the panel shows.
pub fn refusal_code(refusal: &Refusal) -> &'static str {
    match refusal {
        Refusal::MissingInput(code)
        | Refusal::ShapeMismatch(code)
        | Refusal::ValueOutOfRange(code) => code,
        Refusal::UnknownEffect(_) => "PORTRAIT_PROBE_UNKNOWN_EFFECT",
        Refusal::NotImplemented(_) => "PORTRAIT_PROBE_NOT_IMPLEMENTED",
    }
}

/// Splits "no asset to read" from "input never produced".
///
/// The distinction is the whole point of the evidence: an installable file is a
/// different answer for the panel than a same-frame input this photo cannot give.
pub fn classify_refusal(code: &'static str) -> PfInputVerdict {
    if code == MAKEUP_ASSET_ROOT_MISSING
        || code == MATERIAL_PNG_MISSING
        || code == MATERIAL_MISSING
        || code == PF101_MATERIAL
        || code.starts_with("MAKEUP_MATERIAL_PNG_MISSING")
    {
        PfInputVerdict::MissingAsset(code)
    } else {
        PfInputVerdict::MissingInput(code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::pf101_silkworm::{self, SilkwormParams};

    fn synthetic_identity(bytes: &[u8]) -> TeethLutIdentity {
        TeethLutIdentity {
            kind: crate::portrait_stage::TeethLutKind::Whiten,
            config_field: "test",
            source_file: "test.png",
            png_bytes: bytes.len(),
            png_sha256: model_sha256::sha256(bytes),
            rgb8_bytes: bytes.len(),
            rgb8_sha256: model_sha256::sha256(bytes),
            width: 512,
            height: 512,
            format: crate::portrait_stage::TeethLutFormat::Rgb8,
            encoding: crate::portrait_stage::TeethLutEncoding::EncodedSrgb,
        }
    }

    fn teeth_fixture(mask: Vec<u8>, frame: [u32; 2], lut: Vec<u8>) -> TeethProducerInputs {
        TeethProducerInputs {
            job_id: 7,
            mask_width: frame[0],
            mask_height: frame[1],
            mask_source: TeethMaskSource::FaceParsingV2Id2G,
            mask_format: TeethMaskFormat::U8,
            mask_sha256: model_sha256::sha256(&mask),
            whiten_lut_identity: synthetic_identity(&lut),
            bright_lut_identity: synthetic_identity(&lut),
            mask_g: mask,
            bright_ui: 70.0,
            deyellow_ui: 30.0,
            whiten_lut: lut.clone(),
            bright_lut: lut,
        }
    }

    /// A 4x2 frame with two lit teeth pixels, one LUT of 12 bytes.
    fn good_frame() -> ([u32; 2], Vec<u8>, Vec<u8>) {
        ([4, 2], vec![0, 90, 0, 0, 0, 0, 200, 0], vec![1u8; 12])
    }

    #[test]
    fn a_real_id2g_plane_with_matching_luts_is_proven() {
        let (frame, mask, lut) = good_frame();
        let teeth = teeth_fixture(mask, frame, lut.clone());
        let identity = synthetic_identity(&lut);
        let facts = validate_teeth(&teeth, frame, &identity, &identity).expect("proven");
        assert_eq!(facts.mask_pixels, 2);
        assert_eq!(facts.mask_peak, 200);
        assert!(facts.whiten_lut_ok && facts.bright_lut_ok);
    }

    #[test]
    fn an_all_zero_id2g_plane_is_rejected() {
        let (frame, _mask, lut) = good_frame();
        let teeth = teeth_fixture(vec![0u8; 8], frame, lut.clone());
        let identity = synthetic_identity(&lut);
        assert_eq!(
            validate_teeth(&teeth, frame, &identity, &identity),
            Err("PORTRAIT_PROBE_TEETH_MASK_EMPTY")
        );
    }

    #[test]
    fn a_lut_of_the_right_length_but_wrong_content_is_rejected() {
        let (frame, mask, lut) = good_frame();
        let identity = synthetic_identity(&lut);
        let mut teeth = teeth_fixture(mask, frame, lut);
        teeth.whiten_lut[5] = 0xAA;
        assert_eq!(
            validate_teeth(&teeth, frame, &identity, &identity),
            Err("PORTRAIT_PROBE_TEETH_LUT_DIGEST")
        );
    }

    #[test]
    fn a_lut_of_the_wrong_length_is_rejected_before_the_digest() {
        let (frame, mask, lut) = good_frame();
        let identity = synthetic_identity(&lut);
        let mut teeth = teeth_fixture(mask, frame, lut);
        teeth.bright_lut.push(0);
        assert_eq!(
            validate_teeth(&teeth, frame, &identity, &identity),
            Err("PORTRAIT_PROBE_TEETH_LUT_BYTES")
        );
    }

    #[test]
    fn a_stale_mask_digest_or_wrong_shape_is_rejected() {
        let (frame, mask, lut) = good_frame();
        let identity = synthetic_identity(&lut);
        let mut stale = teeth_fixture(mask.clone(), frame, lut.clone());
        stale.mask_sha256 = [0u8; 32];
        assert_eq!(
            validate_teeth(&stale, frame, &identity, &identity),
            Err("PORTRAIT_PROBE_TEETH_MASK_DIGEST")
        );
        let mut short = teeth_fixture(mask, frame, lut);
        short.mask_g.pop();
        assert_eq!(
            validate_teeth(&short, frame, &identity, &identity),
            Err("PORTRAIT_PROBE_TEETH_MASK_SHAPE")
        );
    }

    #[test]
    fn the_shipped_lut_identities_reject_a_synthetic_payload() {
        let payload = vec![0u8; PF108_WHITEN_LUT_IDENTITY.rgb8_bytes];
        assert_eq!(
            lut_matches(&PF108_WHITEN_LUT_IDENTITY, &payload),
            Err("PORTRAIT_PROBE_TEETH_LUT_DIGEST")
        );
        let wrong_len = vec![0u8; PF108_BRIGHT_LUT_IDENTITY.rgb8_bytes - 1];
        assert_eq!(
            lut_matches(&PF108_BRIGHT_LUT_IDENTITY, &wrong_len),
            Err("PORTRAIT_PROBE_TEETH_LUT_BYTES")
        );
    }

    #[test]
    fn a_payload_changed_only_in_the_middle_is_rejected() {
        // Same length, byte-identical first and last 4 KiB, exactly one byte
        // different in the middle. A "length + both ends" fingerprint would call
        // this the same payload; the full digest must not.
        let original: Vec<u8> = (0..PF108_WHITEN_LUT_IDENTITY.rgb8_bytes)
            .map(|index| (index % 251) as u8)
            .collect();
        let identity = synthetic_identity(&original);
        assert_eq!(lut_matches(&identity, &original), Ok(true));

        let mut tampered = original.clone();
        let middle = tampered.len() / 2;
        tampered[middle] ^= 0xFF;
        assert_eq!(tampered.len(), identity.rgb8_bytes);
        assert_eq!(
            lut_matches(&identity, &tampered),
            Err("PORTRAIT_PROBE_TEETH_LUT_DIGEST")
        );
        // Confirm the case really is the one described, not a lucky side effect.
        assert_eq!(tampered[..4096], original[..4096]);
        assert_eq!(
            tampered[tampered.len() - 4096..],
            original[original.len() - 4096..]
        );
    }

    #[test]
    fn asset_absence_is_not_the_same_answer_as_a_missing_input() {
        assert_eq!(
            classify_refusal(MAKEUP_ASSET_ROOT_MISSING),
            PfInputVerdict::MissingAsset(MAKEUP_ASSET_ROOT_MISSING)
        );
        assert_eq!(
            classify_refusal("MAKEUP_MATERIAL_PNG_MISSING:90032"),
            PfInputVerdict::MissingAsset("MAKEUP_MATERIAL_PNG_MISSING:90032")
        );
        assert_eq!(
            classify_refusal("PF108 teeth mask"),
            PfInputVerdict::MissingInput("PF108 teeth mask")
        );
    }

    #[test]
    fn the_teeth_family_is_probed_once_and_the_children_share_it() {
        let pfs = probed_pfs();
        assert_eq!(&pfs[..3], &TEETH_FAMILY_PFS);
        for pf in [108u32, 109, 110, 101] {
            assert!(pfs.contains(&pf), "{pf} must be probed");
        }
        for pf in makeup_rows_2d_declared() {
            assert!(pfs.contains(pf), "{pf} must be probed");
        }
        assert!(!makeup_rows_2d_declared().contains(&109));
        assert!(!makeup_rows_2d_declared().contains(&110));
        assert!(!makeup_rows_2d_declared().contains(&101));
    }

    #[test]
    fn percents_follow_the_document_and_never_invent_a_sibling() {
        let empty = BTreeMap::new();
        assert_eq!(teeth_percents(Some(&empty)), (0.0, 0.0));
        let mut child_only = BTreeMap::new();
        child_only.insert("pc:109".to_string(), 0.5);
        let (bright, deyellow) = teeth_percents(Some(&child_only));
        assert!(bright > 0.0, "pc:109 alone must reach the bright LUT");
        assert_eq!(deyellow, 0.0, "the sibling must not be dragged up");
        assert_eq!(teeth_percents(None), (0.0, 0.0));
    }

    #[test]
    fn owner_mask_hits_use_the_declared_channel_not_any_rgb() {
        // 2x1 mask: the left texel writes only G, the right texel only R.
        let mask = Image::new(2, 1, vec![0, 40, 0, 255, 60, 0, 0, 255]).expect("mask");
        let left = [[0.1f32, 0.5]];
        let right = [[0.9f32, 0.5]];
        assert_eq!(count_uv_hits_channel(&mask, &left, MaskChannel::G), 1);
        assert_eq!(count_uv_hits_channel(&mask, &left, MaskChannel::R), 0);
        assert_eq!(count_uv_hits_channel(&mask, &right, MaskChannel::R), 1);
        assert_eq!(count_uv_hits_channel(&mask, &right, MaskChannel::G), 0);
        assert_eq!(count_uv_hits_channel(&mask, &left, MaskChannel::RG), 1);
        assert_eq!(
            count_uv_hits_channel(&mask, &[[0.1, 0.5], [0.9, 0.5]], MaskChannel::Rgba),
            2
        );
        // Alpha-only coverage is invisible to the declared colour channel and
        // visible to A: the row's own channel decides, never "any of RGB".
        let alpha_only = Image::new(1, 1, vec![0, 0, 0, 255]).expect("mask");
        assert_eq!(
            count_uv_hits_channel(&alpha_only, &[[0.5, 0.5]], MaskChannel::R),
            0
        );
        assert_eq!(
            count_uv_hits_channel(&alpha_only, &[[0.5, 0.5]], MaskChannel::A),
            1
        );
        assert_eq!(mask_channel_bytes(MaskChannel::RB), &[0u8, 2][..]);
    }

    #[test]
    fn material_hits_use_alpha_so_black_opaque_is_content() {
        // Opaque black is a real material texel; transparent black is not.
        let material = Image::new(2, 1, vec![0, 0, 0, 255, 0, 0, 0, 0]).expect("material");
        assert_eq!(count_uv_hits_alpha(&material, &[[0.1f32, 0.5]]), 1);
        assert_eq!(count_uv_hits_alpha(&material, &[[0.9f32, 0.5]]), 0);
        assert_eq!(count_uv_hits_alpha(&material, &[[f32::NAN, 0.5]]), 0);
    }

    fn hit_input<'a>(
        mask: &'a Image,
        material: &'a Image,
        mask_uv: &'a [[f32; 2]],
        material_uv: &'a [[f32; 2]],
        mask_writes: Option<&'a [u16]>,
        material_writes: Option<&'a [u16]>,
    ) -> MakeupInput<'a> {
        MakeupInput {
            pf: 8,
            mix_color: [0, 0, 0, 255],
            opacity_percent: 100.0,
            filter_type: 42,
            material: Some(material),
            material_uv: Some(material_uv),
            material_uv_writes: material_writes,
            mask: Some(mask),
            mask_uv: Some(mask_uv),
            mask_uv_writes: mask_writes,
        }
    }

    #[test]
    fn mask_and_material_hits_on_disjoint_pixels_are_not_a_usable_pair() {
        let mask = Image::new(2, 1, vec![255, 0, 0, 255, 0, 0, 0, 255]).expect("mask");
        let material = Image::new(2, 1, vec![0, 0, 0, 0, 9, 9, 9, 255]).expect("material");
        let uv = [[0.1f32, 0.5], [0.9, 0.5]];
        let input = hit_input(&mask, &material, &uv, &uv, None, None);
        let stats = uv_hit_stats(&input, Some(MaskChannel::R));
        assert_eq!(stats.mask_hits, 1);
        assert_eq!(stats.material_hits, 1);
        assert_eq!(stats.joint_hits, 0);
        assert_eq!(
            require_joint_uv_hits(&stats, true, true),
            Err(PfInputVerdict::MissingInput("PORTRAIT_PROBE_UV_DISJOINT"))
        );
    }

    #[test]
    fn half_texel_interpolation_makes_an_edge_sample_joint() {
        let mask = Image::new(2, 1, vec![255, 0, 0, 255, 0, 0, 0, 255]).expect("mask");
        let material = Image::new(2, 1, vec![9, 9, 9, 255, 9, 9, 9, 255]).expect("material");
        let uv = [[0.5f32, 0.5]];
        let input = hit_input(&mask, &material, &uv, &uv, None, None);
        let stats = uv_hit_stats(&input, Some(MaskChannel::R));
        assert_eq!(stats.mask_hits, 1);
        assert_eq!(stats.material_hits, 1);
        assert_eq!(stats.joint_hits, 1);
        assert_eq!(require_joint_uv_hits(&stats, true, true), Ok(()));
    }

    #[test]
    fn write_masks_are_index_aligned() {
        let mask = Image::new(2, 1, vec![255, 0, 0, 255, 0, 0, 0, 255]).expect("mask");
        let material = Image::new(2, 1, vec![9, 9, 9, 255, 9, 9, 9, 255]).expect("material");
        let uv = [[0.1f32, 0.5], [0.9, 0.5]];
        let input = hit_input(&mask, &material, &uv, &uv, Some(&[1, 0]), Some(&[1, 0]));
        let stats = uv_hit_stats(&input, Some(MaskChannel::R));
        assert_eq!(stats.mask_vertices, 1);
        assert_eq!(stats.material_vertices, 1);
        assert_eq!(stats.joint_hits, 1);
        let second = hit_input(&mask, &material, &uv, &uv, Some(&[0, 1]), Some(&[0, 1]));
        let stats = uv_hit_stats(&second, Some(MaskChannel::R));
        assert_eq!(stats.mask_vertices, 1);
        assert_eq!(stats.material_vertices, 1);
        assert_eq!(stats.joint_hits, 0);
    }

    fn pf101_params() -> SilkwormParams {
        SilkwormParams {
            filter_type: pf101_silkworm::FILTER_TYPE,
            main_mask_scalar: 1.0,
            axis_weight_w1: 1.0,
            adjust_gate: false,
        }
    }

    #[test]
    fn pf101_joint_coverage_requires_same_pixel() {
        let mask = Image::new(2, 1, vec![255, 0, 0, 255, 0, 0, 0, 255]).expect("mask");
        let material = Image::new(2, 1, vec![0, 0, 0, 0, 9, 9, 9, 255]).expect("material");
        let disjoint = Pf101ProducerInputs::from_same_job(
            7,
            material,
            mask,
            vec![[0.1, 0.5], [0.9, 0.5]],
            vec![[0.1, 0.5], [0.9, 0.5]],
            pf101_params(),
        )
        .expect("same-job inputs");
        assert_eq!(pf101_joint_hits(&disjoint), 0);

        let mask = Image::new(2, 1, vec![255, 0, 0, 255, 255, 0, 0, 255]).expect("mask");
        let material = Image::new(2, 1, vec![9, 9, 9, 255, 9, 9, 9, 255]).expect("material");
        let shared = Pf101ProducerInputs::from_same_job(
            7,
            material,
            mask,
            vec![[0.1, 0.5], [0.9, 0.5]],
            vec![[0.1, 0.5], [0.9, 0.5]],
            pf101_params(),
        )
        .expect("same-job inputs");
        assert!(pf101_joint_hits(&shared) > 0);
    }

    #[test]
    fn pf101_style_is_exact_one_based_without_clamping() {
        assert_eq!(pf101_style_from_extra(None), Ok(0));
        for (raw, expected) in [(0.0, 0), (1.0, 1), (2.0, 2), (99.0, 99)] {
            let mut extra = BTreeMap::new();
            extra.insert("pc:101:style".to_string(), raw);
            assert_eq!(pf101_style_from_extra(Some(&extra)), Ok(expected));
        }
        for raw in [1.5, -1.0, 16_777_216.0, f32::NAN] {
            let mut extra = BTreeMap::new();
            extra.insert("pc:101:style".to_string(), raw);
            assert!(pf101_style_from_extra(Some(&extra)).is_err(), "{raw}");
        }
    }

    #[test]
    fn pf101_material_missing_or_corrupt_is_an_asset_verdict() {
        assert_eq!(
            classify_refusal(PF101_MATERIAL),
            PfInputVerdict::MissingAsset(PF101_MATERIAL)
        );
    }
}
