//! Bridges portrait parameters (`pc:<id>`) to `aurora-portrait` CPU kernels.
//!
//! Only effects whose inputs are available are applied. Effects that need face
//! geometry, per-effect model outputs or materials that are not present are
//! skipped with a debug line instead of being approximated.
//!
//! Mask planes come from the local people parser when its ONNX model is
//! available. No external application install is read by this module.

#[cfg(windows)]
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;

use aurora_portrait::cuda::{CudaError, Dispatch};
use aurora_portrait::cuda_pf90021::{
    Direction as Pf90021Direction, RuntimeScalars as Pf90021RuntimeScalars, VolumeLut64,
};
#[cfg(windows)]
use aurora_portrait::cuda_hair_detail::HairDetail;
use aurora_portrait::cuda_hair_detail::{
    GlossProducer, Roi as HairRoi, SmoothParams, SmoothProducer,
};
use aurora_portrait::cuda_makeup::{MakeupRoute, RouteInputs};
use aurora_portrait::cuda_stray_hair::Params as StrayHairParams;
use aurora_portrait::kernels::makeup_material::{
    MakeupLayer, color_makeup, image_makeup, image_makeup_unmasked,
};
use aurora_portrait::kernels::moisture::{FaceParams, Lut1D, MaskChannel, SkinMap};
use aurora_portrait::kernels::smile_warp::{self, Rect};
use aurora_portrait::kernels::suit_makeup::{SuitPass, SuitPassKind};
#[cfg(test)]
use aurora_portrait::kernels::teeth_blend::AlignmentFinalBlendOutcome;
use aurora_portrait::kernels::teeth_blend::AlignmentFinalBlendJob;
use aurora_portrait::{Access, Image, Refusal};

use crate::NormRect;

#[cfg(test)]
const MASK_EDGE: usize = 512;
const LUT_EDGE: usize = 512;
const LUT_BYTES: usize = LUT_EDGE * LUT_EDGE * 3;

/// Masks sampled once per source image, in normalized image coordinates.
pub struct PortraitMasks {
    pub skin: Option<Vec<f32>>,
    pub hair: Option<Vec<f32>>,
    pub face: Option<Vec<f32>>,
    pub body: Option<Vec<f32>>,
    pub width: usize,
    pub height: usize,
}

impl Default for PortraitMasks {
    fn default() -> Self {
        Self {
            skin: None,
            hair: None,
            face: None,
            body: None,
            width: 0,
            height: 0,
        }
    }
}

impl PortraitMasks {
    pub fn from_rgba(rgba: &[u8], width: u32, height: u32) -> Self {
        let parsed = crate::aimask::people_masks(rgba, width, height).ok();
        let face = parsed.as_ref().map(|p| p.face.clone());
        let body = parsed.as_ref().map(|p| p.body.clone());
        let hair = parsed.as_ref().map(|p| p.hair.clone());

        let mut skin = None;
        if face.is_some() || body.is_some() {
            let mut combined = vec![0.0f32; (width as usize) * (height as usize)];
            if let Some(plane) = &face {
                for (dst, src) in combined.iter_mut().zip(plane) {
                    *dst = dst.max(*src);
                }
            }
            if let Some(plane) = &body {
                for (dst, src) in combined.iter_mut().zip(plane) {
                    *dst = dst.max(*src);
                }
            }
            skin = Some(combined);
        }

        Self {
            skin,
            hair,
            face,
            body,
            width: width as usize,
            height: height as usize,
        }
    }

    fn plane(&self, kind: MaskKind) -> Option<&[f32]> {
        match kind {
            MaskKind::Skin => self.skin.as_deref(),
            MaskKind::Hair => self.hair.as_deref(),
            MaskKind::Face => self.face.as_deref(),
            MaskKind::Body => self.body.as_deref(),
        }
    }

    /// Samples one plane at the rendered frame's resolution and normalized region.
    pub fn mask_u8(
        &self,
        kind: MaskKind,
        out_w: u32,
        out_h: u32,
        region: NormRect,
    ) -> Option<Vec<u8>> {
        let plane = self.plane(kind)?;
        if self.width == 0
            || self.height == 0
            || plane.len() != self.width * self.height
            || out_w == 0
            || out_h == 0
        {
            return None;
        }
        let mut out = Vec::with_capacity((out_w as usize) * (out_h as usize));
        for y in 0..out_h {
            let ny = region.y + (y as f32 + 0.5) / out_h as f32 * region.height;
            for x in 0..out_w {
                let nx = region.x + (x as f32 + 0.5) / out_w as f32 * region.width;
                let value = sample_plane(plane, self.width, self.height, nx, ny);
                out.push((value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
        }
        Some(out)
    }

    #[cfg(test)]
    pub fn solid(value: f32) -> Self {
        let n = MASK_EDGE * MASK_EDGE;
        Self {
            skin: Some(vec![value; n]),
            hair: Some(vec![value; n]),
            face: Some(vec![value; n]),
            body: Some(vec![value; n]),
            width: MASK_EDGE,
            height: MASK_EDGE,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskKind {
    Skin,
    Hair,
    Face,
    Body,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortraitBackend {
    Cpu,
    Cuda,
    Skipped,
}

/// Machine-readable reason for a row that produced no pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortraitSkipReason {
    InputRefused(Refusal),
    EdgeGateClosed,
    Cuda(CudaError),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PortraitEffectRun {
    pub pf: u32,
    pub backend: PortraitBackend,
    pub dispatches: Vec<Dispatch>,
    pub fallback_reason: Option<String>,
    pub skipped: bool,
    pub executed: bool,
    pub refusal: Option<Refusal>,
    pub skip_reason: Option<PortraitSkipReason>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PortraitRunReport {
    pub job_id: Option<u64>,
    pub effects: Vec<PortraitEffectRun>,
}

/// Asset bytes needed by rows that cannot use the shared parser masks.
#[derive(Clone, Copy, Debug, Default)]
pub struct PortraitAssetInputs<'a> {
    pub teeth_whiten_lut: Option<&'a [u8]>,
    pub teeth_bright_lut: Option<&'a [u8]>,
}

/// PF108 same-job inputs. `pc:108` supplies the common UI fraction; the other
/// two controls remain explicit fields because no UI-to-native fallback exists.
#[derive(Clone, Copy, Debug)]
pub struct TeethInputs<'a> {
    pub mask_g: &'a [u8],
    pub bright_ui: f32,
    pub deyellow_ui: f32,
}

/// PF30026 final-blend inputs from one worker job.
///
/// Every plane is at the rendered frame size. `roi` identifies the normalized
/// source region and must match the stage request.
#[derive(Clone, Copy, Debug, Default)]
pub struct TeethAlignmentInputs<'a> {
    pub effect_rgb: Option<&'a Image>,
    pub teeth_mask: Option<&'a [u8]>,
    pub edge_plane_a: Option<&'a [f32]>,
    pub edge_plane_b: Option<&'a [f32]>,
    pub roi: Option<NormRect>,
}

/// PF21026 uses two independent beard masks and a same-size repair image.
#[derive(Clone, Copy, Debug)]
pub struct BeardInputs<'a> {
    pub mask_a: &'a [u8],
    pub mask_b: &'a [u8],
    pub repair_rgb: &'a Image,
}

/// PF90002 face-moisture inputs. Every plane is for the rendered frame.
#[derive(Clone, Copy)]
pub struct MoistureFaceInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub mask_rgba: &'a [u8],
    pub blur_y: &'a [u8],
    pub channel: MaskChannel,
    pub skin_map_width: u32,
    pub skin_map_height: u32,
    pub skin_map: &'a [f32],
    pub hardlight: &'a [f32],
    pub beauty: &'a [f32],
    pub pivot: f32,
    pub gain: f32,
}

/// PF90010 body-moisture inputs. Weight planes remain full-frame for a partial
/// ROI; `degree` is the captured native value rather than `pc:90010`.
#[derive(Clone, Copy, Debug)]
pub struct MoistureBodyInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub weight_skin: &'a [f32],
    pub weight_dog: &'a [f32],
    pub beauty: &'a [f32],
    pub degree: f32,
    pub roi: (u32, u32, u32, u32),
}

/// PF90116 aligned model outputs, never the parser's 512-square grid.
#[derive(Clone, Copy, Debug)]
pub struct EyeBagInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub aligned_color_field_rgb_f32: &'a [f32],
    pub aligned_response_mask: &'a [f32],
}

/// PF90139/PF90140 four-plane bundle. Planes are ROI-local with independent strides.
#[derive(Clone, Copy, Debug)]
pub struct HairSmoothInputs<'a> {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub roi: (u32, u32, u32, u32),
    /// Captured PF90140 descriptor `+0x80`, already normalized to `[0,1]`.
    pub mix_strength: f32,
    pub hair_mask: Option<&'a [u8]>,
    pub mix_mask: Option<&'a [u8]>,
    pub blend_rgb: Option<&'a [u8]>,
    pub detail: Option<&'a [u8]>,
    pub hair_stride: u32,
    pub mix_stride: u32,
    pub blend_stride: u32,
    pub detail_stride: u32,
}

/// PF90160 ROI-local mask/detail bundle and captured descriptor `+0x60` scale.
#[derive(Clone, Copy, Debug)]
pub struct HairGlossInputs<'a> {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub roi: (u32, u32, u32, u32),
    pub dynamic_scale: f32,
    pub hair_mask: Option<&'a [u8]>,
    pub model_detail: Option<&'a [u8]>,
    pub hair_stride: u32,
    pub detail_stride: u32,
}

/// PF90088's two hard producers and the per-face values consumed by the final
/// composite. `pc:90088` remains the UI gate; native values come from this job.
#[derive(Clone, Copy, Debug, Default)]
pub struct StrayHairInputs<'a> {
    pub hairfly_v1: Option<&'a Image>,
    pub raw_mask: Option<&'a [u8]>,
    pub person_mask: Option<&'a [u8]>,
    pub background_mask: Option<&'a [u8]>,
    pub native_degree: Option<f32>,
    pub native_person_area: Option<f32>,
    pub native_background_area: Option<f32>,
}

/// PF100007 open-mouth model fields and ROI.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmileInputs<'a> {
    pub material: Option<&'a str>,
    pub lip_mask: Option<&'a [u8]>,
    pub mouth_mask: Option<&'a [u8]>,
    pub model_dx: Option<&'a [f32]>,
    pub model_dy: Option<&'a [f32]>,
    pub roi: Option<Rect>,
}

pub enum SuitPassKindInput<'a> {
    Color,
    Image {
        material: Option<&'a Image>,
        material_uv: Option<&'a [[f32; 2]]>,
    },
    Unavailable(&'static str),
}

/// One PF90033 ordered pass with explicit ownership of every material and UV.
pub struct SuitPassInput<'a> {
    pub kind: SuitPassKindInput<'a>,
    pub mask: Option<&'a Image>,
    pub mask_uv: Option<&'a [[f32; 2]]>,
    pub layer: Option<MakeupLayer>,
    pub shelter: f32,
    pub region: f32,
}

/// One recovered makeup PF. `pc:<pf>` supplies `degree01`; every texture and
/// UV stream stays owned by the same job that produced the frame.
#[derive(Clone, Copy, Debug)]
pub struct MakeupInput<'a> {
    pub pf: u32,
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub filter_type: u32,
    pub material: Option<&'a Image>,
    pub material_uv: Option<&'a [[f32; 2]]>,
    pub mask: Option<&'a Image>,
    pub mask_uv: Option<&'a [[f32; 2]]>,
}

/// PF90021 same-job inputs. The host selects one direction; both LUTs remain
/// explicit so a missing selected asset is not mistaken for neutral.
#[derive(Clone, Copy, Debug)]
pub struct Pf90021Inputs<'a> {
    pub direction: Pf90021Direction,
    pub positive_lut: Option<&'a VolumeLut64>,
    pub negative_lut: Option<&'a VolumeLut64>,
    pub mask_r: Option<&'a [u8]>,
    pub runtime: Option<Pf90021RuntimeScalars>,
}

/// Where a job's inputs came from. Only a renderer job proves that absent
/// fields are absent in the same job, rather than unavailable at this entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortraitJobInputOrigin {
    MasksOnly,
    RendererJob { job_id: u64 },
}

/// Backend input container for effects whose producer is outside this stage.
pub struct PortraitJobInputs<'a> {
    pub origin: PortraitJobInputOrigin,
    pub masks: &'a PortraitMasks,
    pub assets: PortraitAssetInputs<'a>,
    pub teeth: Option<TeethInputs<'a>>,
    pub teeth_alignment: Option<TeethAlignmentInputs<'a>>,
    pub beard: Option<BeardInputs<'a>>,
    pub moisture_face: Option<MoistureFaceInputs<'a>>,
    pub moisture_body: Option<MoistureBodyInputs<'a>>,
    pub eye_bag: Option<EyeBagInputs<'a>>,
    pub hair_smooth: Option<HairSmoothInputs<'a>>,
    pub hair_gloss: Option<HairGlossInputs<'a>>,
    pub stray_hair: StrayHairInputs<'a>,
    pub smile: SmileInputs<'a>,
    pub suit_passes: Option<&'a [SuitPassInput<'a>]>,
    pub makeup: Option<&'a [MakeupInput<'a>]>,
    pub pf90021: Option<Pf90021Inputs<'a>>,
}

impl<'a> PortraitJobInputs<'a> {
    pub fn new(masks: &'a PortraitMasks) -> Self {
        Self::masks_only(masks)
    }

    /// Legacy entry: parser masks only. It cannot execute portrait effects.
    pub fn masks_only(masks: &'a PortraitMasks) -> Self {
        Self {
            origin: PortraitJobInputOrigin::MasksOnly,
            masks,
            assets: PortraitAssetInputs::default(),
            teeth: None,
            teeth_alignment: None,
            beard: None,
            moisture_face: None,
            moisture_body: None,
            eye_bag: None,
            hair_smooth: None,
            hair_gloss: None,
            stray_hair: StrayHairInputs::default(),
            smile: SmileInputs::default(),
            suit_passes: None,
            makeup: None,
            pf90021: None,
        }
    }

    /// A renderer supplies every field from one job. Optional fields mean the
    /// producer found no such input for this job, never a mask fallback.
    pub fn from_renderer_job(job_id: u64, masks: &'a PortraitMasks) -> Self {
        Self {
            origin: PortraitJobInputOrigin::RendererJob { job_id },
            ..Self::masks_only(masks)
        }
    }

    pub fn job_id(&self) -> Option<u64> {
        match self.origin {
            PortraitJobInputOrigin::MasksOnly => None,
            PortraitJobInputOrigin::RendererJob { job_id } => Some(job_id),
        }
    }
}

fn sample_plane(plane: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let px = (x.clamp(0.0, 1.0) * width as f32 - 0.5).clamp(0.0, (width as f32 - 1.001).max(0.0));
    let py = (y.clamp(0.0, 1.0) * height as f32 - 0.5).clamp(0.0, (height as f32 - 1.001).max(0.0));
    let x0 = px as usize;
    let y0 = py as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = px - x0 as f32;
    let ty = py - y0 as f32;
    let at = |xx: usize, yy: usize| plane[yy * width + xx];
    let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
    let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
    top + (bottom - top) * ty
}

/// Effects that can run from the frame, a mask and bundled LUT bytes today.
const SUPPORTED_EFFECTS: &[u32] = &[
    108,    // Teeth whitening
    30026,  // Teeth alignment final blend
    8068,   // Skin gloss
    8069,   // Skin texture (negative surface-blur branch)
    8070,   // Skin clarity
    21015,  // Skin color hue
    21026,  // Stubble removal
    90021,  // Lip makeup volume LUT
    30096,  // Skin temperature
    90002,  // Moisture smooth (face)
    90004,  // Skin radiance
    90005,  // Skin whitening
    90007,  // Skin ruddy
    90010,  // Moisture smooth (body)
    90033,  // Suit makeup ordered passes
    90116,  // Eye bags
    90139,  // Hair smooth
    90160,  // Hair gloss
    90088,  // Stray hair removal
    100007, // Open-mouth smile
    100,    // Eyebrow color makeup, left
    103,    // Upper eyeliner color makeup
    105,    // Upper eyelash color makeup
    111,    // Eye pupil image makeup, no mask
    117,    // Eyebrow color makeup, right
    118,    // Lower eyelash color makeup
    30007,  // Lower eyeliner color makeup
];

fn log_report(report: &PortraitRunReport) {
    for effect in &report.effects {
        if let Some(refusal) = &effect.refusal {
            log::debug!("portrait PF{} refused: {refusal:?}", effect.pf);
            continue;
        }
        match effect.backend {
            PortraitBackend::Cuda => log::debug!(
                "portrait PF{} executed on CUDA: {:?}",
                effect.pf,
                effect
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>()
            ),
            PortraitBackend::Cpu => {
                if let Some(reason) = &effect.fallback_reason {
                    log::debug!("portrait PF{} CPU fallback: {reason}", effect.pf);
                }
            }
            PortraitBackend::Skipped => log::debug!(
                "portrait PF{} skipped: {:?} / {:?}",
                effect.pf,
                effect.skip_reason,
                effect.fallback_reason
            ),
        }
    }
}

/// Applies every portrait parameter whose effect has a CPU stage and whose
/// same-job inputs are available. Returns the image with effects applied in
/// ascending ParamFlag order.
pub fn apply_portrait_effects(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    masks: &PortraitMasks,
) {
    let report = apply_portrait_effects_with_report(
        portrait, width, height, region, extra, masks,
    );
    log_report(&report);
}

/// Production-style entry point for callers that already own the complete
/// same-job bundle.
pub fn apply_portrait_effects_with_job(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    inputs: &PortraitJobInputs<'_>,
) -> PortraitRunReport {
    let report = apply_portrait_effects_with_report_and_inputs(
        portrait, width, height, region, extra, inputs,
    );
    log_report(&report);
    report
}

pub fn apply_portrait_effects_with_report(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    masks: &PortraitMasks,
) -> PortraitRunReport {
    apply_portrait_effects_with_report_and_inputs(
        portrait,
        width,
        height,
        region,
        extra,
        &PortraitJobInputs::new(masks),
    )
}

pub fn apply_portrait_effects_with_report_and_inputs(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    inputs: &PortraitJobInputs<'_>,
) -> PortraitRunReport {
    let mut report = PortraitRunReport {
        job_id: inputs.job_id(),
        effects: Vec::new(),
    };
    let Some(source) = Image::new(width, height, std::mem::take(portrait)) else {
        return report;
    };

    let mut effects: Vec<(u32, f32)> = extra
        .iter()
        .filter_map(|(key, value)| {
            let id: u32 = key.strip_prefix("pc:")?.parse().ok()?;
            Some((id, *value))
        })
        .collect();
    effects.sort_by_key(|&(pf, _)| pf);

    let mut current = source;
    // Keep the main-session optimization: skin effects share one sampled plane.
    // Non-skin rows receive their own explicit planes from `PortraitJobInputs`.
    let skin_mask = inputs.masks.mask_u8(MaskKind::Skin, width, height, region);
    for (pf, degree) in effects {
        let Some(meta) = aurora_portrait::lookup(pf) else {
            continue;
        };
        if meta.access != Access::Cpu || !SUPPORTED_EFFECTS.contains(&pf) {
            log::debug!(
                "portrait PF{pf} skipped; missing same-job inputs: {:?}",
                meta.inputs
            );
            continue;
        }
        let mut run = PortraitEffectRun {
            pf,
            backend: PortraitBackend::Cpu,
            dispatches: Vec::new(),
            fallback_reason: None,
            skipped: true,
            executed: false,
            refusal: None,
            skip_reason: None,
        };
        if inputs.origin == PortraitJobInputOrigin::MasksOnly {
            let refusal = Refusal::MissingInput("same-job input bundle");
            run.backend = PortraitBackend::Skipped;
            run.fallback_reason =
                Some("skipped: masks-only entry has no same-job producer".to_string());
            run.refusal = Some(refusal.clone());
            run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal));
            report.effects.push(run);
            continue;
        }
        if pf == 90021 {
            let (next, pf90021_run) = apply_pf90021_stage(&current, inputs.pf90021);
            if let Some(next) = next {
                current = next;
            }
            report.effects.push(pf90021_run);
            continue;
        }
        if pf == 30026 {
            let (next, run) = apply_pf30026_stage(&current, region, degree, inputs);
            if let Some(next) = next {
                current = next;
            }
            report.effects.push(run);
            continue;
        }
        if pf == 90033 {
            let (next, suit_run) = apply_suit_effect(&current, inputs.suit_passes);
            if let Some(next) = next {
                current = next;
            }
            report.effects.push(suit_run);
            continue;
        }
        if let Err(refusal) =
            validate_effect_inputs(pf, &current, extra, skin_mask.as_deref(), inputs)
        {
            run.fallback_reason = Some(format!("input refused: {refusal:?}"));
            run.refusal = Some(refusal);
            report.effects.push(run);
            continue;
        }
        #[cfg(windows)]
        let cuda_result = try_cuda_effect(&current, pf, degree, skin_mask.as_deref())
            .or_else(|| try_cuda_same_job_effect(&current, pf, degree, inputs));
        #[cfg(windows)]
        if let Some(result) = cuda_result {
            match result {
                Ok((next, dispatches)) if !dispatches.is_empty() => {
                    current = next;
                    run.backend = PortraitBackend::Cuda;
                    run.dispatches = dispatches;
                    run.skipped = false;
                    run.executed = true;
                    report.effects.push(run);
                    continue;
                }
                Ok((next, _)) if matches!(pf, 90088 | 90139 | 90160) => {
                    current = next;
                    run.skipped = false;
                    run.executed = true;
                    report.effects.push(run);
                    continue;
                }
                Ok(_) => {
                    run.fallback_reason = Some(
                        "CUDA returned no dispatch; CPU kernel selected for parity".to_string(),
                    );
                }
                Err(err) if matches!(pf, 90139 | 90160) => {
                    run.refusal = Some(match &err {
                        CudaError::Refused(refusal) => refusal.clone(),
                        _ => Refusal::NotImplemented(pf),
                    });
                    run.fallback_reason = Some(format!("{err:?}"));
                    report.effects.push(run);
                    continue;
                }
                Err(err) => run.fallback_reason = Some(format!("{err:?}")),
            }
        }
        let result = apply_cpu_effect(
            &current,
            pf,
            degree,
            extra,
            skin_mask.as_deref(),
            width,
            height,
            inputs,
        );
        match result {
            Ok(next) => {
                current = next;
                run.skipped = false;
                run.executed = true;
            }
            Err(err) => {
                let cpu_reason = format!("{err:?}");
                run.fallback_reason = Some(match run.fallback_reason.take() {
                    Some(gpu_reason) => format!("{gpu_reason}; CPU: {cpu_reason}"),
                    None => format!("CPU: {cpu_reason}"),
                });
                log::debug!("portrait PF{pf} skipped: {err:?}");
                run.refusal = Some(err);
            }
        }
        report.effects.push(run);
    }

    *portrait = current.pixels;
    report
}

fn required<T>(value: Option<T>, name: &'static str) -> Result<T, Refusal> {
    value.ok_or(Refusal::MissingInput(name))
}

fn check_plane(len: usize, expected: usize, name: &'static str) -> Result<(), Refusal> {
    if len == expected {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(name))
    }
}

fn check_degree01(value: f32, name: &'static str) -> Result<(), Refusal> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn check_ui_percent(value: f32, name: &'static str) -> Result<(), Refusal> {
    if value.is_finite() && (0.0..=100.0).contains(&value) {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn check_roi(width: u32, height: u32, roi: (u32, u32, u32, u32)) -> Result<(), Refusal> {
    let (x, y, w, h) = roi;
    if w == 0
        || h == 0
        || (x as u64 + w as u64) > width as u64
        || (y as u64 + h as u64) > height as u64
    {
        Err(Refusal::ValueOutOfRange("roi"))
    } else {
        Ok(())
    }
}

fn check_norm_rect(rect: NormRect, name: &'static str) -> Result<(), Refusal> {
    if [
        rect.x,
        rect.y,
        rect.width,
        rect.height,
    ]
    .into_iter()
    .all(f32::is_finite)
        && rect.x >= 0.0
        && rect.y >= 0.0
        && rect.width > 0.0
        && rect.height > 0.0
        && rect.x + rect.width <= 1.0
        && rect.y + rect.height <= 1.0
    {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn pf90021_skipped(
    reason: PortraitSkipReason,
    detail: String,
    refusal: Option<Refusal>,
) -> (Option<Image>, PortraitEffectRun) {
    (
        None,
        PortraitEffectRun {
            pf: 90021,
            backend: PortraitBackend::Skipped,
            dispatches: Vec::new(),
            fallback_reason: Some(detail),
            skipped: true,
            executed: false,
            refusal,
            skip_reason: Some(reason),
        },
    )
}

fn resolve_pf90021<'a>(
    current: &Image,
    inputs: Option<Pf90021Inputs<'a>>,
) -> Result<
    Option<(
        Pf90021Direction,
        &'a VolumeLut64,
        &'a [u8],
        Pf90021RuntimeScalars,
    )>,
    Refusal,
> {
    let input = required(inputs, "PF90021_SAME_JOB_INPUTS")?;
    if input.direction == Pf90021Direction::Neutral {
        return Ok(None);
    }
    let (lut, missing_lut) = match input.direction {
        Pf90021Direction::Positive => (input.positive_lut, "PF90021_POSITIVE_LUT64"),
        Pf90021Direction::Negative => (input.negative_lut, "PF90021_NEGATIVE_LUT64"),
        Pf90021Direction::Neutral => unreachable!(),
    };
    let lut = required(lut, missing_lut)?;
    let mask_r = required(input.mask_r, "PF90021_MASK_R")?;
    check_plane(mask_r.len(), current.pixel_count(), "PF90021_MASK_R")?;
    let runtime = required(input.runtime, "PF90021_RUNTIME_SCALARS")?;
    runtime.u_alpha()?;
    Ok(Some((input.direction, lut, mask_r, runtime)))
}

fn apply_pf90021_stage(
    current: &Image,
    inputs: Option<Pf90021Inputs<'_>>,
) -> (Option<Image>, PortraitEffectRun) {
    let resolved = match resolve_pf90021(current, inputs) {
        Ok(Some(resolved)) => resolved,
        Ok(None) => {
            return (
                None,
                PortraitEffectRun {
                    pf: 90021,
                    backend: PortraitBackend::Skipped,
                    dispatches: Vec::new(),
                    fallback_reason: Some("neutral direction; no dispatch".to_string()),
                    skipped: true,
                    executed: false,
                    refusal: None,
                    skip_reason: None,
                },
            );
        }
        Err(error) => {
            return pf90021_skipped(
                PortraitSkipReason::InputRefused(error.clone()),
                format!("input refused: {error:?}"),
                Some(error),
            );
        }
    };
    let (direction, lut, mask_r, runtime) = resolved;

    #[cfg(windows)]
    {
        let result = match forced_cuda_failure() {
            Some(error) => Err(error),
            None => match aurora_portrait::cuda_pf90021::Pf90021::open() {
                Ok(gpu) => gpu.run(aurora_portrait::cuda_pf90021::Request {
                    source: current,
                    direction,
                    lut: Some(lut),
                    mask_r: Some(mask_r),
                    runtime: Some(runtime),
                }),
                Err(error) => Err(error),
            },
        };
        match result {
            Ok(aurora_portrait::cuda_pf90021::Run::Executed { output, dispatch }) => {
                let dispatches = vec![dispatch.lut, dispatch.composite];
                return (
                    Some(output),
                    PortraitEffectRun {
                        pf: 90021,
                        backend: PortraitBackend::Cuda,
                        dispatches,
                        fallback_reason: None,
                        skipped: false,
                        executed: true,
                        refusal: None,
                        skip_reason: None,
                    },
                );
            }
            Ok(aurora_portrait::cuda_pf90021::Run::Bypassed(_)) => {}
            Err(error) => {
                return match aurora_portrait::cuda_pf90021::apply_cpu(
                    current, lut, mask_r, runtime,
                ) {
                    Ok(output) => (
                        Some(output),
                        PortraitEffectRun {
                            pf: 90021,
                            backend: PortraitBackend::Cpu,
                            dispatches: Vec::new(),
                            fallback_reason: Some(format!("{error:?}")),
                            skipped: false,
                            executed: true,
                            refusal: None,
                            skip_reason: None,
                        },
                    ),
                    Err(refusal) => pf90021_skipped(
                        PortraitSkipReason::InputRefused(refusal.clone()),
                        format!("CUDA: {error:?}; CPU: {refusal:?}"),
                        Some(refusal),
                    ),
                };
            }
        }
    }

    match aurora_portrait::cuda_pf90021::apply_cpu(current, lut, mask_r, runtime) {
        Ok(output) => (
            Some(output),
            PortraitEffectRun {
                pf: 90021,
                backend: PortraitBackend::Cpu,
                dispatches: Vec::new(),
                fallback_reason: None,
                skipped: false,
                executed: true,
                refusal: None,
                skip_reason: None,
            },
        ),
        Err(refusal) => pf90021_skipped(
            PortraitSkipReason::InputRefused(refusal.clone()),
            format!("CPU: {refusal:?}"),
            Some(refusal),
        ),
    }
}

fn pf30026_skipped(
    reason: PortraitSkipReason,
    detail: String,
    refusal: Option<Refusal>,
) -> PortraitEffectRun {
    PortraitEffectRun {
        pf: 30026,
        backend: PortraitBackend::Skipped,
        dispatches: Vec::new(),
        fallback_reason: Some(detail),
        skipped: true,
        executed: false,
        refusal,
        skip_reason: Some(reason),
    }
}

fn apply_pf30026_stage<'a>(
    current: &'a Image,
    region: NormRect,
    degree: f32,
    inputs: &'a PortraitJobInputs<'a>,
) -> (Option<Image>, PortraitEffectRun) {
    let refused = |error: Refusal| {
        pf30026_skipped(
            PortraitSkipReason::InputRefused(error.clone()),
            format!("input refused: {error:?}"),
            Some(error),
        )
    };
    if let Err(error) = check_degree01(degree, "PF30026 degree") {
        return (None, refused(error));
    }
    let Some(input) = inputs.teeth_alignment else {
        return (
            None,
            refused(Refusal::MissingInput("PF30026 same-job inputs")),
        );
    };
    let effect_rgb = match required(input.effect_rgb, "PF30026 effect RGB") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    let teeth_mask = match required(input.teeth_mask, "PF30026 teeth mask") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    let edge_plane_a = match required(input.edge_plane_a, "PF30026 edge plane A") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    let edge_plane_b = match required(input.edge_plane_b, "PF30026 edge plane B") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    let roi = match required(input.roi, "PF30026 ROI") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    if let Err(error) = check_norm_rect(roi, "PF30026 ROI") {
        return (None, refused(error));
    }
    if roi != region {
        return (None, refused(Refusal::ValueOutOfRange("PF30026 ROI")));
    }
    let job = AlignmentFinalBlendJob {
        original: Some(current),
        effect_rgb: Some(effect_rgb),
        teeth_mask: Some(teeth_mask),
        edge_plane_a: Some(edge_plane_a),
        edge_plane_b: Some(edge_plane_b),
    };
    match aurora_portrait::kernels::teeth_blend::alignment_final_blend_gate(job) {
        Ok(true) => {}
        Ok(false) => {
            return (
                None,
                pf30026_skipped(
                    PortraitSkipReason::EdgeGateClosed,
                    "PF30026 edge gate closed: valid ratio <= 0.14%".to_string(),
                    None,
                ),
            );
        }
        Err(error) => return (None, refused(error)),
    }

    #[cfg(windows)]
    {
        if let Some(error) = forced_cuda_failure() {
            return (
                None,
                pf30026_skipped(
                    PortraitSkipReason::Cuda(error.clone()),
                    format!("CUDA unavailable: {error:?}"),
                    None,
                ),
            );
        }
        let gpu = match aurora_portrait::cuda_teeth_blend::TeethBlend::open() {
            Ok(gpu) => gpu,
            Err(error) => {
                return (
                    None,
                    pf30026_skipped(
                        PortraitSkipReason::Cuda(error.clone()),
                        format!("CUDA unavailable: {error:?}"),
                        None,
                    ),
                );
            }
        };
        match gpu.apply_alignment_final_blend(job) {
            Ok(aurora_portrait::cuda_teeth_blend::AlignmentFinalBlendGpu::Applied {
                image,
                dispatches,
            }) => (
                Some(image),
                PortraitEffectRun {
                    pf: 30026,
                    backend: PortraitBackend::Cuda,
                    dispatches,
                    fallback_reason: None,
                    skipped: false,
                    executed: true,
                    refusal: None,
                    skip_reason: None,
                },
            ),
            Ok(aurora_portrait::cuda_teeth_blend::AlignmentFinalBlendGpu::Gated) => (
                None,
                pf30026_skipped(
                    PortraitSkipReason::EdgeGateClosed,
                    "PF30026 edge gate closed: valid ratio <= 0.14%".to_string(),
                    None,
                ),
            ),
            Err(error) => (
                None,
                pf30026_skipped(
                    PortraitSkipReason::Cuda(error.clone()),
                    format!("CUDA failed: {error:?}"),
                    None,
                ),
            ),
        }
    }

    #[cfg(not(windows))]
    {
        let error = CudaError::Unavailable("PF30026 requires NVIDIA CUDA");
        (
            None,
            pf30026_skipped(
                PortraitSkipReason::Cuda(error.clone()),
                format!("CUDA unavailable: {error:?}"),
                None,
            ),
        )
    }
}

fn check_roi_plane(
    bytes: &[u8],
    roi: HairRoi,
    stride: u32,
    bytes_per_pixel: u32,
    name: &'static str,
) -> Result<(), Refusal> {
    let row = roi
        .width
        .checked_mul(bytes_per_pixel)
        .ok_or(Refusal::ShapeMismatch(name))?;
    if stride < row {
        return Err(Refusal::ShapeMismatch(name));
    }
    let required = roi
        .height
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|head| head.checked_add(row))
        .ok_or(Refusal::ShapeMismatch(name))?;
    if bytes.len() < required as usize {
        return Err(Refusal::ShapeMismatch(name));
    }
    Ok(())
}

fn resolve_hair_smooth<'a>(
    current: &Image,
    detail_strength: f32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<(HairSmoothInputs<'a>, HairRoi), Refusal> {
    check_degree01(detail_strength, "PF90139_DETAIL_STRENGTH")?;
    let hair = required(
        inputs.hair_smooth,
        "PF90139_PF90140_SAME_JOB_PRODUCER",
    )?;
    if hair.ticket == 0 {
        return Err(Refusal::MissingInput(
            "PF90139_PF90140_SAME_JOB_RECEIPT",
        ));
    }
    if (hair.width, hair.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF90139_PF90140 frame"));
    }
    check_degree01(hair.mix_strength, "PF90140_MIX_STRENGTH")?;
    check_roi(current.width, current.height, hair.roi)?;
    let (x, y, width, height) = hair.roi;
    let roi = HairRoi {
        x,
        y,
        width,
        height,
    };
    check_roi_plane(
        required(hair.hair_mask, "PF90139_HAIR_MASK")?,
        roi,
        hair.hair_stride,
        1,
        "PF90139_HAIR_MASK",
    )?;
    check_roi_plane(
        required(hair.mix_mask, "PF90139_MIX_MASK")?,
        roi,
        hair.mix_stride,
        1,
        "PF90139_MIX_MASK",
    )?;
    check_roi_plane(
        required(hair.blend_rgb, "PF90139_BLEND_SOURCE_RGB8")?,
        roi,
        hair.blend_stride,
        3,
        "PF90139_BLEND_SOURCE_RGB8",
    )?;
    check_roi_plane(
        required(hair.detail, "PF90139_DETAIL")?,
        roi,
        hair.detail_stride,
        1,
        "PF90139_DETAIL",
    )?;
    Ok((hair, roi))
}

fn resolve_hair_gloss<'a>(
    current: &Image,
    enabled: f32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<(HairGlossInputs<'a>, HairRoi), Refusal> {
    check_degree01(enabled, "PF90160_ENABLED")?;
    let hair = required(inputs.hair_gloss, "PF90160_SAME_JOB_PRODUCER")?;
    if hair.ticket == 0 {
        return Err(Refusal::MissingInput("PF90160_SAME_JOB_RECEIPT"));
    }
    if (hair.width, hair.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF90160 frame"));
    }
    check_degree01(hair.dynamic_scale, "PF90160_DYNAMIC_SCALE")?;
    check_roi(current.width, current.height, hair.roi)?;
    let (x, y, width, height) = hair.roi;
    let roi = HairRoi {
        x,
        y,
        width,
        height,
    };
    check_roi_plane(
        required(hair.hair_mask, "PF90160_HAIR_MASK")?,
        roi,
        hair.hair_stride,
        1,
        "PF90160_HAIR_MASK",
    )?;
    check_roi_plane(
        required(hair.model_detail, "PF90160_MODEL_DETAIL")?,
        roi,
        hair.detail_stride,
        1,
        "PF90160_MODEL_DETAIL",
    )?;
    Ok((hair, roi))
}

fn find_makeup_input<'a>(
    pf: u32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<(MakeupRoute, MakeupInput<'a>), Refusal> {
    let route = MakeupRoute::resolve(pf)?;
    let input = inputs
        .makeup
        .and_then(|rows| rows.iter().copied().find(|row| row.pf == pf));
    match input {
        Some(input) => Ok((route, input)),
        None => Err(match route {
            MakeupRoute::Color { .. } => Refusal::MissingInput("makeup mask"),
            MakeupRoute::ImageUnmasked { .. } => Refusal::MissingInput("makeup material"),
        }),
    }
}

fn makeup_layer(
    route: MakeupRoute,
    input: MakeupInput<'_>,
    degree: f32,
) -> Result<MakeupLayer, Refusal> {
    if input.filter_type != route.filter_type() {
        return Err(Refusal::ValueOutOfRange("makeup FilterType"));
    }
    check_ui_percent(input.opacity_percent, "opacity_percent")?;
    check_degree01(degree, "degree01")?;
    Ok(MakeupLayer {
        mix_color: input.mix_color,
        opacity_percent: input.opacity_percent,
        degree01: degree,
        filter_type: route.filter_type(),
        mask_channel: route
            .mask_channel()
            .unwrap_or(aurora_portrait::kernels::makeup_material::MaskChannel::A),
    })
}

fn validate_makeup_inputs(pf: u32, inputs: &PortraitJobInputs<'_>) -> Result<(), Refusal> {
    let (route, input) = find_makeup_input(pf, inputs)?;
    match route {
        MakeupRoute::Color { .. } => {
            required(input.mask, "makeup mask")?.validate()?;
            required(input.mask_uv, "makeup mask UV")?;
        }
        MakeupRoute::ImageUnmasked { .. } => {
            required(input.material, "makeup material")?.validate()?;
            required(input.material_uv, "makeup material UV")?;
            if input.mask.is_some() || input.mask_uv.is_some() {
                return Err(Refusal::ValueOutOfRange("PF111 mask must be absent"));
            }
        }
    }
    Ok(())
}

fn validate_effect_inputs(
    pf: u32,
    current: &Image,
    extra: &BTreeMap<String, f32>,
    skin_mask: Option<&[u8]>,
    inputs: &PortraitJobInputs<'_>,
) -> Result<(), Refusal> {
    let pixels = current.pixel_count();
    if matches!(
        pf,
        8068 | 8069 | 8070 | 21015 | 30096 | 90004 | 90005 | 90007
    ) {
        required(skin_mask, "skin mask")?;
        match pf {
            21015 if tint_green().is_none() || tint_magenta().is_none() => {
                return Err(Refusal::MissingInput("skin tint LUT"));
            }
            30096 if cold_lut().is_none() || warm_lut().is_none() => {
                return Err(Refusal::MissingInput("skin temperature LUT"));
            }
            90005 if whiten_lut().is_none() => {
                return Err(Refusal::MissingInput("whiten LUT"));
            }
            90007 if ruddy_lut().is_none() => {
                return Err(Refusal::MissingInput("ruddy LUT"));
            }
            _ => {}
        }
        return Ok(());
    }

    match pf {
        100 | 103 | 105 | 111 | 117 | 118 | 30007 => validate_makeup_inputs(pf, inputs)?,
        108 => {
            check_degree01(extra.get("pc:108").copied().unwrap_or_default(), "degree")?;
            let teeth = required(inputs.teeth, "teeth mask")?;
            check_plane(teeth.mask_g.len(), pixels, "teeth mask")?;
            let whiten = required(inputs.assets.teeth_whiten_lut, "teeth whiten LUT")?;
            let bright = required(inputs.assets.teeth_bright_lut, "teeth bright LUT")?;
            check_plane(whiten.len(), LUT_BYTES, "teeth whiten LUT")?;
            check_plane(bright.len(), LUT_BYTES, "teeth bright LUT")?;
            check_ui_percent(teeth.bright_ui, "teeth bright UI")?;
            check_ui_percent(teeth.deyellow_ui, "teeth deyellow UI")?;
        }
        21026 => {
            check_degree01(
                extra.get("pc:21026").copied().unwrap_or_default(),
                "strength",
            )?;
            let beard = required(inputs.beard, "mask_a")?;
            check_plane(beard.mask_a.len(), pixels, "mask_a")?;
            check_plane(beard.mask_b.len(), pixels, "mask_b")?;
            beard.repair_rgb.validate()?;
            if (beard.repair_rgb.width, beard.repair_rgb.height) != (current.width, current.height)
            {
                return Err(Refusal::ShapeMismatch("repair RGB"));
            }
        }
        90002 => {
            check_degree01(extra.get("pc:90002").copied().unwrap_or_default(), "degree")?;
            let face = required(inputs.moisture_face, "moisture face inputs")?;
            if (face.width, face.height) != (current.width, current.height) {
                return Err(Refusal::ShapeMismatch("moisture face frame"));
            }
            check_plane(face.mask_rgba.len(), pixels.saturating_mul(4), "mask_rgba")?;
            check_plane(face.blur_y.len(), pixels, "blur_y")?;
            check_plane(
                face.skin_map.len(),
                (face.skin_map_width as usize)
                    .checked_mul(face.skin_map_height as usize)
                    .ok_or(Refusal::ShapeMismatch("skin_map"))?,
                "skin_map",
            )?;
            if face.hardlight.is_empty() {
                return Err(Refusal::MissingInput("hardlight LUT"));
            }
            if face.beauty.is_empty() {
                return Err(Refusal::MissingInput("beauty LUT"));
            }
            if !face.pivot.is_finite() || !face.gain.is_finite() {
                return Err(Refusal::ValueOutOfRange("pivot/gain"));
            }
        }
        90010 => {
            check_degree01(extra.get("pc:90010").copied().unwrap_or_default(), "degree")?;
            let body = required(inputs.moisture_body, "moisture body inputs")?;
            if (body.width, body.height) != (current.width, current.height) {
                return Err(Refusal::ShapeMismatch("moisture body frame"));
            }
            check_plane(body.weight_skin.len(), pixels, "weight_skin")?;
            check_plane(body.weight_dog.len(), pixels, "weight_dog")?;
            if body.beauty.is_empty() {
                return Err(Refusal::MissingInput("beauty LUT"));
            }
            check_degree01(body.degree, "captured degree")?;
            check_roi(current.width, current.height, body.roi)?;
        }
        90116 => {
            check_degree01(extra.get("pc:90116").copied().unwrap_or_default(), "degree")?;
            let eye_bag = required(inputs.eye_bag, "aligned_color_field_rgb_f32")?;
            if (eye_bag.width, eye_bag.height) != (current.width, current.height) {
                return Err(Refusal::ShapeMismatch("aligned_color_field_rgb_f32"));
            }
            check_plane(
                eye_bag.aligned_color_field_rgb_f32.len(),
                pixels.saturating_mul(3),
                "aligned_color_field_rgb_f32",
            )?;
            check_plane(
                eye_bag.aligned_response_mask.len(),
                pixels,
                "aligned_response_mask",
            )?;
        }
        90139 => {
            resolve_hair_smooth(
                current,
                extra.get("pc:90139").copied().unwrap_or_default(),
                inputs,
            )?;
        }
        90160 => {
            resolve_hair_gloss(
                current,
                extra.get("pc:90160").copied().unwrap_or_default(),
                inputs,
            )?;
        }
        90088 => {
            resolve_stray_hair(
                current,
                extra.get("pc:90088").copied().unwrap_or_default(),
                inputs,
            )?;
        }
        100007 => {
            check_degree01(
                extra.get("pc:100007").copied().unwrap_or_default(),
                "pf100007 degree",
            )?;
            let material = required(inputs.smile.material, "pf100007 material")?;
            let lip_mask = required(inputs.smile.lip_mask, "pf100007 lip_mask")?;
            let mouth_mask = required(inputs.smile.mouth_mask, "pf100007 mouth_mask")?;
            let model_dx = required(inputs.smile.model_dx, "pf100007 model_dx")?;
            let model_dy = required(inputs.smile.model_dy, "pf100007 model_dy")?;
            let roi = required(inputs.smile.roi, "pf100007 roi")?;
            if material.is_empty() {
                return Err(Refusal::MissingInput("pf100007 material"));
            }
            check_plane(lip_mask.len(), pixels, "pf100007 lip_mask")?;
            check_plane(mouth_mask.len(), pixels, "pf100007 mouth_mask")?;
            check_plane(model_dx.len(), pixels, "pf100007 model_dx")?;
            check_plane(model_dy.len(), pixels, "pf100007 model_dy")?;
            if roi.x > current.width
                || roi.y > current.height
                || roi.width > current.width - roi.x
                || roi.height > current.height - roi.y
            {
                return Err(Refusal::ValueOutOfRange("pf100007 roi"));
            }
            if model_dx
                .iter()
                .chain(model_dy)
                .any(|value| !value.is_finite())
            {
                return Err(Refusal::ValueOutOfRange("pf100007 displacement"));
            }
        }
        _ => {}
    }
    Ok(())
}

fn apply_cpu_effect(
    current: &Image,
    pf: u32,
    degree: f32,
    _extra: &BTreeMap<String, f32>,
    skin_mask: Option<&[u8]>,
    width: u32,
    height: u32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    match pf {
        100 | 103 | 105 | 111 | 117 | 118 | 30007 => {
            apply_makeup_cpu(current, pf, degree, inputs)
        }
        108 => apply_teeth_cpu(current, degree, inputs),
        8068 => apply_skin_gloss(current, degree, skin_mask),
        8069 => apply_skin_texture(current, degree, skin_mask),
        8070 => apply_skin_clarity(current, degree, skin_mask, width, height),
        21015 => apply_skin_tint(current, degree, skin_mask),
        21026 => apply_beard_cpu(current, degree, inputs),
        30096 => apply_skin_temperature(current, degree, skin_mask),
        90002 => apply_moisture_face_cpu(current, degree, inputs),
        90004 => apply_skin_radiance(current, degree, skin_mask),
        90005 => apply_skin_whiten(current, degree, skin_mask),
        90007 => apply_skin_ruddy(current, degree, skin_mask),
        90010 => apply_moisture_body_cpu(current, degree, inputs),
        90116 => apply_eye_bag_cpu(current, degree, inputs),
        // These rows are GPU-only until the legacy hair_body CPU slots are reconciled.
        90139 | 90160 => Err(Refusal::NotImplemented(pf)),
        90088 => apply_stray_hair_cpu(current, degree, inputs),
        100007 => apply_smile_cpu(current, degree, inputs),
        _ => Ok(current.clone()),
    }
}

fn apply_stray_hair_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let Some(job) = resolve_stray_hair(current, degree, inputs)? else {
        return Ok(current.clone());
    };
    aurora_portrait::kernels::hair_body::stray_hair(
        current,
        job.params.ui,
        job.params.native_degree,
        job.params.native_person_area,
        job.params.native_background_area,
        Some(job.hairfly_v1),
        Some(job.raw_mask),
        job.person_mask,
        job.background_mask,
    )
}

fn apply_makeup_cpu(
    current: &Image,
    pf: u32,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let (route, input) = find_makeup_input(pf, inputs)?;
    let layer = makeup_layer(route, input, degree)?;
    match route {
        MakeupRoute::Color { .. } => color_makeup(
            current,
            required(input.mask, "makeup mask")?,
            required(input.mask_uv, "makeup mask UV")?,
            layer,
        ),
        MakeupRoute::ImageUnmasked { .. } => {
            if input.mask.is_some() || input.mask_uv.is_some() {
                return Err(Refusal::ValueOutOfRange("PF111 mask must be absent"));
            }
            image_makeup_unmasked(
                current,
                required(input.material, "makeup material")?,
                required(input.material_uv, "makeup material UV")?,
                layer,
            )
        }
    }
}

fn apply_teeth_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let teeth = required(inputs.teeth, "teeth mask")?;
    let whiten = required(inputs.assets.teeth_whiten_lut, "teeth whiten LUT")?;
    let bright = required(inputs.assets.teeth_bright_lut, "teeth bright LUT")?;
    Ok(aurora_portrait::kernels::face_detail::teeth_whitening(
        current,
        teeth.mask_g,
        whiten,
        bright,
        degree * 100.0,
        teeth.bright_ui,
        teeth.deyellow_ui,
    )?
    .output)
}

fn apply_beard_cpu(
    current: &Image,
    strength: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let beard = required(inputs.beard, "mask_a")?;
    aurora_portrait::kernels::beard::apply(
        current,
        beard.repair_rgb,
        beard.mask_a,
        beard.mask_b,
        strength.clamp(0.0, 1.0),
    )
}

fn apply_moisture_face_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = required(inputs.moisture_face, "moisture face inputs")?;
    let params = FaceParams {
        ui: degree * 100.0,
        pivot: input.pivot,
        gain: input.gain,
    };
    let skin_map = SkinMap {
        width: input.skin_map_width as usize,
        height: input.skin_map_height as usize,
        samples: input.skin_map,
    };
    aurora_portrait::kernels::moisture::face_apply(
        current,
        input.mask_rgba,
        input.blur_y,
        &params,
        input.channel,
        &skin_map,
        &Lut1D {
            samples: input.hardlight,
        },
        &Lut1D {
            samples: input.beauty,
        },
    )
}

fn apply_moisture_body_cpu(
    current: &Image,
    enabled: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = required(inputs.moisture_body, "moisture body inputs")?;
    let degree = if enabled > 0.0 { input.degree } else { 0.0 };
    aurora_portrait::kernels::moisture::body_apply(
        current,
        input.weight_skin,
        input.weight_dog,
        &Lut1D {
            samples: input.beauty,
        },
        degree,
        input.roi,
    )
}

fn apply_eye_bag_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = required(inputs.eye_bag, "aligned_color_field_rgb_f32")?;
    aurora_portrait::kernels::eye_bag::compose_field_image_pf90116(
        current,
        input.aligned_color_field_rgb_f32,
        input.aligned_response_mask,
        degree,
    )
}

fn apply_smile_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let material = required(inputs.smile.material, "pf100007 material")?;
    let (active, degree) =
        smile_warp::pf100007_gate(degree * 100.0, Some(material)).map_err(smile_refusal)?;
    if !active {
        return Ok(current.clone());
    }
    let lip_mask = required(inputs.smile.lip_mask, "pf100007 lip_mask")?;
    let mouth_mask = required(inputs.smile.mouth_mask, "pf100007 mouth_mask")?;
    let model_dx = required(inputs.smile.model_dx, "pf100007 model_dx")?;
    let model_dy = required(inputs.smile.model_dy, "pf100007 model_dy")?;
    let roi = required(inputs.smile.roi, "pf100007 roi")?;
    let pixels = smile_warp::pf100007_apply(
        &current.pixels,
        current.width,
        current.height,
        degree,
        lip_mask,
        mouth_mask,
        model_dx,
        model_dy,
        roi,
    )
    .map_err(smile_refusal)?;
    Image::new(current.width, current.height, pixels).ok_or(Refusal::ShapeMismatch("pf100007 bgra"))
}

fn smile_refusal(error: smile_warp::Refusal) -> Refusal {
    match error {
        smile_warp::Refusal::MissingInput(name) => Refusal::MissingInput(name),
        smile_warp::Refusal::ShapeMismatch(name) => Refusal::ShapeMismatch(name),
        smile_warp::Refusal::ValueOutOfRange(name) => Refusal::ValueOutOfRange(name),
        smile_warp::Refusal::VendorConsumerOpen(pf) => Refusal::NotImplemented(pf),
    }
}

fn resolve_suit_passes<'a>(
    passes: Option<&'a [SuitPassInput<'a>]>,
) -> Result<Vec<SuitPass<'a>>, Refusal> {
    let passes = required(passes, "suit passes")?;
    if passes.is_empty() {
        return Err(Refusal::MissingInput("suit passes"));
    }
    let mut resolved = Vec::with_capacity(passes.len());
    for pass in passes {
        let mask = required(pass.mask, "suit mask")?;
        let mask_uv = required(pass.mask_uv, "suit mask uv")?;
        let layer = required(pass.layer, "suit layer")?;
        let kind = match &pass.kind {
            SuitPassKindInput::Color => SuitPassKind::Color,
            SuitPassKindInput::Image {
                material: Some(material),
                material_uv: Some(material_uv),
            } => SuitPassKind::Image {
                material,
                material_uv,
            },
            SuitPassKindInput::Image { material: None, .. } => {
                return Err(Refusal::MissingInput("suit material"));
            }
            SuitPassKindInput::Image {
                material_uv: None, ..
            } => return Err(Refusal::MissingInput("suit material uv")),
            SuitPassKindInput::Unavailable(name) => {
                return Err(Refusal::MissingInput(name));
            }
        };
        resolved.push(SuitPass {
            kind,
            mask,
            mask_uv,
            layer,
            shelter: pass.shelter,
            region: pass.region,
        });
    }
    Ok(resolved)
}

fn fold_suit_layer(layer: MakeupLayer, shelter: f32, region: f32) -> Result<MakeupLayer, Refusal> {
    if !shelter.is_finite() || !(0.0..=1.0).contains(&shelter) {
        return Err(Refusal::ValueOutOfRange("suit shelter"));
    }
    if !region.is_finite() || !(0.0..=1.0).contains(&region) {
        return Err(Refusal::ValueOutOfRange("suit region"));
    }
    let degree = layer.degree01 * shelter * region;
    check_degree01(degree, "suit degree")?;
    Ok(MakeupLayer {
        degree01: degree,
        ..layer
    })
}

fn apply_cpu_suit_pass(current: &Image, pass: &SuitPass<'_>) -> Result<Image, Refusal> {
    let layer = fold_suit_layer(pass.layer, pass.shelter, pass.region)?;
    match &pass.kind {
        SuitPassKind::Color => color_makeup(current, pass.mask, pass.mask_uv, layer),
        SuitPassKind::Image {
            material,
            material_uv,
        } => image_makeup(
            current,
            material,
            pass.mask,
            pass.mask_uv,
            material_uv,
            layer,
        ),
    }
}

#[cfg(windows)]
fn apply_gpu_suit_pass(
    current: &Image,
    pass: &SuitPass<'_>,
    gpu: &aurora_portrait::cuda_makeup::MakeupLayer,
) -> Result<(Image, Vec<Dispatch>), CudaError> {
    let layer =
        fold_suit_layer(pass.layer, pass.shelter, pass.region).map_err(CudaError::Refused)?;
    let params = layer.params_of().map_err(CudaError::Refused)?;
    match &pass.kind {
        SuitPassKind::Image {
            material,
            material_uv,
        } => gpu.apply_image_layer_dual(
            current,
            material,
            pass.mask,
            pass.mask_uv,
            material_uv,
            params,
        ),
        SuitPassKind::Color => {
            let [r, g, b, _] = layer.mix_color;
            let material = Image::solid(1, 1, [r, g, b, 255]);
            let flat_uv = vec![[0.5f32, 0.5f32]; current.pixel_count()];
            gpu.apply_image_layer_dual(
                current,
                &material,
                pass.mask,
                pass.mask_uv,
                &flat_uv,
                params,
            )
        }
    }
}

fn apply_suit_effect(
    current: &Image,
    passes: Option<&[SuitPassInput<'_>]>,
) -> (Option<Image>, PortraitEffectRun) {
    let mut run = PortraitEffectRun {
        pf: 90033,
        backend: PortraitBackend::Cpu,
        dispatches: Vec::new(),
        fallback_reason: None,
        skipped: true,
        executed: false,
        refusal: None,
        skip_reason: None,
    };
    let resolved = match resolve_suit_passes(passes) {
        Ok(passes) => passes,
        Err(refusal) => {
            run.fallback_reason = Some(format!("input refused: {refusal:?}"));
            run.refusal = Some(refusal);
            return (None, run);
        }
    };
    let mut next = current.clone();
    let mut reasons = Vec::new();
    let mut all_cuda = true;

    #[cfg(windows)]
    let gpu = match forced_cuda_failure() {
        Some(error) => {
            reasons.push(format!("{error:?}"));
            None
        }
        None => match aurora_portrait::cuda_makeup::MakeupLayer::open() {
            Ok(gpu) => Some(gpu),
            Err(error) => {
                reasons.push(format!("{error:?}"));
                None
            }
        },
    };

    for (index, pass) in resolved.iter().enumerate() {
        #[cfg(windows)]
        if let Some(gpu) = gpu.as_ref() {
            match apply_gpu_suit_pass(&next, pass, gpu) {
                Ok((image, dispatches)) if !dispatches.is_empty() => {
                    next = image;
                    run.dispatches.extend(dispatches);
                    continue;
                }
                Ok(_) => reasons.push(format!("pass {index}: CUDA returned no dispatch")),
                Err(error) => reasons.push(format!("pass {index}: {error:?}")),
            }
        }
        all_cuda = false;
        match apply_cpu_suit_pass(&next, pass) {
            Ok(image) => next = image,
            Err(refusal) => {
                run.refusal = Some(refusal.clone());
                reasons.push(format!("pass {index} CPU: {refusal:?}"));
                run.fallback_reason = (!reasons.is_empty()).then(|| reasons.join("; "));
                return (None, run);
            }
        }
    }
    #[cfg(not(windows))]
    {
        all_cuda = false;
    }

    run.skipped = false;
    run.executed = true;
    run.backend = if all_cuda {
        PortraitBackend::Cuda
    } else {
        PortraitBackend::Cpu
    };
    run.fallback_reason = (!reasons.is_empty()).then(|| reasons.join("; "));
    (Some(next), run)
}

#[cfg(all(windows, test))]
thread_local! {
    static FORCE_CUDA_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(windows)]
fn forced_cuda_failure() -> Option<CudaError> {
    #[cfg(test)]
    if FORCE_CUDA_FAILURE.with(std::cell::Cell::get) {
        return Some(CudaError::Unavailable("test-forced CUDA failure"));
    }
    None
}

#[derive(Clone, Copy)]
struct StrayHairJob<'a> {
    params: StrayHairParams,
    hairfly_v1: &'a Image,
    raw_mask: &'a [u8],
    person_mask: Option<&'a [u8]>,
    background_mask: Option<&'a [u8]>,
}

fn resolve_stray_hair<'a>(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<Option<StrayHairJob<'a>>, Refusal> {
    check_degree01(degree, "PF90088_UI")?;
    if degree == 0.0 {
        return Ok(None);
    }
    let source = &inputs.stray_hair;
    let native_degree = required(source.native_degree, "PF90088_NATIVE_DEGREE")?;
    check_degree01(native_degree, "PF90088_NATIVE_DEGREE")?;
    let native_person_area =
        required(source.native_person_area, "PF90088_NATIVE_PERSON_AREA")?;
    check_degree01(native_person_area, "PF90088_NATIVE_PERSON_AREA")?;
    let native_background_area =
        required(source.native_background_area, "PF90088_NATIVE_BACKGROUND_AREA")?;
    check_degree01(
        native_background_area,
        "PF90088_NATIVE_BACKGROUND_AREA",
    )?;
    if native_degree == 0.0 {
        return Ok(None);
    }

    let hairfly_v1 = required(source.hairfly_v1, "PF90088_HAIRFLY_V1")?;
    hairfly_v1
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("PF90088_HAIRFLY_V1"))?;
    if (hairfly_v1.width, hairfly_v1.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF90088_HAIRFLY_V1"));
    }
    let raw_mask = required(source.raw_mask, "PF90088_RAW_STRAY_HAIR_MASK")?;
    check_plane(
        raw_mask.len(),
        current.pixel_count(),
        "PF90088_RAW_STRAY_HAIR_MASK",
    )?;
    if let Some(mask) = source.person_mask {
        check_plane(mask.len(), current.pixel_count(), "PF90088_PERSON_MASK")?;
    }
    if let Some(mask) = source.background_mask {
        check_plane(
            mask.len(),
            current.pixel_count(),
            "PF90088_BACKGROUND_MASK",
        )?;
    }
    if raw_mask.iter().all(|&value| value == 0) {
        return Ok(None);
    }
    Ok(Some(StrayHairJob {
        params: StrayHairParams {
            ui: degree * 100.0,
            native_degree,
            native_person_area,
            native_background_area,
        },
        hairfly_v1,
        raw_mask,
        person_mask: source.person_mask,
        background_mask: source.background_mask,
    }))
}

#[cfg(windows)]
fn try_cuda_same_job_effect(
    current: &Image,
    pf: u32,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Option<Result<(Image, Vec<Dispatch>), CudaError>> {
    if let Some(error) = forced_cuda_failure() {
        return Some(Err(error));
    }
    match pf {
        100 | 103 | 105 | 111 | 117 | 118 | 30007 => {
            let (_, input) =
                find_makeup_input(pf, inputs).expect("validated makeup same-job input");
            let gpu = match aurora_portrait::cuda_makeup::MakeupLayer::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply_route(
                pf,
                current,
                RouteInputs {
                    mix_color: input.mix_color,
                    opacity_percent: input.opacity_percent,
                    degree01: degree,
                    filter_type: input.filter_type,
                    material: input.material,
                    material_uv: input.material_uv,
                    mask: input.mask,
                    mask_uv: input.mask_uv,
                },
            ))
        }
        108 => {
            let teeth = inputs.teeth.expect("validated PF108 teeth input");
            let whiten = inputs
                .assets
                .teeth_whiten_lut
                .expect("validated PF108 whiten LUT");
            let bright = inputs
                .assets
                .teeth_bright_lut
                .expect("validated PF108 bright LUT");
            let gpu = match aurora_portrait::cuda_teeth::Teeth::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply(
                current,
                teeth.mask_g,
                whiten,
                bright,
                degree * 100.0,
                teeth.bright_ui,
                teeth.deyellow_ui,
            ))
        }
        21026 => {
            let beard = inputs.beard.expect("validated PF21026 beard input");
            let gpu = match aurora_portrait::cuda_beard::Beard::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply(
                current,
                beard.repair_rgb,
                beard.mask_a,
                beard.mask_b,
                degree.clamp(0.0, 1.0),
            ))
        }
        90002 => {
            let input = inputs
                .moisture_face
                .expect("validated PF90002 moisture input");
            let params = FaceParams {
                ui: degree * 100.0,
                pivot: input.pivot,
                gain: input.gain,
            };
            let skin_map = SkinMap {
                width: input.skin_map_width as usize,
                height: input.skin_map_height as usize,
                samples: input.skin_map,
            };
            let gpu = match aurora_portrait::cuda_moisture::Moisture::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.face_apply(
                current,
                input.mask_rgba,
                input.blur_y,
                &params,
                input.channel,
                &skin_map,
                &Lut1D {
                    samples: input.hardlight,
                },
                &Lut1D {
                    samples: input.beauty,
                },
            ))
        }
        90010 => {
            let input = inputs
                .moisture_body
                .expect("validated PF90010 moisture input");
            let native_degree = if degree > 0.0 { input.degree } else { 0.0 };
            let gpu = match aurora_portrait::cuda_moisture::Moisture::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.body_apply(
                current,
                input.weight_skin,
                input.weight_dog,
                &Lut1D {
                    samples: input.beauty,
                },
                native_degree,
                input.roi,
            ))
        }
        90116 => {
            let input = inputs.eye_bag.expect("validated PF90116 eye-bag input");
            let gpu = match aurora_portrait::cuda_eye_bag::EyeBag::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.compose_field(
                current,
                input.aligned_color_field_rgb_f32,
                input.aligned_response_mask,
                degree,
            ))
        }
        90139 => {
            let (hair, roi) =
                resolve_hair_smooth(current, degree, inputs).expect("validated PF90139 inputs");
            let gpu = match HairDetail::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.smooth(
                    current,
                    roi,
                    SmoothParams {
                        mix_strength: hair.mix_strength,
                        detail_strength: degree,
                    },
                    Some(SmoothProducer {
                        ticket: hair.ticket,
                        hair_mask: hair.hair_mask.expect("validated PF90139 hair mask"),
                        mix_mask: hair.mix_mask.expect("validated PF90139 mix mask"),
                        blend_rgb: hair.blend_rgb.expect("validated PF90139 blend RGB8"),
                        detail: hair.detail.expect("validated PF90139 detail"),
                        hair_stride: hair.hair_stride,
                        mix_stride: hair.mix_stride,
                        blend_stride: hair.blend_stride,
                        detail_stride: hair.detail_stride,
                    }),
                )
                .map(|(image, dispatch)| (image, dispatch.into_iter().collect())),
            )
        }
        90160 => {
            let (hair, roi) =
                resolve_hair_gloss(current, degree, inputs).expect("validated PF90160 inputs");
            let gpu = match HairDetail::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.gloss(
                    current,
                    roi,
                    hair.dynamic_scale,
                    Some(GlossProducer {
                        ticket: hair.ticket,
                        hair_mask: hair.hair_mask.expect("validated PF90160 hair mask"),
                        model_detail: hair.model_detail.expect("validated PF90160 model detail"),
                        hair_stride: hair.hair_stride,
                        detail_stride: hair.detail_stride,
                    }),
                )
                .map(|(image, dispatch)| (image, dispatch.into_iter().collect())),
            )
        }
        90088 => {
            let job = match resolve_stray_hair(current, degree, inputs) {
                Ok(Some(job)) => job,
                Ok(None) => return Some(Ok((current.clone(), Vec::new()))),
                Err(error) => return Some(Err(CudaError::Refused(error))),
            };
            let gpu = match aurora_portrait::cuda_stray_hair::Blend::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply(
                current,
                job.params,
                Some(job.hairfly_v1),
                Some(job.raw_mask),
                job.person_mask,
                job.background_mask,
            ))
        }
        100007 => {
            let material = inputs.smile.material.expect("validated PF100007 material");
            let (active, degree) = match smile_warp::pf100007_gate(degree * 100.0, Some(material)) {
                Ok(value) => value,
                Err(error) => return Some(Err(CudaError::Refused(smile_refusal(error)))),
            };
            if !active {
                return Some(Ok((current.clone(), Vec::new())));
            }
            let lip_mask = inputs.smile.lip_mask.expect("validated PF100007 lip mask");
            let mouth_mask = inputs
                .smile
                .mouth_mask
                .expect("validated PF100007 mouth mask");
            let model_dx = inputs.smile.model_dx.expect("validated PF100007 dx");
            let model_dy = inputs.smile.model_dy.expect("validated PF100007 dy");
            let roi = inputs.smile.roi.expect("validated PF100007 roi");
            let gpu = match aurora_portrait::cuda_smile::Smile::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.apply(
                    &current.pixels,
                    current.width,
                    current.height,
                    degree,
                    lip_mask,
                    mouth_mask,
                    model_dx,
                    model_dy,
                    roi,
                )
                .and_then(|(pixels, dispatches)| {
                    let image = Image::new(current.width, current.height, pixels)
                        .ok_or(CudaError::Refused(Refusal::ShapeMismatch("pf100007 bgra")))?;
                    Ok((image, dispatches))
                }),
            )
        }
        _ => None,
    }
}

#[cfg(windows)]
fn try_cuda_effect(
    current: &Image,
    pf: u32,
    degree: f32,
    skin_mask: Option<&[u8]>,
) -> Option<Result<(Image, Vec<Dispatch>), CudaError>> {
    if let Some(error) = forced_cuda_failure() {
        return Some(Err(error));
    }
    if !matches!(pf, 8068 | 8069 | 8070 | 21015 | 90005 | 90007) {
        return None;
    }
    let mask = match skin_mask {
        Some(mask) => mask,
        None => {
            return Some(Err(CudaError::Refused(
                aurora_portrait::Refusal::MissingInput("skin mask"),
            )));
        }
    };
    match pf {
        8069 => {
            let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
            if ui >= 0.0 {
                return None;
            }
            let extra = vec![255u8; mask.len()];
            let gpu = match aurora_portrait::cuda_skin_models::SkinModels::open() {
                Ok(gpu) => gpu,
                Err(err) => return Some(Err(err)),
            };
            Some(gpu.texture_apply(current, mask, &extra, ui))
        }
        8070 => {
            let base = current.clone();
            let blur = blur_rgba(current, 2.0);
            let gpu = match aurora_portrait::cuda_skin_models::SkinModels::open() {
                Ok(gpu) => gpu,
                Err(err) => return Some(Err(err)),
            };
            Some(gpu.clarity_apply(
                current,
                &base,
                &blur,
                mask,
                (0, 0, current.width, current.height),
                degree.clamp(0.0, 1.0),
                0.0,
                4.68,
            ))
        }
        21015 => {
            let (Some(green), Some(magenta)) = (tint_green(), tint_magenta()) else {
                return Some(Err(CudaError::Refused(Refusal::MissingInput(
                    "skin tint LUT",
                ))));
            };
            let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
            Some(with_skin_lut(|gpu| {
                gpu.tint(current, mask, green, magenta, ui)
                    .map(|(image, dispatch)| (image, vec![dispatch]))
            }))
        }
        8068 => {
            let extra = vec![255u8; mask.len()];
            let gpu = match aurora_portrait::cuda_gloss::Gloss::open() {
                Ok(gpu) => gpu,
                Err(err) => return Some(Err(err)),
            };
            Some(gpu.apply(
                current,
                [&mask, &mask, &mask],
                [&extra, &extra, &extra],
                (degree * 100.0).clamp(0.0, 100.0),
            ))
        }
        90005 => {
            let Some(lut) = whiten_lut() else {
                return Some(Err(CudaError::Refused(
                    aurora_portrait::Refusal::MissingInput("whiten LUT"),
                )));
            };
            Some(with_skin_lut(|gpu| {
                gpu.skin_adjust(current, &mask, lut, degree.clamp(0.0, 1.0), pf)
                    .map(|(image, dispatch)| (image, vec![dispatch]))
            }))
        }
        90007 => {
            let Some(lut) = ruddy_lut() else {
                return Some(Err(CudaError::Refused(
                    aurora_portrait::Refusal::MissingInput("ruddy LUT"),
                )));
            };
            Some(with_skin_lut(|gpu| {
                gpu.skin_adjust(current, &mask, lut, degree.clamp(0.0, 1.0), pf)
                    .map(|(image, dispatch)| (image, vec![dispatch]))
            }))
        }
        _ => None,
    }
}

#[cfg(windows)]
thread_local! {
    static SKIN_LUT: RefCell<Option<aurora_portrait::cuda_lut::SkinLut>> =
        const { RefCell::new(None) };
}

#[cfg(windows)]
fn with_skin_lut<T>(
    run: impl FnOnce(&aurora_portrait::cuda_lut::SkinLut) -> Result<T, CudaError>,
) -> Result<T, CudaError> {
    SKIN_LUT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(aurora_portrait::cuda_lut::SkinLut::open()?);
        } else if let Err(error) = slot.as_mut().expect("skin LUT session").resume() {
            *slot = None;
            return Err(error);
        }
        let result = run(slot.as_ref().expect("skin LUT session"));
        if result.is_err() {
            let _ = slot.as_mut().expect("skin LUT session").suspend();
            *slot = None;
            return result;
        }
        match slot.as_mut().expect("skin LUT session").suspend() {
            Ok(()) => result,
            Err(error) => {
                *slot = None;
                if result.is_ok() {
                    Err(error)
                } else {
                    result
                }
            }
        }
    })
}

fn decode_lut(png: &[u8], name: &'static str) -> Option<Vec<u8>> {
    let image = image::load_from_memory(png).ok()?.into_rgb8();
    if image.width() as usize != LUT_EDGE || image.height() as usize != LUT_EDGE {
        log::warn!(
            "portrait LUT {name}: expected {LUT_EDGE}x{LUT_EDGE}, got {}x{}",
            image.width(),
            image.height()
        );
        return None;
    }
    let bytes = image.into_raw();
    (bytes.len() == LUT_BYTES).then_some(bytes)
}

fn tint_green() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/TintGreen.png"
            )),
            "TintGreen.png",
        )
    })
    .as_deref()
}

fn tint_magenta() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/TintMagenta.png"
            )),
            "TintMagenta.png",
        )
    })
    .as_deref()
}

fn whiten_lut() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/whiten.png"
            )),
            "whiten.png",
        )
    })
    .as_deref()
}

fn ruddy_lut() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/ruddy_new.png"
            )),
            "ruddy_new.png",
        )
    })
    .as_deref()
}

fn cold_lut() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/cold.png"
            )),
            "cold.png",
        )
    })
    .as_deref()
}

fn warm_lut() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/warm.png"
            )),
            "warm.png",
        )
    })
    .as_deref()
}

fn apply_skin_tint(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let (Some(green), Some(magenta)) = (tint_green(), tint_magenta()) else {
        return Err(aurora_portrait::Refusal::MissingInput("skin tint LUT"));
    };
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    let signed = ui.clamp(-100.0, 100.0) / 100.0;
    let lut = if signed <= 0.0 { green } else { magenta };
    aurora_portrait::kernels::makeup::skin_adjust_lut(current, mask, lut, signed.abs())
}

fn apply_skin_whiten(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(lut) = whiten_lut() else {
        return Err(aurora_portrait::Refusal::MissingInput("whiten LUT"));
    };
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    aurora_portrait::kernels::makeup::skin_adjust_lut(current, mask, lut, degree.clamp(0.0, 1.0))
}

fn apply_skin_temperature(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let (Some(cold), Some(warm)) = (cold_lut(), warm_lut()) else {
        return Err(aurora_portrait::Refusal::MissingInput(
            "skin temperature LUT",
        ));
    };
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    aurora_portrait::kernels::face_detail::skin_temperature(current, mask, cold, warm, ui)
}

fn apply_skin_ruddy(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(lut) = ruddy_lut() else {
        return Err(aurora_portrait::Refusal::MissingInput("ruddy LUT"));
    };
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    aurora_portrait::kernels::makeup::skin_adjust_lut(current, mask, lut, degree.clamp(0.0, 1.0))
}

fn apply_skin_gloss(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let extra = vec![255u8; mask.len()];
    let native = (degree * 100.0).clamp(0.0, 100.0);
    aurora_portrait::kernels::skin_models::gloss_apply(
        current,
        [&mask, &mask, &mask],
        [&extra, &extra, &extra],
        native,
    )
}

fn apply_skin_texture(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    if ui >= 0.0 {
        return Ok(current.clone());
    }
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let extra = vec![255u8; mask.len()];
    aurora_portrait::kernels::skin_models::texture_surface_blur(current, &mask, &extra, ui)
}

fn apply_skin_clarity(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
    width: u32,
    height: u32,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let base = current.clone();
    let blur = blur_rgba(current, 2.0);
    aurora_portrait::kernels::skin_models::clarity_apply(
        current,
        &base,
        &blur,
        &mask,
        (0, 0, width, height),
        degree.clamp(0.0, 1.0),
        0.0,
        4.68,
    )
}

fn blur_rgba(src: &Image, sigma: f32) -> Image {
    let rgba = image::RgbaImage::from_raw(src.width, src.height, src.pixels.clone())
        .expect("image dimensions match its pixel buffer");
    let out = image::imageops::blur(&rgba, sigma);
    Image::new(src.width, src.height, out.into_raw())
        .expect("blur output keeps the source dimensions")
}

struct RadianceAssets {
    lut_a: Vec<u8>,
    lut_b: Vec<u8>,
    gamma: Vec<f32>,
    matrices: Vec<f32>,
}

fn f32_asset(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn radiance_assets() -> &'static RadianceAssets {
    static ASSETS: OnceLock<RadianceAssets> = OnceLock::new();
    ASSETS.get_or_init(|| RadianceAssets {
        lut_a: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/portrait/radiance/native-lut-a.bin"
        ))
        .to_vec(),
        lut_b: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/portrait/radiance/native-lut-b.bin"
        ))
        .to_vec(),
        gamma: f32_asset(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/portrait/radiance/native-gamma-rgba32f.bin"
        ))),
        matrices: f32_asset(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/portrait/radiance/native-color-matrices.bin"
        ))),
    })
}

fn apply_skin_radiance(
    current: &Image,
    degree: f32,
    skin_mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(skin) = skin_mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let parsing_rgba = vec![0u8; skin.len() * 4];
    let face_rgba = vec![0u8; skin.len() * 4];
    let planes = aurora_portrait::kernels::radiance::MaskPlanes {
        skin: &skin,
        parsing_rgba: &parsing_rgba,
        face_rgba: &face_rgba,
    };
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for (index, &m) in skin.iter().enumerate() {
        if m > 32 {
            let i = index * 4;
            let lum = 0.2126 * current.pixels[i] as f32
                + 0.7152 * current.pixels[i + 1] as f32
                + 0.0722 * current.pixels[i + 2] as f32;
            sum += lum;
            count += 1;
        }
    }
    let value_mean = if count > 0 {
        sum / count as f32
    } else {
        return Ok(current.clone());
    };
    let assets = radiance_assets();
    let fixed = aurora_portrait::kernels::radiance::FixedAssets {
        lut_a: &assets.lut_a,
        lut_b: &assets.lut_b,
        gamma: &assets.gamma,
        matrices: &assets.matrices,
    };
    let ui = (degree * 100.0).clamp(0.0, 100.0);
    Ok(aurora_portrait::kernels::radiance::apply(current, &planes, ui, value_mean, &fixed)?.image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32) -> Vec<u8> {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let v = ((x * 7 + y * 13) % 24) as u8;
                rgba[i] = 190 + v;
                rgba[i + 1] = 150 + v;
                rgba[i + 2] = 110 + v;
                rgba[i + 3] = 255;
            }
        }
        rgba
    }

    fn apply(rgba: &mut Vec<u8>, w: u32, h: u32, extra: &[(&str, f32)]) {
        let map: BTreeMap<String, f32> = extra.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        let _ = apply_portrait_effects_with_job(rgba, w, h, NormRect::FULL, &map, &inputs);
    }

    fn apply_one_with_inputs(
        pf: u32,
        width: u32,
        height: u32,
        inputs: &PortraitJobInputs<'_>,
    ) -> (Vec<u8>, PortraitEffectRun) {
        let mut rgba = frame(width, height);
        let extra = BTreeMap::from([(format!("pc:{pf}"), 0.8f32)]);
        let report = apply_portrait_effects_with_job(
            &mut rgba,
            width,
            height,
            NormRect::FULL,
            &extra,
            inputs,
        );
        assert_eq!(report.effects.len(), 1, "PF{pf}");
        (rgba, report.effects.into_iter().next().unwrap())
    }

    fn strided_plane(
        values: &[u8],
        width: u32,
        height: u32,
        stride: u32,
        bytes_per_pixel: u32,
    ) -> Vec<u8> {
        let mut plane = vec![0xA5; stride as usize * height as usize];
        let row_bytes = width as usize * bytes_per_pixel as usize;
        for y in 0..height as usize {
            let src = y * row_bytes;
            let dst = y * stride as usize;
            plane[dst..dst + row_bytes].copy_from_slice(&values[src..src + row_bytes]);
        }
        plane
    }

    fn hair_frame(width: u32, height: u32) -> Vec<u8> {
        let count = width as usize * height as usize;
        (0..count)
            .flat_map(|index| {
                [
                    (24 + index * 13 % 180) as u8,
                    (35 + index * 17 % 170) as u8,
                    (46 + index * 19 % 160) as u8,
                    (31 + index * 23 % 200) as u8,
                ]
            })
            .collect()
    }

    fn stray_hair_fixture() -> (Image, Image, Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut source = Vec::with_capacity(5 * 4 * 4);
        let mut repair = Vec::with_capacity(5 * 4 * 4);
        for pixel in 0..20u32 {
            source.extend_from_slice(&[
                (pixel * 13 + 2) as u8,
                (pixel * 11 + 20) as u8,
                (pixel * 7 + 30) as u8,
                (17 + pixel * 9) as u8,
            ]);
            repair.extend_from_slice(&[
                (220 - pixel * 3) as u8,
                (220 - pixel * 5) as u8,
                (130 + pixel * 2) as u8,
                (250 - pixel) as u8,
            ]);
        }
        source[0..4].copy_from_slice(&[2, 20, 30, 17]);
        repair[0..4].copy_from_slice(&[3, 220, 130, 250]);
        (
            Image::new(5, 4, source).unwrap(),
            Image::new(5, 4, repair).unwrap(),
            vec![
                255, 128, 0, 255, 64, 200, 17, 33, 99, 245, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            ],
            vec![
                255, 128, 255, 0, 200, 64, 0, 255, 12, 25, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255,
            ],
            vec![
                0, 255, 255, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                150, 160,
            ],
        )
    }

    fn stray_hair_stage_inputs<'a>(
        masks: &'a PortraitMasks,
        repair: &'a Image,
        raw: &'a [u8],
        person: &'a [u8],
        background: &'a [u8],
    ) -> PortraitJobInputs<'a> {
        PortraitJobInputs {
            stray_hair: StrayHairInputs {
                hairfly_v1: Some(repair),
                raw_mask: Some(raw),
                person_mask: Some(person),
                background_mask: Some(background),
                native_degree: Some(0.5),
                native_person_area: Some(1.0),
                native_background_area: Some(0.5),
            },
            ..PortraitJobInputs::from_renderer_job(557, masks)
        }
    }

    fn run_stray_hair_stage(
        source: &Image,
        inputs: &PortraitJobInputs<'_>,
        force_cpu: bool,
    ) -> (Vec<u8>, PortraitEffectRun) {
        let mut actual = source.pixels.clone();
        let extra = BTreeMap::from([("pc:90088".to_string(), 0.8f32)]);
        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);
        #[cfg(not(windows))]
        let _ = force_cpu;
        let report = apply_portrait_effects_with_job(
            &mut actual,
            source.width,
            source.height,
            NormRect::FULL,
            &extra,
            inputs,
        );
        assert_eq!(report.effects.len(), 1);
        (actual, report.effects.into_iter().next().unwrap())
    }

    fn hair_smooth_cpu(
        source: &Image,
        roi: HairRoi,
        detail_strength: f32,
        input: &HairSmoothInputs<'_>,
    ) -> Vec<u8> {
        let mut output = source.pixels.clone();
        for ry in 0..roi.height as usize {
            for rx in 0..roi.width as usize {
                let hair =
                    input.hair_mask.unwrap()[ry * input.hair_stride as usize + rx];
                if hair == 0 {
                    continue;
                }
                let mix = input.mix_mask.unwrap()[ry * input.mix_stride as usize + rx];
                let detail = input.detail.unwrap()[ry * input.detail_stride as usize + rx];
                let gate = hair as f32 * 0.003921568859368563;
                let mix_weight =
                    (mix as f32 * 0.003921568859368563) * input.mix_strength * gate;
                let detail_add =
                    ((detail as f32 - 127.5) * 4.0) * (gate * detail_strength * 0.8);
                let x = roi.x as usize + rx;
                let y = roi.y as usize + ry;
                let offset = (y * source.width as usize + x) * 4;
                let blend_offset =
                    ry * input.blend_stride as usize + rx * 3;
                for channel in 0..3 {
                    let src = source.pixels[offset + channel] as f32;
                    let base = if mix_weight > 0.0 {
                        let blend = input.blend_rgb.unwrap()[blend_offset + channel] as f32;
                        src * (1.0 - mix_weight) + blend * mix_weight
                    } else {
                        src
                    };
                    output[offset + channel] = (base + detail_add)
                        .round_ties_even()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
        output
    }

    fn hair_gloss_cpu(
        source: &Image,
        roi: HairRoi,
        input: &HairGlossInputs<'_>,
    ) -> Vec<u8> {
        let mut output = source.pixels.clone();
        for ry in 0..roi.height as usize {
            for rx in 0..roi.width as usize {
                let hair =
                    input.hair_mask.unwrap()[ry * input.hair_stride as usize + rx];
                if hair == 0 {
                    continue;
                }
                let detail =
                    input.model_detail.unwrap()[ry * input.detail_stride as usize + rx];
                let delta =
                    (detail as f32 - 127.5) * 2.0 * hair as f32 * 0.003921568859368563
                        * input.dynamic_scale;
                let x = roi.x as usize + rx;
                let y = roi.y as usize + ry;
                let offset = (y * source.width as usize + x) * 4;
                for channel in 0..3 {
                    output[offset + channel] =
                        (source.pixels[offset + channel] as f32 + delta)
                            .clamp(0.0, 255.0) as u8;
                }
            }
        }
        output
    }

    #[test]
    fn hair_stage_refuses_each_missing_plane_before_launch() {
        let masks = PortraitMasks::solid(1.0);
        let (width, height) = (7u32, 5u32);
        let (roi_x, roi_y, roi_width, roi_height) = (2u32, 1u32, 3u32, 2u32);
        let roi = HairRoi {
            x: roi_x,
            y: roi_y,
            width: roi_width,
            height: roi_height,
        };
        let hair = strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
        let mix = strided_plane(&[180, 240, 64, 32, 128, 255], roi_width, roi_height, 4, 1);
        let blend = strided_plane(
            &[20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 10, 30, 50, 70, 90, 110],
            roi_width,
            roi_height,
            14,
            3,
        );
        let detail =
            strided_plane(&[96, 127, 128, 160, 255, 0], roi_width, roi_height, 6, 1);
        let base = HairSmoothInputs {
            ticket: 17,
            width,
            height,
            roi: (roi_x, roi_y, roi_width, roi_height),
            mix_strength: 0.6,
            hair_mask: Some(&hair),
            mix_mask: Some(&mix),
            blend_rgb: Some(&blend),
            detail: Some(&detail),
            hair_stride: 5,
            mix_stride: 4,
            blend_stride: 14,
            detail_stride: 6,
        };
        let cases = [
            (
                HairSmoothInputs {
                    hair_mask: None,
                    ..base
                },
                "PF90139_HAIR_MASK",
            ),
            (
                HairSmoothInputs {
                    mix_mask: None,
                    ..base
                },
                "PF90139_MIX_MASK",
            ),
            (
                HairSmoothInputs {
                    blend_rgb: None,
                    ..base
                },
                "PF90139_BLEND_SOURCE_RGB8",
            ),
            (
                HairSmoothInputs {
                    detail: None,
                    ..base
                },
                "PF90139_DETAIL",
            ),
        ];
        for (hair_smooth, missing) in cases {
            let mut rgba = frame(width, height);
            let original = rgba.clone();
            let inputs = PortraitJobInputs {
                hair_smooth: Some(hair_smooth),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let extra = BTreeMap::from([("pc:90139".to_string(), 0.8)]);
            let report = apply_portrait_effects_with_job(
                &mut rgba,
                width,
                height,
                NormRect::FULL,
                &extra,
                &inputs,
            );
            assert_eq!(rgba, original);
            assert_eq!(
                report.effects[0].refusal,
                Some(Refusal::MissingInput(missing))
            );
            assert!(!report.effects[0].executed);
            assert!(report.effects[0].skipped);
            assert!(report.effects[0].dispatches.is_empty());
        }

        let gloss_hair =
            strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
        let gloss_detail =
            strided_plane(&[128, 127, 200, 64, 255, 0], roi_width, roi_height, 7, 1);
        let gloss = HairGlossInputs {
            ticket: 18,
            width,
            height,
            roi: (roi_x, roi_y, roi_width, roi_height),
            dynamic_scale: 0.5,
            hair_mask: Some(&gloss_hair),
            model_detail: Some(&gloss_detail),
            hair_stride: 5,
            detail_stride: 7,
        };
        for (hair_gloss, missing) in [
            (
                HairGlossInputs {
                    hair_mask: None,
                    ..gloss
                },
                "PF90160_HAIR_MASK",
            ),
            (
                HairGlossInputs {
                    model_detail: None,
                    ..gloss
                },
                "PF90160_MODEL_DETAIL",
            ),
        ] {
            let mut rgba = frame(width, height);
            let original = rgba.clone();
            let inputs = PortraitJobInputs {
                hair_gloss: Some(hair_gloss),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let extra = BTreeMap::from([("pc:90160".to_string(), 0.8)]);
            let report = apply_portrait_effects_with_job(
                &mut rgba,
                width,
                height,
                NormRect::FULL,
                &extra,
                &inputs,
            );
            assert_eq!(rgba, original);
            assert_eq!(
                report.effects[0].refusal,
                Some(Refusal::MissingInput(missing))
            );
            assert!(!report.effects[0].executed);
            assert!(report.effects[0].skipped);
            assert!(report.effects[0].dispatches.is_empty());
        }
    }

    #[cfg(windows)]
    #[test]
    fn hair_stage_matches_cpu_and_keeps_roi_and_alpha() {
        let masks = PortraitMasks::solid(1.0);
        let (width, height) = (7u32, 5u32);
        let source = Image::new(width, height, hair_frame(width, height)).unwrap();
        let (roi_x, roi_y, roi_width, roi_height) = (2u32, 1u32, 3u32, 2u32);
        let roi = HairRoi {
            x: roi_x,
            y: roi_y,
            width: roi_width,
            height: roi_height,
        };
        let hair = strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
        let mix = strided_plane(&[180, 240, 64, 32, 128, 255], roi_width, roi_height, 4, 1);
        let blend = strided_plane(
            &[20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 10, 30, 50, 70, 90, 110],
            roi_width,
            roi_height,
            14,
            3,
        );
        let detail =
            strided_plane(&[96, 127, 128, 160, 255, 0], roi_width, roi_height, 6, 1);
        let smooth = HairSmoothInputs {
            ticket: 17,
            width,
            height,
            roi: (roi_x, roi_y, roi_width, roi_height),
            mix_strength: 0.6,
            hair_mask: Some(&hair),
            mix_mask: Some(&mix),
            blend_rgb: Some(&blend),
            detail: Some(&detail),
            hair_stride: 5,
            mix_stride: 4,
            blend_stride: 14,
            detail_stride: 6,
        };
        let expected_smooth = hair_smooth_cpu(&source, roi, 0.8, &smooth);
        let inputs = PortraitJobInputs {
            hair_smooth: Some(smooth),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90139".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        if run.backend != PortraitBackend::Cuda {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF90139 did not dispatch: {:?}",
                run.fallback_reason
            );
            return;
        }
        assert_eq!(actual, expected_smooth);
        assert_eq!(run.dispatches[0].kernel, "pf90139_hair_smooth_kernel");
        eprintln!(
            "PF90139 stage device={} kernel={}",
            run.dispatches[0].device, run.dispatches[0].kernel
        );

        let gloss_hair =
            strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
        let gloss_detail =
            strided_plane(&[128, 127, 200, 64, 255, 0], roi_width, roi_height, 7, 1);
        let gloss = HairGlossInputs {
            ticket: 18,
            width,
            height,
            roi: (roi_x, roi_y, roi_width, roi_height),
            dynamic_scale: 0.5,
            hair_mask: Some(&gloss_hair),
            model_detail: Some(&gloss_detail),
            hair_stride: 5,
            detail_stride: 7,
        };
        let expected_gloss = hair_gloss_cpu(&source, roi, &gloss);
        let inputs = PortraitJobInputs {
            hair_gloss: Some(gloss),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90160".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        assert_eq!(run.backend, PortraitBackend::Cuda, "{:?}", run.fallback_reason);
        assert_eq!(actual, expected_gloss);
        assert_eq!(run.dispatches[0].kernel, "pf90160_hair_gloss_kernel");
        eprintln!(
            "PF90160 stage device={} kernel={}",
            run.dispatches[0].device, run.dispatches[0].kernel
        );

        for y in 0..height {
            for x in 0..width {
                let offset = ((y * width + x) * 4) as usize;
                assert_eq!(actual[offset + 3], source.pixels[offset + 3]);
                if x < roi_x
                    || x >= roi_x + roi_width
                    || y < roi_y
                    || y >= roi_y + roi_height
                {
                    assert_eq!(
                        &actual[offset..offset + 4],
                        &source.pixels[offset..offset + 4]
                    );
                }
            }
        }
    }

    #[test]
    fn pf90088_stage_refuses_missing_same_job_inputs_before_dispatch() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let complete =
            stray_hair_stage_inputs(&masks, &repair, &raw, &person, &background);
        let cases = [
            (
                PortraitJobInputs {
                    stray_hair: StrayHairInputs {
                        hairfly_v1: None,
                        ..complete.stray_hair
                    },
                    ..PortraitJobInputs::from_renderer_job(557, &masks)
                },
                "PF90088_HAIRFLY_V1",
            ),
            (
                PortraitJobInputs {
                    stray_hair: StrayHairInputs {
                        raw_mask: None,
                        ..complete.stray_hair
                    },
                    ..PortraitJobInputs::from_renderer_job(557, &masks)
                },
                "PF90088_RAW_STRAY_HAIR_MASK",
            ),
            (
                PortraitJobInputs {
                    stray_hair: StrayHairInputs {
                        native_degree: None,
                        ..complete.stray_hair
                    },
                    ..PortraitJobInputs::from_renderer_job(557, &masks)
                },
                "PF90088_NATIVE_DEGREE",
            ),
        ];
        for (inputs, missing) in cases {
            let mut actual = source.pixels.clone();
            let report = apply_portrait_effects_with_job(
                &mut actual,
                source.width,
                source.height,
                NormRect::FULL,
                &BTreeMap::from([("pc:90088".to_string(), 0.8f32)]),
                &inputs,
            );
            assert_eq!(actual, source.pixels);
            assert_eq!(
                report.effects[0].refusal,
                Some(Refusal::MissingInput(missing))
            );
            assert!(report.effects[0].skipped);
            assert!(!report.effects[0].executed);
            assert!(report.effects[0].dispatches.is_empty());
        }
    }

    #[test]
    fn pf90088_stage_cpu_preserves_alpha_and_half_away_rounding() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let expected = aurora_portrait::kernels::hair_body::stray_hair(
            &source,
            80.0,
            0.5,
            1.0,
            0.5,
            Some(&repair),
            Some(&raw),
            Some(&person),
            Some(&background),
        )
        .unwrap();
        let inputs =
            stray_hair_stage_inputs(&masks, &repair, &raw, &person, &background);
        let (actual, run) = run_stray_hair_stage(&source, &inputs, true);
        assert_eq!(actual, expected.pixels);
        assert_eq!(&actual[0..4], &[3, 120, 80, 17]);
        for index in 0..source.pixel_count() {
            assert_eq!(actual[index * 4 + 3], source.pixels[index * 4 + 3]);
        }
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn pf90088_stage_cuda_matches_cpu_byte_for_byte() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let expected = aurora_portrait::kernels::hair_body::stray_hair(
            &source,
            80.0,
            0.5,
            1.0,
            0.5,
            Some(&repair),
            Some(&raw),
            Some(&person),
            Some(&background),
        )
        .unwrap();
        let inputs =
            stray_hair_stage_inputs(&masks, &repair, &raw, &person, &background);
        let (actual, run) = run_stray_hair_stage(&source, &inputs, false);
        if run.backend != PortraitBackend::Cuda {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF90088 did not dispatch: {:?}",
                run.fallback_reason
            );
            return;
        }
        assert_eq!(actual, expected.pixels);
        assert_eq!(run.dispatches[0].kernel, "pf90088_stray_hair_kernel");
        assert!(!run.dispatches[0].device.is_empty());
        for index in 0..source.pixel_count() {
            assert_eq!(actual[index * 4 + 3], source.pixels[index * 4 + 3]);
        }
        eprintln!(
            "PF90088 stage device={} kernel={} cc={:?} driver={} ptx_isa={:?}",
            run.dispatches[0].device,
            run.dispatches[0].kernel,
            run.dispatches[0].compute_cap,
            run.dispatches[0].driver_version,
            run.dispatches[0].ptx_isa
        );
    }

    #[cfg(windows)]
    struct ForceCudaFailureGuard;

    #[cfg(windows)]
    impl ForceCudaFailureGuard {
        fn new() -> Self {
            FORCE_CUDA_FAILURE.with(|flag| flag.set(true));
            Self
        }
    }

    #[cfg(windows)]
    impl Drop for ForceCudaFailureGuard {
        fn drop(&mut self) {
            FORCE_CUDA_FAILURE.with(|flag| flag.set(false));
        }
    }

    #[test]
    fn portrait_effects_do_not_crash_on_empty_extra() {
        let mut rgba = frame(16, 16);
        apply(&mut rgba, 16, 16, &[]);
        assert_eq!(rgba.len(), 16 * 16 * 4);
    }

    #[test]
    fn portrait_effects_skip_refused_rows() {
        let mut rgba = frame(16, 16);
        let original = rgba.clone();
        apply(&mut rgba, 16, 16, &[("pc:90001", 50.0)]);
        assert_eq!(rgba, original, "refused rows must not change pixels");
    }

    #[test]
    fn masks_only_entry_is_a_typed_skip_without_dispatch() {
        let masks = PortraitMasks::solid(1.0);
        let extra = BTreeMap::from([
            ("pc:108".to_string(), 0.8f32),
            ("pc:90139".to_string(), 0.8),
            ("pc:90005".to_string(), 0.9),
        ]);
        let mut rgba = frame(16, 16);
        let original = rgba.clone();
        let report = apply_portrait_effects_with_report(
            &mut rgba,
            16,
            16,
            NormRect::FULL,
            &extra,
            &masks,
        );

        assert_eq!(rgba, original, "the legacy entry must not execute effects");
        assert_eq!(report.job_id, None);
        assert_eq!(report.effects.len(), 3);
        for effect in &report.effects {
            assert_eq!(effect.backend, PortraitBackend::Skipped);
            assert!(effect.skipped, "PF{} must be marked skipped", effect.pf);
            assert!(!effect.executed, "PF{} must not execute", effect.pf);
            assert!(
                effect.dispatches.is_empty(),
                "PF{} must not report a fake dispatch",
                effect.pf
            );
            assert_eq!(
                effect.refusal.as_ref(),
                Some(&Refusal::MissingInput("same-job input bundle"))
            );
            assert_eq!(
                effect.skip_reason,
                Some(PortraitSkipReason::InputRefused(Refusal::MissingInput(
                    "same-job input bundle"
                )))
            );
        }
    }

    #[test]
    fn non_skin_rows_never_reuse_parser_masks() {
        let masks = PortraitMasks::solid(1.0);
        let cases = [
            (108u32, "teeth mask"),
            (30026, "PF30026 same-job inputs"),
            (21026, "mask_a"),
            (90021, "PF90021_SAME_JOB_INPUTS"),
            (90002, "moisture face inputs"),
            (90010, "moisture body inputs"),
            (90033, "suit passes"),
            (90116, "aligned_color_field_rgb_f32"),
            (100007, "pf100007 material"),
        ];
        for (pf, missing) in cases {
            let mut rgba = frame(16, 16);
            let original = rgba.clone();
            let extra = BTreeMap::from([(format!("pc:{pf}"), 0.8f32)]);
            let inputs = PortraitJobInputs::from_renderer_job(7, &masks);
            let report =
                apply_portrait_effects_with_job(&mut rgba, 16, 16, NormRect::FULL, &extra, &inputs);
            assert_eq!(rgba, original, "PF{pf} must not zero-fill missing inputs");
            assert_eq!(report.effects.len(), 1);
            let effect = &report.effects[0];
            assert!(effect.skipped, "PF{pf} was skipped");
            assert!(!effect.executed, "PF{pf} was not executed");
            assert!(
                effect.dispatches.is_empty(),
                "PF{pf} must not fake a dispatch"
            );
            assert_eq!(
                effect.refusal.as_ref(),
                Some(&Refusal::MissingInput(missing)),
                "PF{pf} must name its own missing input"
            );
        }
    }

    fn pf90021_volume(offset: f32) -> VolumeLut64 {
        let edge = aurora_portrait::cuda_pf90021::LUT_EDGE;
        let mut nodes = vec![0.0f32; aurora_portrait::cuda_pf90021::LUT_FLOATS];
        let denominator = (3 * (edge - 1)) as f32;
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    let at = ((z * edge + y) * edge + x) * 3;
                    nodes[at] = ((x + y) as f32 / denominator + offset).clamp(0.0, 1.0);
                    nodes[at + 1] =
                        ((y + z) as f32 / denominator + offset).clamp(0.0, 1.0);
                    nodes[at + 2] =
                        ((x + z) as f32 / denominator + offset).clamp(0.0, 1.0);
                }
            }
        }
        VolumeLut64::from_nodes(nodes).unwrap()
    }

    fn pf90021_runtime() -> Pf90021RuntimeScalars {
        Pf90021RuntimeScalars {
            degree01: 0.73,
            opacity01: 0.91,
            shelter: 0.83,
            region: 0.67,
        }
    }

    #[test]
    fn pf90021_stage_refuses_missing_assets_and_runtime_before_launch() {
        let masks = PortraitMasks::solid(1.0);
        let positive = pf90021_volume(0.0);
        let negative = pf90021_volume(0.25);
        let mask = vec![128u8; 16 * 16];
        let base = Pf90021Inputs {
            direction: Pf90021Direction::Positive,
            positive_lut: Some(&positive),
            negative_lut: Some(&negative),
            mask_r: Some(&mask),
            runtime: Some(pf90021_runtime()),
        };
        let cases = [
            (None, "PF90021_SAME_JOB_INPUTS"),
            (
                Some(Pf90021Inputs {
                    positive_lut: None,
                    ..base
                }),
                "PF90021_POSITIVE_LUT64",
            ),
            (
                Some(Pf90021Inputs {
                    direction: Pf90021Direction::Negative,
                    negative_lut: None,
                    ..base
                }),
                "PF90021_NEGATIVE_LUT64",
            ),
            (
                Some(Pf90021Inputs {
                    mask_r: None,
                    ..base
                }),
                "PF90021_MASK_R",
            ),
            (
                Some(Pf90021Inputs {
                    runtime: None,
                    ..base
                }),
                "PF90021_RUNTIME_SCALARS",
            ),
        ];
        for (pf90021, missing) in cases {
            let mut rgba = frame(16, 16);
            let original = rgba.clone();
            let inputs = PortraitJobInputs {
                pf90021,
                ..PortraitJobInputs::from_renderer_job(9, &masks)
            };
            let report = apply_portrait_effects_with_job(
                &mut rgba,
                16,
                16,
                NormRect::FULL,
                &BTreeMap::from([("pc:90021".to_string(), 0.8)]),
                &inputs,
            );
            let run = &report.effects[0];
            assert_eq!(rgba, original, "{missing} must not change pixels");
            assert_eq!(run.backend, PortraitBackend::Skipped, "{missing}");
            assert!(run.skipped, "{missing}");
            assert!(!run.executed, "{missing}");
            assert!(run.dispatches.is_empty(), "{missing}");
            assert_eq!(
                run.refusal,
                Some(Refusal::MissingInput(missing)),
                "{missing}"
            );
            assert_eq!(
                run.skip_reason,
                Some(PortraitSkipReason::InputRefused(Refusal::MissingInput(
                    missing
                ))),
                "{missing}"
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn pf90021_same_job_stage_matches_cpu_on_positive_and_negative() {
        let (width, height) = (16u32, 16u32);
        let masks = PortraitMasks::solid(1.0);
        let positive = pf90021_volume(0.0);
        let negative = pf90021_volume(0.25);
        let mask: Vec<u8> = (0..width as usize * height as usize)
            .map(|index| (index * 37 % 256) as u8)
            .collect();
        let runtime = pf90021_runtime();
        let source = Image::new(width, height, frame(width, height)).unwrap();

        for (direction, lut) in [
            (Pf90021Direction::Positive, &positive),
            (Pf90021Direction::Negative, &negative),
        ] {
            let expected = aurora_portrait::cuda_pf90021::apply_cpu(
                &source, lut, &mask, runtime,
            )
            .unwrap();
            let inputs = PortraitJobInputs {
                pf90021: Some(Pf90021Inputs {
                    direction,
                    positive_lut: Some(&positive),
                    negative_lut: Some(&negative),
                    mask_r: Some(&mask),
                    runtime: Some(runtime),
                }),
                ..PortraitJobInputs::from_renderer_job(21, &masks)
            };
            let (actual, run) = apply_one_with_inputs(90021, width, height, &inputs);
            if run.backend != PortraitBackend::Cuda {
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but PF90021 did not dispatch: {:?}",
                    run.fallback_reason
                );
                return;
            }
            assert_eq!(actual, expected.pixels, "{direction:?}");
            assert!(!run.skipped, "{direction:?}");
            assert!(run.executed, "{direction:?}");
            assert_eq!(
                run.dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>(),
                vec![
                    aurora_portrait::cuda_pf90021::LUT_PASS_KERNEL,
                    aurora_portrait::cuda_pf90021::COMPOSITE_KERNEL,
                ],
                "{direction:?}"
            );
            eprintln!(
                "PF90021 {direction:?} device={} kernels={:?}",
                run.dispatches[0].device,
                run.dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>()
            );
            for (pixel, (actual_pixel, expected_pixel)) in actual
                .chunks_exact(4)
                .zip(expected.pixels.chunks_exact(4))
                .enumerate()
            {
                assert_eq!(actual_pixel[3], expected_pixel[3], "alpha {pixel}");
            }
        }
    }

    #[test]
    fn pf30026_same_job_stage_matches_cpu_or_reports_cuda_unavailable() {
        let (width, height) = (9u32, 7u32);
        let pixels = (width * height) as usize;
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let source = Image::new(width, height, source_pixels.clone()).unwrap();
        let mut effect_pixels = source_pixels.clone();
        for (index, pixel) in effect_pixels.chunks_exact_mut(4).enumerate() {
            pixel[0] = pixel[0].saturating_add((index % 19) as u8);
            pixel[1] = pixel[1].saturating_add(21);
            pixel[2] = pixel[2].saturating_add(17);
            pixel[3] = (index * 13 % 256) as u8;
        }
        let effect = Image::new(width, height, effect_pixels).unwrap();
        let teeth_mask: Vec<u8> = (0..pixels).map(|i| ((i * 29 + 7) % 256) as u8).collect();
        let edge_plane_a = vec![0.5f32; pixels];
        let edge_plane_b = vec![0.5f32; pixels];
        let alignment = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&teeth_mask),
            edge_plane_a: Some(&edge_plane_a),
            edge_plane_b: Some(&edge_plane_b),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source_pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        let job = AlignmentFinalBlendJob {
            original: Some(&source),
            effect_rgb: Some(&effect),
            teeth_mask: Some(&teeth_mask),
            edge_plane_a: Some(&edge_plane_a),
            edge_plane_b: Some(&edge_plane_b),
        };
        let expected =
            aurora_portrait::kernels::teeth_blend::apply_alignment_final_blend(job).unwrap();
        match run.backend {
            PortraitBackend::Cuda => {
                assert!(run.executed);
                assert_eq!(run.dispatches.len(), 1);
                assert_eq!(run.dispatches[0].kernel, aurora_portrait::cuda_teeth_blend::KERNEL);
                let AlignmentFinalBlendOutcome::Applied(expected) = expected else {
                    panic!("open edge gate produced a CPU-gated oracle");
                };
                assert_eq!(actual, expected.pixels, "PF30026 CUDA/CPU byte parity");
                assert!(run.skip_reason.is_none());
                eprintln!(
                    "PF30026 stage device={} kernel={} grid={:?} block={:?} ptx_isa={:?}",
                    run.dispatches[0].device,
                    run.dispatches[0].kernel,
                    run.dispatches[0].grid,
                    run.dispatches[0].block,
                    run.dispatches[0].ptx_isa
                );
            }
            PortraitBackend::Skipped => {
                assert!(!run.executed);
                assert!(run.dispatches.is_empty());
                assert_eq!(actual, source_pixels);
                assert!(matches!(
                    run.skip_reason,
                    Some(PortraitSkipReason::Cuda(CudaError::Unavailable(_)))
                        | Some(PortraitSkipReason::Cuda(CudaError::Driver(_, _)))
                ));
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but PF30026 skipped: {:?}",
                    run.skip_reason
                );
            }
            PortraitBackend::Cpu => panic!("PF30026 must never report a CPU stage success"),
        }
    }

    #[test]
    fn pf30026_closed_gate_is_typed_and_non_destructive() {
        let (width, height) = (5u32, 4u32);
        let pixels = (width * height) as usize;
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let closed = vec![0.1f32; pixels];
        let teeth_mask = vec![128u8; pixels];
        let alignment = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&teeth_mask),
            edge_plane_a: Some(&closed),
            edge_plane_b: Some(&closed),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source_pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        assert_eq!(run.backend, PortraitBackend::Skipped);
        assert!(!run.executed);
        assert!(run.dispatches.is_empty());
        assert_eq!(run.skip_reason, Some(PortraitSkipReason::EdgeGateClosed));
        assert_eq!(actual, source_pixels);
    }

    #[test]
    fn pf30026_missing_inputs_and_roi_mismatch_are_typed() {
        let (width, height) = (4u32, 3u32);
        let pixels = (width * height) as usize;
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let teeth_mask = [128u8; 12];
        let edge = vec![0.5f32; pixels];
        let open = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&teeth_mask),
            edge_plane_a: Some(&edge),
            edge_plane_b: Some(&edge),
            roi: Some(NormRect::FULL),
        };
        for (alignment, missing) in [
            (
                TeethAlignmentInputs {
                    effect_rgb: None,
                    ..open
                },
                "PF30026 effect RGB",
            ),
            (
                TeethAlignmentInputs {
                    teeth_mask: None,
                    ..open
                },
                "PF30026 teeth mask",
            ),
            (
                TeethAlignmentInputs {
                    edge_plane_a: None,
                    ..open
                },
                "PF30026 edge plane A",
            ),
            (
                TeethAlignmentInputs {
                    edge_plane_b: None,
                    ..open
                },
                "PF30026 edge plane B",
            ),
            (
                TeethAlignmentInputs { roi: None, ..open },
                "PF30026 ROI",
            ),
        ] {
            let inputs = PortraitJobInputs {
                teeth_alignment: Some(alignment),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let mut actual = source_pixels.clone();
            let report = apply_portrait_effects_with_job(
                &mut actual,
                width,
                height,
                NormRect::FULL,
                &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
                &inputs,
            );
            let run = &report.effects[0];
            assert_eq!(run.backend, PortraitBackend::Skipped);
            assert!(!run.executed);
            assert_eq!(run.refusal, Some(Refusal::MissingInput(missing)));
            assert_eq!(
                run.skip_reason,
                Some(PortraitSkipReason::InputRefused(
                    Refusal::MissingInput(missing)
                ))
            );
            assert_eq!(actual, source_pixels);
        }

        let alignment = TeethAlignmentInputs {
            roi: Some(NormRect {
                x: 0.1,
                y: 0.0,
                width: 0.8,
                height: 1.0,
            }),
            ..open
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source_pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::ValueOutOfRange("PF30026 ROI"))
        );
        assert_eq!(actual, source_pixels);
    }

    #[cfg(windows)]
    #[test]
    fn pf30026_forced_cuda_failure_is_not_a_cpu_success() {
        let (width, height) = (4u32, 3u32);
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let edge = [0.5f32; 12];
        let alignment = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&[128; 12]),
            edge_plane_a: Some(&edge),
            edge_plane_b: Some(&edge),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source_pixels.clone();
        let _force = ForceCudaFailureGuard::new();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        assert_eq!(run.backend, PortraitBackend::Skipped);
        assert!(!run.executed);
        assert!(run.dispatches.is_empty());
        assert!(matches!(
            run.skip_reason,
            Some(PortraitSkipReason::Cuda(CudaError::Unavailable(_)))
        ));
        assert_eq!(actual, source_pixels);
    }

    #[test]
    fn makeup_routes_refuse_missing_same_job_textures() {
        let masks = PortraitMasks::solid(1.0);
        let material = Image::solid(3, 2, [40, 80, 120, 200]);
        let mask = Image::solid(4, 3, [10, 20, 30, 255]);

        for pf in [100u32, 103, 105, 117, 118, 30007] {
            let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
            let (_, effect) = apply_one_with_inputs(pf, 8, 6, &inputs);
            assert_eq!(
                effect.refusal,
                Some(Refusal::MissingInput("makeup mask")),
                "PF{pf}"
            );

            let no_uv = [MakeupInput {
                pf,
                mix_color: [30, 60, 90, 180],
                opacity_percent: 75.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                mask: Some(&mask),
                mask_uv: None,
            }];
            let inputs = PortraitJobInputs {
                makeup: Some(&no_uv),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let (_, effect) = apply_one_with_inputs(pf, 8, 6, &inputs);
            assert_eq!(
                effect.refusal,
                Some(Refusal::MissingInput("makeup mask UV")),
                "PF{pf}"
            );
        }

        let absent = PortraitJobInputs::from_renderer_job(1, &masks);
        let (_, effect) = apply_one_with_inputs(111, 8, 6, &absent);
        assert_eq!(
            effect.refusal,
            Some(Refusal::MissingInput("makeup material"))
        );

        let no_uv = [MakeupInput {
            pf: 111,
            mix_color: [30, 60, 90, 180],
            opacity_percent: 75.0,
            filter_type: 40,
            material: Some(&material),
            material_uv: None,
            mask: None,
            mask_uv: None,
        }];
        let inputs = PortraitJobInputs {
            makeup: Some(&no_uv),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let (_, effect) = apply_one_with_inputs(111, 8, 6, &inputs);
        assert_eq!(
            effect.refusal,
            Some(Refusal::MissingInput("makeup material UV"))
        );
    }

    fn run_makeup_route_case(force_cpu: bool) -> Vec<u8> {
        let (width, height) = (23u32, 17u32);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let masks = PortraitMasks::solid(1.0);
        let mask = Image::new(
            8,
            6,
            (0..8 * 6 * 4)
                .map(|index| ((index * 37 + 11) % 256) as u8)
                .collect(),
        )
        .unwrap();
        let material = Image::new(
            9,
            7,
            (0..9 * 7 * 4)
                .map(|index| ((index * 43 + 7) % 256) as u8)
                .collect(),
        )
        .unwrap();
        let mask_uv: Vec<[f32; 2]> = (0..pixels)
            .map(|index| {
                [
                    (index * 17 % 97) as f32 / 96.0,
                    (index * 29 % 89) as f32 / 88.0,
                ]
            })
            .collect();
        let material_uv: Vec<[f32; 2]> = (0..pixels)
            .map(|index| {
                [
                    (index * 13 % 83) as f32 / 82.0,
                    (index * 31 % 79) as f32 / 78.0,
                ]
            })
            .collect();

        let rows = [
            MakeupInput {
                pf: 30007,
                mix_color: [31, 61, 91, 211],
                opacity_percent: 68.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
            },
            MakeupInput {
                pf: 111,
                mix_color: [47, 97, 151, 203],
                opacity_percent: 73.0,
                filter_type: 40,
                material: Some(&material),
                material_uv: Some(&material_uv),
                mask: None,
                mask_uv: None,
            },
            MakeupInput {
                pf: 100,
                mix_color: [13, 53, 113, 197],
                opacity_percent: 71.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
            },
            MakeupInput {
                pf: 118,
                mix_color: [19, 59, 109, 199],
                opacity_percent: 77.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
            },
            MakeupInput {
                pf: 103,
                mix_color: [23, 67, 101, 193],
                opacity_percent: 79.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
            },
            MakeupInput {
                pf: 117,
                mix_color: [29, 71, 107, 191],
                opacity_percent: 81.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
            },
            MakeupInput {
                pf: 105,
                mix_color: [37, 73, 109, 189],
                opacity_percent: 83.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
            },
        ];
        let inputs = PortraitJobInputs {
            makeup: Some(&rows),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let degrees = [
            (100u32, 0.61f32),
            (103, 0.62),
            (105, 0.63),
            (111, 0.64),
            (117, 0.65),
            (118, 0.66),
            (30007, 0.67),
        ];
        let extra: BTreeMap<String, f32> = degrees
            .iter()
            .map(|(pf, degree)| (format!("pc:{pf}"), *degree))
            .collect();
        let mut actual = source.pixels.clone();

        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);

        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &extra,
            &inputs,
        );
        let expected_pfs = [100u32, 103, 105, 111, 117, 118, 30007];
        assert_eq!(
            report
                .effects
                .iter()
                .map(|effect| effect.pf)
                .collect::<Vec<_>>(),
            expected_pfs
        );

        let mut expected = source.clone();
        for (pf, degree) in degrees {
            let route = MakeupRoute::resolve(pf).unwrap();
            let input = rows.iter().copied().find(|row| row.pf == pf).unwrap();
            let layer = makeup_layer(route, input, degree).unwrap();
            expected = match route {
                MakeupRoute::Color { .. } => color_makeup(
                    &expected,
                    input.mask.unwrap(),
                    input.mask_uv.unwrap(),
                    layer,
                )
                .unwrap(),
                MakeupRoute::ImageUnmasked { .. } => image_makeup_unmasked(
                    &expected,
                    input.material.unwrap(),
                    input.material_uv.unwrap(),
                    layer,
                )
                .unwrap(),
            };
        }
        assert_eq!(actual, expected.pixels, "stage CPU/GPU route parity");

        for effect in &report.effects {
            eprintln!(
                "makeup PF{} backend={:?} kernels={:?} fallback={:?}",
                effect.pf,
                effect.backend,
                effect
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>(),
                effect.fallback_reason
            );
            assert!(effect.executed, "PF{}", effect.pf);
            assert!(effect.refusal.is_none(), "PF{}: {:?}", effect.pf, effect.refusal);
            if force_cpu {
                assert_eq!(effect.backend, PortraitBackend::Cpu, "PF{}", effect.pf);
                assert!(effect.dispatches.is_empty(), "PF{}", effect.pf);
            } else if effect.backend == PortraitBackend::Cpu {
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but PF{} fell back: {:?}",
                    effect.pf,
                    effect.fallback_reason
                );
            } else {
                assert!(
                    !effect.dispatches.is_empty(),
                    "PF{} must report its dispatch",
                    effect.pf
                );
            }
        }
        actual
    }

    #[test]
    fn makeup_routes_match_cpu_in_pf_order() {
        run_makeup_route_case(true);
    }

    #[cfg(windows)]
    #[test]
    fn makeup_routes_report_real_backend() {
        run_makeup_route_case(false);
    }

    #[test]
    fn missing_skin_plane_is_a_typed_refusal() {
        let mut masks = PortraitMasks::solid(1.0);
        masks.skin = None;
        let mut rgba = frame(8, 8);
        let original = rgba.clone();
        let extra = BTreeMap::from([("pc:90005".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        let report =
            apply_portrait_effects_with_job(&mut rgba, 8, 8, NormRect::FULL, &extra, &inputs);
        assert_eq!(rgba, original);
        assert_eq!(
            report.effects[0].refusal.as_ref(),
            Some(&Refusal::MissingInput("skin mask"))
        );
        assert!(report.effects[0].skipped);
        assert!(!report.effects[0].executed);
    }

    fn run_complete_same_job_case(force_cpu: bool) {
        let (width, height) = (17u32, 13u32);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let masks = PortraitMasks::solid(1.0);

        let teeth_mask: Vec<u8> = (0..pixels).map(|i| ((i * 29 + 7) % 256) as u8).collect();
        let teeth_lut: Vec<u8> = (0..LUT_BYTES).map(|i| (i % 251) as u8).collect();

        let repair = Image::new(
            width,
            height,
            (0..pixels * 4)
                .map(|i| ((i * 43 + 17) % 256) as u8)
                .collect(),
        )
        .unwrap();
        let beard_a: Vec<u8> = (0..pixels).map(|i| ((i * 19 + 3) % 256) as u8).collect();
        let beard_b: Vec<u8> = (0..pixels).map(|i| ((i * 31 + 11) % 256) as u8).collect();

        let face_mask: Vec<u8> = (0..pixels * 4)
            .map(|i| ((i * 13 + 5) % 256) as u8)
            .collect();
        let blur_y: Vec<u8> = (0..pixels).map(|i| ((i * 7 + 9) % 256) as u8).collect();
        let face_skin_map: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let hardlight = [0.0, 0.25, 0.5, 0.75, 1.0];
        let face_beauty = [0.0, 0.1, 0.35, 0.8, 1.0];

        let body_skin: Vec<f32> = (0..pixels)
            .map(|i| ((i * 17 + 1) % 256) as f32 / 255.0)
            .collect();
        let body_dog: Vec<f32> = (0..pixels)
            .map(|i| ((i * 23 + 7) % 256) as f32 / 255.0)
            .collect();
        let body_beauty = [0.0, 0.2, 0.45, 0.7, 1.0];

        let eye_color: Vec<f32> = (0..pixels * 3)
            .map(|i| ((i * 37 + 19) % 256) as f32)
            .collect();
        let eye_response: Vec<f32> = (0..pixels)
            .map(|i| ((i * 11 + 3) % 256) as f32 / 255.0)
            .collect();

        let lip: Vec<u8> = (0..pixels).map(|i| ((i * 17 + 5) % 256) as u8).collect();
        let mouth: Vec<u8> = (0..pixels).map(|i| ((i * 29 + 7) % 256) as u8).collect();
        let model_dx: Vec<f32> = (0..pixels).map(|i| ((i % 9) as f32 - 4.0) * 0.25).collect();
        let model_dy: Vec<f32> = (0..pixels).map(|i| ((i % 7) as f32 - 3.0) * -0.2).collect();
        let smile_roi = Rect {
            x: 2,
            y: 1,
            width: 12,
            height: 9,
        };

        let suit_mask = Image::solid(width, height, [255, 255, 255, 255]);
        let suit_uv = vec![[0.5f32, 0.5f32]; pixels];
        let suit_layer = MakeupLayer {
            mix_color: [250, 40, 20, 255],
            opacity_percent: 100.0,
            degree01: 0.6,
            filter_type: 10,
            mask_channel: aurora_portrait::kernels::makeup_material::MaskChannel::Rgba,
        };
        let suit_passes = [SuitPassInput {
            kind: SuitPassKindInput::Color,
            mask: Some(&suit_mask),
            mask_uv: Some(&suit_uv),
            layer: Some(suit_layer),
            shelter: 1.0,
            region: 1.0,
        }];

        let inputs = PortraitJobInputs {
            origin: PortraitJobInputOrigin::RendererJob { job_id: 9 },
            masks: &masks,
            assets: PortraitAssetInputs {
                teeth_whiten_lut: Some(&teeth_lut),
                teeth_bright_lut: Some(&teeth_lut),
            },
            teeth: Some(TeethInputs {
                mask_g: &teeth_mask,
                bright_ui: 60.0,
                deyellow_ui: 70.0,
            }),
            teeth_alignment: None,
            beard: Some(BeardInputs {
                mask_a: &beard_a,
                mask_b: &beard_b,
                repair_rgb: &repair,
            }),
            moisture_face: Some(MoistureFaceInputs {
                width,
                height,
                mask_rgba: &face_mask,
                blur_y: &blur_y,
                channel: aurora_portrait::kernels::moisture::MaskChannel::R,
                skin_map_width: 4,
                skin_map_height: 4,
                skin_map: &face_skin_map,
                hardlight: &hardlight,
                beauty: &face_beauty,
                pivot: 0.5,
                gain: 1.0,
            }),
            moisture_body: Some(MoistureBodyInputs {
                width,
                height,
                weight_skin: &body_skin,
                weight_dog: &body_dog,
                beauty: &body_beauty,
                degree: 0.5,
                roi: (0, 0, width, height),
            }),
            eye_bag: Some(EyeBagInputs {
                width,
                height,
                aligned_color_field_rgb_f32: &eye_color,
                aligned_response_mask: &eye_response,
            }),
            smile: SmileInputs {
                material: Some("Grinning"),
                lip_mask: Some(&lip),
                mouth_mask: Some(&mouth),
                model_dx: Some(&model_dx),
                model_dy: Some(&model_dy),
                roi: Some(smile_roi),
            },
            suit_passes: Some(&suit_passes),
            hair_smooth: None,
            hair_gloss: None,
            stray_hair: StrayHairInputs::default(),
            makeup: None,
            pf90021: None,
        };

        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);

        let expected = [
            (
                108u32,
                aurora_portrait::kernels::face_detail::teeth_whitening(
                    &source,
                    &teeth_mask,
                    &teeth_lut,
                    &teeth_lut,
                    80.0,
                    60.0,
                    70.0,
                )
                .unwrap()
                .output,
            ),
            (
                21026,
                aurora_portrait::kernels::beard::apply(&source, &repair, &beard_a, &beard_b, 0.8)
                    .unwrap(),
            ),
            (
                90002,
                aurora_portrait::kernels::moisture::face_apply(
                    &source,
                    &face_mask,
                    &blur_y,
                    &FaceParams {
                        ui: 80.0,
                        pivot: 0.5,
                        gain: 1.0,
                    },
                    aurora_portrait::kernels::moisture::MaskChannel::R,
                    &SkinMap {
                        width: 4,
                        height: 4,
                        samples: &face_skin_map,
                    },
                    &Lut1D {
                        samples: &hardlight,
                    },
                    &Lut1D {
                        samples: &face_beauty,
                    },
                )
                .unwrap(),
            ),
            (
                90010,
                aurora_portrait::kernels::moisture::body_apply(
                    &source,
                    &body_skin,
                    &body_dog,
                    &Lut1D {
                        samples: &body_beauty,
                    },
                    0.5,
                    (0, 0, width, height),
                )
                .unwrap(),
            ),
            (
                90116,
                aurora_portrait::kernels::eye_bag::compose_field_image_pf90116(
                    &source,
                    &eye_color,
                    &eye_response,
                    0.8,
                )
                .unwrap(),
            ),
            (
                100007,
                Image::new(
                    width,
                    height,
                    smile_warp::pf100007_apply(
                        &source.pixels,
                        width,
                        height,
                        0.8,
                        &lip,
                        &mouth,
                        &model_dx,
                        &model_dy,
                        smile_roi,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ),
            (
                90033,
                aurora_portrait::kernels::suit_makeup::compose(
                    &source,
                    &[SuitPass {
                        kind: SuitPassKind::Color,
                        mask: &suit_mask,
                        mask_uv: &suit_uv,
                        layer: suit_layer,
                        shelter: 1.0,
                        region: 1.0,
                    }],
                )
                .unwrap(),
            ),
        ];

        for (pf, expected) in expected {
            let (actual, effect) = apply_one_with_inputs(pf, width, height, &inputs);
            assert_eq!(actual, expected.pixels, "PF{pf} CPU fallback pixels");
            assert!(effect.executed, "PF{pf}");
            assert!(effect.refusal.is_none(), "PF{pf}: {:?}", effect.refusal);
            eprintln!(
                "PF{pf} backend={:?} device={:?} kernels={:?} fallback={:?}",
                effect.backend,
                effect
                    .dispatches
                    .first()
                    .map(|dispatch| dispatch.device.as_str()),
                effect
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>(),
                effect.fallback_reason
            );
            if force_cpu {
                assert_eq!(effect.backend, PortraitBackend::Cpu, "PF{pf}");
                assert!(effect.dispatches.is_empty(), "PF{pf}");
                #[cfg(windows)]
                assert!(
                    effect
                        .fallback_reason
                        .as_deref()
                        .is_some_and(|reason| reason.contains("test-forced CUDA failure")),
                    "PF{pf}: {:?}",
                    effect.fallback_reason
                );
            } else {
                match effect.backend {
                    PortraitBackend::Cuda => {
                        assert!(!effect.dispatches.is_empty(), "PF{pf}");
                    }
                    PortraitBackend::Cpu => {
                        assert!(
                            std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                            "CUDA required but PF{pf} fell back: {:?}",
                            effect.fallback_reason
                        );
                        #[cfg(windows)]
                        assert!(
                            effect.fallback_reason.is_some(),
                            "PF{pf} CPU fallback must state why"
                        );
                    }
                    PortraitBackend::Skipped => {
                        panic!("PF{pf} unexpectedly skipped: {:?}", effect.skip_reason)
                    }
                }
            }
        }
    }

    #[test]
    fn complete_same_job_bundle_matches_cpu_oracles() {
        run_complete_same_job_case(true);
    }

    #[cfg(windows)]
    #[test]
    fn complete_same_job_bundle_reports_real_backend() {
        run_complete_same_job_case(false);
    }

    #[test]
    fn skin_lut_effects_change_a_warm_tone() {
        for (pf, degree) in [(21015u32, 0.9f32), (30096, 0.1), (90005, 0.9), (90007, 0.9)] {
            let mut rgba = frame(8, 8);
            let original = rgba.clone();
            apply(&mut rgba, 8, 8, &[(&format!("pc:{pf}"), degree)]);
            assert_ne!(rgba, original, "PF{pf} must change the frame");
        }
    }

    #[test]
    fn skin_texture_and_gloss_change_the_frame() {
        for (pf, degree) in [(8068u32, 0.1f32), (8069, 0.0)] {
            let mut rgba = frame(64, 64);
            let original = rgba.clone();
            apply(&mut rgba, 64, 64, &[(&format!("pc:{pf}"), degree)]);
            assert_ne!(rgba, original, "PF{pf} must change the frame");
        }
    }

    #[test]
    fn skin_texture_neutral_is_identity() {
        let mut rgba = frame(64, 64);
        let original = rgba.clone();
        apply(&mut rgba, 64, 64, &[("pc:8069", 0.5)]);
        assert_eq!(rgba, original);
    }

    #[test]
    fn skin_clarity_and_radiance_run_with_assets() {
        for (pf, degree) in [(8070u32, 0.5f32), (90004, 0.5)] {
            let mut rgba = frame(8, 8);
            apply(&mut rgba, 8, 8, &[(&format!("pc:{pf}"), degree)]);
            assert_eq!(rgba.len(), 8 * 8 * 4, "PF{pf} output size");
        }
    }

    #[test]
    fn portrait_effects_process_in_pf_order() {
        let mut rgba = frame(16, 16);
        apply(&mut rgba, 16, 16, &[("pc:90007", 0.9), ("pc:21015", 0.9)]);
        assert_eq!(rgba.len(), 16 * 16 * 4);
    }

    #[cfg(windows)]
    #[test]
    fn cuda_ported_stages_match_cpu_pixels_when_available() {
        let source = Image::new(8, 8, frame(8, 8)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, 8, 8, NormRect::FULL);
        for (pf, degree) in [(8068u32, 0.1f32), (90005, 0.9), (90007, 0.9)] {
            let cpu = match pf {
                8068 => apply_skin_gloss(&source, degree, skin.as_deref()),
                90005 => apply_skin_whiten(&source, degree, skin.as_deref()),
                90007 => apply_skin_ruddy(&source, degree, skin.as_deref()),
                _ => unreachable!(),
            }
            .expect("cpu");
            match try_cuda_effect(&source, pf, degree, skin.as_deref()) {
                Some(Ok((gpu, dispatches))) => {
                    assert_eq!(gpu.pixels, cpu.pixels, "CUDA/CPU PF{pf} pixels");
                    assert!(!dispatches.is_empty(), "CUDA must report a dispatch");
                    for dispatch in &dispatches {
                        eprintln!(
                            "PF{pf} kernel={} device={} grid={:?} block={:?} ptx_isa={:?}",
                            dispatch.kernel,
                            dispatch.device,
                            dispatch.grid,
                            dispatch.block,
                            dispatch.ptx_isa
                        );
                    }
                }
                Some(Err(err)) => {
                    assert!(
                        std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                        "CUDA required but unavailable: {err:?}"
                    );
                    eprintln!("CUDA unavailable; fallback contract covered elsewhere: {err:?}");
                }
                None => panic!("expected PF{pf} CUDA route"),
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn cuda_skin_model_routes_match_cpu_when_available() {
        let source = Image::solid(16, 16, [127, 113, 91, 255]);
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, 16, 16, NormRect::FULL);
        for (pf, degree) in [(8069u32, 0.1f32), (8070, 0.5)] {
            let cpu = match pf {
                8069 => apply_skin_texture(&source, degree, skin.as_deref()),
                8070 => apply_skin_clarity(&source, degree, skin.as_deref(), 16, 16),
                _ => unreachable!(),
            }
            .expect("cpu");
            match try_cuda_effect(&source, pf, degree, skin.as_deref()) {
                Some(Ok((gpu, dispatches))) => {
                    assert_eq!(gpu, cpu);
                    assert!(!dispatches.is_empty());
                }
                Some(Err(err)) => assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {err:?}"
                ),
                None => panic!("expected PF{pf} CUDA route"),
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn cuda_fixed_point_tint_route_matches_cpu_when_available() {
        let source = Image::solid(16, 16, [181, 139, 103, 255]);
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, 16, 16, NormRect::FULL);
        for degree in [0.1f32, 0.5, 0.9] {
            let cpu = apply_skin_tint(&source, degree, skin.as_deref()).expect("cpu");
            match try_cuda_effect(&source, 21015, degree, skin.as_deref()) {
                Some(Ok((gpu, dispatches))) => {
                    assert_eq!(gpu, cpu);
                    assert_eq!(dispatches.len(), 1);
                }
                Some(Err(err)) => assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {err:?}"
                ),
                None => panic!("expected PF21015 CUDA route"),
            }
        }
    }

    #[test]
    fn run_report_does_not_label_cpu_as_cuda() {
        let mut rgba = frame(8, 8);
        let map = BTreeMap::from([("pc:90005".to_string(), 0.9)]);
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs::from_renderer_job(42, &masks);
        let report = apply_portrait_effects_with_report_and_inputs(
            &mut rgba,
            8,
            8,
            NormRect::FULL,
            &map,
            &inputs,
        );
        assert_eq!(report.job_id, Some(42));
        assert_eq!(report.effects.len(), 1);
        let effect = &report.effects[0];
        assert_eq!(effect.pf, 90005);
        assert!(!effect.skipped);
        assert!(effect.executed);
        match effect.backend {
            PortraitBackend::Cuda => assert!(!effect.dispatches.is_empty()),
            PortraitBackend::Cpu => assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but report fell back to CPU: {:?}",
                effect.fallback_reason
            ),
            PortraitBackend::Skipped => panic!("PF90005 unexpectedly skipped"),
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "GPU timing; run with --ignored and PORTRAIT_CUDA_REQUIRED=1"]
    fn cuda_skin_lut_timing_p50_p95() {
        use std::time::Instant;

        let (width, height) = (256u32, 256u32);
        let source = Image::new(width, height, frame(width, height)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, width, height, NormRect::FULL);
        let mut cpu_ms = Vec::new();
        let mut gpu_ms = Vec::new();
        for _ in 0..10 {
            let started = Instant::now();
            apply_skin_whiten(&source, 0.9, skin.as_deref()).expect("cpu");
            cpu_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        for _ in 0..10 {
            let started = Instant::now();
            match try_cuda_effect(&source, 90005, 0.9, skin.as_deref()) {
                Some(Ok(_)) => gpu_ms.push(started.elapsed().as_secs_f64() * 1000.0),
                _ => {
                    assert!(
                        std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                        "CUDA timing requested but unavailable"
                    );
                    return;
                }
            }
        }
        cpu_ms.sort_by(|a, b| a.total_cmp(b));
        gpu_ms.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "PF90005 256x256 cpu p50={:.3}ms p95={:.3}ms; cuda p50={:.3}ms p95={:.3}ms",
            cpu_ms[5], cpu_ms[9], gpu_ms[5], gpu_ms[9]
        );
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "GPU session timing; run with --ignored and PORTRAIT_CUDA_REQUIRED=1"]
    fn cuda_skin_lut_session_cold_warm_p50_p95() {
        use std::time::Instant;

        let (width, height) = (256u32, 256u32);
        let source = Image::new(width, height, frame(width, height)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, width, height, NormRect::FULL);
        let run = || {
            let started = Instant::now();
            let routed = try_cuda_effect(&source, 90005, 0.9, skin.as_deref());
            (started.elapsed().as_secs_f64() * 1000.0, routed)
        };
        let (cold_ms, cold) = run();
        assert!(matches!(cold, Some(Ok(_))), "cold route must use CUDA");
        for _ in 0..10 {
            assert!(matches!(run().1, Some(Ok(_))), "warmup must stay on CUDA");
        }
        let mut warm_ms = Vec::with_capacity(30);
        for _ in 0..30 {
            let (elapsed, routed) = run();
            assert!(matches!(routed, Some(Ok(_))), "sample must stay on CUDA");
            warm_ms.push(elapsed);
        }
        warm_ms.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "PF90005 256x256 session cold={cold_ms:.3}ms warm_p50={:.3}ms warm_p95={:.3}ms",
            warm_ms[14], warm_ms[28]
        );
    }
}
