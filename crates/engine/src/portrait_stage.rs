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
use aurora_portrait::cuda_pf8050::{self, MESH_POINTS};
use aurora_portrait::cuda_pf90021::{
    Direction as Pf90021Direction, RuntimeScalars as Pf90021RuntimeScalars, VolumeLut64,
};
use aurora_portrait::device::{Backend as DeviceBackend, DeviceRefusal, Selection};
#[cfg(windows)]
use aurora_portrait::cuda_hair_detail::HairDetail;
use aurora_portrait::cuda_hair_detail::{
    GlossProducer, Roi as HairRoi, SmoothParams, SmoothProducer,
};
use aurora_portrait::cuda_makeup::{MakeupRoute, RouteInputs};
use aurora_portrait::cuda_stray_hair::Params as StrayHairParams;
use aurora_portrait::kernels::face_detail::{Rect as ToothRect, ToothRepairProducer};
use aurora_portrait::kernels::makeup_material::{
    MakeupLayer, MaskChannel as MakeupMaskChannel, color_makeup, image_makeup,
    image_makeup_unmasked,
};
use aurora_portrait::kernels::moisture::{FaceParams, Lut1D, MaskChannel, SkinMap};
use aurora_portrait::kernels::skin_adjust::{
    self, AdjustLut, BodyHlfUniforms, Roi, StoreMode, WrinkleUniforms,
};
use aurora_portrait::kernels::smile_warp::{self, Rect};
use aurora_portrait::kernels::suit_makeup::{SuitPass, SuitPassKind};
use aurora_portrait::kernels::teeth_blend::AlignmentFinalBlendOutcome;
use aurora_portrait::kernels::teeth_blend::AlignmentFinalBlendJob;
use aurora_portrait::{Access, Image, Refusal};

use crate::NormRect;

#[cfg(test)]
#[path = "portrait_hlf_wrinkle_stage_tests.rs"]
mod hlf_wrinkle_stage_tests;

#[cfg(test)]
#[path = "portrait_hlf_24mp_bench_tests.rs"]
mod hlf_24mp_bench_tests;

#[cfg(test)]
#[path = "portrait_pf8050_stage_tests.rs"]
mod pf8050_stage_tests;

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
        self.mask_f32(kind, out_w, out_h, region).map(|out| {
            out.into_iter()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                .collect()
        })
    }

    /// Samples a mask without quantizing it, so automatic HLF producers keep
    /// the same float mask domain as explicit same-job inputs.
    pub fn mask_f32(
        &self,
        kind: MaskKind,
        out_w: u32,
        out_h: u32,
        region: NormRect,
    ) -> Option<Vec<f32>> {
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
                out.push(sample_plane(plane, self.width, self.height, nx, ny).clamp(0.0, 1.0));
            }
        }
        Some(out)
    }

    /// Build the PF90004 fields that the local people parser can prove.
    /// Vendor parsing/face packings remain absent rather than being guessed.
    pub fn pf90004_parser_inputs(
        &self,
        rgba: &[u8],
        out_w: u32,
        out_h: u32,
        region: NormRect,
    ) -> Result<Pf90004AutoInputs, Refusal> {
        if region != NormRect::FULL {
            return Err(Refusal::ValueOutOfRange("PF90004_ROI"));
        }
        let pixels = usize::try_from(out_w)
            .ok()
            .and_then(|w| usize::try_from(out_h).ok().and_then(|h| w.checked_mul(h)))
            .filter(|&pixels| pixels != 0)
            .ok_or(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))?;
        if rgba.len() != pixels.saturating_mul(4) {
            return Err(Refusal::ShapeMismatch("PF90004_FRAME_RGBA"));
        }
        let skin_mask = self
            .mask_u8(MaskKind::Skin, out_w, out_h, region)
            .ok_or(Refusal::MissingInput("PF90004_PARSER_SKIN_MASK"))?;
        let value_mean = pf90004_value_mean(rgba, out_w, out_h, &skin_mask)?;
        Ok(Pf90004AutoInputs {
            source: Pf90004AutoSource::PortraitParser,
            skin_mask,
            parsing_rgba: None,
            face_rgba: None,
            value_mean,
            roi: NormRect::FULL,
        })
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

/// Provenance is part of the value: parser coverage is not a vendor plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pf90004AutoSource {
    PortraitParser,
}

/// The subset of PF90004 inputs derivable without inventing vendor packing.
#[derive(Clone, Debug, PartialEq)]
pub struct Pf90004AutoInputs {
    pub source: Pf90004AutoSource,
    pub skin_mask: Vec<u8>,
    pub parsing_rgba: Option<Vec<u8>>,
    pub face_rgba: Option<Vec<u8>>,
    pub value_mean: f32,
    pub roi: NormRect,
}

/// The native statistic at `0x702E3F0`: max-side 320 resize to even
/// dimensions, then the mean red byte over pixels whose resized mask is nonzero.
/// Vendor input surfaces are BGR; Aurora's `Image` is RGBA8, so red is byte 0.
fn pf90004_value_mean(
    rgba: &[u8],
    width: u32,
    height: u32,
    skin_mask: &[u8],
) -> Result<f32, Refusal> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))?;
    if skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF90004_SKIN_MASK"));
    }
    if rgba.len() != pixels.saturating_mul(4) {
        return Err(Refusal::ShapeMismatch("PF90004_FRAME_RGBA"));
    }

    let max_side = width.max(height) as f64;
    let scale = (320.0_f64 / max_side).min(1.0) as f32;
    let stats_w = (2.0 * (width as f32 * scale * 0.5).round()) as u32;
    let stats_h = (2.0 * (height as f32 * scale * 0.5).round()) as u32;
    if stats_w == 0 || stats_h == 0 {
        return Ok(0.0);
    }

    let sx_step = width as f32 / stats_w as f32;
    let sy_step = height as f32 / stats_h as f32;
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for y in 0..stats_h {
        let sy = (y as f32 + 0.5) * sy_step - 0.5;
        let y0 = sy.max(0.0) as u32;
        let y1 = (y0 + 1).min(height - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..stats_w {
            let sx = (x as f32 + 0.5) * sx_step - 0.5;
            let x0 = sx.max(0.0) as u32;
            let x1 = (x0 + 1).min(width - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            let at = |xx: u32, yy: u32| {
                let i = yy as usize * width as usize + xx as usize;
                (rgba[i * 4] as f32, skin_mask[i] as f32)
            };
            let (r00, m00) = at(x0, y0);
            let (r10, m10) = at(x1, y0);
            let (r01, m01) = at(x0, y1);
            let (r11, m11) = at(x1, y1);
            let red = (r00 * (1.0 - fx) + r10 * fx) * (1.0 - fy)
                + (r01 * (1.0 - fx) + r11 * fx) * fy;
            let mask = (m00 * (1.0 - fx) + m10 * fx) * (1.0 - fy)
                + (m01 * (1.0 - fx) + m11 * fx) * fy;
            if (mask + 0.5).floor() != 0.0 {
                sum += (red + 0.5).floor();
                count += 1;
            }
        }
    }
    Ok(if count == 0 { 0.0 } else { sum / count as f32 })
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

pub type Pf8050GeometryInputs<'a> = cuda_pf8050::Inputs<'a>;

/// Vendor GL receipts are separate from the six-stage geometry contract.
/// s934 proved PF8033's Vec3/ROI family cannot substitute for PF8050.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pf8050VendorRuntime<'a> {
    pub attach: Option<Pf8050AttachReceipt>,
    pub vec3: Option<&'a [[f32; 3]]>,
    pub roi: Option<NormRect>,
    pub vbo: Option<&'a [u8]>,
    pub fbo: Option<Pf8050FboReceipt>,
    pub readback: Option<&'a [u8]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf8050AttachReceipt {
    pub job_id: u64,
    pub face_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf8050FboReceipt {
    pub framebuffer_id: u32,
    pub color_attachment: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Pf8050StageInputs<'a> {
    pub geometry: Pf8050GeometryInputs<'a>,
    pub runtime: Pf8050VendorRuntime<'a>,
}

/// Candidate six-stage geometry result. `candidate_only` must remain true
/// until same-job vendor VBO/FBO/readback bytes are captured and consumed.
#[derive(Clone, Debug, PartialEq)]
pub struct PortraitGeometryRun {
    pub backend: PortraitBackend,
    pub selection: Selection,
    pub dispatches: Vec<Dispatch>,
    pub fallback_reason: Option<String>,
    pub refusal: Option<Refusal>,
    pub stages: Vec<[[f32; 2]; MESH_POINTS]>,
    pub candidate_only: bool,
}

/// Asset bytes needed by rows that cannot use the shared parser masks.
#[derive(Clone, Copy, Debug, Default)]
pub struct PortraitAssetInputs<'a> {
    pub teeth_whiten_lut: Option<&'a [u8]>,
    pub teeth_bright_lut: Option<&'a [u8]>,
}

/// Explicit producer output for one renderer job. Missing rows stay absent.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProducerInputs {
    pub teeth: Option<TeethProducerInputs>,
    pub pf30025: Option<Pf30025ProducerInputs>,
    pub teeth_alignment: Option<TeethAlignmentProducerInputs>,
    pub stray_hair: Option<StrayHairProducerInputs>,
    pub pf90021: Option<Pf90021ProducerInputs>,
    pub pf90004: Option<Pf90004ProducerInputs>,
    pub hlf_body_adjust_lut: Option<HlfBodyAdjustLutProducerInputs>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TeethProducerInputs {
    pub mask_g: Vec<u8>,
    pub bright_ui: f32,
    pub deyellow_ui: f32,
    pub whiten_lut: Vec<u8>,
    pub bright_lut: Vec<u8>,
}

/// PF90088 same-job native values and planes. Every field stays optional so an
/// incomplete producer row retains the consumer's typed refusal order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrayHairProducerInputs {
    pub hairfly_v1: Option<Image>,
    pub raw_mask: Option<Vec<u8>>,
    pub person_mask: Option<Vec<u8>>,
    pub background_mask: Option<Vec<u8>>,
    pub native_degree: Option<f32>,
    pub native_person_area: Option<f32>,
    pub native_background_area: Option<f32>,
}

/// PF30025 same-job model outputs and support inputs.
#[derive(Clone, Copy, Debug)]
pub struct Pf30025Inputs<'a> {
    pub repair_780929437_bgra: &'a [u8],
    pub gate_3204548346: Option<&'a [u8]>,
    pub teeth_mask: &'a [f32],
    pub roi: ToothRect,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf30025ProducerInputs {
    /// Model output `780929437`, per-face slot `+0x228`, packed BGRA8.
    pub repair_780929437_bgra: Vec<u8>,
    /// Model output `3204548346`, per-face slot `+0x280`, raw host bytes.
    pub gate_3204548346: Option<Vec<u8>>,
    /// Explicit same-job teeth support normalized to `[0,1]`.
    pub teeth_mask: Vec<f32>,
    pub roi: ToothRect,
}

/// PF30026 producer bytes for one renderer job. The edge planes are candidate
/// record wrappers C/D; unproven gate binding stays absent rather than defaulted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeethAlignmentProducerInputs {
    pub effect_rgb: Option<Image>,
    pub teeth_mask: Option<Vec<u8>>,
    pub edge_plane_a: Option<Vec<f32>>,
    pub edge_plane_b: Option<Vec<f32>>,
    pub roi: Option<NormRect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf90021ProducerInputs {
    pub direction: Pf90021Direction,
    pub positive_lut: VolumeLut64,
    pub negative_lut: VolumeLut64,
    pub mask_r: Vec<u8>,
    pub runtime: Pf90021RuntimeScalars,
}

/// The five PF90004 producer outputs plus the exact rendered-frame size they
/// belong to. Planes are RGBA8/R8 coverage values in the consumer's byte domain.
#[derive(Clone, Debug, PartialEq)]
pub struct Pf90004ProducerInputs {
    width: u32,
    height: u32,
    skin_mask: Vec<u8>,
    parsing_rgba: Vec<u8>,
    face_rgba: Vec<u8>,
    value_mean: f32,
    roi: NormRect,
}

impl Pf90004ProducerInputs {
    pub fn from_same_job(
        width: u32,
        height: u32,
        skin_mask: Vec<u8>,
        parsing_rgba: Vec<u8>,
        face_rgba: Vec<u8>,
        value_mean: f32,
        roi: NormRect,
    ) -> Result<Self, Refusal> {
        let bundle = Self {
            width,
            height,
            skin_mask,
            parsing_rgba,
            face_rgba,
            value_mean,
            roi,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    fn validate(&self) -> Result<(), Refusal> {
        let pixels = usize::try_from(self.width)
            .ok()
            .and_then(|width| {
                usize::try_from(self.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .filter(|&pixels| pixels != 0)
            .ok_or(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))?;
        if self.skin_mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF90004_SKIN_MASK"));
        }
        let rgba = pixels
            .checked_mul(4)
            .ok_or(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))?;
        if self.parsing_rgba.len() != rgba {
            return Err(Refusal::ShapeMismatch("PF90004_PARSING_RGBA"));
        }
        if self.face_rgba.len() != rgba {
            return Err(Refusal::ShapeMismatch("PF90004_FACE_RGBA"));
        }
        if !self.value_mean.is_finite() || !(0.0..=255.0).contains(&self.value_mean) {
            return Err(Refusal::ValueOutOfRange("PF90004_VALUE_MEAN"));
        }
        if self.roi != NormRect::FULL {
            return Err(Refusal::ValueOutOfRange("PF90004_ROI"));
        }
        Ok(())
    }
}

/// Vendor PF30022 L8 curve. The source, median, gauss, face and skin planes
/// remain separate same-job inputs and are never synthesized from this LUT.
#[derive(Clone, Debug, PartialEq)]
pub struct HlfBodyAdjustLutProducerInputs {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<u8>,
}

/// Candidate filter settings for the automatic HLF producer.
///
/// s923 recovered the kernel shapes but not the vendor interval values. These
/// defaults are explicitly candidate values, not a vendor_equivalent claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HlfAutoProducerOptions {
    pub median_interval: u32,
    pub gauss_interval: u32,
}

impl Default for HlfAutoProducerOptions {
    fn default() -> Self {
        Self {
            median_interval: 1,
            gauss_interval: 1,
        }
    }
}

impl HlfAutoProducerOptions {
    fn validate(self) -> Result<Self, Refusal> {
        if self.median_interval == 0 || self.gauss_interval == 0 {
            return Err(Refusal::ValueOutOfRange("HLF auto interval"));
        }
        Ok(self)
    }
}

const HLF_MEDIAN_RADIUS: i32 = 1;
const HLF_GAUSS_SUPPORT: i32 = 4;
const HLF_GAUSS_WEIGHTS: [f32; 9] = [
    0.0162, 0.054, 0.1216, 0.1945, 0.227, 0.1945, 0.1216, 0.054, 0.0162,
];

#[derive(Clone, Debug, PartialEq)]
struct HlfAutoInputs {
    width: u32,
    height: u32,
    source_rgb_full_f32: Vec<f32>,
    median_rgb_full_f32: Vec<f32>,
    gauss_rgb_full_f32: Vec<f32>,
    face_mask_full_f32: Vec<f32>,
    skin_mask_full_f32: Vec<f32>,
    face_mask_rgba_full_f32: Vec<f32>,
    roi: Roi,
    preprocess_dispatches: Vec<Dispatch>,
    preprocess_fallback: Option<String>,
}

impl HlfAutoInputs {
    fn face(&self) -> HlfFaceInputs<'_> {
        HlfFaceInputs {
            width: self.width,
            height: self.height,
            source_rgb_full_f32: &self.source_rgb_full_f32,
            median_rgb_full_f32: &self.median_rgb_full_f32,
            gauss_rgb_full_f32: &self.gauss_rgb_full_f32,
            face_mask_full_f32: &self.face_mask_full_f32,
            skin_mask_full_f32: &self.skin_mask_full_f32,
            roi: self.roi,
        }
    }

    fn body(&self) -> HlfBodyInputs<'_> {
        HlfBodyInputs {
            width: self.width,
            height: self.height,
            median_rgb_full_f32: &self.median_rgb_full_f32,
            gauss_rgb_full_f32: &self.gauss_rgb_full_f32,
            face_mask_rgba_full_f32: &self.face_mask_rgba_full_f32,
            skin_mask_full_f32: &self.skin_mask_full_f32,
            roi: self.roi,
            uniforms: hlf_auto_body_uniforms(),
            adjust_lut: None,
        }
    }
}

fn hlf_auto_body_uniforms() -> BodyHlfUniforms {
    BodyHlfUniforms {
        spotless_degree: 0.0,
        retouch_degree: 1.0,
        spotless_acne_threshold: 0.0,
        spotless_high_freq_factor: 1.0,
        spotless_low_freq_alpha: 0.0,
        retouch_high_freq_factor: 0.0,
        diff_min_range: -0.2,
        diff_max_range: 0.2,
    }
}

fn hlf_auto_roi(region: NormRect, width: u32, height: u32) -> Result<Roi, Refusal> {
    check_norm_rect(region, "HLF auto region")?;
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("HLF auto frame"));
    }
    let x0 = (region.x * width as f32).floor() as u32;
    let y0 = (region.y * height as f32).floor() as u32;
    let x1 = ((region.x + region.width) * width as f32).ceil() as u32;
    let y1 = ((region.y + region.height) * height as f32).ceil() as u32;
    let x0 = x0.min(width - 1);
    let y0 = y0.min(height - 1);
    let x1 = x1.clamp(x0 + 1, width);
    let y1 = y1.clamp(y0 + 1, height);
    Ok(Roi {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

fn hlf_median_rgb(source: &[f32], width: usize, height: usize, interval: u32) -> Vec<f32> {
    let step = i64::from(interval);
    let mut out = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let dst = (y * width + x) * 3;
            for channel in 0..3 {
                let mut samples = [0.0f32; 9];
                let mut index = 0;
                for dy in -HLF_MEDIAN_RADIUS..=HLF_MEDIAN_RADIUS {
                    let sy = (y as i64 + i64::from(dy) * step)
                        .clamp(0, height as i64 - 1) as usize;
                    for dx in -HLF_MEDIAN_RADIUS..=HLF_MEDIAN_RADIUS {
                        let sx = (x as i64 + i64::from(dx) * step)
                            .clamp(0, width as i64 - 1) as usize;
                        samples[index] = source[(sy * width + sx) * 3 + channel];
                        index += 1;
                    }
                }
                samples.sort_by(f32::total_cmp);
                out[dst + channel] = samples[4];
            }
        }
    }
    out
}

fn hlf_gauss_rgb(source: &[f32], width: usize, height: usize, interval: u32) -> Vec<f32> {
    let step = i64::from(interval);
    let mut horizontal = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let dst = (y * width + x) * 3;
            for channel in 0..3 {
                let mut value = 0.0;
                for dx in -HLF_GAUSS_SUPPORT..=HLF_GAUSS_SUPPORT {
                    let sx = (x as i64 + i64::from(dx) * step)
                        .clamp(0, width as i64 - 1) as usize;
                    let wx = (dx + HLF_GAUSS_SUPPORT) as usize;
                    value += source[(y * width + sx) * 3 + channel] * HLF_GAUSS_WEIGHTS[wx];
                }
                horizontal[dst + channel] = value;
            }
        }
    }
    let mut out = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let dst = (y * width + x) * 3;
            for channel in 0..3 {
                let mut value = 0.0;
                for dy in -HLF_GAUSS_SUPPORT..=HLF_GAUSS_SUPPORT {
                    let sy = (y as i64 + i64::from(dy) * step)
                        .clamp(0, height as i64 - 1) as usize;
                    let wy = (dy + HLF_GAUSS_SUPPORT) as usize;
                    value += horizontal[(sy * width + x) * 3 + channel] * HLF_GAUSS_WEIGHTS[wy];
                }
                out[dst + channel] = value;
            }
        }
    }
    out
}

fn hlf_cpu_preprocess(
    source: &[f32],
    width: u32,
    height: u32,
    options: HlfAutoProducerOptions,
) -> (Vec<f32>, Vec<f32>) {
    (
        hlf_median_rgb(source, width as usize, height as usize, options.median_interval),
        hlf_gauss_rgb(source, width as usize, height as usize, options.gauss_interval),
    )
}

fn hlf_preprocess(
    source: &[f32],
    width: u32,
    height: u32,
    options: HlfAutoProducerOptions,
) -> (Vec<f32>, Vec<f32>, Vec<Dispatch>, Option<String>) {
    #[cfg(windows)]
    {
        if let Some(error) = forced_cuda_failure() {
            let (median, gauss) = hlf_cpu_preprocess(source, width, height, options);
            return (median, gauss, Vec::new(), Some(format!("{error:?}")));
        }
        match aurora_portrait::cuda_hlf::Hlf::open().and_then(|gpu| {
            gpu.preprocess_source_rgb(
                source,
                width,
                height,
                options.median_interval,
                options.gauss_interval,
            )
        }) {
            Ok((median, gauss, dispatches)) => {
                return (median, gauss, dispatches, None);
            }
            Err(error) => {
                let (median, gauss) = hlf_cpu_preprocess(source, width, height, options);
                return (median, gauss, Vec::new(), Some(format!("{error:?}")));
            }
        }
    }
    #[cfg(not(windows))]
    {
        let (median, gauss) = hlf_cpu_preprocess(source, width, height, options);
        (median, gauss, Vec::new(), None)
    }
}

fn produce_hlf_auto(
    current: &Image,
    region: NormRect,
    masks: &PortraitMasks,
    options: HlfAutoProducerOptions,
) -> Result<HlfAutoInputs, Refusal> {
    options.validate()?;
    current.validate()?;
    let pixels = current.pixel_count();
    let face_mask = masks
        .mask_f32(MaskKind::Face, current.width, current.height, region)
        .ok_or(Refusal::MissingInput("HLF auto face mask"))?;
    let skin_mask = masks
        .mask_f32(MaskKind::Skin, current.width, current.height, region)
        .ok_or(Refusal::MissingInput("HLF auto skin mask"))?;
    if face_mask.len() != pixels || skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("HLF auto masks"));
    }
    let mut source = Vec::with_capacity(pixels * 3);
    for pixel in current.pixels.chunks_exact(4) {
        source.push(pixel[0] as f32 / 255.0);
        source.push(pixel[1] as f32 / 255.0);
        source.push(pixel[2] as f32 / 255.0);
    }
    let (median, gauss, preprocess_dispatches, preprocess_fallback) =
        hlf_preprocess(&source, current.width, current.height, options);
    let mut face_rgba = Vec::with_capacity(pixels * 4);
    for value in &face_mask {
        face_rgba.extend_from_slice(&[*value, 0.0, 0.0, 1.0]);
    }
    Ok(HlfAutoInputs {
        width: current.width,
        height: current.height,
        source_rgb_full_f32: source,
        median_rgb_full_f32: median,
        gauss_rgb_full_f32: gauss,
        face_mask_full_f32: face_mask,
        skin_mask_full_f32: skin_mask,
        face_mask_rgba_full_f32: face_rgba,
        roi: hlf_auto_roi(region, current.width, current.height)?,
        preprocess_dispatches,
        preprocess_fallback,
    })
}

#[cfg(windows)]
fn try_cuda_hlf_auto(
    current: &Image,
    pf: u32,
    degree: f32,
    region: NormRect,
    masks: &PortraitMasks,
) -> Option<(Image, Vec<Dispatch>)> {
    if forced_cuda_failure().is_some() {
        return None;
    }
    if forced_cuda_empty_dispatch() {
        return None;
    }
    let options = HlfAutoProducerOptions::default();
    options.validate().ok()?;
    current.validate().ok()?;
    let pixels = current.pixel_count();
    let face_mask =
        masks.mask_f32(MaskKind::Face, current.width, current.height, region)?;
    let skin_mask =
        masks.mask_f32(MaskKind::Skin, current.width, current.height, region)?;
    if face_mask.len() != pixels || skin_mask.len() != pixels {
        return None;
    }
    let mut source_rgb = Vec::with_capacity(pixels * 3);
    for pixel in current.pixels.chunks_exact(4) {
        source_rgb.push(pixel[0] as f32 / 255.0);
        source_rgb.push(pixel[1] as f32 / 255.0);
        source_rgb.push(pixel[2] as f32 / 255.0);
    }
    let roi = hlf_auto_roi(region, current.width, current.height).ok()?;
    let gpu = aurora_portrait::cuda_hlf::Hlf::open().ok()?;
    let (next, dispatches) = match pf {
        30021 => {
            let (rgb, dispatches) = gpu
                .auto_face_apply(
                    &source_rgb,
                    &face_mask,
                    &skin_mask,
                    current.width,
                    current.height,
                    options.median_interval,
                    options.gauss_interval,
                    degree * 100.0,
                )
                .ok()?;
            (hlf_rgb_f32_to_image(current, &rgb).ok()?, dispatches)
        }
        30022 => {
            let mut face_rgba = Vec::with_capacity(pixels * 4);
            for value in &face_mask {
                face_rgba.extend_from_slice(&[*value, 0.0, 0.0, 1.0]);
            }
            gpu.auto_body_apply(
                current,
                &source_rgb,
                &face_rgba,
                &skin_mask,
                roi,
                options.median_interval,
                options.gauss_interval,
                degree * 100.0,
                hlf_auto_body_uniforms(),
            )
            .ok()?
        }
        _ => return None,
    };
    (!dispatches.is_empty()).then_some((next, dispatches))
}

impl ProducerInputs {
    pub fn write_pf90004(&mut self, bundle: Pf90004ProducerInputs) {
        self.pf90004 = Some(bundle);
    }

    pub fn fingerprint(&self) -> u64 {
        let mut hash = fnv1a64(FNV1A64_OFFSET_BASIS, b"aurora-producer-inputs-v2");
        hash = fnv_u64(hash, u64::from(self.teeth.is_some()));
        if let Some(teeth) = self.teeth.as_ref() {
            hash = fnv_usize(hash, teeth.mask_g.len());
            hash = fnv1a64(hash, &teeth.mask_g);
            hash = fnv_u64(hash, teeth.bright_ui.to_bits() as u64);
            hash = fnv_u64(hash, teeth.deyellow_ui.to_bits() as u64);
            hash = fnv_usize(hash, teeth.whiten_lut.len());
            hash = fnv1a64(hash, &teeth.whiten_lut);
            hash = fnv_usize(hash, teeth.bright_lut.len());
            hash = fnv1a64(hash, &teeth.bright_lut);
        }
        hash = match self.stray_hair.as_ref() {
            None => fnv_u64(hash, 0),
            Some(stray_hair) => {
                hash = fnv_u64(hash, 1);
                hash = match stray_hair.hairfly_v1.as_ref() {
                    None => fnv_u64(hash, 0),
                    Some(hairfly_v1) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(hash, hairfly_v1.width as u64);
                        hash = fnv_u64(hash, hairfly_v1.height as u64);
                        hash = fnv_usize(hash, hairfly_v1.pixels.len());
                        fnv1a64(hash, &hairfly_v1.pixels)
                    }
                };
                hash = fnv_option_bytes(hash, stray_hair.raw_mask.as_deref());
                hash = fnv_option_bytes(hash, stray_hair.person_mask.as_deref());
                hash = fnv_option_bytes(hash, stray_hair.background_mask.as_deref());
                hash = fnv_option_f32(hash, stray_hair.native_degree);
                hash = fnv_option_f32(hash, stray_hair.native_person_area);
                fnv_option_f32(hash, stray_hair.native_background_area)
            }
        };
        hash = match self.pf30025.as_ref() {
            None => fnv_u64(hash, 0),
            Some(pf30025) => {
                hash = fnv_u64(hash, 1);
                hash = fnv_usize(hash, pf30025.repair_780929437_bgra.len());
                hash = fnv1a64(hash, &pf30025.repair_780929437_bgra);
                hash = match pf30025.gate_3204548346.as_ref() {
                    None => fnv_u64(hash, 0),
                    Some(gate) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_usize(hash, gate.len());
                        fnv1a64(hash, gate)
                    }
                };
                hash = fnv_usize(hash, pf30025.teeth_mask.len());
                for value in &pf30025.teeth_mask {
                    hash = fnv_u64(hash, value.to_bits() as u64);
                }
                for value in [
                    pf30025.roi.x as u32,
                    pf30025.roi.y as u32,
                    pf30025.roi.width as u32,
                    pf30025.roi.height as u32,
                ] {
                    hash = fnv_u64(hash, value as u64);
                }
                hash
            }
        };
        hash = match self.teeth_alignment.as_ref() {
            None => fnv_u64(hash, 0),
            Some(alignment) => {
                hash = fnv_u64(hash, 1);
                hash = match alignment.effect_rgb.as_ref() {
                    None => fnv_u64(hash, 0),
                    Some(effect) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(hash, effect.width as u64);
                        hash = fnv_u64(hash, effect.height as u64);
                        hash = fnv_usize(hash, effect.pixels.len());
                        fnv1a64(hash, &effect.pixels)
                    }
                };
                hash = fnv_option_bytes(hash, alignment.teeth_mask.as_deref());
                hash = fnv_option_f32_plane(hash, alignment.edge_plane_a.as_deref());
                hash = fnv_option_f32_plane(hash, alignment.edge_plane_b.as_deref());
                match alignment.roi {
                    None => fnv_u64(hash, 0),
                    Some(roi) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(hash, roi.x.to_bits() as u64);
                        hash = fnv_u64(hash, roi.y.to_bits() as u64);
                        hash = fnv_u64(hash, roi.width.to_bits() as u64);
                        fnv_u64(hash, roi.height.to_bits() as u64)
                    }
                }
            }
        };
        hash = fnv_u64(hash, u64::from(self.pf90021.is_some()));
        if let Some(pf90021) = self.pf90021.as_ref() {
            hash = fnv_u64(
                hash,
                match pf90021.direction {
                    Pf90021Direction::Neutral => 0,
                    Pf90021Direction::Positive => 1,
                    Pf90021Direction::Negative => 2,
                },
            );
            for lut in [&pf90021.positive_lut, &pf90021.negative_lut] {
                hash = fnv_usize(hash, lut.nodes().len());
                for node in lut.nodes() {
                    hash = fnv_u64(hash, node.to_bits() as u64);
                }
            }
            hash = fnv_usize(hash, pf90021.mask_r.len());
            hash = fnv1a64(hash, &pf90021.mask_r);
            for value in [
                pf90021.runtime.degree01,
                pf90021.runtime.opacity01,
                pf90021.runtime.shelter,
                pf90021.runtime.region,
            ] {
                hash = fnv_u64(hash, value.to_bits() as u64);
            }
        }
        hash = fnv_u64(hash, u64::from(self.pf90004.is_some()));
        if let Some(pf90004) = self.pf90004.as_ref() {
            hash = fnv_u64(hash, u64::from(pf90004.width));
            hash = fnv_u64(hash, u64::from(pf90004.height));
            hash = fnv_usize(hash, pf90004.skin_mask.len());
            hash = fnv1a64(hash, &pf90004.skin_mask);
            hash = fnv_usize(hash, pf90004.parsing_rgba.len());
            hash = fnv1a64(hash, &pf90004.parsing_rgba);
            hash = fnv_usize(hash, pf90004.face_rgba.len());
            hash = fnv1a64(hash, &pf90004.face_rgba);
            hash = fnv_u64(hash, u64::from(pf90004.value_mean.to_bits()));
            for value in [
                pf90004.roi.x,
                pf90004.roi.y,
                pf90004.roi.width,
                pf90004.roi.height,
            ] {
                hash = fnv_u64(hash, u64::from(value.to_bits()));
            }
        }
        hash = fnv_u64(hash, u64::from(self.hlf_body_adjust_lut.is_some()));
        if let Some(lut) = self.hlf_body_adjust_lut.as_ref() {
            hash = fnv_u64(hash, lut.width as u64);
            hash = fnv_u64(hash, lut.height as u64);
            hash = fnv_usize(hash, lut.luma.len());
            hash = fnv1a64(hash, &lut.luma);
        }
        hash
    }
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

/// PF90004 same-job binding. The parser's skin mask is not a substitute for
/// the vendor's skin/parsing/face planes or its valueMean receipt.
#[derive(Clone, Copy, Debug)]
pub struct RadianceInputs<'a> {
    pub frame_width: Option<u32>,
    pub frame_height: Option<u32>,
    pub skin_mask: Option<&'a [u8]>,
    pub parsing_rgba: Option<&'a [u8]>,
    pub face_rgba: Option<&'a [u8]>,
    pub value_mean: Option<f32>,
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

/// Static PF90033 fields carried by the exported suit preset. Runtime pixels,
/// UVs and weights live in [`SuitPassSameJobInputs`].
#[derive(Clone, Copy, Debug)]
pub struct SuitPassPlan<'a> {
    pub order: u32,
    pub effect: &'a str,
    pub degree: Option<f32>,
    pub opacity: Option<f32>,
    pub filter_type: Option<u32>,
    pub material_path: Option<&'a str>,
    pub mask_field: Option<&'a str>,
    pub mask_value: Option<u32>,
    pub shelter_type: Option<i32>,
    pub force_use_shelter: Option<bool>,
    /// Geometry metadata only. This field never becomes the region weight.
    pub region_rect: Option<[i32; 4]>,
    pub mix_color: Option<[u8; 4]>,
}

/// Inputs that must come from the same renderer job as the output frame.
#[derive(Clone, Copy, Debug)]
pub struct SuitPassSameJobInputs<'a> {
    pub mask: Option<&'a Image>,
    pub mask_uv: Option<&'a [[f32; 2]]>,
    pub material: Option<&'a Image>,
    pub material_uv: Option<&'a [[f32; 2]]>,
    pub shelter: Option<f32>,
    pub region: Option<f32>,
}

fn suit_mask_channel(field: &str, value: u32) -> Result<MakeupMaskChannel, Refusal> {
    match (field, value) {
        ("BlusherMaskType", 15) => Ok(MakeupMaskChannel::Rgba),
        ("EyeBrowMaskType", 5) => Ok(MakeupMaskChannel::RB),
        ("EyeMaskType", 10) => Ok(MakeupMaskChannel::Rgb),
        ("EyeMaskType", 4) | ("EyeLashMaskType", 4) => Ok(MakeupMaskChannel::RG),
        ("LipMaskType", 3) => Ok(MakeupMaskChannel::R),
        _ => Err(Refusal::MissingInput("suit mask channel")),
    }
}

fn validate_suit_uv(
    uv: &[[f32; 2]],
    pixel_count: usize,
    name: &'static str,
) -> Result<(), Refusal> {
    if uv.len() != pixel_count {
        return Err(Refusal::ShapeMismatch(name));
    }
    if !uv
        .iter()
        .all(|point| point[0].is_finite() && point[1].is_finite())
    {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(())
}

fn suit_weight(value: Option<f32>, name: &'static str) -> Result<f32, Refusal> {
    let value = required(value, name)?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(value)
}

/// Converts one static pass plus same-job data into the existing stage input.
/// Missing runtime data is refused by name rather than defaulted.
pub fn produce_suit_pass<'a>(
    pixel_count: usize,
    plan: &SuitPassPlan<'_>,
    inputs: SuitPassSameJobInputs<'a>,
) -> Result<SuitPassInput<'a>, Refusal> {
    let mask = required(inputs.mask, "suit mask")?;
    mask.validate()?;
    let mask_uv = required(inputs.mask_uv, "suit mask uv")?;
    validate_suit_uv(mask_uv, pixel_count, "suit mask uv")?;

    let (mask_field, mask_value) = match (plan.mask_field, plan.mask_value) {
        (Some(field), Some(value)) => (field, value),
        (None, None) => return Err(Refusal::MissingInput("suit mask spec")),
        _ => return Err(Refusal::ShapeMismatch("suit mask spec")),
    };
    let mask_channel = suit_mask_channel(mask_field, mask_value)?;

    let degree = required(plan.degree, "suit degree")?;
    if !degree.is_finite() || !(0.0..=100.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("suit degree"));
    }
    let opacity = required(plan.opacity, "suit opacity")?;
    if !opacity.is_finite() || !(0.0..=100.0).contains(&opacity) {
        return Err(Refusal::ValueOutOfRange("suit opacity"));
    }
    let filter_type = required(plan.filter_type, "suit filter type")?;
    let mix_color = required(plan.mix_color, "suit mix color")?;
    let layer = MakeupLayer {
        mix_color,
        opacity_percent: opacity,
        degree01: degree / 100.0,
        filter_type,
        mask_channel,
    };

    let kind = match plan.effect {
        "BlusherImageMakeup"
        | "NormalEyeBrowImageMakeup"
        | "SymEyeImageMakeup"
        | "SymEyeLashImageMakeup"
        | "LipStickImageMakeup" => {
            required(plan.material_path, "suit material path")?;
            let material = required(inputs.material, "suit material")?;
            material.validate()?;
            let material_uv = required(inputs.material_uv, "suit material uv")?;
            validate_suit_uv(material_uv, pixel_count, "suit material uv")?;
            SuitPassKindInput::Image {
                material: Some(material),
                material_uv: Some(material_uv),
            }
        }
        "LipColor" => SuitPassKindInput::Color,
        _ => SuitPassKindInput::Unavailable("suit pass effect"),
    };

    let skip_shelter_query = plan.force_use_shelter == Some(true)
        || (plan.shelter_type.is_some() && plan.mask_field.is_none());
    let shelter = if skip_shelter_query {
        1.0
    } else {
        suit_weight(inputs.shelter, "suit shelter")?
    };
    let region = suit_weight(inputs.region, "suit region")?;

    Ok(SuitPassInput {
        kind,
        mask: Some(mask),
        mask_uv: Some(mask_uv),
        layer: Some(layer),
        shelter,
        region,
    })
}

/// Fail-fast 12-pass adapter. Plan and same-job arrays are positional and
/// ordered; the returned slice is accepted directly by `PortraitJobInputs`.
pub fn produce_suit_passes<'a>(
    pixel_count: usize,
    plans: Option<&[SuitPassPlan<'_>]>,
    inputs: Option<&'a [SuitPassSameJobInputs<'a>]>,
) -> Result<Vec<SuitPassInput<'a>>, Refusal> {
    let plans = required(plans, "suit passes")?;
    let inputs = required(inputs, "suit job inputs")?;
    if plans.is_empty() {
        return Err(Refusal::MissingInput("suit passes"));
    }
    if plans.len() != inputs.len() {
        return Err(Refusal::ShapeMismatch("suit passes"));
    }
    plans
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(index, (plan, inputs))| {
            if plan.order != index as u32 + 1 {
                return Err(Refusal::ValueOutOfRange("suit pass order"));
            }
            produce_suit_pass(pixel_count, plan, *inputs)
        })
        .collect()
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
    /// PF111 coverage; `material_uv[p]` is valid only when `writes[p] > 0`.
    pub material_uv_writes: Option<&'a [u16]>,
    pub mask: Option<&'a Image>,
    pub mask_uv: Option<&'a [[f32; 2]]>,
}

/// PF30021 consumes explicit full-frame f32 source/filter/mask planes. The
/// current RGBA8 stage is used only to preserve alpha in the final byte store.
#[derive(Clone, Copy, Debug)]
pub struct HlfFaceInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub source_rgb_full_f32: &'a [f32],
    pub median_rgb_full_f32: &'a [f32],
    pub gauss_rgb_full_f32: &'a [f32],
    pub face_mask_full_f32: &'a [f32],
    pub skin_mask_full_f32: &'a [f32],
    pub roi: Roi,
}

/// PF30022 keeps the current RGBA8 frame as source and requires full-frame
/// f32 median/gauss/mask planes. Only RGB inside `roi` is replaced.
#[derive(Clone, Copy, Debug)]
pub struct HlfBodyInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub median_rgb_full_f32: &'a [f32],
    pub gauss_rgb_full_f32: &'a [f32],
    pub face_mask_rgba_full_f32: &'a [f32],
    pub skin_mask_full_f32: &'a [f32],
    pub roi: Roi,
    pub uniforms: BodyHlfUniforms,
    pub adjust_lut: Option<AdjustLut<'a>>,
}

/// PF21025/21069/21070/21071 share one full-frame wrinkle bundle. `base`,
/// `residual`, masks and ROI all come from the same renderer job.
#[derive(Clone, Copy, Debug)]
pub struct WrinkleInputs<'a> {
    pub base_full: &'a Image,
    pub residual_full: &'a Image,
    pub mask_rgba_full: &'a [u8],
    pub skin_mask_full: &'a [u8],
    pub roi: Roi,
    pub uniforms: WrinkleUniforms,
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

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x100000001b3;

fn fnv1a64(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn fnv_usize(hash: u64, value: usize) -> u64 {
    fnv1a64(hash, &(value as u64).to_le_bytes())
}

fn fnv_u64(hash: u64, value: u64) -> u64 {
    fnv1a64(hash, &value.to_le_bytes())
}

fn fnv_option_bytes(hash: u64, bytes: Option<&[u8]>) -> u64 {
    match bytes {
        None => fnv_u64(hash, 0),
        Some(bytes) => {
            let hash = fnv_u64(hash, 1);
            fnv1a64(fnv_usize(hash, bytes.len()), bytes)
        }
    }
}

fn fnv_option_f32_plane(hash: u64, values: Option<&[f32]>) -> u64 {
    match values {
        None => fnv_u64(hash, 0),
        Some(values) => {
            let mut hash = fnv_usize(fnv_u64(hash, 1), values.len());
            for value in values {
                hash = fnv_u64(hash, value.to_bits() as u64);
            }
            hash
        }
    }
}

fn fnv_option_f32(hash: u64, value: Option<f32>) -> u64 {
    match value {
        None => fnv_u64(hash, 0),
        Some(value) => fnv_u64(fnv_u64(hash, 1), value.to_bits() as u64),
    }
}

/// Producer-supplied fingerprint for one immutable portrait input bundle.
///
/// The fingerprint binds the pre-effect RGBA8 frame, dimensions, normalized
/// region, the complete parameter table and all four mask planes. A receipt has
/// no public constructor with a caller-chosen id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortraitJobReceipt {
    fingerprint: u64,
    width: u32,
    height: u32,
    frame_len: usize,
    region_bits: [u32; 4],
    mask_width: usize,
    mask_height: usize,
    plane_lengths: [Option<usize>; 4],
}

impl PortraitJobReceipt {
    /// Builds a receipt from the current renderer inputs before portrait effects.
    pub fn from_renderer_receipt(
        frame: &[u8],
        width: u32,
        height: u32,
        region: NormRect,
        params: &BTreeMap<String, f32>,
        masks: &PortraitMasks,
    ) -> Self {
        let region_bits = [
            region.x.to_bits(),
            region.y.to_bits(),
            region.width.to_bits(),
            region.height.to_bits(),
        ];
        let planes = [&masks.skin, &masks.hair, &masks.face, &masks.body];
        let plane_lengths = [
            masks.skin.as_ref().map(Vec::len),
            masks.hair.as_ref().map(Vec::len),
            masks.face.as_ref().map(Vec::len),
            masks.body.as_ref().map(Vec::len),
        ];

        let mut fingerprint = fnv1a64(FNV1A64_OFFSET_BASIS, b"aurora-portrait-receipt-v1");
        fingerprint = fnv_usize(fingerprint, frame.len());
        fingerprint = fnv1a64(fingerprint, frame);
        fingerprint = fnv_u64(fingerprint, width as u64);
        fingerprint = fnv_u64(fingerprint, height as u64);
        for bits in region_bits {
            fingerprint = fnv_u64(fingerprint, bits as u64);
        }
        fingerprint = fnv_usize(fingerprint, params.len());
        for (key, value) in params {
            fingerprint = fnv_usize(fingerprint, key.len());
            fingerprint = fnv1a64(fingerprint, key.as_bytes());
            fingerprint = fnv_u64(fingerprint, value.to_bits() as u64);
        }
        fingerprint = fnv_usize(fingerprint, masks.width);
        fingerprint = fnv_usize(fingerprint, masks.height);
        for (index, plane) in planes.into_iter().enumerate() {
            fingerprint = fnv_u64(fingerprint, index as u64);
            fingerprint = fnv_u64(fingerprint, plane_lengths[index].unwrap_or(usize::MAX) as u64);
            if let Some(plane) = plane {
                for value in plane {
                    fingerprint = fnv_u64(fingerprint, value.to_bits() as u64);
                }
            }
        }

        Self {
            fingerprint,
            width,
            height,
            frame_len: frame.len(),
            region_bits,
            mask_width: masks.width,
            mask_height: masks.height,
            plane_lengths,
        }
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn job_id(&self) -> u64 {
        self.fingerprint
    }

    fn refusal(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        region: NormRect,
        params: &BTreeMap<String, f32>,
        masks: &PortraitMasks,
    ) -> Option<Refusal> {
        let expected_frame_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4));
        if Some(self.frame_len) != expected_frame_len
            || frame.len() != self.frame_len
            || self.width != width
            || self.height != height
            || self.region_bits
                != [
                    region.x.to_bits(),
                    region.y.to_bits(),
                    region.width.to_bits(),
                    region.height.to_bits(),
                ]
        {
            return Some(Refusal::ShapeMismatch("PortraitJobReceipt frame/region"));
        }

        let expected_plane_len = masks.width.checked_mul(masks.height);
        if self.mask_width != masks.width
            || self.mask_height != masks.height
            || self.plane_lengths
                != [
                    masks.skin.as_ref().map(Vec::len),
                    masks.hair.as_ref().map(Vec::len),
                    masks.face.as_ref().map(Vec::len),
                    masks.body.as_ref().map(Vec::len),
                ]
        {
            return Some(Refusal::ShapeMismatch("PortraitJobReceipt masks"));
        }
        for plane in [&masks.skin, &masks.hair, &masks.face, &masks.body] {
            if plane
                .as_ref()
                .is_some_and(|plane| Some(plane.len()) != expected_plane_len)
            {
                return Some(Refusal::ShapeMismatch("PortraitJobReceipt mask plane"));
            }
        }

        let actual = Self::from_renderer_receipt(frame, width, height, region, params, masks);
        (actual.fingerprint != self.fingerprint)
            .then_some(Refusal::ValueOutOfRange("PortraitJobReceipt digest"))
    }
}

/// Where a job's inputs came from. Only a renderer receipt or test-only
/// renderer job proves that absent fields are absent in the same job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortraitJobInputOrigin {
    MasksOnly,
    RendererJob { job_id: u64 },
}

/// Backend input container for effects whose producer is outside this stage.
pub struct PortraitJobInputs<'a> {
    pub origin: PortraitJobInputOrigin,
    pub receipt: Option<PortraitJobReceipt>,
    pub masks: &'a PortraitMasks,
    pub assets: PortraitAssetInputs<'a>,
    pub teeth: Option<TeethInputs<'a>>,
    pub pf30025: Option<Pf30025Inputs<'a>>,
    pub teeth_alignment: Option<TeethAlignmentInputs<'a>>,
    pub radiance: Option<RadianceInputs<'a>>,
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
    pub hlf_face: Option<HlfFaceInputs<'a>>,
    pub hlf_body: Option<HlfBodyInputs<'a>>,
    pub wrinkle: Option<WrinkleInputs<'a>>,
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
            receipt: None,
            masks,
            assets: PortraitAssetInputs::default(),
            teeth: None,
            pf30025: None,
            teeth_alignment: None,
            radiance: None,
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
            hlf_face: None,
            hlf_body: None,
            wrinkle: None,
            pf90021: None,
        }
    }

    /// Test-only legacy constructor with a caller-chosen label.
    #[cfg(test)]
    pub fn from_renderer_job(job_id: u64, masks: &'a PortraitMasks) -> Self {
        Self {
            origin: PortraitJobInputOrigin::RendererJob { job_id },
            ..Self::masks_only(masks)
        }
    }

    /// Production entry: the job id is the producer receipt's input fingerprint.
    pub fn from_renderer_receipt(
        receipt: PortraitJobReceipt,
        masks: &'a PortraitMasks,
    ) -> Self {
        Self {
            origin: PortraitJobInputOrigin::RendererJob {
                job_id: receipt.job_id(),
            },
            receipt: Some(receipt),
            ..Self::masks_only(masks)
        }
    }

    pub fn fill_from_producer_inputs(&mut self, producer: &'a ProducerInputs) {
        if let Some(teeth) = producer.teeth.as_ref() {
            self.teeth = Some(TeethInputs {
                mask_g: &teeth.mask_g,
                bright_ui: teeth.bright_ui,
                deyellow_ui: teeth.deyellow_ui,
            });
            self.assets.teeth_whiten_lut = Some(&teeth.whiten_lut);
            self.assets.teeth_bright_lut = Some(&teeth.bright_lut);
        }
        if let Some(stray_hair) = producer.stray_hair.as_ref() {
            self.stray_hair = StrayHairInputs {
                hairfly_v1: stray_hair.hairfly_v1.as_ref(),
                raw_mask: stray_hair.raw_mask.as_deref(),
                person_mask: stray_hair.person_mask.as_deref(),
                background_mask: stray_hair.background_mask.as_deref(),
                native_degree: stray_hair.native_degree,
                native_person_area: stray_hair.native_person_area,
                native_background_area: stray_hair.native_background_area,
            };
        }
        if let Some(pf30025) = producer.pf30025.as_ref() {
            self.pf30025 = Some(Pf30025Inputs {
                repair_780929437_bgra: &pf30025.repair_780929437_bgra,
                gate_3204548346: pf30025.gate_3204548346.as_deref(),
                teeth_mask: &pf30025.teeth_mask,
                roi: pf30025.roi,
            });
        }
        if let Some(alignment) = producer.teeth_alignment.as_ref() {
            self.teeth_alignment = Some(TeethAlignmentInputs {
                effect_rgb: alignment.effect_rgb.as_ref(),
                teeth_mask: alignment.teeth_mask.as_deref(),
                edge_plane_a: alignment.edge_plane_a.as_deref(),
                edge_plane_b: alignment.edge_plane_b.as_deref(),
                roi: alignment.roi,
            });
        }
        if let Some(pf90021) = producer.pf90021.as_ref() {
            self.pf90021 = Some(Pf90021Inputs {
                direction: pf90021.direction,
                positive_lut: Some(&pf90021.positive_lut),
                negative_lut: Some(&pf90021.negative_lut),
                mask_r: Some(&pf90021.mask_r),
                runtime: Some(pf90021.runtime),
            });
        }
        if let Some(pf90004) = producer.pf90004.as_ref() {
            self.radiance = Some(RadianceInputs {
                frame_width: Some(pf90004.width),
                frame_height: Some(pf90004.height),
                skin_mask: Some(&pf90004.skin_mask),
                parsing_rgba: Some(&pf90004.parsing_rgba),
                face_rgba: Some(&pf90004.face_rgba),
                value_mean: Some(pf90004.value_mean),
                roi: Some(pf90004.roi),
            });
        }
        if let (Some(body), Some(lut)) = (
            self.hlf_body.as_mut(),
            producer.hlf_body_adjust_lut.as_ref(),
        ) {
            body.adjust_lut = Some(AdjustLut {
                width: lut.width,
                height: lut.height,
                luma: &lut.luma,
            });
        }
    }

    pub fn job_id(&self) -> Option<u64> {
        if let Some(receipt) = self.receipt {
            return Some(receipt.job_id());
        }
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
    30025,  // Tooth repair
    30026,  // Teeth alignment final blend
    21025,  // Puppet wrinkle
    21069,  // Cheek wrinkle
    21070,  // Nose-back wrinkle
    21071,  // Perioral wrinkle
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
    30021,  // High-low frequency face
    30022,  // High-low frequency body
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
                match &effect.fallback_reason {
                    Some(reason) => log::debug!("portrait PF{} CPU fallback: {reason}", effect.pf),
                    None => log::debug!("portrait PF{} executed on CPU", effect.pf),
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
    apply_portrait_effects_with_report_and_inputs_policy(
        portrait,
        width,
        height,
        region,
        extra,
        inputs,
        cfg!(not(test)),
    )
}

fn apply_portrait_effects_with_report_and_inputs_policy(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    inputs: &PortraitJobInputs<'_>,
    require_renderer_receipt: bool,
) -> PortraitRunReport {
    let mut report = PortraitRunReport {
        job_id: inputs.job_id(),
        effects: Vec::new(),
    };
    let missing_renderer_receipt = require_renderer_receipt
        && inputs.receipt.is_none()
        && matches!(
            inputs.origin,
            PortraitJobInputOrigin::RendererJob { .. }
        );
    let receipt_refusal = if missing_renderer_receipt {
        Some(Refusal::MissingInput("renderer receipt"))
    } else {
        inputs.receipt.as_ref().and_then(|receipt| {
            receipt.refusal(portrait, width, height, region, extra, inputs.masks)
        })
    };
    let mut effects: Vec<(u32, f32)> = extra
        .iter()
        .filter_map(|(key, value)| {
            let id: u32 = key.strip_prefix("pc:")?.parse().ok()?;
            Some((id, *value))
        })
        .collect();
    effects.sort_by_key(|&(pf, _)| pf);

    if let Some(refusal) = receipt_refusal {
        for &(pf, _) in &effects {
            let Some(meta) = aurora_portrait::lookup(pf) else {
                continue;
            };
            if meta.access != Access::Cpu || !SUPPORTED_EFFECTS.contains(&pf) {
                continue;
            }
            report.effects.push(PortraitEffectRun {
                pf,
                backend: PortraitBackend::Skipped,
                dispatches: Vec::new(),
                fallback_reason: Some(format!("receipt refused: {refusal:?}")),
                skipped: true,
                executed: false,
                refusal: Some(refusal.clone()),
                skip_reason: Some(PortraitSkipReason::InputRefused(refusal.clone())),
            });
        }
        return report;
    }

    let Some(source) = Image::new(width, height, std::mem::take(portrait)) else {
        return report;
    };

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
        let needs_hlf_auto = (pf == 30021 && inputs.hlf_face.is_none())
            || (pf == 30022 && inputs.hlf_body.is_none());
        #[cfg(windows)]
        if needs_hlf_auto {
            if let Some((next, dispatches)) =
                try_cuda_hlf_auto(&current, pf, degree, region, inputs.masks)
            {
                current = next;
                run.backend = PortraitBackend::Cuda;
                run.dispatches = dispatches;
                run.skipped = false;
                run.executed = true;
                report.effects.push(run);
                continue;
            }
        }
        let hlf_auto = if needs_hlf_auto {
            match produce_hlf_auto(
                &current,
                region,
                inputs.masks,
                HlfAutoProducerOptions::default(),
            ) {
                Ok(auto) => Some(auto),
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                    run.refusal = Some(refusal.clone());
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal));
                    report.effects.push(run);
                    continue;
                }
            }
        } else {
            None
        };
        if let Some(auto) = hlf_auto.as_ref() {
            if let Some(reason) = auto.preprocess_fallback.as_ref() {
                run.fallback_reason = Some(format!("HLF preprocessing CPU fallback: {reason}"));
            }
        }
        if let Err(refusal) =
            validate_effect_inputs(
                pf,
                &current,
                region,
                extra,
                skin_mask.as_deref(),
                inputs,
                hlf_auto.as_ref(),
            )
        {
            run.fallback_reason = Some(format!("input refused: {refusal:?}"));
            run.refusal = Some(refusal);
            report.effects.push(run);
            continue;
        }
        #[cfg(windows)]
        let cuda_result =
            try_cuda_effect(&current, pf, degree, skin_mask.as_deref()).or_else(|| {
                try_cuda_same_job_effect(&current, pf, degree, inputs, hlf_auto.as_ref())
            });
        #[cfg(windows)]
        if let Some(result) = cuda_result {
            match result {
                Ok((next, dispatches)) if !dispatches.is_empty() => {
                    current = next;
                    run.backend = PortraitBackend::Cuda;
                    if let Some(auto) = hlf_auto.as_ref() {
                        run.dispatches = auto.preprocess_dispatches.clone();
                        run.dispatches.extend(dispatches);
                    } else {
                        run.dispatches = dispatches;
                    }
                    run.skipped = false;
                    run.executed = true;
                    report.effects.push(run);
                    continue;
                }
                Ok(_) => {
                    let reason = "CUDA returned no dispatch; CPU kernel selected for parity";
                    run.fallback_reason = Some(match run.fallback_reason.take() {
                        Some(previous) => format!("{previous}; {reason}"),
                        None => reason.to_string(),
                    });
                }
                Err(err) => {
                    let reason = format!("{err:?}");
                    run.fallback_reason = Some(match run.fallback_reason.take() {
                        Some(previous) => format!("{previous}; CUDA effect: {reason}"),
                        None => reason,
                    });
                }
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
            hlf_auto.as_ref(),
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

/// Runs the PF8050/PF8051 six-stage candidate geometry path.
pub fn apply_pf8050_pf8051_geometry(
    inputs: Pf8050GeometryInputs<'_>,
) -> PortraitGeometryRun {
    let resolved = match cuda_pf8050::resolve(inputs) {
        Ok(resolved) => resolved,
        Err(refusal) => {
            return PortraitGeometryRun {
                backend: PortraitBackend::Skipped,
                selection: Selection::Refused(DeviceRefusal::EffectUnsupported),
                dispatches: Vec::new(),
                fallback_reason: Some(format!("input refused: {refusal:?}")),
                refusal: Some(refusal),
                stages: Vec::new(),
                candidate_only: true,
            };
        }
    };

    #[cfg(windows)]
    let fallback_reason = {
        let cuda_result = match forced_cuda_failure() {
            Some(error) => Err(error),
            None => match cuda_pf8050::Pf8050Stage::open() {
                Ok(gpu) => gpu.apply(resolved),
                Err(error) => Err(error),
            },
        };
        match cuda_result {
            Ok((output, dispatch)) if !dispatch.kernel.is_empty() => {
                return PortraitGeometryRun {
                    backend: PortraitBackend::Cuda,
                    selection: Selection::Selected(DeviceBackend::Cuda),
                    dispatches: vec![dispatch],
                    fallback_reason: None,
                    refusal: None,
                    stages: output.stages,
                    candidate_only: true,
                };
            }
            Ok(_) => Some("CUDA returned no dispatch".to_string()),
            Err(error) => Some(format!("{error:?}")),
        }
    };
    #[cfg(not(windows))]
    let fallback_reason = None;

    let output = cuda_pf8050::apply_cpu(resolved);
    PortraitGeometryRun {
        backend: PortraitBackend::Cpu,
        selection: Selection::Selected(DeviceBackend::Cpu),
        dispatches: Vec::new(),
        fallback_reason,
        refusal: None,
        stages: output.stages,
        candidate_only: true,
    }
}

/// Full product stage entry. Missing vendor receipts are typed refusals; a
/// complete receipt set still reports the open raster consumer, never success.
pub fn apply_pf8050_pf8051_stage(inputs: Pf8050StageInputs<'_>) -> PortraitGeometryRun {
    if let Err(refusal) = require_pf8050_vendor_inputs(inputs.runtime) {
        return PortraitGeometryRun {
            backend: PortraitBackend::Skipped,
            selection: Selection::Refused(DeviceRefusal::EffectUnsupported),
            dispatches: Vec::new(),
            fallback_reason: Some(format!("vendor input refused: {refusal:?}")),
            refusal: Some(refusal),
            stages: Vec::new(),
            candidate_only: false,
        };
    }

    let mut run = apply_pf8050_pf8051_geometry(inputs.geometry);
    run.refusal = Some(Refusal::NotImplemented(8050));
    run.fallback_reason = Some(
        "vendor VBO/FBO/readback consumer is not wired; candidate geometry only".to_string(),
    );
    run
}

fn require_pf8050_vendor_inputs(runtime: Pf8050VendorRuntime<'_>) -> Result<(), Refusal> {
    required(runtime.attach, "PF8050_ATTACH")?;
    required(runtime.vec3, "PF8050_VEC3")?;
    required(runtime.roi, "PF8050_ROI")?;
    required(runtime.vbo, "PF8050_VBO")?;
    required(runtime.fbo, "PF8050_FBO")?;
    required(runtime.readback, "PF8050_READBACK")?;
    Ok(())
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

fn check_f32_plane(plane: &[f32], expected: usize, name: &'static str) -> Result<(), Refusal> {
    check_plane(plane.len(), expected, name)?;
    if plane.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
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

fn check_skin_roi(width: u32, height: u32, roi: Roi) -> Result<(), Refusal> {
    check_roi(width, height, (roi.x, roi.y, roi.width, roi.height))
}

fn validate_hlf_face(current: &Image, input: HlfFaceInputs<'_>) -> Result<(), Refusal> {
    if (input.width, input.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF30021_HLF frame"));
    }
    check_skin_roi(current.width, current.height, input.roi)?;
    let pixels = current.pixel_count();
    check_f32_plane(
        input.source_rgb_full_f32,
        pixels * 3,
        "PF30021_source_rgb_full_f32",
    )?;
    check_f32_plane(input.median_rgb_full_f32, pixels * 3, "median_rgb_full_f32")?;
    check_f32_plane(input.gauss_rgb_full_f32, pixels * 3, "gauss_rgb_full_f32")?;
    check_f32_plane(input.face_mask_full_f32, pixels, "face_mask_full_f32")?;
    check_f32_plane(input.skin_mask_full_f32, pixels, "skin_mask_full_f32")
}

fn validate_hlf_body(current: &Image, input: HlfBodyInputs<'_>) -> Result<(), Refusal> {
    if (input.width, input.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF30022_HLF frame"));
    }
    check_skin_roi(current.width, current.height, input.roi)?;
    let pixels = current.pixel_count();
    check_f32_plane(input.median_rgb_full_f32, pixels * 3, "median_rgb_full_f32")?;
    check_f32_plane(input.gauss_rgb_full_f32, pixels * 3, "gauss_rgb_full_f32")?;
    check_f32_plane(
        input.face_mask_rgba_full_f32,
        pixels * 4,
        "face_mask_rgba_full_f32",
    )?;
    check_f32_plane(input.skin_mask_full_f32, pixels, "skin_mask_full_f32")?;
    let uniforms = [
        input.uniforms.spotless_degree,
        input.uniforms.retouch_degree,
        input.uniforms.spotless_acne_threshold,
        input.uniforms.spotless_high_freq_factor,
        input.uniforms.spotless_low_freq_alpha,
        input.uniforms.retouch_high_freq_factor,
        input.uniforms.diff_min_range,
        input.uniforms.diff_max_range,
    ];
    if !uniforms.iter().all(|value| value.is_finite())
        || input.uniforms.diff_min_range > input.uniforms.diff_max_range
    {
        return Err(Refusal::ValueOutOfRange("PF30022_HLF uniforms"));
    }
    if let Some(lut) = input.adjust_lut.as_ref() {
        lut.validate()?;
    } else if input.uniforms.spotless_degree > 0.0 {
        return Err(Refusal::MissingInput("adjust_lut"));
    }
    Ok(())
}

fn validate_wrinkle(current: &Image, input: WrinkleInputs<'_>) -> Result<(), Refusal> {
    input.base_full.validate()?;
    input.residual_full.validate()?;
    if (input.base_full.width, input.base_full.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("wrinkle_base_full"));
    }
    if (input.residual_full.width, input.residual_full.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("wrinkle_residual_full"));
    }
    let pixels = current.pixel_count();
    check_plane(
        input.mask_rgba_full.len(),
        pixels * 4,
        "wrinkle_mask_rgba_full",
    )?;
    check_plane(input.skin_mask_full.len(), pixels, "wrinkle_skin_mask_full")?;
    check_skin_roi(current.width, current.height, input.roi)?;
    let uniforms = [
        input.uniforms.legal_alpha,
        input.uniforms.dark_circle_alpha,
        input.uniforms.diff_factor,
        input.uniforms.diff_factor2,
        input.uniforms.color_threshold_factor,
        input.uniforms.mix_alpha,
        input.uniforms.diff_min_threshold,
        input.uniforms.diff_max_threshold,
        input.uniforms.high_freq_min_threshold,
        input.uniforms.high_freq_max_threshold,
    ];
    if !uniforms.iter().all(|value| value.is_finite())
        || input.uniforms.diff_min_threshold > input.uniforms.diff_max_threshold
        || input.uniforms.high_freq_min_threshold > input.uniforms.high_freq_max_threshold
    {
        return Err(Refusal::ValueOutOfRange("wrinkle_uniforms"));
    }
    Ok(())
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
            None => with_pf90021(|gpu| {
                gpu.run(aurora_portrait::cuda_pf90021::Request {
                    source: current,
                    direction,
                    lut: Some(lut),
                    mask_r: Some(mask_r),
                    runtime: Some(runtime),
                })
            }),
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

#[cfg(windows)]
thread_local! {
    static PF90021: RefCell<Option<aurora_portrait::cuda_pf90021::Pf90021>> =
        const { RefCell::new(None) };
}

#[cfg(windows)]
fn with_pf90021<T>(
    run: impl FnOnce(&aurora_portrait::cuda_pf90021::Pf90021) -> Result<T, CudaError>,
) -> Result<T, CudaError> {
    PF90021.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(aurora_portrait::cuda_pf90021::Pf90021::open()?);
        } else if let Err(error) = slot.as_mut().expect("PF90021 session").resume() {
            *slot = None;
            return Err(error);
        }
        let result = run(slot.as_ref().expect("PF90021 session"));
        if result.is_err() {
            let _ = slot.as_mut().expect("PF90021 session").suspend();
            *slot = None;
            return result;
        }
        match slot.as_mut().expect("PF90021 session").suspend() {
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

fn pf30026_cpu_fallback(
    job: AlignmentFinalBlendJob<'_>,
    cuda_reason: String,
) -> (Option<Image>, PortraitEffectRun) {
    match aurora_portrait::kernels::teeth_blend::apply_alignment_final_blend(job) {
        Ok(AlignmentFinalBlendOutcome::Applied(image)) => (
            Some(image),
            PortraitEffectRun {
                pf: 30026,
                backend: PortraitBackend::Cpu,
                dispatches: Vec::new(),
                fallback_reason: Some(cuda_reason),
                skipped: false,
                executed: true,
                refusal: None,
                skip_reason: None,
            },
        ),
        Ok(AlignmentFinalBlendOutcome::Gated) => (
            None,
            pf30026_skipped(
                PortraitSkipReason::EdgeGateClosed,
                format!("{cuda_reason}; CPU edge gate closed"),
                None,
            ),
        ),
        Err(error) => (
            None,
            pf30026_skipped(
                PortraitSkipReason::InputRefused(error.clone()),
                format!("{cuda_reason}; CPU: {error:?}"),
                Some(error),
            ),
        ),
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
            return pf30026_cpu_fallback(job, format!("CUDA unavailable: {error:?}"));
        }
        let gpu = match aurora_portrait::cuda_teeth_blend::TeethBlend::open() {
            Ok(gpu) => gpu,
            Err(error) => return pf30026_cpu_fallback(job, format!("CUDA unavailable: {error:?}")),
        };
        match gpu.apply_alignment_final_blend(job) {
            Ok(aurora_portrait::cuda_teeth_blend::AlignmentFinalBlendGpu::Applied {
                image,
                dispatches,
            }) if !dispatches.is_empty() => (
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
            Ok(aurora_portrait::cuda_teeth_blend::AlignmentFinalBlendGpu::Applied { .. }) => {
                pf30026_cpu_fallback(
                    job,
                    "CUDA returned no dispatch; CPU kernel selected for parity".to_string(),
                )
            }
            Ok(aurora_portrait::cuda_teeth_blend::AlignmentFinalBlendGpu::Gated) => (
                None,
                pf30026_skipped(
                    PortraitSkipReason::EdgeGateClosed,
                    "PF30026 edge gate closed: valid ratio <= 0.14%".to_string(),
                    None,
                ),
            ),
            Err(error) => pf30026_cpu_fallback(job, format!("CUDA failed: {error:?}")),
        }
    }

    #[cfg(not(windows))]
    {
        let error = CudaError::Unavailable("PF30026 requires NVIDIA CUDA");
        pf30026_cpu_fallback(job, format!("CUDA unavailable: {error:?}"))
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

fn validate_makeup_inputs(
    pf: u32,
    pixels: usize,
    inputs: &PortraitJobInputs<'_>,
) -> Result<(), Refusal> {
    let (route, input) = find_makeup_input(pf, inputs)?;
    match route {
        MakeupRoute::Color { .. } => {
            required(input.mask, "makeup mask")?.validate()?;
            required(input.mask_uv, "makeup mask UV")?;
        }
        MakeupRoute::ImageUnmasked { .. } => {
            required(input.material, "makeup material")?.validate()?;
            let material_uv = required(input.material_uv, "makeup material UV")?;
            check_plane(material_uv.len(), pixels, "makeup material UV")?;
            let writes = required(input.material_uv_writes, "makeup material UV writes")?;
            check_plane(writes.len(), pixels, "makeup material UV writes")?;
            if input.mask.is_some() || input.mask_uv.is_some() {
                return Err(Refusal::ValueOutOfRange("PF111 mask must be absent"));
            }
        }
    }
    Ok(())
}

struct ResolvedRadiance<'a> {
    skin: &'a [u8],
    parsing_rgba: &'a [u8],
    face_rgba: &'a [u8],
    value_mean: f32,
}

fn resolve_radiance_inputs<'a>(
    current: &Image,
    job: Option<RadianceInputs<'a>>,
) -> Result<ResolvedRadiance<'a>, Refusal> {
    let job = required(job, "PF90004_SAME_JOB_INPUTS")?;
    let frame_width = required(job.frame_width, "PF90004_FRAME_WIDTH")?;
    let frame_height = required(job.frame_height, "PF90004_FRAME_HEIGHT")?;
    if frame_width != current.width || frame_height != current.height {
        return Err(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"));
    }
    let skin = required(job.skin_mask, "PF90004_SKIN_MASK")?;
    let parsing_rgba = required(job.parsing_rgba, "PF90004_PARSING_RGBA")?;
    let face_rgba = required(job.face_rgba, "PF90004_FACE_RGBA")?;
    let value_mean = required(job.value_mean, "PF90004_VALUE_MEAN")?;
    let roi = required(job.roi, "PF90004_ROI")?;
    if roi != NormRect::FULL {
        return Err(Refusal::ValueOutOfRange("PF90004_ROI"));
    }
    let pixels = current.pixel_count();
    check_plane(skin.len(), pixels, "skin_mask")?;
    check_plane(parsing_rgba.len(), pixels.saturating_mul(4), "parsing_rgba")?;
    check_plane(face_rgba.len(), pixels.saturating_mul(4), "face_rgba")?;
    if !value_mean.is_finite() || !(0.0..=255.0).contains(&value_mean) {
        return Err(Refusal::ValueOutOfRange("PF90004_VALUE_MEAN"));
    }
    Ok(ResolvedRadiance {
        skin,
        parsing_rgba,
        face_rgba,
        value_mean,
    })
}

fn validate_effect_inputs(
    pf: u32,
    current: &Image,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    skin_mask: Option<&[u8]>,
    inputs: &PortraitJobInputs<'_>,
    hlf_auto: Option<&HlfAutoInputs>,
) -> Result<(), Refusal> {
    let pixels = current.pixel_count();
    if pf == 90004 {
        check_degree01(
            extra.get("pc:90004").copied().unwrap_or_default(),
            "degree",
        )?;
        if inputs.radiance.is_none() {
            let automatic = inputs.masks.pf90004_parser_inputs(
                &current.pixels,
                current.width,
                current.height,
                region,
            )?;
            if automatic.parsing_rgba.is_none() {
                return Err(Refusal::MissingInput("PF90004_PARSING_RGBA"));
            }
        }
        resolve_radiance_inputs(current, inputs.radiance)?;
        return Ok(());
    }
    if matches!(
        pf,
        8068 | 8069 | 8070 | 21015 | 30096 | 90005 | 90007
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
        100 | 103 | 105 | 111 | 117 | 118 | 30007 => {
            validate_makeup_inputs(pf, pixels, inputs)?
        }
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
        30025 => {
            check_degree01(
                extra.get("pc:30025").copied().unwrap_or_default(),
                "PF30025 degree",
            )?;
            let pf30025 = required(inputs.pf30025, "pf30025 repair_bgra")?;
            if pf30025.repair_780929437_bgra.is_empty() {
                return Err(Refusal::MissingInput("pf30025 repair_bgra"));
            }
            let bytes = pixels
                .checked_mul(4)
                .ok_or(Refusal::ShapeMismatch("pf30025 repair_bgra"))?;
            check_plane(
                pf30025.repair_780929437_bgra.len(),
                bytes,
                "pf30025 repair_bgra",
            )?;
            check_f32_plane(pf30025.teeth_mask, pixels, "pf30025 teeth mask")?;
            if !pf30025.roi.inside(current.width, current.height) {
                return Err(Refusal::ValueOutOfRange("pf30025 roi"));
            }
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
        21025 | 21069 | 21070 | 21071 => {
            check_degree01(
                extra.get(&format!("pc:{pf}")).copied().unwrap_or_default(),
                "wrinkle ui",
            )?;
            let input = required(
                inputs.wrinkle,
                "PF21025_PF21069_PF21070_PF21071_WRINKLE_SAME_JOB_INPUTS",
            )?;
            validate_wrinkle(current, input)?;
        }
        30021 => {
            check_degree01(
                extra.get("pc:30021").copied().unwrap_or_default(),
                "PF30021 ui",
            )?;
            let input = match inputs.hlf_face {
                Some(input) => input,
                None => hlf_auto
                    .ok_or(Refusal::MissingInput("PF30021_HLF_SAME_JOB_INPUTS"))?
                    .face(),
            };
            validate_hlf_face(current, input)?;
        }
        30022 => {
            check_degree01(
                extra.get("pc:30022").copied().unwrap_or_default(),
                "PF30022 ui",
            )?;
            let input = match inputs.hlf_body {
                Some(input) => input,
                None => hlf_auto
                    .ok_or(Refusal::MissingInput("PF30022_HLF_SAME_JOB_INPUTS"))?
                    .body(),
            };
            validate_hlf_body(current, input)?;
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
    hlf_auto: Option<&HlfAutoInputs>,
) -> Result<Image, Refusal> {
    match pf {
        100 | 103 | 105 | 111 | 117 | 118 | 30007 => {
            apply_makeup_cpu(current, pf, degree, inputs)
        }
        108 => apply_teeth_cpu(current, degree, inputs),
        30025 => apply_pf30025_cpu(current, degree, inputs),
        21025 | 21069 | 21070 | 21071 => apply_wrinkle_cpu(current, pf, degree, inputs),
        30021 => apply_hlf_face_cpu(current, degree, inputs, hlf_auto),
        30022 => apply_hlf_body_cpu(current, degree, inputs, hlf_auto),
        8068 => apply_skin_gloss(current, degree, skin_mask),
        8069 => apply_skin_texture(current, degree, skin_mask),
        8070 => apply_skin_clarity(current, degree, skin_mask, width, height),
        21015 => apply_skin_tint(current, degree, skin_mask),
        21026 => apply_beard_cpu(current, degree, inputs),
        30096 => apply_skin_temperature(current, degree, skin_mask),
        90002 => apply_moisture_face_cpu(current, degree, inputs),
        90004 => apply_skin_radiance(current, degree, inputs.radiance),
        90005 => apply_skin_whiten(current, degree, skin_mask),
        90007 => apply_skin_ruddy(current, degree, skin_mask),
        90010 => apply_moisture_body_cpu(current, degree, inputs),
        90116 => apply_eye_bag_cpu(current, degree, inputs),
        90139 => apply_hair_smooth_cpu(current, degree, inputs),
        90160 => apply_hair_gloss_cpu(current, degree, inputs),
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

fn apply_hair_smooth_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let (input, roi) = resolve_hair_smooth(current, degree, inputs)?;
    let hair_mask = required(input.hair_mask, "PF90139_HAIR_MASK")?;
    let mix_mask = required(input.mix_mask, "PF90139_MIX_MASK")?;
    let blend = required(input.blend_rgb, "PF90139_BLEND_SOURCE_RGB8")?;
    let detail = required(input.detail, "PF90139_DETAIL")?;
    let mut output = current.clone();
    for y in 0..roi.height as usize {
        for x in 0..roi.width as usize {
            let hair = hair_mask[y * input.hair_stride as usize + x];
            if hair == 0 {
                continue;
            }
            let mix = mix_mask[y * input.mix_stride as usize + x];
            let detail = detail[y * input.detail_stride as usize + x];
            let gate = hair as f32 * 0.003921568859368563;
            let mix_weight = mix as f32 * 0.003921568859368563 * input.mix_strength * gate;
            let detail_add = (detail as f32 - 127.5) * 4.0 * (gate * degree * 0.8);
            let offset =
                ((roi.y as usize + y) * current.width as usize + roi.x as usize + x) * 4;
            let blend_offset = y * input.blend_stride as usize + x * 3;
            for channel in 0..3 {
                let source = current.pixels[offset + channel] as f32;
                let base = if mix_weight > 0.0 {
                    source * (1.0 - mix_weight)
                        + blend[blend_offset + channel] as f32 * mix_weight
                } else {
                    source
                };
                output.pixels[offset + channel] = (base + detail_add)
                    .round_ties_even()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

fn apply_hair_gloss_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let (input, roi) = resolve_hair_gloss(current, degree, inputs)?;
    let hair_mask = required(input.hair_mask, "PF90160_HAIR_MASK")?;
    let detail = required(input.model_detail, "PF90160_MODEL_DETAIL")?;
    let mut output = current.clone();
    for y in 0..roi.height as usize {
        for x in 0..roi.width as usize {
            let hair = hair_mask[y * input.hair_stride as usize + x];
            if hair == 0 {
                continue;
            }
            let detail = detail[y * input.detail_stride as usize + x];
            let delta = (detail as f32 - 127.5)
                * 2.0
                * hair as f32
                * 0.003921568859368563
                * input.dynamic_scale;
            let offset =
                ((roi.y as usize + y) * current.width as usize + roi.x as usize + x) * 4;
            for channel in 0..3 {
                output.pixels[offset + channel] =
                    (current.pixels[offset + channel] as f32 + delta).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

fn apply_wrinkle_cpu(
    current: &Image,
    pf: u32,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = required(
        inputs.wrinkle,
        "PF21025_PF21069_PF21070_PF21071_WRINKLE_SAME_JOB_INPUTS",
    )?;
    validate_wrinkle(current, input)?;
    let degree = skin_adjust::wrinkle_degree(pf, degree * 100.0)?;
    skin_adjust::wrinkle_apply(
        current,
        input.base_full,
        input.residual_full,
        input.mask_rgba_full,
        input.skin_mask_full,
        input.roi,
        degree,
        input.uniforms,
    )
}

fn hlf_rgb_f32_to_image(current: &Image, rgb: &[f32]) -> Result<Image, Refusal> {
    check_plane(
        rgb.len(),
        current.pixel_count() * 3,
        "PF30021_HLF output",
    )?;
    let mut pixels = current.pixels.clone();
    for (index, rgb) in rgb.chunks_exact(3).enumerate() {
        let offset = index * 4;
        for channel in 0..3 {
            pixels[offset + channel] = StoreMode::TruncU8.quantize(rgb[channel]);
        }
    }
    Ok(Image {
        width: current.width,
        height: current.height,
        pixels,
    })
}

fn apply_hlf_face_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
    hlf_auto: Option<&HlfAutoInputs>,
) -> Result<Image, Refusal> {
    let input = match inputs.hlf_face {
        Some(input) => input,
        None => hlf_auto
            .ok_or(Refusal::MissingInput("PF30021_HLF_SAME_JOB_INPUTS"))?
            .face(),
    };
    validate_hlf_face(current, input)?;
    let rgb = skin_adjust::hlf_face_apply(
        input.source_rgb_full_f32,
        input.median_rgb_full_f32,
        input.gauss_rgb_full_f32,
        input.face_mask_full_f32,
        input.skin_mask_full_f32,
        current.width,
        current.height,
        input.roi,
        degree * 100.0,
    )?;
    hlf_rgb_f32_to_image(current, &rgb)
}

fn apply_hlf_body_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
    hlf_auto: Option<&HlfAutoInputs>,
) -> Result<Image, Refusal> {
    let input = match inputs.hlf_body {
        Some(input) => input,
        None => hlf_auto
            .ok_or(Refusal::MissingInput("PF30022_HLF_SAME_JOB_INPUTS"))?
            .body(),
    };
    validate_hlf_body(current, input)?;
    skin_adjust::body_hlf_apply(
        current,
        input.roi,
        input.median_rgb_full_f32,
        input.gauss_rgb_full_f32,
        input.face_mask_rgba_full_f32,
        input.skin_mask_full_f32,
        degree * 100.0,
        input.uniforms,
        input.adjust_lut.as_ref(),
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
            .and_then(|output| {
                gate_makeup_writes(
                    current,
                    output,
                    required(input.material_uv_writes, "makeup material UV writes")?,
                )
            })
        }
    }
}

fn gate_makeup_writes(source: &Image, mut output: Image, writes: &[u16]) -> Result<Image, Refusal> {
    check_plane(writes.len(), source.pixel_count(), "makeup material UV writes")?;
    for (pixel, writes) in writes.iter().enumerate() {
        if *writes == 0 {
            let base = pixel * 4;
            output.pixels[base..base + 4].copy_from_slice(&source.pixels[base..base + 4]);
        }
    }
    Ok(output)
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

fn apply_pf30025_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let pf30025 = required(inputs.pf30025, "pf30025 repair_bgra")?;
    aurora_portrait::kernels::face_detail::tooth_repair_same_job(
        current,
        Some(ToothRepairProducer {
            repair_780929437_bgra: pf30025.repair_780929437_bgra,
            gate_3204548346: pf30025.gate_3204548346,
            teeth_mask: pf30025.teeth_mask,
            roi: pf30025.roi,
        }),
        degree * 100.0,
    )
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
    static FORCE_CUDA_EMPTY_DISPATCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(windows)]
fn forced_cuda_failure() -> Option<CudaError> {
    #[cfg(test)]
    if FORCE_CUDA_FAILURE.with(std::cell::Cell::get) {
        return Some(CudaError::Unavailable("test-forced CUDA failure"));
    }
    None
}

#[cfg(windows)]
fn forced_cuda_empty_dispatch() -> bool {
    #[cfg(test)]
    if FORCE_CUDA_EMPTY_DISPATCH.with(std::cell::Cell::get) {
        return true;
    }
    false
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
    hlf_auto: Option<&HlfAutoInputs>,
) -> Option<Result<(Image, Vec<Dispatch>), CudaError>> {
    if let Some(error) = forced_cuda_failure() {
        return Some(Err(error));
    }
    if forced_cuda_empty_dispatch() {
        return Some(Ok((current.clone(), Vec::new())));
    }
    match pf {
        90004 => {
            let resolved =
                resolve_radiance_inputs(current, inputs.radiance).expect("validated PF90004 inputs");
            let assets = radiance_assets();
            let fixed = aurora_portrait::kernels::radiance::FixedAssets {
                lut_a: &assets.lut_a,
                lut_b: &assets.lut_b,
                gamma: &assets.gamma,
                matrices: &assets.matrices,
            };
            let gpu = match aurora_portrait::cuda_radiance::Radiance::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.apply(aurora_portrait::cuda_radiance::Request {
                    source: current,
                    skin_r: resolved.skin,
                    parsing_rgba: resolved.parsing_rgba,
                    face_rgba: resolved.face_rgba,
                    ui: (degree * 100.0).clamp(0.0, 100.0),
                    value_mean: resolved.value_mean,
                    assets: &fixed,
                })
                .map(|(image, dispatch)| (image, dispatch.into_iter().collect())),
            )
        }
        21025 | 21069 | 21070 | 21071 => {
            let input = inputs
                .wrinkle
                .expect("validated wrinkle same-job inputs");
            validate_wrinkle(current, input).expect("validated wrinkle inputs");
            let gpu = match aurora_portrait::cuda_wrinkle::WrinkleGpu::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply_route(
                pf,
                current,
                input.base_full,
                input.residual_full,
                input.mask_rgba_full,
                input.skin_mask_full,
                input.roi,
                degree * 100.0,
                input.uniforms,
                StoreMode::TruncU8,
            ))
        }
        30021 => {
            let input = match inputs.hlf_face {
                Some(input) => input,
                None => hlf_auto.expect("validated PF30021 HLF inputs").face(),
            };
            validate_hlf_face(current, input).expect("validated PF30021 HLF inputs");
            let gpu = match aurora_portrait::cuda_hlf::Hlf::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.face_apply(
                    input.source_rgb_full_f32,
                    input.median_rgb_full_f32,
                    input.gauss_rgb_full_f32,
                    input.face_mask_full_f32,
                    input.skin_mask_full_f32,
                    current.width,
                    current.height,
                    input.roi,
                    degree * 100.0,
                )
                .and_then(|(rgb, dispatches)| {
                    let image =
                        hlf_rgb_f32_to_image(current, &rgb).map_err(CudaError::Refused)?;
                    Ok((image, dispatches))
                }),
            )
        }
        30022 => {
            let input = match inputs.hlf_body {
                Some(input) => input,
                None => hlf_auto.expect("validated PF30022 HLF inputs").body(),
            };
            validate_hlf_body(current, input).expect("validated PF30022 HLF inputs");
            let gpu = match aurora_portrait::cuda_hlf::Hlf::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.body_apply(
                current,
                input.roi,
                input.median_rgb_full_f32,
                input.gauss_rgb_full_f32,
                input.face_mask_rgba_full_f32,
                input.skin_mask_full_f32,
                degree * 100.0,
                input.uniforms,
                input.adjust_lut.as_ref(),
            ))
        }
        100 | 103 | 105 | 111 | 117 | 118 | 30007 => {
            let (_, input) =
                find_makeup_input(pf, inputs).expect("validated makeup same-job input");
            let gpu = match aurora_portrait::cuda_makeup::MakeupLayer::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            let result = gpu.apply_route(
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
            );
            Some(if pf == 111 {
                let writes = input
                    .material_uv_writes
                    .expect("validated PF111 material UV writes");
                result.and_then(|(next, dispatches)| {
                    let next =
                        gate_makeup_writes(current, next, writes).map_err(CudaError::Refused)?;
                    Ok((next, dispatches))
                })
            } else {
                result
            })
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
        30025 => {
            let input = inputs.pf30025.expect("validated PF30025 repair input");
            let gpu = match aurora_portrait::cuda_pf30025::ToothRepair::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply(&aurora_portrait::cuda_pf30025::Job {
                source: current,
                repair_780929437_bgra: input.repair_780929437_bgra,
                gate_3204548346: input.gate_3204548346,
                teeth_mask: input.teeth_mask,
                roi: input.roi,
                ui: degree * 100.0,
            }))
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
    if !matches!(pf, 8068 | 8069 | 8070 | 21015 | 30096 | 90005 | 90007) {
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
        30096 => {
            let (Some(cold), Some(warm)) = (cold_lut(), warm_lut()) else {
                return Some(Err(CudaError::Refused(Refusal::MissingInput(
                    "skin temperature LUT",
                ))));
            };
            let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
            Some(with_skin_lut(|gpu| {
                gpu.temperature(current, mask, cold, warm, ui)
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
    job: Option<RadianceInputs<'_>>,
) -> Result<Image, aurora_portrait::Refusal> {
    let resolved = resolve_radiance_inputs(current, job)?;
    let planes = aurora_portrait::kernels::radiance::MaskPlanes {
        skin: resolved.skin,
        parsing_rgba: resolved.parsing_rgba,
        face_rgba: resolved.face_rgba,
    };
    let assets = radiance_assets();
    let fixed = aurora_portrait::kernels::radiance::FixedAssets {
        lut_a: &assets.lut_a,
        lut_b: &assets.lut_b,
        gamma: &assets.gamma,
        matrices: &assets.matrices,
    };
    let ui = (degree * 100.0).clamp(0.0, 100.0);
    Ok(aurora_portrait::kernels::radiance::apply(
        current,
        &planes,
        ui,
        resolved.value_mean,
        &fixed,
    )?
    .image)
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

    #[test]
    fn renderer_receipt_rejects_changed_frame_or_params_before_dispatch() {
        let masks = PortraitMasks::solid(1.0);
        let (width, height) = (8, 8);
        let source = frame(width, height);
        let params = BTreeMap::from([("pc:90005".to_string(), 0.9f32)]);
        let receipt = PortraitJobReceipt::from_renderer_receipt(
            &source,
            width,
            height,
            NormRect::FULL,
            &params,
            &masks,
        );
        let inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);

        let mut changed_frame = source.clone();
        changed_frame[0] ^= 1;
        let before = changed_frame.clone();
        let report = apply_portrait_effects_with_job(
            &mut changed_frame,
            width,
            height,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(changed_frame, before);
        assert_eq!(report.job_id, Some(receipt.job_id()));
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::ValueOutOfRange("PortraitJobReceipt digest"))
        );
        assert!(report.effects[0].skipped);
        assert!(!report.effects[0].executed);
        assert!(report.effects[0].dispatches.is_empty());

        let mut changed_params = params.clone();
        changed_params.insert("pc:90005".to_string(), 0.8);
        let mut unchanged_frame = source.clone();
        let before = unchanged_frame.clone();
        let report = apply_portrait_effects_with_job(
            &mut unchanged_frame,
            width,
            height,
            NormRect::FULL,
            &changed_params,
            &inputs,
        );
        assert_eq!(unchanged_frame, before);
        assert_eq!(report.job_id, Some(receipt.job_id()));
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::ValueOutOfRange("PortraitJobReceipt digest"))
        );
        assert!(report.effects[0].dispatches.is_empty());
    }

    #[test]
    fn renderer_receipt_does_not_promote_masks_only() {
        let masks = PortraitMasks::solid(1.0);
        let mut rgba = frame(8, 8);
        let before = rgba.clone();
        let params = BTreeMap::from([("pc:90005".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::masks_only(&masks);
        let report = apply_portrait_effects_with_job(
            &mut rgba,
            8,
            8,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(rgba, before);
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::MissingInput("same-job input bundle"))
        );
        assert!(report.effects[0].skipped);
        assert!(report.effects[0].dispatches.is_empty());
    }

    #[test]
    fn renderer_job_without_receipt_is_refused_before_dispatch() {
        let masks = PortraitMasks::solid(1.0);
        let mut rgba = frame(8, 8);
        let before = rgba.clone();
        let params = BTreeMap::from([("pc:90005".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::from_renderer_job(9, &masks);
        let report = apply_portrait_effects_with_report_and_inputs_policy(
            &mut rgba,
            8,
            8,
            NormRect::FULL,
            &params,
            &inputs,
            true,
        );
        assert_eq!(rgba, before);
        assert_eq!(report.job_id, Some(9));
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::MissingInput("renderer receipt"))
        );
        assert!(report.effects[0].skipped);
        assert!(!report.effects[0].executed);
        assert!(report.effects[0].dispatches.is_empty());
    }

    #[test]
    fn pf30025_uses_existing_producer_inputs_and_fails_closed() {
        let masks = PortraitMasks::solid(1.0);
        let mut inputs = PortraitJobInputs::from_renderer_job(12, &masks);
        let (before, run) = apply_one_with_inputs(30025, 1, 1, &inputs);
        assert_eq!(
            run.refusal,
            Some(Refusal::MissingInput("pf30025 repair_bgra"))
        );
        assert!(run.skipped);
        assert!(!run.executed);

        inputs.pf30025 = Some(Pf30025Inputs {
            repair_780929437_bgra: &[30, 20, 200, 255],
            gate_3204548346: None,
            teeth_mask: &[0.0],
            roi: ToothRect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
        });
        let (after, run) = apply_one_with_inputs(30025, 1, 1, &inputs);
        assert_eq!(after, before);
        assert_eq!(run.refusal, None);
        assert!(run.executed);
        assert!(!run.skipped);
    }

    #[cfg(windows)]
    #[test]
    fn pf30025_cuda_matches_cpu_for_bgra_roi_alpha_and_gate() {
        struct Case {
            width: u32,
            height: u32,
            source: Vec<u8>,
            repair_bgra: Vec<u8>,
            teeth_mask: Vec<f32>,
            roi: ToothRect,
            degree01: f32,
            gate: Option<Vec<u8>>,
        }

        fn run(case: &Case, force_cpu: bool) -> (Vec<u8>, PortraitEffectRun) {
            let masks = PortraitMasks::solid(1.0);
            let mut inputs = PortraitJobInputs::from_renderer_job(30025, &masks);
            inputs.pf30025 = Some(Pf30025Inputs {
                repair_780929437_bgra: &case.repair_bgra,
                gate_3204548346: case.gate.as_deref(),
                teeth_mask: &case.teeth_mask,
                roi: case.roi,
            });
            let mut rgba = case.source.clone();
            let _force = force_cpu.then(ForceCudaFailureGuard::new);
            let report = apply_portrait_effects_with_job(
                &mut rgba,
                case.width,
                case.height,
                NormRect::FULL,
                &BTreeMap::from([("pc:30025".to_string(), case.degree01)]),
                &inputs,
            );
            assert_eq!(report.effects.len(), 1);
            (rgba, report.effects.into_iter().next().unwrap())
        }

        let cases = [
            Case {
                width: 1,
                height: 1,
                source: vec![10, 20, 30, 91],
                // R rounds from 10.5 to 11: midpoint away from zero.
                repair_bgra: vec![30, 20, 12, 255],
                teeth_mask: vec![0.25],
                roi: ToothRect {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
                degree01: 1.0,
                gate: Some(vec![7]),
            },
            Case {
                width: 4,
                height: 3,
                source: vec![
                    10, 20, 30, 41, 50, 60, 70, 81, 90, 100, 110, 121, 130, 140, 150, 161,
                    12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162,
                    14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154, 164,
                ],
                repair_bgra: vec![
                    200, 180, 160, 0, 210, 190, 170, 1, 220, 200, 180, 2, 230, 210, 190, 3,
                    240, 220, 200, 4, 250, 230, 210, 5, 20, 30, 40, 6, 30, 40, 50, 7,
                    40, 50, 60, 8, 50, 60, 70, 9, 60, 70, 80, 10, 70, 80, 90, 11,
                ],
                teeth_mask: vec![
                    0.0, 0.2, 0.5, 1.0, 0.7, 0.33, 0.25, 0.9, 0.6, 0.15, 0.45, 0.8,
                ],
                roi: ToothRect {
                    x: 1,
                    y: 1,
                    width: 2,
                    height: 1,
                },
                degree01: 0.37,
                gate: None,
            },
            Case {
                width: 3,
                height: 2,
                source: vec![
                    8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118,
                    13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123,
                ],
                repair_bgra: vec![
                    180, 160, 140, 255, 190, 170, 150, 254, 200, 180, 160, 253,
                    210, 190, 170, 252, 220, 200, 180, 251, 230, 210, 190, 250,
                ],
                teeth_mask: vec![0.1, 0.4, 0.9, 1.0, 0.55, 0.75],
                roi: ToothRect {
                    x: 0,
                    y: 0,
                    width: 3,
                    height: 2,
                },
                degree01: 1.0,
                gate: Some(vec![1, 2, 3]),
            },
        ];

        for case in &cases {
            let (cpu, cpu_run) = run(case, true);
            let (actual, actual_run) = run(case, false);
            let differing_bytes = actual
                .iter()
                .zip(&cpu)
                .filter(|(left, right)| left != right)
                .count();
            assert_eq!(actual, cpu, "PF30025 CPU/CUDA byte parity");
            assert_eq!(differing_bytes, 0);
            assert_eq!(cpu_run.backend, PortraitBackend::Cpu);
            assert!(cpu_run.dispatches.is_empty());
            assert!(actual_run.executed);
            assert!(actual_run.refusal.is_none());
            for pixel in 0..case.width as usize * case.height as usize {
                assert_eq!(
                    actual[pixel * 4 + 3],
                    case.source[pixel * 4 + 3],
                    "PF30025 alpha pixel {pixel}"
                );
                let x = pixel as i32 % case.width as i32;
                let y = pixel as i32 / case.width as i32;
                let inside = x >= case.roi.x
                    && y >= case.roi.y
                    && x < case.roi.x + case.roi.width
                    && y < case.roi.y + case.roi.height;
                if !inside {
                    assert_eq!(
                        &actual[pixel * 4..pixel * 4 + 4],
                        &case.source[pixel * 4..pixel * 4 + 4],
                        "PF30025 ROI pixel {pixel}"
                    );
                }
            }
            if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
                assert_eq!(
                    actual_run.backend,
                    PortraitBackend::Cuda,
                    "{:?}",
                    actual_run.fallback_reason
                );
                assert_eq!(actual_run.dispatches.len(), 1);
                assert_eq!(
                    actual_run.dispatches[0].kernel,
                    aurora_portrait::cuda_pf30025::KERNEL
                );
                assert!(!actual_run.dispatches[0].device.is_empty());
                let dispatch = &actual_run.dispatches[0];
                eprintln!(
                    "PF30025 case={}x{} roi={}x{}+{}+{} backend=Cuda kernel={} device={} cc={}.{} driver={} ptx_isa={}.{} differing_bytes={}",
                    case.width,
                    case.height,
                    case.roi.width,
                    case.roi.height,
                    case.roi.x,
                    case.roi.y,
                    dispatch.kernel,
                    dispatch.device,
                    dispatch.compute_cap.0,
                    dispatch.compute_cap.1,
                    dispatch.driver_version,
                    dispatch.ptx_isa.0,
                    dispatch.ptx_isa.1,
                    differing_bytes,
                );
            } else if actual_run.backend == PortraitBackend::Cpu {
                assert!(actual_run.fallback_reason.is_some());
            } else {
                assert_eq!(
                    actual_run.dispatches[0].kernel,
                    aurora_portrait::cuda_pf30025::KERNEL
                );
            }
        }
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

    #[cfg(windows)]
    fn run_hair_route_cuda_fault(
        pf: u32,
        empty_dispatch: bool,
    ) -> (Vec<u8>, PortraitEffectRun, Vec<u8>) {
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
        let run_case = |inputs: &PortraitJobInputs<'_>, expected: Vec<u8>| {
            let _empty_guard = empty_dispatch.then(ForceCudaEmptyDispatchGuard::new);
            let _failure_guard = (!empty_dispatch).then(ForceCudaFailureGuard::new);
            let mut actual = source.pixels.clone();
            let report = apply_portrait_effects_with_job(
                &mut actual,
                width,
                height,
                NormRect::FULL,
                &BTreeMap::from([(format!("pc:{pf}"), 0.8)]),
                inputs,
            );
            (
                actual,
                report.effects.into_iter().next().unwrap(),
                expected,
            )
        };
        if pf == 90139 {
            let hair =
                strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
            let mix =
                strided_plane(&[180, 240, 64, 32, 128, 255], roi_width, roi_height, 4, 1);
            let blend = strided_plane(
                &[
                    20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 10, 30, 50, 70,
                    90, 110,
                ],
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
            let expected = hair_smooth_cpu(&source, roi, 0.8, &smooth);
            let inputs = PortraitJobInputs {
                hair_smooth: Some(smooth),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            run_case(&inputs, expected)
        } else if pf == 90160 {
            let hair =
                strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
            let detail =
                strided_plane(&[128, 127, 200, 64, 255, 0], roi_width, roi_height, 7, 1);
            let gloss = HairGlossInputs {
                ticket: 18,
                width,
                height,
                roi: (roi_x, roi_y, roi_width, roi_height),
                dynamic_scale: 0.5,
                hair_mask: Some(&hair),
                model_detail: Some(&detail),
                hair_stride: 5,
                detail_stride: 7,
            };
            let expected = hair_gloss_cpu(&source, roi, &gloss);
            let inputs = PortraitJobInputs {
                hair_gloss: Some(gloss),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            run_case(&inputs, expected)
        } else {
            unreachable!("hair CUDA fault test only covers PF90139/PF90160")
        }
    }

    #[cfg(windows)]
    #[test]
    fn n855_pf90139_forced_cuda_failure_falls_back_to_cpu() {
        let (actual, run, expected) = run_hair_route_cuda_fault(90139, false);
        assert_eq!(actual, expected);
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
        assert!(run
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("test-forced CUDA failure")));
    }

    #[cfg(windows)]
    #[test]
    fn n855_pf90160_forced_cuda_failure_falls_back_to_cpu() {
        let (actual, run, expected) = run_hair_route_cuda_fault(90160, false);
        assert_eq!(actual, expected);
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
        assert!(run
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("test-forced CUDA failure")));
    }

    #[cfg(windows)]
    #[test]
    fn n855_empty_cuda_dispatch_falls_back_to_cpu_for_hair_routes() {
        for pf in [90139, 90160] {
            let (actual, run, expected) = run_hair_route_cuda_fault(pf, true);
            assert_eq!(actual, expected, "PF{pf}");
            assert_eq!(run.backend, PortraitBackend::Cpu, "PF{pf}");
            assert!(run.executed, "PF{pf}");
            assert!(run.dispatches.is_empty(), "PF{pf}");
            assert!(
                run.fallback_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("CUDA returned no dispatch")),
                "PF{pf}: {:?}",
                run.fallback_reason
            );
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
    fn pf90088_producer_binds_fields_and_keeps_typed_refusal_order() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let complete = StrayHairProducerInputs {
            hairfly_v1: Some(repair.clone()),
            raw_mask: Some(raw.clone()),
            person_mask: Some(person.clone()),
            background_mask: Some(background.clone()),
            native_degree: Some(0.5),
            native_person_area: Some(1.0),
            native_background_area: Some(0.5),
        };
        let producer = ProducerInputs {
            stray_hair: Some(complete.clone()),
            ..Default::default()
        };
        let mut inputs = PortraitJobInputs::from_renderer_job(941, &masks);
        inputs.fill_from_producer_inputs(&producer);
        let staged = inputs.stray_hair;
        assert_eq!(staged.hairfly_v1, Some(&repair));
        assert_eq!(staged.raw_mask, Some(raw.as_slice()));
        assert_eq!(staged.person_mask, Some(person.as_slice()));
        assert_eq!(staged.background_mask, Some(background.as_slice()));
        assert_eq!(staged.native_degree, Some(0.5));
        assert_eq!(staged.native_person_area, Some(1.0));
        assert_eq!(staged.native_background_area, Some(0.5));

        let fingerprint = producer.fingerprint();
        assert_ne!(fingerprint, ProducerInputs::default().fingerprint());
        let changed_rows = [
            StrayHairProducerInputs {
                hairfly_v1: None,
                ..complete.clone()
            },
            StrayHairProducerInputs {
                raw_mask: None,
                ..complete.clone()
            },
            StrayHairProducerInputs {
                person_mask: None,
                ..complete.clone()
            },
            StrayHairProducerInputs {
                background_mask: None,
                ..complete.clone()
            },
            StrayHairProducerInputs {
                native_degree: Some(0.25),
                ..complete.clone()
            },
            StrayHairProducerInputs {
                native_degree: Some(0.0),
                ..complete.clone()
            },
            StrayHairProducerInputs {
                native_person_area: Some(0.75),
                ..complete.clone()
            },
            StrayHairProducerInputs {
                native_background_area: Some(0.25),
                ..complete
            },
        ];
        for changed in changed_rows {
            assert_ne!(
                fingerprint,
                ProducerInputs {
                    stray_hair: Some(changed),
                    ..Default::default()
                }
                .fingerprint()
            );
        }

        let cases = [
            (
                StrayHairProducerInputs::default(),
                "PF90088_NATIVE_DEGREE",
            ),
            (
                StrayHairProducerInputs {
                    native_degree: Some(0.5),
                    ..Default::default()
                },
                "PF90088_NATIVE_PERSON_AREA",
            ),
            (
                StrayHairProducerInputs {
                    native_degree: Some(0.5),
                    native_person_area: Some(1.0),
                    ..Default::default()
                },
                "PF90088_NATIVE_BACKGROUND_AREA",
            ),
            (
                StrayHairProducerInputs {
                    native_degree: Some(0.5),
                    native_person_area: Some(1.0),
                    native_background_area: Some(0.5),
                    ..Default::default()
                },
                "PF90088_HAIRFLY_V1",
            ),
            (
                StrayHairProducerInputs {
                    hairfly_v1: Some(repair.clone()),
                    native_degree: Some(0.5),
                    native_person_area: Some(1.0),
                    native_background_area: Some(0.5),
                    ..Default::default()
                },
                "PF90088_RAW_STRAY_HAIR_MASK",
            ),
        ];
        for (candidate, missing) in cases {
            let producer = ProducerInputs {
                stray_hair: Some(candidate),
                ..Default::default()
            };
            let mut inputs = PortraitJobInputs::from_renderer_job(941, &masks);
            inputs.fill_from_producer_inputs(&producer);
            let (actual, run) = run_stray_hair_stage(&source, &inputs, true);
            assert_eq!(actual, source.pixels);
            assert_eq!(run.refusal, Some(Refusal::MissingInput(missing)));
            assert!(run.skipped);
            assert!(!run.executed);
            assert!(run.dispatches.is_empty());
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
    fn n855_pf90088_empty_cuda_dispatch_falls_back_to_cpu() {
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
        let _empty = ForceCudaEmptyDispatchGuard::new();
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            source.width,
            source.height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90088".to_string(), 0.8f32)]),
            &inputs,
        );
        let run = &report.effects[0];
        assert_eq!(actual, expected.pixels);
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
        assert!(run
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("CUDA returned no dispatch")));
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
    pub(super) struct ForceCudaFailureGuard;

    #[cfg(windows)]
    impl ForceCudaFailureGuard {
        pub(super) fn new() -> Self {
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

    #[cfg(windows)]
    struct ForceCudaEmptyDispatchGuard;

    #[cfg(windows)]
    impl ForceCudaEmptyDispatchGuard {
        fn new() -> Self {
            FORCE_CUDA_EMPTY_DISPATCH.with(|flag| flag.set(true));
            Self
        }
    }

    #[cfg(windows)]
    impl Drop for ForceCudaEmptyDispatchGuard {
        fn drop(&mut self) {
            FORCE_CUDA_EMPTY_DISPATCH.with(|flag| flag.set(false));
        }
    }

    #[cfg(windows)]
    #[test]
    fn hlf_wrinkle_forced_cpu_paths_match_their_oracles() {
        let width = 7u32;
        let height = 5u32;
        let pixels = width as usize * height as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let base = Image::new(width, height, source.pixels.iter().rev().copied().collect()).unwrap();
        let residual = Image::solid(width, height, [51, 102, 153, 77]);
        let roi = Roi {
            x: 1,
            y: 1,
            width: 4,
            height: 3,
        };
        let source_rgb: Vec<f32> = (0..pixels * 3)
            .map(|i| (i * 17 % 251) as f32 / 255.0)
            .collect();
        let median_rgb: Vec<f32> = (0..pixels * 3)
            .map(|i| (i * 19 % 241) as f32 / 255.0)
            .collect();
        let gauss_rgb: Vec<f32> = (0..pixels * 3)
            .map(|i| (i * 23 % 239) as f32 / 255.0)
            .collect();
        let face_mask: Vec<f32> = (0..pixels).map(|i| (i % 7) as f32 / 6.0).collect();
        let face_mask_rgba: Vec<f32> = (0..pixels * 4)
            .map(|i| (i % 11) as f32 / 10.0)
            .collect();
        let skin_mask: Vec<f32> = (0..pixels).map(|i| (i % 5) as f32 / 4.0).collect();
        let wrinkle_mask: Vec<u8> = (0..pixels * 4).map(|i| (i * 13 % 256) as u8).collect();
        let wrinkle_skin: Vec<u8> = (0..pixels).map(|i| (i * 29 % 256) as u8).collect();
        let wrinkle_uniforms = WrinkleUniforms {
            legal_alpha: 1.0,
            dark_circle_alpha: 1.0,
            diff_factor: 1.0,
            diff_factor2: 0.0,
            color_threshold_factor: 0.0,
            mix_alpha: 1.0,
            diff_min_threshold: -1.0,
            diff_max_threshold: 1.0,
            high_freq_min_threshold: -1.0,
            high_freq_max_threshold: 1.0,
        };
        let hlf_body_uniforms = BodyHlfUniforms {
            spotless_degree: 0.0,
            retouch_degree: 1.0,
            spotless_acne_threshold: 0.0,
            spotless_high_freq_factor: 1.0,
            spotless_low_freq_alpha: 0.0,
            retouch_high_freq_factor: 0.0,
            diff_min_range: -0.2,
            diff_max_range: 0.2,
        };
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs {
            hlf_face: Some(HlfFaceInputs {
                width,
                height,
                source_rgb_full_f32: &source_rgb,
                median_rgb_full_f32: &median_rgb,
                gauss_rgb_full_f32: &gauss_rgb,
                face_mask_full_f32: &face_mask,
                skin_mask_full_f32: &skin_mask,
                roi,
            }),
            hlf_body: Some(HlfBodyInputs {
                width,
                height,
                median_rgb_full_f32: &median_rgb,
                gauss_rgb_full_f32: &gauss_rgb,
                face_mask_rgba_full_f32: &face_mask_rgba,
                skin_mask_full_f32: &skin_mask,
                roi,
                uniforms: hlf_body_uniforms,
                adjust_lut: None,
            }),
            wrinkle: Some(WrinkleInputs {
                base_full: &base,
                residual_full: &residual,
                mask_rgba_full: &wrinkle_mask,
                skin_mask_full: &wrinkle_skin,
                roi,
                uniforms: wrinkle_uniforms,
            }),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let _force = ForceCudaFailureGuard::new();
        for pf in [21025, 21069, 21070, 21071, 30021, 30022] {
            let expected = match pf {
                21025 | 21069 | 21070 | 21071 => skin_adjust::wrinkle_apply(
                    &source,
                    &base,
                    &residual,
                    &wrinkle_mask,
                    &wrinkle_skin,
                    roi,
                    skin_adjust::wrinkle_degree(pf, 50.0).unwrap(),
                    wrinkle_uniforms,
                )
                .unwrap(),
                30021 => {
                    let rgb = skin_adjust::hlf_face_apply(
                        &source_rgb,
                        &median_rgb,
                        &gauss_rgb,
                        &face_mask,
                        &skin_mask,
                        width,
                        height,
                        roi,
                        50.0,
                    )
                    .unwrap();
                    hlf_rgb_f32_to_image(&source, &rgb).unwrap()
                }
                30022 => skin_adjust::body_hlf_apply(
                    &source,
                    roi,
                    &median_rgb,
                    &gauss_rgb,
                    &face_mask_rgba,
                    &skin_mask,
                    50.0,
                    hlf_body_uniforms,
                    None,
                )
                .unwrap(),
                _ => unreachable!(),
            };
            let mut actual = source.pixels.clone();
            let extra = BTreeMap::from([(format!("pc:{pf}"), 0.5)]);
            let report = apply_portrait_effects_with_job(
                &mut actual,
                width,
                height,
                NormRect::FULL,
                &extra,
                &inputs,
            );
            assert_eq!(actual, expected.pixels, "PF{pf}");
            assert_eq!(report.effects[0].backend, PortraitBackend::Cpu);
            assert!(report.effects[0].executed);
            assert!(report.effects[0].dispatches.is_empty());
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

    fn suit_plan(order: u32) -> SuitPassPlan<'static> {
        let (effect, mask_field, mask_value) = match order {
            1..=3 => ("BlusherImageMakeup", "BlusherMaskType", 15),
            4..=5 => ("NormalEyeBrowImageMakeup", "EyeBrowMaskType", 5),
            6..=7 => ("SymEyeImageMakeup", "EyeMaskType", 10),
            8..=9 => ("SymEyeImageMakeup", "EyeMaskType", 4),
            10..=11 => ("SymEyeLashImageMakeup", "EyeLashMaskType", 4),
            12 => ("LipStickImageMakeup", "LipMaskType", 3),
            _ => panic!("suit order must be 1..=12"),
        };
        SuitPassPlan {
            order,
            effect,
            degree: Some(70.0),
            opacity: Some(49.0),
            filter_type: Some(42),
            material_path: Some("Resources/material.png"),
            mask_field: Some(mask_field),
            mask_value: Some(mask_value),
            shelter_type: Some(7),
            force_use_shelter: Some((4..=11).contains(&order)),
            region_rect: Some([95, 40, 901, 1081]),
            mix_color: Some([0, 0, 0, 255]),
        }
    }

    fn suit_stage_pass<'a>(
        mask: Option<&'a Image>,
        mask_uv: Option<&'a [[f32; 2]]>,
        material: Option<&'a Image>,
        material_uv: Option<&'a [[f32; 2]]>,
    ) -> SuitPassInput<'a> {
        SuitPassInput {
            kind: SuitPassKindInput::Image {
                material,
                material_uv,
            },
            mask,
            mask_uv,
            layer: Some(MakeupLayer {
                mix_color: [0, 0, 0, 255],
                opacity_percent: 49.0,
                degree01: 0.7,
                filter_type: 42,
                mask_channel: MakeupMaskChannel::Rgba,
            }),
            shelter: 0.5,
            region: 0.75,
        }
    }

    #[test]
    fn suit_producer_names_every_missing_same_job_input() {
        let plan = suit_plan(1);
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let material = Image::solid(1, 1, [20, 30, 40, 128]);
        let uv = [[0.5, 0.5]];
        let complete = SuitPassSameJobInputs {
            mask: Some(&mask),
            mask_uv: Some(&uv),
            material: Some(&material),
            material_uv: Some(&uv),
            shelter: Some(0.5),
            region: Some(0.75),
        };
        assert!(produce_suit_pass(1, &plan, complete).is_ok());

        let cases = [
            (
                SuitPassSameJobInputs {
                    mask: None,
                    ..complete
                },
                "suit mask",
            ),
            (
                SuitPassSameJobInputs {
                    mask_uv: None,
                    ..complete
                },
                "suit mask uv",
            ),
            (
                SuitPassSameJobInputs {
                    material: None,
                    ..complete
                },
                "suit material",
            ),
            (
                SuitPassSameJobInputs {
                    material_uv: None,
                    ..complete
                },
                "suit material uv",
            ),
            (
                SuitPassSameJobInputs {
                    shelter: None,
                    ..complete
                },
                "suit shelter",
            ),
            (
                SuitPassSameJobInputs {
                    region: None,
                    ..complete
                },
                "suit region",
            ),
        ];
        for (inputs, missing) in cases {
            assert_eq!(
                produce_suit_pass(1, &plan, inputs).err(),
                Some(Refusal::MissingInput(missing))
            );
        }
    }

    #[test]
    fn suit_stage_refuses_missing_existing_inputs() {
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let material = Image::solid(1, 1, [20, 30, 40, 128]);
        let uv = [[0.5, 0.5]];
        let cases = [
            (
                suit_stage_pass(None, Some(&uv), Some(&material), Some(&uv)),
                "suit mask",
            ),
            (
                suit_stage_pass(Some(&mask), None, Some(&material), Some(&uv)),
                "suit mask uv",
            ),
            (
                suit_stage_pass(Some(&mask), Some(&uv), None, Some(&uv)),
                "suit material",
            ),
            (
                suit_stage_pass(Some(&mask), Some(&uv), Some(&material), None),
                "suit material uv",
            ),
        ];
        for (pass, missing) in cases {
            assert_eq!(
                resolve_suit_passes(Some(&[pass])).err(),
                Some(Refusal::MissingInput(missing))
            );
        }
    }

    #[test]
    fn suit_producer_uses_only_pinned_mask_channels() {
        let cases = [
            ("BlusherMaskType", 15, MakeupMaskChannel::Rgba),
            ("EyeBrowMaskType", 5, MakeupMaskChannel::RB),
            ("EyeMaskType", 10, MakeupMaskChannel::Rgb),
            ("EyeMaskType", 4, MakeupMaskChannel::RG),
            ("EyeLashMaskType", 4, MakeupMaskChannel::RG),
            ("LipMaskType", 3, MakeupMaskChannel::R),
        ];
        for (field, value, expected) in cases {
            assert_eq!(suit_mask_channel(field, value), Ok(expected));
        }
        assert_eq!(
            suit_mask_channel("UnknownMaskType", 99),
            Err(Refusal::MissingInput("suit mask channel"))
        );
    }

    #[test]
    fn suit_producer_keeps_skip_shelter_and_region_rect_separate() {
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let material = Image::solid(1, 1, [20, 30, 40, 128]);
        let uv = [[0.5, 0.5]];
        let inputs = SuitPassSameJobInputs {
            mask: Some(&mask),
            mask_uv: Some(&uv),
            material: Some(&material),
            material_uv: Some(&uv),
            shelter: None,
            region: Some(0.75),
        };
        let mut plan = suit_plan(1);
        plan.force_use_shelter = Some(true);
        let before = produce_suit_pass(1, &plan, inputs).unwrap();
        plan.region_rect = Some([0, 0, 0, 0]);
        let after = produce_suit_pass(1, &plan, inputs).unwrap();
        assert_eq!(before.shelter, 1.0);
        assert_eq!(after.shelter, 1.0);
        assert_eq!(before.region, 0.75);
        assert_eq!(after.region, 0.75);
    }

    #[test]
    fn suit_producer_emits_twelve_ordered_existing_inputs() {
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let material = Image::solid(1, 1, [20, 30, 40, 128]);
        let uv = [[0.5, 0.5]];
        let plans: Vec<_> = (1..=12).map(suit_plan).collect();
        let jobs = vec![
            SuitPassSameJobInputs {
                mask: Some(&mask),
                mask_uv: Some(&uv),
                material: Some(&material),
                material_uv: Some(&uv),
                shelter: Some(0.5),
                region: Some(0.75),
            };
            12
        ];
        let produced = produce_suit_passes(1, Some(&plans), Some(&jobs)).unwrap();
        assert_eq!(produced.len(), 12);
        assert!(produced
            .iter()
            .all(|pass| pass.mask.is_some() && pass.mask_uv.is_some()));
        assert!(produced.iter().all(|pass| matches!(
            &pass.kind,
            SuitPassKindInput::Image {
                material: Some(_),
                material_uv: Some(_)
            }
        )));

        let resolved = resolve_suit_passes(Some(&produced)).unwrap();
        assert_eq!(resolved.len(), 12);
        assert_eq!(
            resolved
                .iter()
                .map(|pass| pass.layer.mask_channel)
                .collect::<Vec<_>>(),
            vec![
                MakeupMaskChannel::Rgba,
                MakeupMaskChannel::Rgba,
                MakeupMaskChannel::Rgba,
                MakeupMaskChannel::RB,
                MakeupMaskChannel::RB,
                MakeupMaskChannel::Rgb,
                MakeupMaskChannel::Rgb,
                MakeupMaskChannel::RG,
                MakeupMaskChannel::RG,
                MakeupMaskChannel::RG,
                MakeupMaskChannel::RG,
                MakeupMaskChannel::R,
            ]
        );
        assert!(resolved.iter().all(|pass| pass.shelter == 1.0
            || (pass.shelter == 0.5 && pass.region == 0.75)));
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
            PortraitBackend::Cpu => {
                assert!(run.executed);
                assert!(run.dispatches.is_empty());
                assert!(run.fallback_reason.is_some());
                let AlignmentFinalBlendOutcome::Applied(expected) = expected else {
                    panic!("open edge gate produced a CPU-gated oracle");
                };
                assert_eq!(actual, expected.pixels, "PF30026 CPU fallback parity");
            }
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

    #[test]
    fn pf30026_producer_preserves_fields_and_typed_missing() {
        let (width, height) = (4u32, 3u32);
        let pixels = (width * height) as usize;
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let teeth_mask = vec![128u8; pixels];
        let edge = vec![0.5f32; pixels];
        let alignment = TeethAlignmentProducerInputs {
            effect_rgb: Some(effect.clone()),
            teeth_mask: Some(teeth_mask.clone()),
            edge_plane_a: Some(edge.clone()),
            edge_plane_b: Some(edge.clone()),
            roi: Some(NormRect::FULL),
        };
        let producer = ProducerInputs {
            teeth_alignment: Some(alignment.clone()),
            ..Default::default()
        };

        let mut inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        inputs.fill_from_producer_inputs(&producer);
        let staged = inputs.teeth_alignment.expect("producer row is present");
        assert_eq!(staged.effect_rgb, Some(&effect));
        assert_eq!(staged.teeth_mask, Some(teeth_mask.as_slice()));
        assert_eq!(staged.edge_plane_a, Some(edge.as_slice()));
        assert_eq!(staged.edge_plane_b, Some(edge.as_slice()));
        assert_eq!(staged.roi, Some(NormRect::FULL));

        let mut changed = producer.clone();
        changed.teeth_alignment.as_mut().unwrap().edge_plane_b = Some(vec![0.25f32; pixels]);
        assert_ne!(producer.fingerprint(), changed.fingerprint());

        let cases = [
            (
                TeethAlignmentProducerInputs {
                    effect_rgb: None,
                    ..alignment.clone()
                },
                "PF30026 effect RGB",
            ),
            (
                TeethAlignmentProducerInputs {
                    teeth_mask: None,
                    ..alignment.clone()
                },
                "PF30026 teeth mask",
            ),
            (
                TeethAlignmentProducerInputs {
                    edge_plane_a: None,
                    ..alignment.clone()
                },
                "PF30026 edge plane A",
            ),
            (
                TeethAlignmentProducerInputs {
                    edge_plane_b: None,
                    ..alignment.clone()
                },
                "PF30026 edge plane B",
            ),
            (
                TeethAlignmentProducerInputs {
                    roi: None,
                    ..alignment
                },
                "PF30026 ROI",
            ),
        ];
        for (alignment, missing) in cases {
            let producer = ProducerInputs {
                teeth_alignment: Some(alignment),
                ..Default::default()
            };
            let mut inputs = PortraitJobInputs::from_renderer_job(1, &masks);
            inputs.fill_from_producer_inputs(&producer);
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
                Some(Refusal::MissingInput(missing))
            );
            assert_eq!(actual, source_pixels);
        }
    }

    #[cfg(windows)]
    #[test]
    fn n855_pf30026_forced_cuda_failure_falls_back_to_cpu() {
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
        let expected = aurora_portrait::kernels::teeth_blend::apply_alignment_final_blend(
            AlignmentFinalBlendJob {
                original: Some(&Image::new(width, height, source_pixels.clone()).unwrap()),
                effect_rgb: Some(&effect),
                teeth_mask: Some(&[128; 12]),
                edge_plane_a: Some(&edge),
                edge_plane_b: Some(&edge),
            },
        )
        .unwrap();
        let AlignmentFinalBlendOutcome::Applied(expected) = expected else {
            panic!("open edge gate produced a CPU-gated oracle");
        };
        assert_eq!(actual, expected.pixels);
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
        assert!(run
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("test-forced CUDA failure")));
        assert!(run.refusal.is_none());
        assert!(run.skip_reason.is_none());
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
                material_uv_writes: None,
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
            material_uv_writes: None,
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

        let material_uv = vec![[0.0, 0.0]; 8 * 6];
        let no_writes = [MakeupInput {
            pf: 111,
            mix_color: [30, 60, 90, 180],
            opacity_percent: 75.0,
            filter_type: 40,
            material: Some(&material),
            material_uv: Some(&material_uv),
            material_uv_writes: None,
            mask: None,
            mask_uv: None,
        }];
        let inputs = PortraitJobInputs {
            makeup: Some(&no_writes),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let (_, effect) = apply_one_with_inputs(111, 8, 6, &inputs);
        assert_eq!(
            effect.refusal,
            Some(Refusal::MissingInput("makeup material UV writes"))
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
        let material_uv_writes = vec![1u16; pixels];

        let rows = [
            MakeupInput {
                pf: 30007,
                mix_color: [31, 61, 91, 211],
                opacity_percent: 68.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
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
                material_uv_writes: Some(&material_uv_writes),
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
                material_uv_writes: None,
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
                material_uv_writes: None,
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
                material_uv_writes: None,
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
                material_uv_writes: None,
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
                material_uv_writes: None,
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

    struct Pf111StageResult {
        before: Vec<u8>,
        after: Vec<u8>,
        writes: Vec<u16>,
        effect: PortraitEffectRun,
    }

    fn run_pf111_producer_stage(force_cpu: bool) -> Pf111StageResult {
        use aurora_portrait::kernels::pf111_uv::{
            EyePupilPosition49, EyeSide, NormalEyeEbo, NormalEyeUv49, PF111_VERTEX_COUNT,
            Pf111RasterInputs, RasterExtent, rasterize_pixels,
        };

        let (width, height) = (8u32, 8u32);
        let mut positions = vec![[0.0; 2]; PF111_VERTEX_COUNT];
        let mut normal_eye_uv = vec![[0.0; 2]; PF111_VERTEX_COUNT];
        positions[0] = [0.0, 0.0];
        positions[1] = [width as f32, 0.0];
        positions[2] = [0.0, height as f32];
        normal_eye_uv[0] = [0.5, 0.5];
        normal_eye_uv[1] = [0.5, 0.5];
        normal_eye_uv[2] = [0.5, 0.5];
        let ebo = [0u16, 1, 2];
        let uv = rasterize_pixels(Pf111RasterInputs {
            side: EyeSide::Left,
            positions: Some(EyePupilPosition49::new(&positions).unwrap()),
            normal_eye_uv: Some(NormalEyeUv49::new(&normal_eye_uv).unwrap()),
            extent: Some(RasterExtent::new(width, height).unwrap()),
            normal_eye_ebo: Some(NormalEyeEbo::new(&ebo).unwrap()),
        })
        .unwrap();
        let material = Image::solid(1, 1, [0, 0, 255, 255]);
        let rows = [MakeupInput {
            pf: 111,
            mix_color: [255, 255, 255, 255],
            opacity_percent: 100.0,
            filter_type: 40,
            material: Some(&material),
            material_uv: Some(uv.material_uv()),
            material_uv_writes: Some(uv.writes()),
            mask: None,
            mask_uv: None,
        }];
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs {
            makeup: Some(&rows),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut rgba = frame(width, height);
        let before = rgba.clone();
        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);
        #[cfg(not(windows))]
        let _ = force_cpu;
        let report = apply_portrait_effects_with_job(
            &mut rgba,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:111".to_string(), 0.8f32)]),
            &inputs,
        );
        assert_eq!(report.effects.len(), 1);
        Pf111StageResult {
            before,
            after: rgba,
            writes: uv.writes().to_vec(),
            effect: report.effects.into_iter().next().unwrap(),
        }
    }

    fn assert_pf111_uncovered_pixels_are_not_injected(case: &Pf111StageResult) {
        assert!(case.writes.iter().any(|writes| *writes == 0));
        assert!(case.writes.iter().any(|writes| *writes > 0));
        let mut covered_changed = false;
        for (pixel, writes) in case.writes.iter().enumerate() {
            let base = pixel * 4;
            if *writes == 0 {
                assert_eq!(
                    &case.after[base..base + 4],
                    &case.before[base..base + 4],
                    "uncovered pixel {pixel} sampled the [0,0] material slot"
                );
            } else if case.after[base..base + 4] != case.before[base..base + 4] {
                covered_changed = true;
            }
        }
        assert!(covered_changed, "covered PF111 pixels must be affected");
    }

    #[test]
    fn pf111_producer_stage_gate_matches_cpu() {
        let cpu = run_pf111_producer_stage(true);
        assert_eq!(cpu.effect.backend, PortraitBackend::Cpu);
        assert!(cpu.effect.executed);
        assert!(cpu.effect.dispatches.is_empty());
        assert_pf111_uncovered_pixels_are_not_injected(&cpu);
    }

    #[cfg(windows)]
    #[test]
    fn pf111_producer_stage_gate_matches_real_backend() {
        let cpu = run_pf111_producer_stage(true);
        let actual = run_pf111_producer_stage(false);
        assert_eq!(actual.after, cpu.after, "PF111 producer-stage parity");
        assert_pf111_uncovered_pixels_are_not_injected(&actual);
        eprintln!(
            "PF111 producer-stage backend={:?} kernels={:?} fallback={:?}",
            actual.effect.backend,
            actual
                .effect
                .dispatches
                .iter()
                .map(|dispatch| dispatch.kernel)
                .collect::<Vec<_>>(),
            actual.effect.fallback_reason
        );
        if actual.effect.backend == PortraitBackend::Cpu {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF111 fell back: {:?}",
                actual.effect.fallback_reason
            );
        } else {
            assert!(!actual.effect.dispatches.is_empty());
        }
    }

    #[test]
    fn pf111_missing_same_job_producer_input_stays_typed() {
        use aurora_portrait::kernels::pf111_uv::{
            EYE_PUPIL_POSITION_MISSING, EyeSide, Pf111RasterInputs, rasterize_pixels,
        };

        assert_eq!(
            rasterize_pixels(Pf111RasterInputs {
                side: EyeSide::Left,
                positions: None,
                normal_eye_uv: None,
                extent: None,
                normal_eye_ebo: None,
            })
            .unwrap_err(),
            Refusal::MissingInput(EYE_PUPIL_POSITION_MISSING)
        );
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
            receipt: None,
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
            pf30025: None,
            teeth_alignment: None,
            radiance: None,
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
            hlf_face: None,
            hlf_body: None,
            wrinkle: None,
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
    fn skin_clarity_runs_with_assets() {
        let mut rgba = frame(8, 8);
        apply(&mut rgba, 8, 8, &[("pc:8070", 0.5)]);
        assert_eq!(rgba.len(), 8 * 8 * 4);
    }

    #[test]
    fn pf90004_requires_an_owned_same_job_bundle() {
        let masks = PortraitMasks::solid(1.0);
        let absent = PortraitJobInputs::from_renderer_job(1, &masks);
        let (_, run) = apply_one_with_inputs(90004, 8, 8, &absent);
        assert_eq!(
            run.refusal,
            Some(Refusal::MissingInput("PF90004_PARSING_RGBA"))
        );

        let skin = vec![255u8; 64];
        let rgba = vec![0u8; 64 * 4];
        let complete = RadianceInputs {
            frame_width: Some(8),
            frame_height: Some(8),
            skin_mask: Some(&skin),
            parsing_rgba: Some(&rgba),
            face_rgba: Some(&rgba),
            value_mean: Some(128.0),
            roi: Some(NormRect::FULL),
        };
        for (radiance, missing) in [
            (
                RadianceInputs {
                    frame_width: None,
                    ..complete
                },
                "PF90004_FRAME_WIDTH",
            ),
            (
                RadianceInputs {
                    frame_height: None,
                    ..complete
                },
                "PF90004_FRAME_HEIGHT",
            ),
            (
                RadianceInputs {
                    skin_mask: None,
                    ..complete
                },
                "PF90004_SKIN_MASK",
            ),
            (
                RadianceInputs {
                    parsing_rgba: None,
                    ..complete
                },
                "PF90004_PARSING_RGBA",
            ),
            (
                RadianceInputs {
                    face_rgba: None,
                    ..complete
                },
                "PF90004_FACE_RGBA",
            ),
            (
                RadianceInputs {
                    value_mean: None,
                    ..complete
                },
                "PF90004_VALUE_MEAN",
            ),
            (
                RadianceInputs { roi: None, ..complete },
                "PF90004_ROI",
            ),
        ] {
            let inputs = PortraitJobInputs {
                radiance: Some(radiance),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let (_, run) = apply_one_with_inputs(90004, 8, 8, &inputs);
            assert_eq!(run.refusal, Some(Refusal::MissingInput(missing)));
        }

        let inputs = PortraitJobInputs {
            radiance: Some(RadianceInputs {
                roi: Some(NormRect {
                    x: 0.0,
                    y: 0.0,
                    width: 0.5,
                    height: 1.0,
                }),
                ..complete
            }),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let (_, run) = apply_one_with_inputs(90004, 8, 8, &inputs);
        assert_eq!(
            run.refusal,
            Some(Refusal::ValueOutOfRange("PF90004_ROI"))
        );

        let inputs = PortraitJobInputs {
            radiance: Some(complete),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let (_, run) = apply_one_with_inputs(90004, 8, 8, &inputs);
        assert!(run.refusal.is_none());
        assert!(run.executed);
    }

    #[test]
    fn pf90004_producer_writer_is_atomic_and_frame_exact() {
        let bundle = || {
            Pf90004ProducerInputs::from_same_job(
                8,
                8,
                vec![0u8; 64],
                vec![0u8; 64 * 4],
                vec![0u8; 64 * 4],
                82.42559051513672,
                NormRect::FULL,
            )
            .expect("valid PF90004 bundle")
        };

        for (result, expected) in [
            (
                Pf90004ProducerInputs::from_same_job(
                    0,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4],
                    128.0,
                    NormRect::FULL,
                ),
                Refusal::ShapeMismatch("PF90004_FRAME_SIZE"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 63],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4],
                    128.0,
                    NormRect::FULL,
                ),
                Refusal::ShapeMismatch("PF90004_SKIN_MASK"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4 - 1],
                    vec![0; 64 * 4],
                    128.0,
                    NormRect::FULL,
                ),
                Refusal::ShapeMismatch("PF90004_PARSING_RGBA"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4 - 1],
                    128.0,
                    NormRect::FULL,
                ),
                Refusal::ShapeMismatch("PF90004_FACE_RGBA"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4],
                    f32::NAN,
                    NormRect::FULL,
                ),
                Refusal::ValueOutOfRange("PF90004_VALUE_MEAN"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4],
                    128.0,
                    NormRect {
                        x: 0.0,
                        y: 0.0,
                        width: 0.5,
                        height: 1.0,
                    },
                ),
                Refusal::ValueOutOfRange("PF90004_ROI"),
            ),
        ] {
            assert_eq!(result.expect_err("invalid bundle"), expected);
        }

        let mut producer = ProducerInputs::default();
        let empty_fingerprint = producer.fingerprint();
        producer.write_pf90004(bundle());
        assert_ne!(producer.fingerprint(), empty_fingerprint);

        let masks = PortraitMasks::solid(1.0);
        let mut inputs = PortraitJobInputs::from_renderer_job(7, &masks);
        inputs.fill_from_producer_inputs(&producer);
        let radiance = inputs.radiance.expect("PF90004 bundle");
        assert_eq!(
            (radiance.frame_width, radiance.frame_height),
            (Some(8), Some(8))
        );
        assert_eq!(radiance.skin_mask.map(<[u8]>::len), Some(64));
        assert_eq!(radiance.parsing_rgba.map(<[u8]>::len), Some(64 * 4));
        assert_eq!(radiance.face_rgba.map(<[u8]>::len), Some(64 * 4));
        assert_eq!(radiance.value_mean, Some(82.42559051513672));
        assert_eq!(radiance.roi, Some(NormRect::FULL));

        let (_, run) = apply_one_with_inputs(90004, 8, 8, &inputs);
        assert!(run.refusal.is_none());
        assert!(run.executed);

        let shape_bundle = Pf90004ProducerInputs::from_same_job(
            4,
            16,
            vec![0; 64],
            vec![0; 64 * 4],
            vec![0; 64 * 4],
            128.0,
            NormRect::FULL,
        )
        .expect("same byte count, different frame shape");
        let mut wrong_shape = ProducerInputs::default();
        wrong_shape.write_pf90004(shape_bundle);
        let mut wrong_inputs = PortraitJobInputs::from_renderer_job(8, &masks);
        wrong_inputs.fill_from_producer_inputs(&wrong_shape);
        let (_, run) = apply_one_with_inputs(90004, 8, 8, &wrong_inputs);
        assert_eq!(
            run.refusal,
            Some(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))
        );
    }

    #[test]
    fn pf90004_parser_producer_keeps_unproven_planes_absent() {
        let masks = PortraitMasks {
            skin: Some(vec![1.0, 0.0, 0.5, 0.0]),
            hair: None,
            face: Some(vec![1.0, 1.0, 1.0, 1.0]),
            body: Some(vec![0.0; 4]),
            width: 2,
            height: 2,
        };
        let mut rgba = vec![0u8; 16];
        for (pixel, red) in [100u8, 200, 50, 20].into_iter().enumerate() {
            rgba[pixel * 4] = red;
            rgba[pixel * 4 + 3] = 255;
        }
        let auto = masks
            .pf90004_parser_inputs(&rgba, 2, 2, NormRect::FULL)
            .expect("parser fields");
        assert_eq!(auto.source, Pf90004AutoSource::PortraitParser);
        assert_eq!(auto.skin_mask, vec![255, 0, 128, 0]);
        assert_eq!(auto.parsing_rgba, None);
        assert_eq!(auto.face_rgba, None);
        assert_eq!(auto.value_mean, 75.0);
        assert_eq!(auto.roi, NormRect::FULL);
    }

    #[test]
    fn pf90004_value_mean_uses_native_even_statistics_size() {
        let masks = PortraitMasks {
            skin: Some(vec![1.0; 15]),
            hair: None,
            face: None,
            body: None,
            width: 5,
            height: 3,
        };
        let mut rgba = vec![0u8; 15 * 4];
        for pixel in 0..15 {
            rgba[pixel * 4] = 101;
            rgba[pixel * 4 + 1] = 202;
            rgba[pixel * 4 + 2] = 17;
            rgba[pixel * 4 + 3] = 255;
        }
        let auto = masks
            .pf90004_parser_inputs(&rgba, 5, 3, NormRect::FULL)
            .expect("parser fields");
        assert_eq!(auto.value_mean, 101.0);
    }

    #[cfg(windows)]
    #[test]
    fn pf90004_cuda_route_matches_cpu_oracle_and_reports_dispatch() {
        let (width, height) = (35u32, 27u32);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let masks = PortraitMasks::solid(1.0);
        let skin: Vec<u8> = (0..pixels).map(|i| (40 + (i * 19) % 216) as u8).collect();
        let parsing: Vec<u8> = (0..pixels * 4)
            .map(|i| ((i * 13 + 7) % 256) as u8)
            .collect();
        let face: Vec<u8> = (0..pixels * 4)
            .map(|i| ((i * 29 + 11) % 256) as u8)
            .collect();
        let radiance = RadianceInputs {
            frame_width: Some(width),
            frame_height: Some(height),
            skin_mask: Some(&skin),
            parsing_rgba: Some(&parsing),
            face_rgba: Some(&face),
            value_mean: Some(143.25),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            radiance: Some(radiance),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let expected = apply_skin_radiance(&source, 0.47, Some(radiance)).unwrap();
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90004".to_string(), 0.47f32)]),
            &inputs,
        );
        assert_eq!(actual, expected.pixels);
        let effect = &report.effects[0];
        assert!(effect.executed);
        assert!(effect.refusal.is_none());
        if effect.backend == PortraitBackend::Cuda {
            assert_eq!(
                effect
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>(),
                vec![aurora_portrait::cuda_radiance::KERNEL]
            );
            let dispatch = &effect.dispatches[0];
            assert_eq!(dispatch.block, (16, 16));
            assert!(!dispatch.device.is_empty());
            eprintln!(
                "PF90004 stage device={} kernel={} cc={:?} driver={} ptx_isa={:?}",
                dispatch.device,
                dispatch.kernel,
                dispatch.compute_cap,
                dispatch.driver_version,
                dispatch.ptx_isa
            );
        } else {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF90004 fell back: {:?}",
                effect.fallback_reason
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn pf90004_forced_cuda_failure_uses_cpu_oracle() {
        let (width, height) = (8u32, 8u32);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let masks = PortraitMasks::solid(1.0);
        let skin = vec![255u8; pixels];
        let parsing = vec![0u8; pixels * 4];
        let face = vec![0u8; pixels * 4];
        let radiance = RadianceInputs {
            frame_width: Some(width),
            frame_height: Some(height),
            skin_mask: Some(&skin),
            parsing_rgba: Some(&parsing),
            face_rgba: Some(&face),
            value_mean: Some(128.0),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            radiance: Some(radiance),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let expected = apply_skin_radiance(&source, 0.5, Some(radiance)).unwrap();
        let mut actual = source.pixels.clone();
        let _force = ForceCudaFailureGuard::new();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90004".to_string(), 0.5f32)]),
            &inputs,
        );
        assert_eq!(actual, expected.pixels);
        let effect = &report.effects[0];
        assert_eq!(effect.backend, PortraitBackend::Cpu);
        assert!(effect.dispatches.is_empty());
        assert!(effect.fallback_reason.is_some());
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
        for (pf, degree) in [
            (8068u32, 0.1f32),
            (30096, 0.1),
            (30096, 0.9),
            (90005, 0.9),
            (90007, 0.9),
        ] {
            let cpu = match pf {
                8068 => apply_skin_gloss(&source, degree, skin.as_deref()),
                30096 => apply_skin_temperature(&source, degree, skin.as_deref()),
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
    fn pf30096_forced_cuda_failure_falls_back_to_cpu() {
        let (width, height) = (8u32, 8u32);
        let source = Image::new(width, height, frame(width, height)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, width, height, NormRect::FULL);
        let expected = apply_skin_temperature(&source, 0.9, skin.as_deref()).expect("cpu");
        let params = BTreeMap::from([("pc:30096".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        let mut actual = source.pixels.clone();
        let _force = ForceCudaFailureGuard::new();
        let report = apply_portrait_effects_with_report_and_inputs(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(actual, expected.pixels);
        let effect = &report.effects[0];
        assert_eq!(effect.backend, PortraitBackend::Cpu);
        assert!(effect.dispatches.is_empty());
        assert!(effect.executed);
        assert!(
            effect
                .fallback_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("test-forced CUDA failure"))
        );
    }

    #[cfg(windows)]
    #[test]
    fn pf30096_report_marks_the_real_backend() {
        let (width, height) = (8u32, 8u32);
        let source = Image::new(width, height, frame(width, height)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, width, height, NormRect::FULL);
        let expected = apply_skin_temperature(&source, 0.9, skin.as_deref()).expect("cpu");
        let params = BTreeMap::from([("pc:30096".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_report_and_inputs(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(actual, expected.pixels);
        let effect = &report.effects[0];
        assert!(effect.executed);
        assert!(effect.refusal.is_none());
        match effect.backend {
            PortraitBackend::Cuda => {
                assert_eq!(effect.dispatches.len(), 1);
                assert_eq!(
                    effect.dispatches[0].kernel,
                    aurora_portrait::cuda_lut::TEMPERATURE_KERNEL
                );
            }
            PortraitBackend::Cpu => {
                assert!(std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none());
                assert!(effect.fallback_reason.is_some());
            }
            PortraitBackend::Skipped => panic!("PF30096 unexpectedly skipped"),
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
