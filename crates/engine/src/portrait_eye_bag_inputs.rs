//! PF101 (卧蚕 / Silkworm) basic same-frame assembly.
//!
//! The recovered consumer is `OverLay/BlendImageLayerWithMask.frag` with a
//! face-mask input id `0x15` sampled on `MASK_R`, a FilterType 42 material, and
//! separate mask/material UV streams (see `pf101_silkworm`). This module builds
//! the one `producer.pf101` slot from a `FaceInputBundle`:
//!
//! * Lower-eyelid band anchors use the same 194-point indices the crate already
//!   uses for the lower-lid Type1011 row (`PF30073`): eye corners `57/63` and
//!   `77/83`. No separate lower-lid ring index table is recovered, so the band
//!   is a tapered strip from the corner line downward, one per eye.
//! * `mask_uv[p]` is the pixel's normalized source coordinate inside a band and
//!   `[0, 0]` outside; `material_uv[p]` is the band-local `[0, 1]` rectangle
//!   inside a band and `[0, 0]` outside. The face mask reserves texel `(0, 0)`
//!   as zero so the background UV can never pick up mask weight.
//! * The face mask itself is the bundle's face ROI filled with 255, per the
//!   PF101 contract; other pixels are 0.
//! * `control` is the recovered render degree in `[0, 1]` (evidence
//!   `parameter_contract.PF101.render_param_range`), not UI percent. It is
//!   encoded as `main_mask_scalar` so `SilkwormParams::degree01() == control`
//!   bit-for-bit. `adjust_gate` stays `false`: the Geometry predicate
//!   `0x60C17C0` that would multiply the 0.7 AdjustFactor is still unproven
//!   (`U-PF101-01`), and `Pf101ProducerInputs::apply` receives the raw control.
//!
//! Material provenance: `SILK00000001/wcz1.png` is the recovered left-eye
//! material and `wcy1.png` the right-eye material. The on-disk vendor files are
//! container-obfuscated, so they are loaded only as decoded PNGs from the
//! established external roots: `AURORA_PF101_MATERIAL` (a direct PNG) or
//! `AURORA_MAKEUP_MATERIAL_ROOT` + `Makeups/Silkworm/SILK00000001/wcz1.png`.
//! The single-material PF101 slot uses wcz1 for both bands and mirrors `u` for
//! the right eye; per-side asset selection/flip is still unproven
//! (`U-PF101-04`). No asset is synthesized: missing/undecodable material is a
//! named `MissingInput("PF101_MATERIAL")`.

use std::path::PathBuf;

use aurora_portrait::kernels::pf101_silkworm::{self, SilkwormParams};
use aurora_portrait::{Image, Refusal};

use crate::portrait_stage::{Pf101ProducerInputs, ProducerInputs};

/// Named refusal when no decrypted PF101 material PNG is available.
pub const PF101_MATERIAL: &str = "PF101_MATERIAL";
const PF101_SOURCE: &str = "PF101_SOURCE";
const PF101_CONTROL: &str = "PF101_CONTROL";
const PF101_FACE_MASK: &str = "PF101_FACE_MASK";
const PF101_EYEBAG_LANDMARKS: &str = "PF101_EYEBAG_LANDMARKS";

/// Direct path to one already-decrypted Silkworm material PNG.
pub const PF101_MATERIAL_ENV: &str = "AURORA_PF101_MATERIAL";
/// Root of the decrypted vendor `effect/config` tree (shared with makeup).
pub const PF101_MATERIAL_ROOT_ENV: &str = "AURORA_MAKEUP_MATERIAL_ROOT";
/// Recovered left-eye material under the decrypted config root.
pub const PF101_MATERIAL_RELATIVE: &str = "Makeups/Silkworm/SILK00000001/wcz1.png";
/// Private-asset layout for the same decrypted PNG, matching the other
/// portrait materials (`private-assets/portrait/...`).
pub const PF101_PRIVATE_RELATIVE: &str =
    "portrait/makeup/Makeups/Silkworm/SILK00000001/wcz1.png";

/// Left eye outer/inner corner indices in the 142-point primary template.
pub const PF101_LEFT_EYE_CORNERS: (usize, usize) = (18, 22);
/// Right eye inner/outer corner indices in the 142-point primary template.
pub const PF101_RIGHT_EYE_CORNERS: (usize, usize) = (40, 36);

impl crate::portrait_face_input_stage::FaceInputBundle {
    /// Builds and attaches PF101 inputs for this exact frame/job.
    ///
    /// `control` is the render degree in `[0, 1]`. The same value must be passed
    /// to `Pf101ProducerInputs::apply`, which requires it to equal
    /// `degree01()`.
    pub fn attach_pf101_inputs(
        &self,
        source: &Image,
        control: f32,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.attach_pf101_inputs_with_style(source, control, 0, producer)
    }

    pub fn attach_pf101_inputs_with_style(
        &self,
        source: &Image,
        control: f32,
        style_index: usize,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.attach_pf101_selection(source, control, style_index, None, producer)
    }

    pub(crate) fn attach_pf101_from_adjustments(
        &self, source: &Image, control: f32, extra: &std::collections::BTreeMap<String, f32>, producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        let style = extra.get("pc:101:style").copied().unwrap_or(0.0);
        if !(0.0..=16_777_215.0).contains(&style) || style.fract() != 0.0 {
            return Err(Refusal::ValueOutOfRange("PF101 style ordinal"));
        }
        let tint = crate::portrait_makeup_wiring::material_tint(101, Some(extra))?;
        self.attach_pf101_selection(source, control, style as usize, tint, producer)
    }

    fn attach_pf101_selection(&self, source: &Image, control: f32, style_index: usize,
        tint: Option<[u8; 3]>, producer: &mut ProducerInputs) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        source
            .validate()
            .map_err(|_| Refusal::ShapeMismatch(PF101_SOURCE))?;
        let params = pf101_params(control)?;
        let mut material = if style_index >= 1 {
            let presets = crate::portrait_makeup_wiring::makeup_presets_for_pf(101);
            let preset = presets.get(style_index - 1).ok_or(Refusal::MissingInput("PF101 selected material"))?;
            load_pf101_material_from(std::slice::from_ref(&preset.path))?
        } else {
            load_pf101_material()?
        };
        if let Some(tint) = tint { crate::portrait_makeup_wiring::tint_material_pixels(&mut material, tint); }
        let face_mask = build_face_mask(source, &self.points194, self.roi_xywh)?;
        let (mask_uv, material_uv) = build_band_uvs(source, &self.points194)?;
        producer.pf101 = Some(Pf101ProducerInputs::from_same_job(
            self.job_id,
            material,
            face_mask,
            mask_uv,
            material_uv,
            params,
        )?);
        Ok(())
    }
}

/// Recovered FilterType 42 (SoftLight) with the control as maskIndex=0 scalar.
fn pf101_params(control: f32) -> Result<SilkwormParams, Refusal> {
    if !control.is_finite() || !(0.0..=1.0).contains(&control) {
        return Err(Refusal::ValueOutOfRange(PF101_CONTROL));
    }
    let params = SilkwormParams {
        filter_type: pf101_silkworm::FILTER_TYPE,
        main_mask_scalar: control,
        axis_weight_w1: 1.0,
        adjust_gate: false,
    };
    // Validate the FilterType/degree contract before it reaches the producer.
    params.degree01()?;
    Ok(params)
}

fn material_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os(PF101_MATERIAL_ENV)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    {
        candidates.push(path);
    }
    if let Some(root) = std::env::var_os(PF101_MATERIAL_ROOT_ENV).map(PathBuf::from) {
        candidates.push(root.join(PF101_MATERIAL_RELATIVE));
    }
    if let Some(root) = crate::portrait_makeup_wiring::makeup_asset_root() {
        candidates.push(root.join(PF101_MATERIAL_RELATIVE));
    }
    if let Some(path) = aurora_portrait::private_assets::path(PF101_PRIVATE_RELATIVE) {
        candidates.push(path);
    }
    candidates
}

/// Default PF101 material resolution, shared by the producer and the cheap
/// probe: direct env file, the decoded config/default candidates, then the
/// first real Silkworm preset. A broken candidate is skipped here; an explicit
/// style is resolved alone and never falls through to another material.
pub(crate) fn pf101_default_material_candidates() -> Vec<PathBuf> {
    let mut candidates = material_candidates();
    candidates.extend(
        crate::portrait_makeup_wiring::makeup_presets_for_pf(101)
            .into_iter()
            .map(|preset| preset.path),
    );
    candidates
}

fn load_pf101_material() -> Result<Image, Refusal> {
    load_pf101_material_from(&pf101_default_material_candidates())
}

fn load_pf101_material_from(candidates: &[PathBuf]) -> Result<Image, Refusal> {
    for path in candidates {
        if !path.is_file() {
            continue;
        }
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let Ok(decoded) = image::load_from_memory(&bytes) else {
            log::warn!(
                "PF101 material {} is not a decrypted PNG; refusing",
                path.display()
            );
            continue;
        };
        let rgba = decoded.into_rgba8();
        let (width, height) = rgba.dimensions();
        if let Some(image) = Image::new(width, height, rgba.into_raw()) {
            return Ok(image);
        }
    }
    log::warn!(
        "PF101 material missing; expected a decrypted PNG via {} or {} + {}",
        PF101_MATERIAL_ENV,
        PF101_MATERIAL_ROOT_ENV,
        PF101_MATERIAL_RELATIVE
    );
    Err(Refusal::MissingInput(PF101_MATERIAL))
}

/// Face mask: 255 inside the bundle's face ROI, 0 elsewhere. Texel `(0, 0)` is
/// reserved as zero because the background UV is `[0, 0]`.
fn build_face_mask(
    source: &Image,
    points194: &[[f32; 2]],
    roi_xywh: [f32; 4],
) -> Result<Image, Refusal> {
    let width = source.width;
    let height = source.height;
    let pixels = source
        .pixel_count()
        .checked_mul(4)
        .ok_or(Refusal::ShapeMismatch(PF101_FACE_MASK))?;
    let (x0, y0, x1, y1) = face_roi_bounds(width, height, points194, roi_xywh)?;
    let mut rgba = vec![0u8; pixels];
    for y in y0..y1 {
        for x in x0..x1 {
            let offset = (y as usize * width as usize + x as usize) * 4;
            rgba[offset..offset + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    if rgba.len() >= 4 {
        rgba[0..4].fill(0);
    }
    Image::new(width, height, rgba).ok_or(Refusal::ShapeMismatch(PF101_FACE_MASK))
}

fn face_roi_bounds(
    width: u32,
    height: u32,
    points194: &[[f32; 2]],
    roi_xywh: [f32; 4],
) -> Result<(u32, u32, u32, u32), Refusal> {
    let (low, high);
    if roi_xywh.iter().all(|value| value.is_finite())
        && roi_xywh[2] > 0.0
        && roi_xywh[3] > 0.0
    {
        low = [roi_xywh[0].floor(), roi_xywh[1].floor()];
        high = [
            (roi_xywh[0] + roi_xywh[2]).ceil(),
            (roi_xywh[1] + roi_xywh[3]).ceil(),
        ];
    } else {
        let mut min = [f32::INFINITY; 2];
        let mut max = [f32::NEG_INFINITY; 2];
        for point in points194 {
            if point.iter().any(|value| !value.is_finite()) {
                return Err(Refusal::ValueOutOfRange(PF101_FACE_MASK));
            }
            min[0] = min[0].min(point[0]);
            min[1] = min[1].min(point[1]);
            max[0] = max[0].max(point[0]);
            max[1] = max[1].max(point[1]);
        }
        if !min[0].is_finite() || !min[1].is_finite() {
            return Err(Refusal::MissingInput(PF101_FACE_MASK));
        }
        low = [min[0].floor(), min[1].floor()];
        high = [max[0].ceil(), max[1].ceil()];
    }
    let clamp = |value: f32, limit: u32| value.clamp(0.0, limit as f32) as u32;
    let x0 = clamp(low[0], width);
    let y0 = clamp(low[1], height);
    let x1 = clamp(high[0], width);
    let y1 = clamp(high[1], height);
    if x1 <= x0 || y1 <= y0 {
        return Err(Refusal::ShapeMismatch(PF101_FACE_MASK));
    }
    Ok((x0, y0, x1, y1))
}

/// Per-pixel UV streams of length `source.pixel_count()`.
///
/// Inside a band: `mask_uv = ((x + 0.5) / width, (y + 0.5) / height)` (sample
/// the face mask at this pixel) and `material_uv` spans the material's `[0, 1]`
/// rectangle (`u` across the eye, mirrored for the right eye; `v` from lid to
/// band bottom). Outside every band both streams are `[0, 0]`, where the face
/// mask is guaranteed to be zero.
fn build_band_uvs(
    source: &Image,
    points194: &[[f32; 2]],
) -> Result<(Vec<[f32; 2]>, Vec<[f32; 2]>), Refusal> {
    let width = source.width as usize;
    let height = source.height as usize;
    let pixels = width
        .checked_mul(height)
        .ok_or(Refusal::ShapeMismatch(PF101_SOURCE))?;
    if source.pixels.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch(PF101_SOURCE));
    }
    let mut mask_uv = vec![[0.0f32; 2]; pixels];
    let mut material_uv = vec![[0.0f32; 2]; pixels];
    for (eye, (first, second)) in [
        PF101_LEFT_EYE_CORNERS,
        PF101_RIGHT_EYE_CORNERS,
    ]
    .into_iter()
    .enumerate()
    {
        let corner = |index: usize| {
            points194
                .get(index)
                .copied()
                .ok_or(Refusal::MissingInput(PF101_EYEBAG_LANDMARKS))
        };
        let a = corner(first)?;
        let b = corner(second)?;
        if a.iter().chain(&b).any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange(PF101_EYEBAG_LANDMARKS));
        }
        let center_x = (a[0] + b[0]) * 0.5;
        let center_y = (a[1] + b[1]) * 0.5;
        let span = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if !span.is_finite() || span < 2.0 {
            return Err(Refusal::ValueOutOfRange(PF101_EYEBAG_LANDMARKS));
        }
        let half = span * 0.5;
        let thickness = (span * 0.42).max(1.5);
        for y in 0..height {
            let py = y as f32 + 0.5;
            let dy = py - center_y;
            if dy < 0.0 {
                continue;
            }
            for x in 0..width {
                let px = x as f32 + 0.5;
                let t = (px - center_x) / half;
                if t <= -1.0 || t >= 1.0 {
                    continue;
                }
                let taper = (1.0 - t * t).max(0.0).sqrt();
                let bottom = thickness * taper;
                if bottom <= f32::EPSILON || dy > bottom {
                    continue;
                }
                let local_u = (t * 0.5 + 0.5).clamp(0.0, 1.0);
                let u = if eye == 1 { 1.0 - local_u } else { local_u };
                let v = (dy / bottom).clamp(0.0, 1.0);
                let index = y * width + x;
                mask_uv[index] = [px / source.width as f32, py / source.height as f32];
                material_uv[index] = [u, v];
            }
        }
    }
    Ok((mask_uv, material_uv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portrait_face_input_stage::FaceInputBundle;
    use crate::portrait_stage::{
        C1FaceDetection, C1FaceMesh, FaceLandmarkProducerInputs, FaceModelIdentity,
        FaceNormalization,
    };

    const JOB: u64 = 41;

    fn source() -> Image {
        Image::solid(64, 48, [10, 20, 30, 255])
    }

    fn points194() -> Vec<[f32; 2]> {
        let mut points = vec![[32.0f32, 24.0]; 194];
        points[PF101_LEFT_EYE_CORNERS.0] = [18.0, 18.0];
        points[PF101_LEFT_EYE_CORNERS.1] = [30.0, 18.0];
        points[PF101_RIGHT_EYE_CORNERS.0] = [34.0, 18.0];
        points[PF101_RIGHT_EYE_CORNERS.1] = [46.0, 18.0];
        points
    }

    fn bundle(job: u64, source: &Image) -> FaceInputBundle {
        let normalization = FaceNormalization {
            channel_order: "BGR",
            mean: [127.5; 3],
            scale: [128.0; 3],
        };
        let model = FaceModelIdentity {
            model_id: "test",
            sha256: [0; 32],
        };
        FaceInputBundle {
            job_id: job,
            source_size: [source.width, source.height],
            source_sha256: crate::sha256_32(&source.pixels),
            bound_digest: None,
            detection: C1FaceDetection {
                score: 0.9,
                landmarks_source: [[2.0, 4.0]; 5],
                source_to_input: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                input_size: [640, 640],
                normalization,
                provider: "cpu",
                model,
            },
            mesh: C1FaceMesh {
                score: 0.9,
                landmarks_source: vec![[2.0, 2.0, 0.0]],
                source_to_crop: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                crop_side: 4.0,
                input_size: [256, 256],
                normalization,
                provider: "cpu",
                model,
            },
            roi_xywh: [4.0, 4.0, 56.0, 40.0],
            roi_normalized_xywh: [0.0625, 0.0833, 0.875, 0.8333],
            points17: Vec::new(),
            points142: Vec::new(),
            points194: points194(),
            points414: vec![[4.0, 4.0]; 414],
            landmark_producer: FaceLandmarkProducerInputs {
                resource_id: 1,
                provider: "test".to_string(),
                model_sha256: [0; 32],
                source_sha256: [0; 32],
                source_size: [source.width, source.height],
                points142: Vec::new(),
                crop_bgr: Vec::new(),
                crop_sha256: [0; 32],
                raw52: Vec::new(),
                points52: Vec::new(),
                points194: Vec::new(),
                fit_pre: [[0.0; 3]; 2],
                fit_inverse: [[0.0; 3]; 2],
            },
            model_inputs: None,
        }
    }

    #[test]
    fn uv_streams_cover_pixels_and_map_band_into_material_rect() {
        let source = source();
        let points = points194();
        let (mask_uv, material_uv) = build_band_uvs(&source, &points).unwrap();
        let pixels = source.pixel_count();
        assert_eq!(mask_uv.len(), pixels);
        assert_eq!(material_uv.len(), pixels);
        assert!(mask_uv.iter().chain(&material_uv).flatten().all(|v| v.is_finite()));

        // Background is zero in both streams.
        assert_eq!(mask_uv[0], [0.0, 0.0]);
        assert_eq!(material_uv[0], [0.0, 0.0]);

        // Directly under each eye centre, one row below the corner line.
        let left = 19 * 64 + 22;
        let right = 19 * 64 + 38;
        let bottom = 12.0f32 * 0.42 * (1.0 - 0.25f32 * 0.25).sqrt();
        let expected_v = 1.5 / bottom;
        assert!((material_uv[left][0] - 0.375).abs() < 1e-5);
        assert!((material_uv[left][1] - expected_v).abs() < 1e-5);
        // The right band mirrors u with the same single left-eye material.
        assert!((material_uv[right][0] - 0.625).abs() < 1e-5);
        assert!((material_uv[right][1] - expected_v).abs() < 1e-5);
        // Band pixels sample the face mask at their own normalized coordinate.
        assert!((mask_uv[left][0] - 22.5 / 64.0).abs() < 1e-6);
        assert!((mask_uv[left][1] - 19.5 / 48.0).abs() < 1e-6);

        let inside = material_uv
            .iter()
            .filter(|uv| **uv != [0.0, 0.0])
            .count();
        assert!(inside > 0);
        assert!(inside < pixels);
        assert!(material_uv
            .iter()
            .filter(|uv| **uv != [0.0, 0.0])
            .all(|uv| (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1])));
    }

    #[test]
    fn params_encode_control_as_degree01() {
        for control in [0.0f32, 0.35, 0.7, 1.0] {
            let params = pf101_params(control).unwrap();
            assert_eq!(params.filter_type, 42);
            assert_eq!(params.degree01().unwrap(), control);
            assert!(!params.adjust_gate);
            assert_eq!(params.main_mask_scalar, control);
            assert_eq!(params.axis_weight_w1, 1.0);
        }
        for bad in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
            assert_eq!(
                pf101_params(bad),
                Err(Refusal::ValueOutOfRange(PF101_CONTROL))
            );
        }
    }

    #[test]
    fn face_mask_fills_roi_and_reserves_origin_for_background_uv() {
        let source = source();
        let mask = build_face_mask(&source, &points194(), [4.0, 4.0, 56.0, 40.0]).unwrap();
        assert_eq!((mask.width, mask.height), (64, 48));
        assert_eq!(mask.pixel(4, 4), [255, 255, 255, 255]);
        assert_eq!(mask.pixel(5, 5), [255, 255, 255, 255]);
        assert_eq!(mask.pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(mask.pixel(1, 1), [0, 0, 0, 0]);
        assert_eq!(mask.pixel(63, 47), [0, 0, 0, 0]);
    }

    #[test]
    fn missing_material_is_a_named_refusal() {
        let missing = std::env::temp_dir().join(format!(
            "aurora-pf101-material-does-not-exist-{}.png",
            std::process::id()
        ));
        assert_eq!(
            load_pf101_material_from(&[missing]),
            Err(Refusal::MissingInput(PF101_MATERIAL))
        );
        assert_eq!(
            load_pf101_material_from(&[]),
            Err(Refusal::MissingInput(PF101_MATERIAL))
        );
    }

    #[test]
    fn attach_validates_ticket_before_material() {
        let source = source();
        let mut producer = ProducerInputs::default();
        assert_eq!(
            bundle(0, &source).attach_pf101_inputs(&source, 0.7, &mut producer),
            Err(Refusal::ValueOutOfRange("face input job ticket"))
        );
        assert!(producer.pf101.is_none());
        // Degree outside the recovered render range is refused by name before
        // any asset lookup.
        assert_eq!(
            bundle(JOB, &source).attach_pf101_inputs(&source, 70.0, &mut producer),
            Err(Refusal::ValueOutOfRange(PF101_CONTROL))
        );
    }
}
