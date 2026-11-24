//! PF107 `LipColor` and PF115 `3DMMFaceSticker` consumers over decoded vendor
//! preset rows.
//!
//! Both functions read only the already-decoded `VendorEffect` rows and bytes
//! supplied by the caller. Nothing here walks a vendor install directory.
//!
//! PF107
//! -----
//! `LipColor` rows carry no `FilterType`; their pixel law is
//! `aurora_portrait::kernels::lip_color` (`LipColorNormal.frag`). The five
//! uniforms without a located config writer and the mode-to-program selection
//! stay caller-owned; see that module. `LipStickImageMakeup` rows are a
//! different family (`FilterType 42`) and are refused here by name.
//!
//! PF115
//! -----
//! The five `3DMMFaceSticker` rows name `01.png`..`05.png` and
//! `bfm_head_meanshape_reduce_with_uv_smooth_nose.obj` (9,794 `v`, 9,794 `vt`,
//! 19,512 triangles; five byte-identical copies, MD5
//! `c21b29e032c5c3a82c5dc771443441d0`). The decoded preset carries **no**
//! placement data: no per-element `pos/scale/rot` list, no global scale, no
//! `fl`/`fk`, no flip-mask flag, and no 3DMM projection from `ModelPath`
//! vertices into the same-job camera frame. [`apply_vendor_sticker`] therefore
//! refuses with [`PF115_3DMM_TRANSFORM`] unless the caller supplies an
//! evidence-backed [`StickerTransform`]; no default transform is synthesised.
//! The three `BlusherImageMakeup` rows in the same directory are the 2D
//! `BlusherMaskType=15` family and are refused by this module.

use aurora_portrait::kernels::lip_color::{
    apply_lip_color, LipColorParams, LipColorRow, LipColorUniforms, LipColorUv, LIP_COLOR_EFFECT,
};
use aurora_portrait::kernels::makeup_material::MaskChannel;
use aurora_portrait::kernels::sticker::{self, Element, StickerParams};
use aurora_portrait::{Image, Refusal};

use crate::portrait_vendor_config::VendorEffect;

pub const LIP_COLOR_PF: u32 = 107;
pub const FACE_STICKER_PF: u32 = 115;
pub const LIP_COLOR: &str = LIP_COLOR_EFFECT;
pub const FACE_STICKER: &str = "3DMMFaceSticker";
pub const BLUSHER_IMAGE: &str = "BlusherImageMakeup";

/// `LipColor` uniforms with no located config writer were not supplied.
pub const LIP_COLOR_UNIFORMS_MISSING: &str = "PF107_LIP_COLOR_UNIFORMS";

/// Decoded OBJ shape of the vendor head mesh. Every `3DMMFaceSticker` row in
/// the corpus names a byte-identical copy of this file.
pub const STICKER_OBJ_VERTICES: usize = 9_794;
pub const STICKER_OBJ_UVS: usize = 9_794;
pub const STICKER_OBJ_FACES: usize = 19_512;
pub const STICKER_VENDOR_OBJ: &str =
    "bfm_head_meanshape_reduce_with_uv_smooth_nose.obj";

/// Named refusal for a 3DMM sticker that has texture and OBJ static inputs but
/// no placement/projection evidence. The detailed list of missing fields is in
/// [`PF115_3DMM_TRANSFORM_DETAIL`].
pub const PF115_3DMM_TRANSFORM: &str = "PF115_3DMM_TRANSFORM";
pub const PF115_3DMM_TRANSFORM_DETAIL: &str = "PF115_3DMM_TRANSFORM: effect.json \
    has Paths + ModelPath but no sticker element list (per-element pos/scale/rot), \
    no global scale / fl / fk, no flip-mask flag, and no 3DMM projection from \
    ModelPath vertices to the same-job camera frame";
pub const STICKER_TEXTURE_MISSING: &str = "PF115_3DMM_TEXTURE";
pub const STICKER_OBJ_MISSING: &str = "PF115_3DMM_OBJ";
pub const STICKER_OBJ_INVALID: &str = "PF115_3DMM_OBJ_INVALID";
pub const STICKER_OPACITY_UNSUPPORTED: &str = "PF115_3DMM_OPACITY";
pub const STICKER_MASK_UNSUPPORTED: &str = "PF115_3DMM_MASK";
pub const STICKER_PARAM_FLAG: &str = "3DMMFaceSticker ParamFlag";
pub const LIP_COLOR_PARAM_FLAG: &str = "LipColor ParamFlag";

fn as_string(value: Option<&serde_json::Value>) -> Option<String> {
    value.and_then(serde_json::Value::as_str).map(str::to_owned)
}

fn as_bool(value: Option<&serde_json::Value>) -> Option<bool> {
    value.and_then(serde_json::Value::as_bool)
}

fn as_f64(value: Option<&serde_json::Value>) -> Option<f64> {
    value.and_then(serde_json::Value::as_f64)
}

fn as_u32(value: Option<&serde_json::Value>) -> Option<u32> {
    value
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn as_rgba(value: Option<&serde_json::Value>) -> Option<[u8; 4]> {
    let items = value?.as_array()?;
    if items.len() != 4 {
        return None;
    }
    let mut out = [0u8; 4];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = u8::try_from(item.as_i64()?).ok()?;
    }
    Some(out)
}

/// Decodes one `EffectName=LipColor` preset row. Only fields named by the
/// vendor preset are read; a missing field is a named refusal rather than a
/// substituted default.
pub fn lip_color_row_from_vendor(effect: &VendorEffect) -> Result<LipColorRow, Refusal> {
    if effect.effect_name != LIP_COLOR {
        return Err(Refusal::ValueOutOfRange(LIP_COLOR_EFFECT));
    }
    if effect.param_flag != Some(LIP_COLOR_PF as i64) {
        return Err(Refusal::ValueOutOfRange(LIP_COLOR_PARAM_FLAG));
    }
    let raw = effect.raw();
    Ok(LipColorRow {
        effect_name: LIP_COLOR_EFFECT,
        config_path: effect.config_rel_path.clone(),
        lip_color_mode: as_u32(raw.get("LipColorMode"))
            .ok_or(Refusal::MissingInput("LipColorMode"))?,
        multi_color_switch: as_bool(raw.get("MultiColorSwitch")).unwrap_or(false),
        teeth_whiten_switch: as_bool(raw.get("TeethWhitenSwitch")).unwrap_or(false),
        makeup_degree_factor: as_f64(raw.get("MakeupDegreeFactor"))
            .ok_or(Refusal::MissingInput("MakeupDegreeFactor"))?,
        light_alpha_percent: as_f64(raw.get("LightAlpha")),
        gloss_alpha_percent: as_f64(raw.get("GlossAlpha")),
        mix_color0: as_rgba(raw.get("MixColor0")).ok_or(Refusal::MissingInput("MixColor0"))?,
        opacity0_percent: as_f64(raw.get("Opacity0"))
            .ok_or(Refusal::MissingInput("Opacity0"))?,
        adapt_env: as_bool(raw.get("AdaptEnv")).unwrap_or(false),
        remove_highlight: as_bool(raw.get("RemoveHighlight")).unwrap_or(false),
        radius_scale_factor: as_f64(raw.get("RadiusScaleFactor")),
        metal_color: as_rgba(raw.get("MetalColor")),
        shimmer_metal_path: as_string(raw.get("ShimmerMetalPath")),
        highlight_param: match raw.get("HighlightParam") {
            None => None,
            Some(value) if value.is_null() => None,
            Some(value) => {
                let weight = as_f64(value.get("Weight"))
                    .ok_or(Refusal::MissingInput("HighlightParam.Weight"))?;
                let threshold = as_f64(value.get("Threshold"))
                    .ok_or(Refusal::MissingInput("HighlightParam.Threshold"))?;
                Some(aurora_portrait::kernels::lip_color::HighlightParam {
                    weight,
                    threshold,
                })
            }
        },
        default_degree: effect
            .default_degree
            .ok_or(Refusal::MissingInput("LipColor DefaultDegree"))?,
        degree_range: effect
            .degree_range
            .ok_or(Refusal::MissingInput("LipColor DegreeRange"))?,
    })
}

/// Decodes a `LipColor` row and applies the recovered `LipColorNormal` law.
/// `uniforms` is `None` when the caller has no evidence for the uniforms whose
/// config writer was not located; that is a named refusal.
#[allow(clippy::too_many_arguments)]
pub fn apply_vendor_lip_color(
    effect: &VendorEffect,
    source: &Image,
    combine_mask: &Image,
    smooth: &Image,
    blend_weight: &Image,
    level_weight: &Image,
    uv: Option<LipColorUv>,
    uniforms: Option<LipColorUniforms>,
) -> Result<Image, Refusal> {
    let row = lip_color_row_from_vendor(effect)?;
    let uniforms = uniforms.ok_or(Refusal::MissingInput(LIP_COLOR_UNIFORMS_MISSING))?;
    let params = LipColorParams::from_vendor(&row, uniforms)?;
    apply_lip_color(
        source,
        combine_mask,
        smooth,
        blend_weight,
        level_weight,
        uv,
        params,
    )
}

/// One vertex of the decoded vendor OBJ. Normals are ignored: the recovered
/// sticker shader samples the material at the transformed UV and uses its
/// alpha/colour only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StickerVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct StickerMesh {
    pub vertices: Vec<StickerVertex>,
    /// Zero-based `v` indices, three per triangle.
    pub indices: Vec<u32>,
    pub uv_count: usize,
    pub source_model_path: String,
}

fn invalid_obj() -> Refusal {
    Refusal::ShapeMismatch(STICKER_OBJ_INVALID)
}

fn parse_f32(field: Option<&str>) -> Result<f32, Refusal> {
    field
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .ok_or_else(invalid_obj)
}

fn parse_index(part: Option<&str>) -> Result<usize, Refusal> {
    part.and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .map(|value| value - 1)
        .ok_or_else(invalid_obj)
}

fn parse_corner(token: &str) -> Result<(usize, usize), Refusal> {
    let mut parts = token.split('/');
    let position = parse_index(parts.next())?;
    let uv = parse_index(parts.next())?;
    // A trailing normal index (`v/vt/vn`) is allowed and ignored.
    Ok((position, uv))
}

impl StickerMesh {
    /// Parses the plaintext OBJ subset in the corpus: `v x y z`, `vt u v`,
    /// `f v/vt[/vn] v/vt[/vn] v/vt[/vn]`. Anything else is
    /// [`STICKER_OBJ_INVALID`].
    pub fn parse_obj(text: &str, model_path: &str) -> Result<Self, Refusal> {
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut corners: Vec<(usize, usize)> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split_whitespace();
            match fields.next() {
                Some("v") => {
                    let mut values = [0.0f32; 3];
                    for slot in &mut values {
                        *slot = parse_f32(fields.next())?;
                    }
                    if fields.next().is_some() {
                        return Err(invalid_obj());
                    }
                    positions.push(values);
                }
                Some("vt") => {
                    let mut values = [0.0f32; 2];
                    for slot in &mut values {
                        *slot = parse_f32(fields.next())?;
                    }
                    // An optional third component is ignored.
                    uvs.push(values);
                }
                Some("f") => {
                    let mut face = [(0usize, 0usize); 3];
                    for slot in &mut face {
                        let token = fields.next().ok_or_else(invalid_obj)?;
                        *slot = parse_corner(token)?;
                    }
                    if fields.next().is_some() {
                        return Err(invalid_obj());
                    }
                    corners.extend_from_slice(&face);
                }
                // `vn`, `o`, `g`, `usemtl`, `s`, `mtllib` and comments are inert.
                _ => {}
            }
        }
        if positions.is_empty() || uvs.is_empty() || corners.is_empty() {
            return Err(invalid_obj());
        }
        let mut uv_index_of_vertex = vec![usize::MAX; positions.len()];
        let mut indices = Vec::with_capacity(corners.len());
        for (position, uv) in corners {
            if position >= positions.len() || uv >= uvs.len() {
                return Err(invalid_obj());
            }
            if uv_index_of_vertex[position] == usize::MAX {
                uv_index_of_vertex[position] = uv;
            } else if uv_index_of_vertex[position] != uv {
                return Err(invalid_obj());
            }
            indices.push(position as u32);
        }
        let mut vertices = Vec::with_capacity(positions.len());
        for (position, uv_index) in positions.into_iter().zip(uv_index_of_vertex) {
            if uv_index == usize::MAX {
                return Err(invalid_obj());
            }
            vertices.push(StickerVertex {
                position,
                uv: uvs[uv_index],
            });
        }
        Ok(Self {
            vertices,
            indices,
            uv_count: uvs.len(),
            source_model_path: model_path.to_owned(),
        })
    }

    /// True when the mesh matches the vendor head mesh named by every decoded
    /// `3DMMFaceSticker` row.
    pub fn is_vendor_corpus_shape(&self) -> bool {
        self.vertices.len() == STICKER_OBJ_VERTICES
            && self.uv_count == STICKER_OBJ_UVS
            && self.indices.len() == STICKER_OBJ_FACES * 3
    }
}

/// PF115 `MaskType=15` → `dec 14` → `table[14] = 13` → `MASK_RGBA`
/// (`backend-r76/s54_pf115_sticker_summary/REPORT.md`).
pub fn sticker_mask_channel(mask_type: u32) -> Result<MaskChannel, Refusal> {
    if mask_type == 15 {
        Ok(MaskChannel::Rgba)
    } else {
        Err(Refusal::MissingInput(STICKER_MASK_UNSUPPORTED))
    }
}

/// Static inputs of one `3DMMFaceSticker` row. `erode_interval`,
/// `face_3dmm_type` and `degree_range` are carried verbatim; no located pixel
/// consumer reads them, so they are not applied here.
#[derive(Clone, Debug, PartialEq)]
pub struct StickerStaticInputs {
    pub config_path: String,
    pub texture_path: String,
    pub model_path: String,
    pub mesh: StickerMesh,
    pub mask_type: u32,
    pub mask_channel: MaskChannel,
    pub erode_interval: Option<f64>,
    pub face_3dmm_type: Option<u32>,
    pub filter_type: u32,
    pub opacity: f64,
    pub default_degree: f64,
    pub degree_range: [f64; 2],
}

/// Resolves the static texture/OBJ/parameter side of a `3DMMFaceSticker` row.
/// This is `Ok` only because placement is a separate input: see
/// [`apply_vendor_sticker`], which cannot run without [`StickerTransform`].
pub fn resolve_3dmm_sticker(
    effect: &VendorEffect,
    obj_text: &str,
) -> Result<StickerStaticInputs, Refusal> {
    if effect.effect_name != FACE_STICKER {
        return Err(Refusal::ValueOutOfRange(FACE_STICKER));
    }
    if effect.param_flag != Some(FACE_STICKER_PF as i64) {
        return Err(Refusal::ValueOutOfRange(STICKER_PARAM_FLAG));
    }
    let raw = effect.raw();
    let texture_path = effect
        .paths
        .first()
        .cloned()
        .ok_or(Refusal::MissingInput(STICKER_TEXTURE_MISSING))?;
    let model_path = as_string(raw.get("ModelPath"))
        .ok_or(Refusal::MissingInput(STICKER_OBJ_MISSING))?;
    let mask_type = as_u32(raw.get("MaskType"))
        .ok_or(Refusal::MissingInput("3DMMFaceSticker MaskType"))?;
    let filter_type = effect
        .filter_type
        .and_then(|value| u32::try_from(value).ok())
        .ok_or(Refusal::MissingInput("3DMMFaceSticker FilterType"))?;
    let mesh = StickerMesh::parse_obj(obj_text, &model_path)?;
    if model_path == STICKER_VENDOR_OBJ && !mesh.is_vendor_corpus_shape() {
        return Err(Refusal::ShapeMismatch(STICKER_OBJ_INVALID));
    }
    Ok(StickerStaticInputs {
        config_path: effect.config_rel_path.clone(),
        texture_path,
        model_path,
        mesh,
        mask_type,
        mask_channel: sticker_mask_channel(mask_type)?,
        erode_interval: as_f64(raw.get("ErodeInterval")),
        face_3dmm_type: as_u32(raw.get("Face3DMMType")),
        filter_type,
        opacity: effect
            .opacity
            .ok_or(Refusal::MissingInput("3DMMFaceSticker Opacity"))?,
        default_degree: effect
            .default_degree
            .ok_or(Refusal::MissingInput("3DMMFaceSticker DefaultDegree"))?,
        degree_range: effect
            .degree_range
            .ok_or(Refusal::MissingInput("3DMMFaceSticker DegreeRange"))?,
    })
}

/// Explicit placement evidence the sticker shader requires. No decoded
/// `effect.json` value populates any of these fields; the type exists so a
/// future evidence-backed caller cannot omit them silently.
#[derive(Clone, Debug, PartialEq)]
pub struct StickerTransform {
    pub elements: Vec<Element>,
    pub global_scale: f32,
    pub fl: f32,
    pub fk: f32,
    /// `true` keeps `GetMaskWeight()`; `false` applies the shader's
    /// `#ifndef FLIPMASK` default `1 - p`.
    pub flip_mask: bool,
}

/// Applies the recovered sticker law only when the caller supplies an
/// evidence-backed transform. The static preset alone always refuses with
/// [`PF115_3DMM_TRANSFORM`]. `u_alpha` is not an input of the ported kernel, so
/// a preset `Opacity` other than 100 is refused rather than ignored.
pub fn apply_vendor_sticker(
    effect: &VendorEffect,
    obj_text: &str,
    source: &Image,
    texture: &Image,
    mask: &Image,
    transform: Option<StickerTransform>,
) -> Result<Image, Refusal> {
    let static_inputs = resolve_3dmm_sticker(effect, obj_text)?;
    if !static_inputs.opacity.is_finite() || (static_inputs.opacity - 100.0).abs() > 1e-9 {
        return Err(Refusal::MissingInput(STICKER_OPACITY_UNSUPPORTED));
    }
    let transform = transform.ok_or(Refusal::MissingInput(PF115_3DMM_TRANSFORM))?;
    if transform.elements.is_empty() {
        return Err(Refusal::MissingInput(PF115_3DMM_TRANSFORM));
    }
    sticker::apply(
        source,
        texture,
        mask,
        &transform.elements,
        StickerParams {
            global_scale: transform.global_scale,
            fl: transform.fl,
            fk: transform.fk,
            filter_type: static_inputs.filter_type,
            flip_mask: transform.flip_mask,
            blend_alpha: false,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portrait_vendor_config::parse_config;

    const LIP_COLOR_UNIFORMS: LipColorUniforms = LipColorUniforms {
        factor_scale: 1.0,
        factor_shift: 0.0,
        factor_shrink: 1.0,
        factor_threshold: 1.0,
        metallight_alpha: 0.0,
        enable_hsl_adjust: false,
    };

    const SATIN_ROW: &str = r#"{"Effects":[{
        "EffectName":"LipColor","LipColorMode":1,"MultiColorSwitch":false,
        "TeethWhitenSwitch":false,"TeethWhitenDegree":0,"MakeupDegreeFactor":1.0,
        "MixColor0":[179,67,74,255],"MixColorFlag":30040,"Opacity0":100,
        "AdaptEnv":false,"RemoveHighlight":false,"RadiusScaleFactor":1.0,
        "EnableColorFormat":true,"ParamFlag":107,"DefaultDegree":50,
        "DegreeRange":[0,100]
    }]}"#;

    const MATTE_ROW: &str = r#"{"Effects":[{
        "EffectName":"LipColor","LipColorMode":0,"MultiColorSwitch":false,
        "TeethWhitenSwitch":false,"TeethWhitenDegree":0.0,"MakeupDegreeFactor":1.0,
        "MixColor0":[200,66,74,255],"MixColorFlag":30040,"Opacity0":100,
        "AdaptEnv":false,"RemoveHighlight":true,"RadiusScaleFactor":1.0,
        "HighlightParam":{"Weight":0.95,"Threshold":0.1},
        "EnableColorFormat":true,"ParamFlag":107,"DefaultDegree":50,
        "DegreeRange":[0,65]
    }]}"#;

    const STICKER_ROW: &str = r#"{"Effects":[{
        "EffectName":"3DMMFaceSticker","FilterType":41,"Paths":["01.png"],
        "ModelPath":"bfm_head_meanshape_reduce_with_uv_smooth_nose.obj",
        "MixColor":[0,0,0,255],"Opacity":100,"MaskType":15,
        "ErodeInterval":3.0999999046325684,"EnableColorFormat":true,
        "Face3DMMType":2,"ParamFlag":115,"DefaultDegree":70,
        "DegreeRange":[0,100]
    }]}"#;

    fn effects(json: &str) -> Vec<VendorEffect> {
        parse_config("x/effect.json", json).unwrap()
    }

    fn corpus_obj() -> String {
        let mut obj = String::with_capacity(2_000_000);
        for index in 0..STICKER_OBJ_VERTICES {
            let value = index as f32 / 1000.0;
            obj.push_str(&format!("v {value} {value} {value}\nvt {value} {value}\n"));
        }
        for face in 0..STICKER_OBJ_FACES {
            let a = face % STICKER_OBJ_VERTICES + 1;
            let b = (face + 1) % STICKER_OBJ_VERTICES + 1;
            let c = (face + 2) % STICKER_OBJ_VERTICES + 1;
            obj.push_str(&format!("f {a}/{a} {b}/{b} {c}/{c}\n"));
        }
        obj
    }

    #[test]
    fn satin_row_decodes_without_a_filter_type() {
        let rows = effects(SATIN_ROW);
        let row = lip_color_row_from_vendor(&rows[0]).unwrap();
        assert_eq!(row.lip_color_mode, 1);
        assert_eq!(row.mix_color0, [179, 67, 74, 255]);
        assert_eq!(row.opacity0_percent, 100.0);
        assert!(row.highlight_param.is_none());
        assert!(row.light_alpha_percent.is_none());
        assert!(rows[0].filter_type.is_none());
        let params = LipColorParams::from_vendor(&row, LIP_COLOR_UNIFORMS).unwrap();
        assert_eq!(params.mode, aurora_portrait::kernels::lip_color::LipColorMode::Satin);
    }

    #[test]
    fn highlight_param_row_refuses_by_name() {
        let rows = effects(MATTE_ROW);
        let row = lip_color_row_from_vendor(&rows[0]).unwrap();
        assert_eq!(
            row.highlight_param,
            Some(aurora_portrait::kernels::lip_color::HighlightParam {
                weight: 0.95,
                threshold: 0.1,
            })
        );
        assert_eq!(
            LipColorParams::from_vendor(&row, LIP_COLOR_UNIFORMS).err(),
            Some(Refusal::MissingInput(
                aurora_portrait::kernels::lip_color::LIP_COLOR_HIGHLIGHT_OPEN
            ))
        );
    }

    #[test]
    fn lip_color_missing_fields_are_named() {
        let rows = effects(r#"{"Effects":[{
            "EffectName":"LipColor","LipColorMode":1,"MakeupDegreeFactor":1.0,
            "Opacity0":100,"ParamFlag":107,"DefaultDegree":50,"DegreeRange":[0,100]
        }]}"#);
        assert_eq!(
            lip_color_row_from_vendor(&rows[0]).err(),
            Some(Refusal::MissingInput("MixColor0"))
        );
        let wrong = effects(r#"{"Effects":[{
            "EffectName":"LipColor","LipColorMode":1,"MixColor0":[1,2,3,255],
            "MakeupDegreeFactor":1.0,"Opacity0":100,"ParamFlag":107,
            "DefaultDegree":50,"DegreeRange":[0,100]
        }]}"#);
        let mut effect = wrong[0].clone();
        effect.param_flag = Some(106);
        assert_eq!(
            lip_color_row_from_vendor(&effect).err(),
            Some(Refusal::ValueOutOfRange(LIP_COLOR_PARAM_FLAG))
        );
    }

    #[test]
    fn lip_color_apply_refuses_without_uniforms_or_uv() {
        let rows = effects(SATIN_ROW);
        let image = Image::solid(1, 1, [100, 110, 120, 130]);
        assert_eq!(
            apply_vendor_lip_color(
                &rows[0], &image, &image, &image, &image, &image, None, None
            ),
            Err(Refusal::MissingInput(LIP_COLOR_UNIFORMS_MISSING))
        );
        assert_eq!(
            apply_vendor_lip_color(
                &rows[0],
                &image,
                &image,
                &image,
                &image,
                &image,
                None,
                Some(LIP_COLOR_UNIFORMS)
            ),
            Err(Refusal::MissingInput(
                aurora_portrait::kernels::lip_color::LIP_COLOR_UV_MISSING
            ))
        );
    }

    #[test]
    fn lip_stick_image_row_is_not_a_lip_color_row() {
        let rows = effects(r#"{"Effects":[{
            "EffectName":"LipStickImageMakeup","FilterType":42,
            "Paths":["staintexture.png"],"LipMaskType":3,"ParamFlag":107,
            "DefaultDegree":50,"DegreeRange":[0,100]
        }]}"#);
        assert_eq!(
            lip_color_row_from_vendor(&rows[0]).err(),
            Some(Refusal::ValueOutOfRange(LIP_COLOR_EFFECT))
        );
    }

    #[test]
    fn sticker_static_inputs_parse_the_vendor_obj() {
        let rows = effects(STICKER_ROW);
        let inputs = resolve_3dmm_sticker(&rows[0], &corpus_obj()).unwrap();
        assert_eq!(inputs.texture_path, "01.png");
        assert_eq!(inputs.model_path, STICKER_VENDOR_OBJ);
        assert_eq!(inputs.mask_type, 15);
        assert_eq!(inputs.mask_channel, MaskChannel::Rgba);
        assert_eq!(inputs.filter_type, 41);
        assert_eq!(inputs.opacity, 100.0);
        assert_eq!(inputs.default_degree, 70.0);
        assert_eq!(inputs.face_3dmm_type, Some(2));
        assert!(inputs.mesh.is_vendor_corpus_shape());
        assert_eq!(inputs.mesh.vertices.len(), STICKER_OBJ_VERTICES);
        assert_eq!(inputs.mesh.indices.len(), STICKER_OBJ_FACES * 3);
        assert_eq!(inputs.mesh.vertices[0].uv, [0.0, 0.0]);
    }

    #[test]
    fn sticker_apply_refuses_without_a_transform() {
        let rows = effects(STICKER_ROW);
        let image = Image::solid(1, 1, [10, 20, 30, 255]);
        assert_eq!(
            apply_vendor_sticker(&rows[0], &corpus_obj(), &image, &image, &image, None),
            Err(Refusal::MissingInput(PF115_3DMM_TRANSFORM))
        );
    }

    #[test]
    fn sticker_apply_with_a_supplied_transform_matches_the_kernel() {
        let rows = effects(STICKER_ROW);
        let source = Image::solid(1, 1, [10, 20, 30, 255]);
        let texture = Image::solid(1, 1, [200, 100, 50, 255]);
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let transform = StickerTransform {
            elements: vec![Element {
                pos: [0.5, 0.5],
                scale: 1.0,
                rot: 0.0,
            }],
            global_scale: 1.0,
            fl: 1.0,
            fk: 1.0,
            flip_mask: true,
        };
        let out = apply_vendor_sticker(
            &rows[0],
            &corpus_obj(),
            &source,
            &texture,
            &mask,
            Some(transform),
        )
        .unwrap();
        // FilterType 41 is multiply: base * material, alpha copied.
        assert_eq!(out.pixel(0, 0), [8, 8, 6, 255]);
    }

    #[test]
    fn sticker_without_a_flip_flag_keeps_the_mask() {
        let rows = effects(STICKER_ROW);
        let source = Image::solid(1, 1, [10, 20, 30, 255]);
        let texture = Image::solid(1, 1, [200, 100, 50, 255]);
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let transform = StickerTransform {
            elements: vec![Element {
                pos: [0.5, 0.5],
                scale: 1.0,
                rot: 0.0,
            }],
            global_scale: 1.0,
            fl: 1.0,
            fk: 1.0,
            flip_mask: false,
        };
        let out = apply_vendor_sticker(
            &rows[0],
            &corpus_obj(),
            &source,
            &texture,
            &mask,
            Some(transform),
        )
        .unwrap();
        assert_eq!(out, source);
    }

    #[test]
    fn sticker_obj_shape_and_index_errors_are_named() {
        // Same model path but not the corpus shape.
        let rows = effects(STICKER_ROW);
        let small = "v 0 0 0\nvt 0 0\nv 1 0 0\nvt 1 0\nv 0 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";
        assert_eq!(
            resolve_3dmm_sticker(&rows[0], small).err(),
            Some(Refusal::ShapeMismatch(STICKER_OBJ_INVALID))
        );
        // A different model path skips the corpus-shape gate, and an
        // out-of-range face index is still refused.
        let custom = effects(r#"{"Effects":[{
            "EffectName":"3DMMFaceSticker","FilterType":41,"Paths":["01.png"],
            "ModelPath":"custom.obj","Opacity":100,"MaskType":15,"ParamFlag":115,
            "DefaultDegree":70,"DegreeRange":[0,100]
        }]}"#);
        assert_eq!(
            resolve_3dmm_sticker(&custom[0], "v 0 0 0\nvt 0 0\nf 1/1 2/2 3/3\n").err(),
            Some(Refusal::ShapeMismatch(STICKER_OBJ_INVALID))
        );
    }

    #[test]
    fn sticker_opacity_other_than_100_is_refused() {
        let rows = effects(r#"{"Effects":[{
            "EffectName":"3DMMFaceSticker","FilterType":41,"Paths":["01.png"],
            "ModelPath":"bfm_head_meanshape_reduce_with_uv_smooth_nose.obj",
            "Opacity":50,"MaskType":15,"ParamFlag":115,"DefaultDegree":70,
            "DegreeRange":[0,100]
        }]}"#);        let image = Image::solid(1, 1, [10, 20, 30, 255]);
        let transform = StickerTransform {
            elements: vec![Element {
                pos: [0.5, 0.5],
                scale: 1.0,
                rot: 0.0,
            }],
            global_scale: 1.0,
            fl: 1.0,
            fk: 1.0,
            flip_mask: true,
        };
        assert_eq!(
            apply_vendor_sticker(
                &rows[0],
                &corpus_obj(),
                &image,
                &image,
                &image,
                Some(transform)
            ),
            Err(Refusal::MissingInput(STICKER_OPACITY_UNSUPPORTED))
        );
    }

    #[test]
    fn blusher_row_is_not_a_sticker_row() {
        let rows = effects(r#"{"Effects":[{
            "EffectName":"BlusherImageMakeup","FilterType":41,"Paths":["z01.png"],
            "BlusherMaskType":15,"MeshType":2,"ROIRect":[95,40,901,1081],
            "ParamFlag":115,"DefaultDegree":70,"DegreeRange":[0,100]
        }]}"#);
        assert_eq!(
            resolve_3dmm_sticker(&rows[0], &corpus_obj()).err(),
            Some(Refusal::ValueOutOfRange(FACE_STICKER))
        );
    }

    #[test]
    fn unsupported_sticker_mask_type_is_named() {
        assert_eq!(
            sticker_mask_channel(15),
            Ok(MaskChannel::Rgba)
        );
        assert_eq!(
            sticker_mask_channel(3),
            Err(Refusal::MissingInput(STICKER_MASK_UNSUPPORTED))
        );
    }
}
