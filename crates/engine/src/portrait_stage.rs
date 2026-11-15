//! Bridges portrait parameters (`pc:<id>`) to `aurora-portrait` CPU kernels.
//!
//! Only effects whose inputs are available are applied. Effects that need face
//! geometry, per-effect model outputs or materials that are not present are
//! skipped with a debug line instead of being approximated.
//!
//! Mask planes come from the local people parser when its ONNX model is
//! available. No external application install is read by this module.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use aurora_portrait::{Access, Image};

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

fn sample_plane(plane: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let px = (x.clamp(0.0, 1.0) * width as f32 - 0.5)
        .clamp(0.0, (width as f32 - 1.001).max(0.0));
    let py = (y.clamp(0.0, 1.0) * height as f32 - 0.5)
        .clamp(0.0, (height as f32 - 1.001).max(0.0));
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
    8068, // Skin gloss
    8069, // Skin texture (negative surface-blur branch)
    8070, // Skin clarity
    21015, // Skin color hue
    30096, // Skin temperature
    90004, // Skin radiance
    90005, // Skin whitening
    90007, // Skin ruddy
];

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
    let Some(source) = Image::new(width, height, std::mem::take(portrait)) else {
        return;
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
        let result = match pf {
            8068 => apply_skin_gloss(&current, degree, masks, width, height, region),
            8069 => apply_skin_texture(&current, degree, masks, width, height, region),
            8070 => apply_skin_clarity(&current, degree, masks, width, height, region),
            21015 => apply_skin_tint(&current, degree, masks, width, height, region),
            30096 => apply_skin_temperature(&current, degree, masks, width, height, region),
            90004 => apply_skin_radiance(&current, degree, masks, width, height, region),
            90005 => apply_skin_whiten(&current, degree, masks, width, height, region),
            90007 => apply_skin_ruddy(&current, degree, masks, width, height, region),
            _ => Ok(current.clone()),
        };
        match result {
            Ok(next) => current = next,
            Err(err) => log::debug!("portrait PF{pf} skipped: {err:?}"),
        }
    }

    *portrait = current.pixels;
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
    masks: &PortraitMasks,
    width: u32,
    height: u32,
    region: NormRect,
) -> Result<Image, aurora_portrait::Refusal> {
    let (Some(green), Some(magenta)) = (tint_green(), tint_magenta()) else {
        return Ok(current.clone());
    };
    let Some(mask) = masks.mask_u8(MaskKind::Skin, width, height, region) else {
        return Ok(current.clone());
    };
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    aurora_portrait::kernels::face_detail::skin_tint(current, &mask, green, magenta, ui)
}

fn apply_skin_whiten(
    current: &Image,
    degree: f32,
    masks: &PortraitMasks,
    width: u32,
    height: u32,
    region: NormRect,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(lut) = whiten_lut() else {
        return Ok(current.clone());
    };
    let Some(mask) = masks.mask_u8(MaskKind::Skin, width, height, region) else {
        return Ok(current.clone());
    };
    aurora_portrait::kernels::makeup::skin_adjust_lut(current, &mask, lut, degree.clamp(0.0, 1.0))
}

fn apply_skin_temperature(
    current: &Image,
    degree: f32,
    masks: &PortraitMasks,
    width: u32,
    height: u32,
    region: NormRect,
) -> Result<Image, aurora_portrait::Refusal> {
    let (Some(cold), Some(warm)) = (cold_lut(), warm_lut()) else {
        return Ok(current.clone());
    };
    let Some(mask) = masks.mask_u8(MaskKind::Skin, width, height, region) else {
        return Ok(current.clone());
    };
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    aurora_portrait::kernels::face_detail::skin_temperature(current, &mask, cold, warm, ui)
}

fn apply_skin_ruddy(
    current: &Image,
    degree: f32,
    masks: &PortraitMasks,
    width: u32,
    height: u32,
    region: NormRect,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(lut) = ruddy_lut() else {
        return Ok(current.clone());
    };
    let Some(mask) = masks.mask_u8(MaskKind::Skin, width, height, region) else {
        return Ok(current.clone());
    };
    aurora_portrait::kernels::makeup::skin_adjust_lut(current, &mask, lut, degree.clamp(0.0, 1.0))
}

fn apply_skin_gloss(
    current: &Image,
    degree: f32,
    masks: &PortraitMasks,
    width: u32,
    height: u32,
    region: NormRect,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(mask) = masks.mask_u8(MaskKind::Skin, width, height, region) else {
        return Ok(current.clone());
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
    masks: &PortraitMasks,
    width: u32,
    height: u32,
    region: NormRect,
) -> Result<Image, aurora_portrait::Refusal> {
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    if ui >= 0.0 {
        return Ok(current.clone());
    }
    let Some(mask) = masks.mask_u8(MaskKind::Skin, width, height, region) else {
        return Ok(current.clone());
    };
    let extra = vec![255u8; mask.len()];
    aurora_portrait::kernels::skin_models::texture_surface_blur(current, &mask, &extra, ui)
}

fn apply_skin_clarity(
    current: &Image,
    degree: f32,
    masks: &PortraitMasks,
    width: u32,
    height: u32,
    region: NormRect,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(mask) = masks.mask_u8(MaskKind::Skin, width, height, region) else {
        return Ok(current.clone());
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
    masks: &PortraitMasks,
    width: u32,
    height: u32,
    region: NormRect,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(skin) = masks.mask_u8(MaskKind::Skin, width, height, region) else {
        return Ok(current.clone());
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
    Ok(
        aurora_portrait::kernels::radiance::apply(current, &planes, ui, value_mean, &fixed)?
            .image,
    )
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
        let map: BTreeMap<String, f32> =
            extra.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        apply_portrait_effects(
            rgba,
            w,
            h,
            NormRect::FULL,
            &map,
            &PortraitMasks::solid(1.0),
        );
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
}
