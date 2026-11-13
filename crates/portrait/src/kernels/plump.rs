// FacePlump rows PF30191 (forehead) and PF30194 (mid-face).
//
// Proven consumer (render 0x5E0FC00):
//   degree   = PF map["midface_degree"] / forehead PF30191 map value
//   smooth   = mapSmooth * degree      @ 0x5E11981 (mulss xmm0, xmm8)
//   brighten = degree * mapBrighten    @ 0x5E12361 (mulss xmm8, xmm0)
//
// `xmm8` is consumed only by those two multiplies; no extra clamp, mask or
// alpha rule is proven before the model request. The request builder
// 0x5E1AB30 -> FacialPlumpingPipeline 0x5B78280 -> final pixel blend is OPEN.
// This module therefore executes the scalar consumer and refuses the pixel
// blend by name instead of inventing one.

use crate::{Image, Refusal};

pub const PF_FOREHEAD: u32 = 30191;
pub const PF_MIDFACE: u32 = 30194;
pub const RENDER_RVA: u32 = 0x5E0FC00;
pub const PIPELINE_CTOR_RVA: u32 = 0x5E1AB30;
pub const PIPELINE_RUNNER_RVA: u32 = 0x5B78280;
pub const SMOOTH_MUL_RVA: u32 = 0x5E11981;
pub const BRIGHTEN_MUL_RVA: u32 = 0x5E12361;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlumpRow {
    Forehead,
    MidFace,
}

impl PlumpRow {
    pub fn pf(self) -> u32 {
        match self {
            PlumpRow::Forehead => PF_FOREHEAD,
            PlumpRow::MidFace => PF_MIDFACE,
        }
    }
}

/// Same-job native maps. Values are taken as delivered; no remap is guessed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Maps {
    pub degree: f32,
    pub smooth: f32,
    pub brighten: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Effective {
    pub degree: f32,
    pub smooth: f32,
    pub brighten: f32,
    pub active: bool,
}

fn finite(value: f32, what: &'static str) -> Result<f32, Refusal> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Refusal::ValueOutOfRange(what))
    }
}

/// Exact consumer order: `smooth * degree`, then `degree * brighten`.
pub fn effective(_row: PlumpRow, maps: Maps) -> Result<Effective, Refusal> {
    let degree = finite(maps.degree, "degree")?;
    let smooth = finite(maps.smooth, "smooth")?;
    let brighten = finite(maps.brighten, "brighten")?;
    let smooth_effective = smooth * degree;
    let brighten_effective = degree * brighten;
    Ok(Effective {
        degree,
        smooth: if degree == 0.0 { 0.0 } else { smooth_effective },
        brighten: if degree == 0.0 { 0.0 } else { brighten_effective },
        active: degree != 0.0,
    })
}

fn check_image(image: &Image, what: &'static str) -> Result<usize, Refusal> {
    if image.pixels.len() != image.width as usize * image.height as usize * 4 {
        return Err(Refusal::ShapeMismatch(what));
    }
    Ok(image.pixel_count())
}

/// FacePlump pixel entry.
///
/// Degree 0 is the proven identity gate. Any active degree validates the
/// explicit ROI, mask and already-rendered plump layer, then refuses
/// `facial_plumping_pixel_blend`: the pipeline's final blend/alpha/colour
/// space has no static evidence.
pub fn apply_pixels(
    row: PlumpRow,
    source: &Image,
    mask_r: &[u8],
    roi: (u32, u32, u32, u32),
    maps: Maps,
    plump_layer: Option<&Image>,
) -> Result<Image, Refusal> {
    check_image(source, "source")?;
    let bound = effective(row, maps)?;
    if !bound.active {
        // Alpha and every byte stay untouched on the disabled path.
        return Ok(source.clone());
    }

    let (x, y, width, height) = roi;
    if width == 0 || height == 0 || x + width > source.width || y + height > source.height {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    if mask_r.len() != width as usize * height as usize {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    let layer = plump_layer.ok_or(Refusal::MissingInput("facial_plumping_layer"))?;
    check_image(layer, "facial_plumping_layer")?;
    if layer.width != source.width || layer.height != source.height {
        return Err(Refusal::ShapeMismatch("facial_plumping_layer"));
    }
    Err(Refusal::MissingInput("facial_plumping_pixel_blend"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps() -> Maps {
        Maps { degree: 0.5, smooth: 0.8, brighten: 0.6 }
    }

    #[test]
    fn effective_is_a_real_non_identity_product() {
        for row in [PlumpRow::Forehead, PlumpRow::MidFace] {
            let bound = effective(row, maps()).unwrap();
            assert!(bound.active);
            assert!((bound.smooth - 0.4).abs() < 1e-6);
            assert!((bound.brighten - 0.3).abs() < 1e-6);
            assert_ne!(bound.smooth, 1.0);
            assert_ne!(bound.brighten, 0.0);
        }
    }

    #[test]
    fn zero_degree_is_the_identity_gate() {
        let bound = effective(
            PlumpRow::Forehead,
            Maps { degree: 0.0, smooth: 0.8, brighten: 0.6 },
        )
        .unwrap();
        assert!(!bound.active);
        assert_eq!((bound.smooth, bound.brighten), (0.0, 0.0));
    }

    #[test]
    fn non_finite_maps_are_refused() {
        let mut bad = maps();
        bad.degree = f32::NAN;
        assert_eq!(
            effective(PlumpRow::MidFace, bad),
            Err(Refusal::ValueOutOfRange("degree"))
        );
        let mut bad = maps();
        bad.smooth = f32::INFINITY;
        assert_eq!(
            effective(PlumpRow::MidFace, bad),
            Err(Refusal::ValueOutOfRange("smooth"))
        );
    }

    #[test]
    fn disabled_path_preserves_alpha_and_bytes() {
        let src = Image::solid(2, 2, [20, 40, 60, 77]);
        let out = apply_pixels(
            PlumpRow::Forehead,
            &src,
            &[],
            (0, 0, 0, 0),
            Maps { degree: 0.0, smooth: 0.8, brighten: 0.6 },
            None,
        )
        .unwrap();
        assert_eq!(out, src);
        assert_eq!(out.pixel(1, 1)[3], 77);
    }

    #[test]
    fn active_path_validates_inputs_then_refuses_the_open_blend() {
        let src = Image::solid(2, 2, [20, 40, 60, 77]);
        let layer = Image::solid(2, 2, [30, 50, 70, 255]);
        let mask = [255u8; 4];
        assert_eq!(
            apply_pixels(PlumpRow::MidFace, &src, &mask, (0, 0, 2, 2), maps(), None),
            Err(Refusal::MissingInput("facial_plumping_layer"))
        );
        assert_eq!(
            apply_pixels(PlumpRow::MidFace, &src, &mask, (0, 0, 2, 2), maps(), Some(&layer)),
            Err(Refusal::MissingInput("facial_plumping_pixel_blend"))
        );
        assert_eq!(
            apply_pixels(PlumpRow::MidFace, &src, &[], (0, 0, 2, 2), maps(), Some(&layer)),
            Err(Refusal::ShapeMismatch("mask_r"))
        );
        assert_eq!(
            apply_pixels(PlumpRow::MidFace, &src, &mask, (0, 0, 3, 3), maps(), Some(&layer)),
            Err(Refusal::ValueOutOfRange("roi"))
        );
        let packed = Image { width: 2, height: 2, pixels: vec![0; 5] };
        assert_eq!(
            apply_pixels(PlumpRow::MidFace, &packed, &mask, (0, 0, 2, 2), maps(), Some(&layer)),
            Err(Refusal::ShapeMismatch("source"))
        );
    }
}
