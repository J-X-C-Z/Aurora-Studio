// PF8069 positive branch (`FaceSticker`, FilterType 45).
//
// Evidence: r8 p07 `PE-EVIDENCE.json` + `Pf8069TextureDegreeProducer.cs`.
// The positive render entry is `0x4C83CF0`; it stores the degree at
// `filter+0x328` (`0x4C841DE`), sets the dirty flag (`0x4C84200`) and draws
// through `[r10+0xD0]` (`0x4C84233`). The final blend is the recovered type10
// consumer. The texture.png/FaceMask.png projection and face transform are
// NOT statically recovered, so this module never synthesizes a material layer:
// it validates the binding and refuses by name.

use crate::{Image, Refusal};

pub const PF: u32 = 8069;
pub const FILTER_TYPE: u32 = 45;
pub const MATERIAL_PATHS: [&str; 2] = ["texture.png", "FaceMask.png"];
/// Config `ROIRect` for the positive branch.
pub const ROI_RECT: [u32; 4] = [95, 40, 901, 1081];
/// Config `MaterialGrayData`: texture.png carries gray material data.
pub const MATERIAL_GRAY_DATA: bool = true;

pub const POSITIVE_RENDER_RVA: u32 = 0x4c83cf0;
pub const DEGREE_STORE_RVA: u32 = 0x4c841de;
pub const DIRTY_FLAG_RVA: u32 = 0x4c84200;
pub const DRAW_RVA: u32 = 0x4c84233;
pub const CONFIG_SHA256: &str = "5D2063217AB83FC00E1D1557FEF2D17A5470487B9738AE4A73033B57C364966E";
pub const TYPE10_FRAGMENT_SHA256: &str =
    "934C3ECE5979EFDE94D3FA4F84D881344BF14F4DA26EE5AEA4DCE86036429EB6";

const BLEND_GATE: f32 = 0.01;

fn check_image(image: &Image, what: &'static str) -> Result<usize, Refusal> {
    image.validate().map_err(|_| Refusal::ShapeMismatch(what))?;
    Ok(image.pixel_count())
}

/// `EnableDegreeRemap=true` with `ParamRange [0,0.5,1]` and
/// `EffectRange [0,0,1]`: `fe=clamp(0.5+0.005*ui,0,1)`, then
/// `0` for `fe<=0.5` and `2*fe-1` above. Mathematically `0.01*ui`, but the
/// f32 rounding differs on 135 of 201 half-step samples, so the vendor path is
/// kept literally.
pub fn face_sticker_degree(ui: f32) -> Result<f32, Refusal> {
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let fe = (0.5 + 0.005 * ui).clamp(0.0, 1.0);
    Ok(if fe <= 0.5 { 0.0 } else { 2.0 * fe - 1.0 })
}

/// PF8069 positive entry: derived degree from the UI value.
pub fn face_sticker_apply(
    source: &Image,
    material_layer: &Image,
    mask_r: &[u8],
    extra_mask_r: &[u8],
    ui: f32,
) -> Result<Image, Refusal> {
    let degree = face_sticker_degree(ui)?;
    face_sticker_compose(source, material_layer, mask_r, extra_mask_r, degree)
}

fn to_byte(value: f32) -> u8 {
    (value * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Exact PF8069 FaceSticker type10 consumer.
///
/// `material_layer` is the already projected/face-transformed RGBA8 layer; its
/// alpha is the material coverage. `mask_r`/`extra_mask_r` are the same-job
/// R8 masks. `degree` is `max(0.01*ui, 0)`.
pub fn face_sticker_compose(
    source: &Image,
    material_layer: &Image,
    mask_r: &[u8],
    extra_mask_r: &[u8],
    degree: f32,
) -> Result<Image, Refusal> {
    let pixels = check_image(source, "source")?;
    check_image(material_layer, "material_layer")?;
    if material_layer.width != source.width || material_layer.height != source.height {
        return Err(Refusal::ShapeMismatch("material_layer"));
    }
    if mask_r.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    if extra_mask_r.len() != pixels {
        return Err(Refusal::ShapeMismatch("extra_mask_r"));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }

    let mut out = source.clone();
    for (index, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        let offset = index * 4;
        let mix = (material_layer.pixels[offset + 3] as f32 / 255.0)
            * degree
            * (mask_r[index] as f32 / 255.0)
            * (extra_mask_r[index] as f32 / 255.0);
        if mix <= BLEND_GATE {
            continue;
        }
        for channel in 0..3 {
            let base = source.pixels[offset + channel] as f32 / 255.0;
            let layer = material_layer.pixels[offset + channel] as f32 / 255.0;
            px[channel] = to_byte(base + (layer - base) * mix);
        }
        px[3] = source.pixels[offset + 3];
    }
    Ok(out)
}

/// Protected binder for the OPEN texture.png/FaceMask.png projection.
///
/// Validates the real inputs, then refuses `face_transform`: the vendor's
/// FilterType45 material transform is not statically closed, so no UV or
/// landmark-to-material mapping is guessed. Callers that do hold a validated
/// projection feed its output to `face_sticker_compose`.
pub fn project_sticker_material(
    texture: &Image,
    face_mask: &Image,
    landmarks: &[[f32; 2]],
    roi: [u32; 4],
) -> Result<Image, Refusal> {
    check_image(texture, "texture")?;
    check_image(face_mask, "face_mask")?;
    let [_, _, width, height] = roi;
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    if landmarks.is_empty() {
        return Err(Refusal::MissingInput("face_landmarks"));
    }
    if landmarks.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("face_landmarks"));
    }
    Err(Refusal::MissingInput("face_transform"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Image {
        Image::solid(1, 1, [100, 100, 100, 200])
    }

    fn layer(rgba: [u8; 4]) -> Image {
        Image::solid(1, 1, rgba)
    }

    #[test]
    fn degree_remap_matches_the_vendor_f32_path() {
        assert_eq!(face_sticker_degree(0.0).unwrap(), 0.0);
        assert_eq!(face_sticker_degree(50.0).unwrap(), 0.5);
        assert_eq!(face_sticker_degree(100.0).unwrap(), 1.0);
        assert_eq!(face_sticker_degree(75.0).unwrap(), 0.75);
        assert_ne!(face_sticker_degree(0.5).unwrap(), 0.01 * 0.5);
        assert_eq!(face_sticker_degree(-1.0), Err(Refusal::ValueOutOfRange("ui")));
    }

    #[test]
    fn positive_full_degree_replaces_rgb_and_keeps_base_alpha() {
        let out = face_sticker_compose(
            &base(),
            &layer([200, 50, 20, 255]),
            &[255],
            &[255],
            1.0,
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [200, 50, 20, 200]);
    }

    #[test]
    fn positive_half_degree_and_masks_interpolate() {
        let out = face_sticker_compose(
            &base(),
            &layer([200, 50, 20, 255]),
            &[255],
            &[255],
            0.5,
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [150, 75, 60, 200]);

        let out = face_sticker_compose(
            &base(),
            &layer([200, 50, 20, 255]),
            &[128],
            &[255],
            1.0,
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0)[0], 150);
    }

    #[test]
    fn positive_gate_and_zero_masks_are_identity() {
        // layer alpha 2/255 * degree 1 = 0.0078 <= 0.01
        let gated = face_sticker_compose(&base(), &layer([200, 50, 20, 2]), &[255], &[255], 1.0)
            .unwrap();
        assert_eq!(gated, base());

        let zero = face_sticker_compose(&base(), &layer([200, 50, 20, 255]), &[0], &[255], 1.0)
            .unwrap();
        assert_eq!(zero, base());

        let zero_extra =
            face_sticker_compose(&base(), &layer([200, 50, 20, 255]), &[255], &[0], 1.0).unwrap();
        assert_eq!(zero_extra, base());
    }

    #[test]
    fn positive_refuses_bad_shapes_and_degree() {
        assert_eq!(
            face_sticker_compose(&base(), &layer([200, 50, 20, 255]), &[], &[255], 1.0),
            Err(Refusal::ShapeMismatch("mask_r"))
        );
        assert_eq!(
            face_sticker_compose(&base(), &layer([200, 50, 20, 255]), &[255], &[255], f32::NAN),
            Err(Refusal::ValueOutOfRange("degree"))
        );
        assert_eq!(
            face_sticker_compose(&base(), &layer([200, 50, 20, 255]), &[255], &[255], 1.5),
            Err(Refusal::ValueOutOfRange("degree"))
        );
        let wide = Image::solid(2, 1, [0, 0, 0, 255]);
        assert_eq!(
            face_sticker_compose(&base(), &wide, &[255], &[255], 1.0),
            Err(Refusal::ShapeMismatch("material_layer"))
        );
        let packed = Image { width: 1, height: 1, pixels: vec![0; 3] };
        assert_eq!(
            face_sticker_compose(&packed, &layer([0, 0, 0, 255]), &[255], &[255], 1.0),
            Err(Refusal::ShapeMismatch("source"))
        );
    }

    #[test]
    fn projection_binder_refuses_with_named_missing_transform() {
        let texture = Image::solid(4, 4, [128, 128, 128, 255]);
        let face_mask = Image::solid(4, 4, [255, 0, 0, 255]);
        let landmarks = [[0.5f32, 0.5]];
        assert_eq!(
            project_sticker_material(&texture, &face_mask, &landmarks, ROI_RECT),
            Err(Refusal::MissingInput("face_transform"))
        );
        assert_eq!(
            project_sticker_material(&texture, &face_mask, &[], ROI_RECT),
            Err(Refusal::MissingInput("face_landmarks"))
        );
        assert_eq!(
            project_sticker_material(&texture, &face_mask, &[[f32::NAN, 0.0]], ROI_RECT),
            Err(Refusal::ValueOutOfRange("face_landmarks"))
        );
        assert_eq!(
            project_sticker_material(&texture, &face_mask, &landmarks, [0, 0, 0, 10]),
            Err(Refusal::ValueOutOfRange("roi"))
        );
        let packed = Image { width: 4, height: 4, pixels: vec![0; 8] };
        assert_eq!(
            project_sticker_material(&packed, &face_mask, &landmarks, ROI_RECT),
            Err(Refusal::ShapeMismatch("texture"))
        );
    }
}
