//! Skin-color / LUT pixel backend for queue rows 26-34.
//!
//! Truth sources, read-only from `work/dispatch-r9-20261109`:
//! - `p13-rows26-29-vendor-parse/source/OilShinePf90080Core.cs`
//! - `p13-rows26-29-vendor-parse/source/Pf5DarkCircle.cs`
//! - `p13-rows26-29-vendor-parse/source/EyeBagPf90116Kernel.cs`
//! - `p13-rows26-29-vendor-parse/source/RemoveSweatCore.cs`
//! - `p13-rows26-29-vendor-parse/source/RowCompatibility.cs`
//! - `p14-rows30-34-vendor-parse/src/NostrilPf21104.cs`
//! - `p14-rows30-34-vendor-parse/src/LipLinePF90118.cs`
//! - `p14-rows30-34-vendor-parse/src/BeardPf21026Parser.cs`
//! - `o3b-apple-masked-whiten-integration/src/o3b_masked_lut_whiten.metal`
//!
//! Every entry validates packed image length, plane/mask/LUT shape and degree
//! range, and refuses a missing input. Nothing here synthesizes a model, a
//! mask, an ROI or a LUT: a caller must supply the same-job data.

use crate::{Access, EffectMeta, Image, Refusal};

/// Registry rows for priorities 26-34. `Cpu` is only used where a recovered
/// pixel consumer is complete and tested; open final stages stay `Refused`.
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 90080,
        key: "",
        label: "祛油光（脸部）",
        access: Access::Cpu,
        inputs: &["effect_rgb", "mask", "degree"],
    },
    EffectMeta {
        pf: 5,
        key: "",
        label: "祛黑眼圈",
        access: Access::Cpu,
        inputs: &[
            "processed residual RGB f32 (warped + masked)",
            "bias 80",
        ],
    },
    crate::kernels::eye_bag::ROW29,
    EffectMeta {
        pf: 21104,
        key: "",
        label: "祛鼻孔瑕疵",
        access: Access::Cpu,
        inputs: &[
            "M_plane",
            "A_plane",
            "V_plane",
            "T_plane",
            "source_rgb",
            "degree",
        ],
    },
    EffectMeta {
        pf: 90118,
        key: "",
        label: "唇纹修整",
        access: Access::Cpu,
        inputs: &[
            "source_bgra",
            "repair_bgra",
            "mask_a",
            "mask_b",
            "lip_points",
        ],
    },
    EffectMeta {
        pf: 30188,
        key: "removeDoubleChin_degree",
        label: "祛双下巴-程度",
        access: Access::Refused,
        inputs: &["double_chin_degree", "face_parsing_masks", "model_output", "roi"],
    },
    EffectMeta {
        pf: 30189,
        key: "removeDoubleChin_shadow",
        label: "祛双下巴-阴影",
        access: Access::Refused,
        inputs: &["double_chin_shadow", "face_parsing_masks", "model_output", "roi"],
    },
    EffectMeta {
        pf: 30209,
        key: "",
        label: "下颌线增强",
        access: Access::Refused,
        inputs: &["face_194_packet", "jawline_indices", "mask", "model_output"],
    },
];

/// Integer ROI in source pixels; `w == 0 || h == 0` means empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Roi {
    pub fn empty() -> Self {
        Self {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.w <= 0 || self.h <= 0
    }
}

fn pixels_of(image: &Image) -> Result<usize, Refusal> {
    image.validate()?;
    Ok(image.pixel_count())
}

fn require_len(name: &'static str, actual: usize, expected: usize) -> Result<(), Refusal> {
    if actual != expected {
        return Err(Refusal::ShapeMismatch(name));
    }
    Ok(())
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), Refusal> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(())
}

fn require_degree(degree: f32) -> Result<f32, Refusal> {
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    Ok(degree)
}

fn clamp_byte(value: f32) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value.round_ties_even() as u8
    }
}

/// Vendor rounding at `0x82AA2E0`: `+/-0.5` then truncation toward zero.
fn vendor_round_byte(value: f32) -> u8 {
    let adjusted = if value >= 0.0 {
        value + 0.5
    } else {
        value - 0.5
    };
    (adjusted as i32).clamp(0, 255) as u8
}

// ---------------------------------------------------------------------------
// Priority 26 - PF90080 face oil shine
// ---------------------------------------------------------------------------

pub const OIL_SHINE_PF: u32 = 90080;
pub const OIL_SHINE_MODEL_WIDTH: usize = 512;
pub const OIL_SHINE_MODEL_HEIGHT: usize = 640;
pub const OIL_SHINE_ROI_WIDTH: u32 = 1024;
pub const OIL_SHINE_ROI_HEIGHT: u32 = 1280;

pub struct OilShineModelInput {
    pub b: Vec<f32>,
    pub g: Vec<f32>,
    pub r: Vec<f32>,
}

/// `roi` must be the explicit native 1024x1280 face ROI. Each model sample is
/// the integer 2x2 mean of the ROI, normalized `(mean - 127.5) * (2 / 255)`.
pub fn oil_shine_model_input(roi: &Image) -> Result<OilShineModelInput, Refusal> {
    let pixels = pixels_of(roi)?;
    if roi.width != OIL_SHINE_ROI_WIDTH || roi.height != OIL_SHINE_ROI_HEIGHT {
        return Err(Refusal::ShapeMismatch("roi"));
    }
    let out_pixels = OIL_SHINE_MODEL_WIDTH * OIL_SHINE_MODEL_HEIGHT;
    let mut b = vec![0.0f32; out_pixels];
    let mut g = vec![0.0f32; out_pixels];
    let mut r = vec![0.0f32; out_pixels];

    const SCALE: f32 = 2.0 / 255.0;
    for oy in 0..OIL_SHINE_MODEL_HEIGHT {
        for ox in 0..OIL_SHINE_MODEL_WIDTH {
            let sx = ox * 2;
            let sy = oy * 2;
            let i00 = (sy * roi.width as usize + sx) * 4;
            let i01 = i00 + 4;
            let i10 = i00 + roi.width as usize * 4;
            let i11 = i10 + 4;
            let p = &roi.pixels;
            let mean_b = (p[i00] as u32 + p[i01] as u32 + p[i10] as u32 + p[i11] as u32) / 4;
            let mean_g =
                (p[i00 + 1] as u32 + p[i01 + 1] as u32 + p[i10 + 1] as u32 + p[i11 + 1] as u32) / 4;
            let mean_r =
                (p[i00 + 2] as u32 + p[i01 + 2] as u32 + p[i10 + 2] as u32 + p[i11 + 2] as u32) / 4;
            let o = oy * OIL_SHINE_MODEL_WIDTH + ox;
            b[o] = (mean_b as f32 - 127.5) * SCALE;
            g[o] = (mean_g as f32 - 127.5) * SCALE;
            r[o] = (mean_r as f32 - 127.5) * SCALE;
        }
    }
    debug_assert_eq!(
        pixels,
        OIL_SHINE_ROI_WIDTH as usize * OIL_SHINE_ROI_HEIGHT as usize
    );
    Ok(OilShineModelInput { b, g, r })
}

/// PF90080 output is `[2][640][512][4]`; the first six lane planes are the
/// model's `[6,640,512]` output and lanes 6/7 are padding.
pub fn oil_shine_decode_nc4hw4(raw: &[f32]) -> Result<Vec<Vec<f32>>, Refusal> {
    let plane_len = OIL_SHINE_MODEL_WIDTH * OIL_SHINE_MODEL_HEIGHT;
    let expected = 2 * plane_len * 4;
    require_len("nc4hw4", raw.len(), expected)?;
    let mut planes = vec![vec![0.0f32; plane_len]; 6];
    for block in 0..2 {
        for y in 0..OIL_SHINE_MODEL_HEIGHT {
            for x in 0..OIL_SHINE_MODEL_WIDTH {
                let src = ((block * OIL_SHINE_MODEL_HEIGHT + y) * OIL_SHINE_MODEL_WIDTH + x) * 4;
                let pixel = y * OIL_SHINE_MODEL_WIDTH + x;
                for lane in 0..4 {
                    let plane = block * 4 + lane;
                    if plane < 6 {
                        planes[plane][pixel] = raw[src + lane];
                    }
                }
            }
        }
    }
    Ok(planes)
}

/// Normal branch of `MaskBlendToEffect.frag` (`0x5B78090`): each RGB channel is
/// `source + (mask/255) * degree * (effect - source)`; alpha is copied.
pub fn oil_shine_mask_blend(
    source: &Image,
    effect_rgb: &[u8],
    mask: &[u8],
    degree: f32,
) -> Result<Image, Refusal> {
    let pixels = pixels_of(source)?;
    let degree = require_degree(degree)?;
    require_len("effect_rgb", effect_rgb.len(), pixels * 3)?;
    require_len("mask", mask.len(), pixels)?;

    let mut out = source.clone();
    if degree == 0.0 {
        return Ok(out);
    }
    for i in 0..pixels {
        let q = mask[i] as f32 / 255.0 * degree;
        for c in 0..3 {
            let src = source.pixels[i * 4 + c] as f32;
            let effect = effect_rgb[i * 3 + c] as f32;
            out.pixels[i * 4 + c] = clamp_byte(src + q * (effect - src));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Priority 27 - PF30259 remove sweat
// ---------------------------------------------------------------------------

pub const SWEAT_MODEL_WIDTH: i32 = 384;
pub const SWEAT_MODEL_HEIGHT: i32 = 256;

/// `0x8616F40`: invert the explicit job affine, transform the four model
/// corners, clamp to the frame and truncate the bounding rectangle.
pub fn sweat_crop_rect(
    affine: [f32; 6],
    input_width: i32,
    input_height: i32,
) -> Result<Roi, Refusal> {
    if input_width <= 0 || input_height <= 0 {
        return Err(Refusal::ValueOutOfRange("frame"));
    }
    if affine.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("affine"));
    }
    let [m0, m1, m2, m3, m4, m5] = affine;
    let det = m0 * m4 - m1 * m3;
    if !det.is_finite() || det.abs() < 1e-12 {
        return Err(Refusal::ValueOutOfRange("affine"));
    }
    let i0 = m4 / det;
    let i1 = -m1 / det;
    let i2 = (m1 * m5 - m4 * m2) / det;
    let i3 = -m3 / det;
    let i4 = m0 / det;
    let i5 = (m3 * m2 - m0 * m5) / det;

    let corners = [
        (0.0f32, 0.0f32),
        (SWEAT_MODEL_WIDTH as f32, 0.0),
        (0.0, SWEAT_MODEL_HEIGHT as f32),
        (SWEAT_MODEL_WIDTH as f32, SWEAT_MODEL_HEIGHT as f32),
    ];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for (x, y) in corners {
        let sx = (i0 * x + i1 * y + i2).clamp(0.0, input_width as f32);
        let sy = (i3 * x + i4 * y + i5).clamp(0.0, input_height as f32);
        min_x = min_x.min(sx);
        min_y = min_y.min(sy);
        max_x = max_x.max(sx);
        max_y = max_y.max(sy);
    }
    let (x0, y0) = (min_x.trunc() as i32, min_y.trunc() as i32);
    let (x1, y1) = (max_x.trunc() as i32, max_y.trunc() as i32);
    if x1 <= x0 || y1 <= y0 {
        return Ok(Roi::empty());
    }
    Ok(Roi {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    })
}

/// Reference blend only: the vendor final binding is not proven. RGB is mixed
/// with truncation, alpha is copied.
pub fn sweat_reference_blend(
    source: &Image,
    repair: &Image,
    detection_mask: &[u8],
    degree: f32,
) -> Result<Image, Refusal> {
    let pixels = pixels_of(source)?;
    if repair.width != source.width || repair.height != source.height {
        return Err(Refusal::ShapeMismatch("repair"));
    }
    pixels_of(repair)?;
    let degree = require_degree(degree)?;
    require_len("detection_mask", detection_mask.len(), pixels)?;

    let mut out = source.clone();
    if degree == 0.0 {
        return Ok(out);
    }
    for i in 0..pixels {
        let alpha = detection_mask[i] as f32 / 255.0 * degree;
        for c in 0..3 {
            let src = source.pixels[i * 4 + c] as f32;
            let rep = repair.pixels[i * 4 + c] as f32;
            out.pixels[i * 4 + c] = (src + (rep - src) * alpha).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

pub fn sweat_auxiliary_weights() -> [f32; 4] {
    [0.0, 0.25, 0.5, 0.75]
}

// ---------------------------------------------------------------------------
// Priority 28 - PF5 dark circle
// ---------------------------------------------------------------------------

pub const DARK_CIRCLE_MODEL_SIZE: usize = 384;

/// `0x46D5DB0`: `residual[c,p] = (target3[c,p] - source_rgb[c,p]) * blend1[p]`.
/// `target3` is channel-planar, `blend1` is one weight per pixel.
pub fn dark_circle_residual(
    source_rgb: &[u8],
    width: u32,
    height: u32,
    target3: &[f32],
    blend1: &[f32],
) -> Result<Vec<f32>, Refusal> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("source_rgb"))?;
    require_len("source_rgb", source_rgb.len(), pixels * 3)?;
    require_len("target3", target3.len(), pixels * 3)?;
    require_len("blend1", blend1.len(), pixels)?;
    require_finite("target3", target3)?;
    require_finite("blend1", blend1)?;
    if blend1.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err(Refusal::ValueOutOfRange("blend1"));
    }

    let mut out = vec![0.0f32; pixels * 3];
    for p in 0..pixels {
        let weight = blend1[p];
        for c in 0..3 {
            let target = target3[c * pixels + p];
            out[p * 3 + c] = (target - source_rgb[p * 3 + c] as f32) * weight;
        }
    }
    Ok(out)
}

/// Model 1-channel tail: `0.5 * clip(x, -1, 1) + 0.5`.
pub fn dark_circle_blend_from_pre_affine(pre: &[f32]) -> Result<Vec<f32>, Refusal> {
    require_len(
        "blend_weight",
        pre.len(),
        DARK_CIRCLE_MODEL_SIZE * DARK_CIRCLE_MODEL_SIZE,
    )?;
    require_finite("blend_weight", pre)?;
    Ok(pre
        .iter()
        .map(|value| 0.5 * value.clamp(-1.0, 1.0) + 0.5)
        .collect())
}

/// Model 3-channel tail: `127.5 * x + 127.5`.
pub fn dark_circle_field_from_pre_scale(pre: &[f32]) -> Result<Vec<f32>, Refusal> {
    require_len(
        "residual_field",
        pre.len(),
        DARK_CIRCLE_MODEL_SIZE * DARK_CIRCLE_MODEL_SIZE * 3,
    )?;
    require_finite("residual_field", pre)?;
    Ok(pre.iter().map(|value| 127.5 * value + 127.5).collect())
}

/// Consumer-side add/clamp candidate, not a proven vendor edge. Uses the C#
/// `MathF.Round` default (banker's rounding) and clamps to `[0,255]`.
pub fn dark_circle_add_clamp(source_rgb: &[u8], residual: &[f32]) -> Result<Vec<u8>, Refusal> {
    require_len("residual", residual.len(), source_rgb.len())?;
    require_finite("residual", residual)?;
    Ok(source_rgb
        .iter()
        .zip(residual)
        .map(|(source, residual)| clamp_byte(*source as f32 + residual))
        .collect())
}

// ---------------------------------------------------------------------------
// Priority 29 - PF90116 eye bag
// ---------------------------------------------------------------------------

pub const EYE_BAG_PF: u32 = 90116;
pub const EYE_BAG_IDENTITY_THRESHOLD: f32 = 0.001;
pub const EYE_BAG_AREA_LIMIT: f32 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyeBagRoiDecision {
    pub raw: Roi,
    pub expanded: Roi,
    pub selected: Roi,
    pub use_crop: bool,
}

fn eye_bag_expand(rect: Roi, width: i32, height: i32) -> Roi {
    if rect.is_empty() {
        return Roi {
            x: 0,
            y: 0,
            w: width,
            h: height,
        };
    }
    let x = rect.x.max(4) - 4;
    let y = rect.y.max(4) - 4;
    let w = (rect.w + 8).min(width - x);
    let h = (rect.h + 8).min(height - y);
    Roi { x, y, w, h }
}

/// `0x5C1BB80`: strict `> 0.001` response threshold, inclusive bounding rect,
/// `+4/-8` expansion and a strict `< 0.9` area gate.
pub fn eye_bag_response_roi(
    response: &[f32],
    width: u32,
    height: u32,
    reference_width: u32,
    reference_height: u32,
) -> Result<EyeBagRoiDecision, Refusal> {
    if width == 0 || height == 0 || reference_width == 0 || reference_height == 0 {
        return Err(Refusal::ValueOutOfRange("frame"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("response"))?;
    require_len("response", response.len(), pixels)?;
    require_finite("response", response)?;

    let (width, height) = (width as i32, height as i32);
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = -1;
    let mut max_y = -1;
    for y in 0..height {
        for x in 0..width {
            if response[(y * width + x) as usize] <= EYE_BAG_IDENTITY_THRESHOLD {
                continue;
            }
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    if max_x < 0 {
        return Ok(EyeBagRoiDecision {
            raw: Roi::empty(),
            expanded: Roi {
                x: 0,
                y: 0,
                w: width,
                h: height,
            },
            selected: Roi::empty(),
            use_crop: false,
        });
    }
    let raw = Roi {
        x: min_x,
        y: min_y,
        w: max_x - min_x + 1,
        h: max_y - min_y + 1,
    };
    let expanded = eye_bag_expand(raw, width, height);
    let image_area = width as i64 * height as i64;
    let reference_area = reference_width as i64 * reference_height as i64;
    let allowed = reference_area > 0
        && reference_area < image_area
        && expanded.w.min(expanded.h).min(width).min(height) > 0
        && !(expanded.x == 0 && expanded.y == 0 && expanded.w == width && expanded.h == height)
        && ((expanded.w as i64 * expanded.h as i64) as f32)
            < image_area as f32 * EYE_BAG_AREA_LIMIT;
    Ok(EyeBagRoiDecision {
        raw,
        expanded,
        selected: if allowed { expanded } else { Roi::empty() },
        use_crop: allowed,
    })
}

/// `cv::resize` INTER_LINEAR half-pixel mapping as used at `0x5C1C610`.
pub fn eye_bag_bilinear(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<f32>, Refusal> {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Err(Refusal::ValueOutOfRange("dimensions"));
    }
    require_len("source", src.len(), src_w as usize * src_h as usize)?;
    require_finite("source", src)?;

    let sx = src_w as f64 / dst_w as f64;
    let sy = src_h as f64 / dst_h as f64;
    let mut dst = vec![0.0f32; dst_w as usize * dst_h as usize];
    for y in 0..dst_h as usize {
        let fy = (((y as f64 + 0.5) * sy - 0.5).clamp(0.0, src_h as f64 - 1.0)) as f32;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(src_h as usize - 1);
        let wy = fy - y0 as f32;
        for x in 0..dst_w as usize {
            let fx = (((x as f64 + 0.5) * sx - 0.5).clamp(0.0, src_w as f64 - 1.0)) as f32;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(src_w as usize - 1);
            let wx = fx - x0 as f32;

            let v00 = src[y0 * src_w as usize + x0];
            let v10 = src[y0 * src_w as usize + x1];
            let v01 = src[y1 * src_w as usize + x0];
            let v11 = src[y1 * src_w as usize + x1];
            let top = v00 + (v10 - v00) * wx;
            let bottom = v01 + (v11 - v01) * wx;
            dst[y * dst_w as usize + x] = top + (bottom - top) * wy;
        }
    }
    Ok(dst)
}

/// Split three interleaved planes at offsets `0, w*h*4, 2*w*h*4`, resize each
/// with INTER_LINEAR and merge back in plane order.
pub fn eye_bag_repair_image(
    color_field3: &[f32],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<f32>, Refusal> {
    let src_pixels = (src_w as usize)
        .checked_mul(src_h as usize)
        .ok_or(Refusal::ShapeMismatch("color_field"))?;
    require_len("color_field", color_field3.len(), src_pixels * 3)?;
    require_finite("color_field", color_field3)?;

    let mut planes: [Vec<f32>; 3] = [
        vec![0.0; src_pixels],
        vec![0.0; src_pixels],
        vec![0.0; src_pixels],
    ];
    for p in 0..src_pixels {
        for c in 0..3 {
            planes[c][p] = color_field3[p * 3 + c];
        }
    }
    let resized: Vec<Vec<f32>> = planes
        .iter()
        .map(|plane| eye_bag_bilinear(plane, src_w, src_h, dst_w, dst_h))
        .collect::<Result<_, _>>()?;

    let dst_pixels = dst_w as usize * dst_h as usize;
    let mut out = vec![0.0f32; dst_pixels * 3];
    for p in 0..dst_pixels {
        for c in 0..3 {
            out[p * 3 + c] = resized[c][p];
        }
    }
    Ok(out)
}

/// Explicit bounded consumer: `original += (repair - original) * clamp(mask) *
/// degree`. The `0x5C0A400` post-body (filter/LUT ordering) stays open.
pub fn eye_bag_linear_mask_blend(
    original3: &mut [f32],
    repair3: &[f32],
    mask1: &[f32],
    degree: f32,
) -> Result<(), Refusal> {
    let degree = require_degree(degree)?;
    if original3.len() % 3 != 0 {
        return Err(Refusal::ShapeMismatch("original"));
    }
    let pixels = original3.len() / 3;
    require_len("repair", repair3.len(), pixels * 3)?;
    require_len("mask", mask1.len(), pixels)?;
    require_finite("repair", repair3)?;
    require_finite("mask", mask1)?;
    require_finite("original", original3)?;
    for p in 0..pixels {
        let weight = mask1[p].clamp(0.0, 1.0) * degree;
        for c in 0..3 {
            let index = p * 3 + c;
            original3[index] += (repair3[index] - original3[index]) * weight;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Priority 30 - PF21104 nostril
// ---------------------------------------------------------------------------

pub const NOSTRIL_PF: u32 = 21104;
pub const NOSTRIL_LOWER_GATE: f32 = 1.0e-5;
pub const NOSTRIL_UPPER_GATE: f32 = 1.00001;

pub struct NostrilOutput {
    pub rgb: Vec<u8>,
    pub mask: Vec<u8>,
    pub active: bool,
}

/// Exact pixel/mask law from `0x861AD08..0x861B130`:
/// `mask = clamp(trunc(A * degree * 255))`,
/// `rgb = clamp(trunc((M*V + T) * degree + source))`.
pub fn nostril_apply(
    width: u32,
    height: u32,
    source_rgb: &[u8],
    m: &[f32],
    a: &[f32],
    v: &[f32],
    t: &[f32],
    degree: f32,
) -> Result<NostrilOutput, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("frame"));
    }
    if !degree.is_finite() || degree < 0.0 {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("frame"))?;
    let rgb_len = pixels * 3;
    require_len("source_rgb", source_rgb.len(), rgb_len)?;
    require_len("M", m.len(), pixels)?;
    require_len("A", a.len(), pixels)?;
    require_len("V", v.len(), rgb_len)?;
    require_len("T", t.len(), rgb_len)?;
    require_finite("M", m)?;
    require_finite("A", a)?;
    require_finite("V", v)?;
    require_finite("T", t)?;

    let active = degree > NOSTRIL_LOWER_GATE && degree <= NOSTRIL_UPPER_GATE;
    if !active {
        return Ok(NostrilOutput {
            rgb: source_rgb.to_vec(),
            mask: vec![0; pixels],
            active: false,
        });
    }

    let mut rgb = vec![0u8; rgb_len];
    let mut mask = vec![0u8; pixels];
    for p in 0..pixels {
        mask[p] = (a[p] * degree * 255.0).clamp(0.0, 255.0) as u8;
        for c in 0..3 {
            let delta = m[p] * v[p * 3 + c] + t[p * 3 + c];
            let value = delta * degree + source_rgb[p * 3 + c] as f32;
            rgb[p * 3 + c] = value.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(NostrilOutput {
        rgb,
        mask,
        active: true,
    })
}

// ---------------------------------------------------------------------------
// Priority 31 / 34 - shared vendor compositor (`0x82AA2E0`)
// ---------------------------------------------------------------------------

/// Exact RGB compositor: `w = min(maskA, maskB)`, B/G/R written, alpha kept.
/// `Image` stores channels in RGBA order; the per-channel law is identical, so
/// the caller's channel order is preserved.
pub fn shared_min_mask_composite(
    source: &Image,
    repair: &Image,
    mask_a: &[u8],
    mask_b: &[u8],
) -> Result<Image, Refusal> {
    let pixels = pixels_of(source)?;
    if source.width != repair.width || source.height != repair.height {
        return Err(Refusal::ShapeMismatch("repair"));
    }
    pixels_of(repair)?;
    require_len("mask_a", mask_a.len(), pixels)?;
    require_len("mask_b", mask_b.len(), pixels)?;

    let mut out = source.clone();
    for p in 0..pixels {
        let weight = mask_a[p].min(mask_b[p]) as i32;
        if weight == 0 {
            continue;
        }
        let inverse = 255 - weight;
        for c in 0..3 {
            let source_c = source.pixels[p * 4 + c] as i32;
            let repair_c = repair.pixels[p * 4 + c] as i32;
            let weighted = inverse * source_c + weight * repair_c;
            let delta = weighted as f32 * (1.0 / 255.0) - source_c as f32;
            out.pixels[p * 4 + c] = vendor_round_byte(source_c as f32 + delta);
        }
    }
    Ok(out)
}

/// `0x82A6250`: 11-point lip ROI with a 10% expansion.
pub fn lip_line_roi(
    points: &[(f32, f32)],
    source_width: u32,
    source_height: u32,
) -> Result<Roi, Refusal> {
    if points.len() != 11 {
        return Err(Refusal::ValueOutOfRange("points"));
    }
    if source_width == 0 || source_height == 0 {
        return Err(Refusal::ValueOutOfRange("frame"));
    }
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for (x, y) in points {
        if !x.is_finite() || !y.is_finite() {
            return Err(Refusal::ValueOutOfRange("points"));
        }
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }
    let expand_x = (max_x - min_x) * 0.1;
    let expand_y = (max_y - min_y) * 0.1;
    let left = (min_x - expand_x).max(0.0) as i32;
    let top = (min_y - expand_y).max(0.0) as i32;
    let right = ((max_x + expand_x) as i32).clamp(0, source_width as i32);
    let bottom = ((max_y + expand_y) as i32).clamp(0, source_height as i32);
    Ok(Roi {
        x: left,
        y: top,
        w: (right - left).max(0),
        h: (bottom - top).max(0),
    })
}

// ---------------------------------------------------------------------------
// 3D LUT sampling (`o3_sample_lut`, SHA256 7744FB..10F2)
// ---------------------------------------------------------------------------

pub const LUT_SIZE: usize = 64;
pub const LUT_TILE: usize = 8;
pub const LUT_ATLAS_WIDTH: usize = 512;
pub const LUT_ATLAS_FLOATS: usize = LUT_ATLAS_WIDTH * LUT_ATLAS_WIDTH * 4;

/// Trilinear 3D-LUT sample: 512x512 RGBA float4 atlas, 8x8 tiles of 64x64.
/// `rgb` is clamped to `[0,1]`; missing or short atlases are refused.
pub fn sample_lut3d(atlas: &[f32], rgb: [f32; 3]) -> Result<[f32; 3], Refusal> {
    require_len("lut", atlas.len(), LUT_ATLAS_FLOATS)?;
    if rgb.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("rgb"));
    }
    let scaled = [
        rgb[0].clamp(0.0, 1.0) * 63.0,
        rgb[1].clamp(0.0, 1.0) * 63.0,
        rgb[2].clamp(0.0, 1.0) * 63.0,
    ];
    let base = [
        scaled[0].floor() as usize,
        scaled[1].floor() as usize,
        scaled[2].floor() as usize,
    ];
    let frac = [
        scaled[0] - base[0] as f32,
        scaled[1] - base[1] as f32,
        scaled[2] - base[2] as f32,
    ];
    let next = [
        (base[0] + 1).min(63),
        (base[1] + 1).min(63),
        (base[2] + 1).min(63),
    ];

    let mut out = [0.0f32; 3];
    for bz in 0..2 {
        let z = if bz == 0 { base[2] } else { next[2] };
        let wz = if bz == 0 { 1.0 - frac[2] } else { frac[2] };
        let y_base = (z >> 3) * LUT_SIZE;
        let x_base = (z & 7) * LUT_SIZE;
        for gy in 0..2 {
            let y = y_base + if gy == 0 { base[1] } else { next[1] };
            let wy = if gy == 0 { 1.0 - frac[1] } else { frac[1] };
            for gx in 0..2 {
                let x = x_base + if gx == 0 { base[0] } else { next[0] };
                let wx = if gx == 0 { 1.0 - frac[0] } else { frac[0] };
                let weight = wz * wy * wx;
                let index = (y * LUT_ATLAS_WIDTH + x) * 4;
                let texel = &atlas[index..index + 4];
                if texel.iter().any(|value| !value.is_finite()) {
                    return Err(Refusal::ValueOutOfRange("lut"));
                }
                for c in 0..3 {
                    out[c] += texel[c] * weight;
                }
            }
        }
    }
    Ok(out)
}

/// Masked 3D-LUT blend (`o3_masked_lut_whiten`): `alpha = mask/255 * degree`,
/// skipped when `alpha <= 0.001`; RGB blends source and LUT sample, alpha is
/// copied byte-for-byte, LUT round is half-even.
pub fn masked_lut_blend(
    source: &Image,
    mask: &[u8],
    atlas: &[f32],
    degree: f32,
) -> Result<Image, Refusal> {
    let pixels = pixels_of(source)?;
    let degree = require_degree(degree)?;
    require_len("mask", mask.len(), pixels)?;
    require_len("lut", atlas.len(), LUT_ATLAS_FLOATS)?;

    let mut out = source.clone();
    for p in 0..pixels {
        let alpha = mask[p] as f32 / 255.0 * degree;
        if !(alpha > 0.001) {
            continue;
        }
        let color = [
            source.pixels[p * 4] as f32 / 255.0,
            source.pixels[p * 4 + 1] as f32 / 255.0,
            source.pixels[p * 4 + 2] as f32 / 255.0,
        ];
        let lut = sample_lut3d(atlas, color)?;
        for c in 0..3 {
            let mixed = color[c] * (1.0 - alpha) + lut[c] * alpha;
            out.pixels[p * 4 + c] = clamp_byte(mixed * 255.0);
        }
    }
    Ok(out)
}
