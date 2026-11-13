// Makeup, plump and skin-tone rows 51-70.
//
// Parameter laws are ported from the p21/p22 settings adapters; the
// SkinAdjustV2 pixel law is ported from `Pf90005ExactWhiten.metal`. Every live
// value is an explicit argument: a missing native degree, model plane, mask or
// LUT is a typed refusal, never a guessed default.

use crate::{Access, EffectMeta, Image, Refusal};

const FACE_PLUMP_ACTIVE: f32 = 0.0001;
const SKIN_ADJUST_GATE: f32 = 0.001;
const LUT_EDGE: usize = 512;
const LUT_BYTES: usize = LUT_EDGE * LUT_EDGE * 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    MoistFace,
    MoistBody,
    Collarbone,
    FullFacePlump,
    Forehead,
    MidFace,
    Cheek,
    NasalBase,
    BrowRidge,
    Chin,
    MouthCorner,
    SkinGloss,
    SkinTexture,
    SkinClarity,
    SkinRadiance,
    SkinWhitening,
    SkinRosy,
    SkinToneMulti,
    SkinToneFace,
    SkinToneBody,
}

impl Row {
    pub const ALL: [Row; 20] = [
        Row::MoistFace,
        Row::MoistBody,
        Row::Collarbone,
        Row::FullFacePlump,
        Row::Forehead,
        Row::MidFace,
        Row::Cheek,
        Row::NasalBase,
        Row::BrowRidge,
        Row::Chin,
        Row::MouthCorner,
        Row::SkinGloss,
        Row::SkinTexture,
        Row::SkinClarity,
        Row::SkinRadiance,
        Row::SkinWhitening,
        Row::SkinRosy,
        Row::SkinToneMulti,
        Row::SkinToneFace,
        Row::SkinToneBody,
    ];

    pub fn pf(self) -> u32 {
        match self {
            Row::MoistFace => 90002,
            Row::MoistBody => 90010,
            Row::Collarbone => 30017,
            Row::FullFacePlump => 30210,
            Row::Forehead => 30191,
            Row::MidFace => 30194,
            Row::Cheek => 30197,
            Row::NasalBase => 30198,
            Row::BrowRidge => 30203,
            Row::Chin => 30204,
            Row::MouthCorner => 30207,
            Row::SkinGloss => 8068,
            Row::SkinTexture => 8069,
            Row::SkinClarity => 8070,
            Row::SkinRadiance => 90004,
            Row::SkinWhitening => 90005,
            Row::SkinRosy => 90007,
            Row::SkinToneMulti => 30059,
            Row::SkinToneFace => 90085,
            Row::SkinToneBody => 90086,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Row::MoistFace => "moistureSmoothSkin",
            Row::MoistBody => "moistureSmoothSkin_body",
            Row::Collarbone => "poreEnhance",
            Row::FullFacePlump => "fullFacePlump",
            Row::Forehead => "foreheadPlump",
            Row::MidFace => "midFacePlump",
            Row::Cheek => "cheekPlump",
            Row::NasalBase => "nasalBasePlump",
            Row::BrowRidge => "geisonPlump",
            Row::Chin => "chinPlump",
            Row::MouthCorner => "mouthAngulusPlump",
            Row::SkinGloss => "skinGloss",
            Row::SkinTexture => "skinTexture",
            Row::SkinClarity => "skinClarity",
            Row::SkinRadiance => "skinRadiance",
            Row::SkinWhitening => "skinWhitening",
            Row::SkinRosy => "skinRosy",
            Row::SkinToneMulti => "skinToneUnifyMulti",
            Row::SkinToneFace => "skinToneUnifyFace",
            Row::SkinToneBody => "skinToneUnifyBody",
        }
    }

    pub fn from_pf(pf: u32) -> Option<Row> {
        Row::ALL.into_iter().find(|row| row.pf() == pf)
    }

    fn ui_range(self) -> (f32, f32) {
        match self {
            Row::FullFacePlump => (0.0, 4.0),
            Row::SkinGloss | Row::SkinTexture => (-100.0, 100.0),
            _ => (0.0, 100.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Live {
    /// PF30210 native degree, PF30207 raw mouthAngulus, and per-face native
    /// degree for rows 57/59/60.
    pub native_degree: Option<f32>,
    /// PF90002 degree pivot/gain writers.
    pub pivot: Option<f32>,
    pub gain: Option<f32>,
    /// PF90010 captured `cell+0x8C` degree.
    pub captured_degree: Option<f32>,
    /// PF30191 degree/flatten/brighten maps.
    pub degree: Option<f32>,
    pub flatten: Option<f32>,
    pub brighten: Option<f32>,
    /// PF30194 smooth map.
    pub smooth: Option<f32>,
    /// PF90004 `valueMean` at `0x702E3F0`.
    pub value_mean: Option<f32>,
    /// PF8068 explicit native percent in the config domain.
    pub native_percent: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bound {
    pub active: bool,
    pub display: f32,
    pub degree: f32,
    /// Row 55 `flatten*degree`, row 56 `degree*smooth`, row 62 negative part,
    /// row 63 surface-blur degree.
    pub second: Option<f32>,
    /// Row 55/56 `degree*brighten`.
    pub brighten: Option<f32>,
}

fn finite(value: f32, what: &'static str) -> Result<f32, Refusal> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Refusal::ValueOutOfRange(what))
    }
}

fn need(value: Option<f32>, what: &'static str) -> Result<f32, Refusal> {
    match value {
        Some(value) if value.is_finite() => Ok(value),
        Some(_) => Err(Refusal::ValueOutOfRange(what)),
        None => Err(Refusal::MissingInput(what)),
    }
}

fn identity() -> Bound {
    Bound { active: false, display: 0.0, degree: 0.0, second: None, brighten: None }
}

pub fn bind(row: Row, ui: f32, live: &Live) -> Result<Bound, Refusal> {
    let (min, max) = row.ui_range();
    if !ui.is_finite() || ui < min || ui > max {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let display = ui / 100.0;

    if row == Row::FullFacePlump && (ui - ui.round()).abs() > 1e-6 {
        return Err(Refusal::ValueOutOfRange("ui_level"));
    }

    match row {
        Row::MoistFace => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let pivot = need(live.pivot, "pivot")?;
            let gain = need(live.gain, "gain")?;
            let degree = (display - pivot) * gain + pivot;
            Ok(Bound { active: degree > 0.0, display, degree, ..identity() })
        }
        Row::MoistBody => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let degree = need(live.captured_degree, "captured_degree")?;
            if !(0.0..=1.0).contains(&degree) {
                return Err(Refusal::ValueOutOfRange("captured_degree"));
            }
            Ok(Bound { active: degree >= 0.01, display, degree, ..identity() })
        }
        Row::Collarbone => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let degree = display.clamp(0.0, 1.0);
            Ok(Bound { active: true, display, degree, ..identity() })
        }
        Row::FullFacePlump => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let degree = need(live.native_degree, "native_degree")?;
            Ok(Bound {
                active: degree > FACE_PLUMP_ACTIVE,
                display: ui.round(),
                degree,
                ..identity()
            })
        }
        Row::Forehead => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let degree = need(live.degree, "degree")?;
            let flatten = need(live.flatten, "flatten")?;
            let brighten = need(live.brighten, "brighten")?;
            Ok(Bound {
                active: true,
                display,
                degree,
                second: Some(flatten * degree),
                brighten: Some(degree * brighten),
            })
        }
        Row::MidFace => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let degree = need(live.degree, "degree")?;
            let smooth = need(live.smooth, "smooth")?;
            let brighten = need(live.brighten, "brighten")?;
            Ok(Bound {
                active: true,
                display,
                degree,
                second: Some(degree * smooth),
                brighten: Some(degree * brighten),
            })
        }
        Row::Cheek | Row::BrowRidge | Row::Chin => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let degree = need(live.native_degree, "native_degree")?;
            Ok(Bound {
                active: degree > FACE_PLUMP_ACTIVE,
                display,
                degree,
                ..identity()
            })
        }
        Row::NasalBase => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let degree = need(live.native_degree, "native_degree")?;
            Ok(Bound { active: degree.abs() > 0.01, display, degree, ..identity() })
        }
        Row::MouthCorner => {
            if ui == 0.0 {
                return Ok(identity());
            }
            let degree = need(live.native_degree, "native_degree")?;
            Ok(Bound { active: true, display, degree, ..identity() })
        }
        Row::SkinGloss => {
            let native_percent = match (ui, live.native_percent) {
                (0.0, None) => 50.0,
                (_, Some(value)) => finite(value, "native_percent")?,
                (_, None) => return Err(Refusal::MissingInput("native_percent")),
            };
            if !(0.0..=100.0).contains(&native_percent) {
                return Err(Refusal::ValueOutOfRange("native_percent"));
            }
            let part = native_percent / 100.0;
            let positive = (2.0 * part - 1.0).max(0.0);
            let negative = (1.0 - 2.0 * part).max(0.0);
            Ok(Bound {
                active: positive > 0.0 || negative > 0.0,
                display,
                degree: positive,
                second: Some(negative),
                brighten: None,
            })
        }
        Row::SkinTexture => {
            let sticker = (0.01 * ui).max(0.0);
            let blur = (-0.0049 * ui).max(0.0);
            Ok(Bound {
                active: sticker > 0.0 || blur > 0.0,
                display,
                degree: sticker,
                second: Some(blur),
                brighten: None,
            })
        }
        Row::SkinClarity => {
            if ui == 0.0 {
                return Ok(identity());
            }
            Ok(Bound { active: true, display, degree: display, ..identity() })
        }
        Row::SkinRadiance => {
            let degree = (display * 100.0).round().clamp(0.0, 100.0);
            if degree == 0.0 {
                return Ok(identity());
            }
            let value_mean = need(live.value_mean, "value_mean")?;
            if value_mean < 0.0 {
                return Err(Refusal::ValueOutOfRange("value_mean"));
            }
            Ok(Bound { active: true, display, degree, ..identity() })
        }
        Row::SkinWhitening | Row::SkinRosy => {
            if ui == 0.0 {
                return Ok(identity());
            }
            Ok(Bound { active: true, display, degree: display, ..identity() })
        }
        Row::SkinToneMulti | Row::SkinToneFace => {
            let degree = (ui.abs() / 100.0).clamp(0.0, 1.0);
            Ok(Bound { active: degree > SKIN_ADJUST_GATE, display, degree, ..identity() })
        }
        Row::SkinToneBody => {
            if ui == 0.0 {
                return Ok(identity());
            }
            Ok(Bound { active: display > SKIN_ADJUST_GATE, display, degree: display, ..identity() })
        }
    }
}

fn check_image(image: &Image, what: &'static str) -> Result<usize, Refusal> {
    image.validate().map_err(|_| Refusal::ShapeMismatch(what))?;
    Ok(image.pixel_count())
}

fn lut_byte(lut: &[u8], x: usize, y: usize, channel: usize) -> i32 {
    lut[(y * LUT_EDGE + x) * 3 + channel] as i32
}

fn tap(lut: &[u8], channel: usize, px: f32, py: f32) -> f32 {
    let u = px * 512.0 - 0.5;
    let v = py * 512.0 - 0.5;
    let i0 = u.floor();
    let j0 = v.floor();
    let kx = ((u - i0) * 256.0 + 0.5).floor() as i32;
    let ky = ((v - j0) * 256.0 + 0.5).floor() as i32;
    let t = (kx * ky + 128) >> 8;
    let w00 = 256 - kx - ky + t;
    let w10 = kx - t;
    let w01 = ky - t;
    let w11 = t;
    let x0 = (i0 as i32).clamp(0, 511) as usize;
    let x1 = (i0 as i32 + 1).clamp(0, 511) as usize;
    let y0 = (j0 as i32).clamp(0, 511) as usize;
    let y1 = (j0 as i32 + 1).clamp(0, 511) as usize;
    let q = lut_byte(lut, x0, y0, channel) * w00
        + lut_byte(lut, x1, y0, channel) * w10
        + lut_byte(lut, x0, y1, channel) * w01
        + lut_byte(lut, x1, y1, channel) * w11;
    (((q * 257 + 128) >> 8) as f32) / 65535.0
}

fn sample_lut(lut: &[u8], r: f32, g: f32, b: f32, channel: usize) -> f32 {
    let blue = b * 63.0;
    let bf = blue.floor();
    let q1y = (bf * 0.125).floor();
    let q1x = bf - q1y * 8.0;
    let bc = blue.ceil();
    let q2y = (bc * 0.125).floor();
    let q2x = bc - q2y * 8.0;
    let k = 0.000_976_562_5;
    let span = f32::from_bits(0x3dfc_0001);
    let x1 = span * r + (q1x * 0.125 + k);
    let y1 = span * g + (q1y * 0.125 + k);
    let x2 = q2x * 0.125 + span * r + k;
    let y2 = q2y * 0.125 + span * g + k;
    let low = tap(lut, channel, x1, y1);
    let high = tap(lut, channel, x2, y2);
    low + (blue - bf) * (high - low)
}

fn quantize(value: f32) -> u8 {
    let value12 = (value * 4096.0).floor() as i32;
    ((value12 * 255 + 2047) >> 12).clamp(0, 255) as u8
}

/// SkinAdjustV2 WHITEN/RUDDY: `alpha = mask/255 * mix`, shared 512x512x3 LUT
/// atlas (8x8 tiles of 64x64). Pixels at or below `0.001` are copied.
pub fn skin_adjust_lut(
    source: &Image,
    mask_r: &[u8],
    lut_rgb512: &[u8],
    mix: f32,
) -> Result<Image, Refusal> {
    let pixels = check_image(source, "source")?;
    if mask_r.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    if lut_rgb512.len() != LUT_BYTES {
        return Err(Refusal::ShapeMismatch("lut_rgb512"));
    }
    if !mix.is_finite() || !(0.0..=1.0).contains(&mix) {
        return Err(Refusal::ValueOutOfRange("mix"));
    }

    let mut out = source.clone();
    for (index, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        let weight = mask_r[index] as f32 / 255.0 * mix;
        if !(weight > SKIN_ADJUST_GATE) {
            continue;
        }
        let r = px[0] as f32 / 255.0;
        let g = px[1] as f32 / 255.0;
        let b = px[2] as f32 / 255.0;
        let mapped = [
            sample_lut(lut_rgb512, r, g, b, 0),
            sample_lut(lut_rgb512, r, g, b, 1),
            sample_lut(lut_rgb512, r, g, b, 2),
        ];
        for channel in 0..3 {
            let source_channel = [r, g, b][channel];
            let blended = source_channel + weight * (mapped[channel] - source_channel);
            px[channel] = quantize(blended);
        }
    }
    Ok(out)
}

/// PF90086 body tone unify pixel law:
/// `out = trunc(clamp(src + 255*(wM*M*(src/255) + wO*O)))`.
/// `m` and `o` are per-pixel linear planes, three floats per pixel.
pub fn body_tone_unify(
    source: &Image,
    m: &[f32],
    o: &[f32],
    w_m: f32,
    w_o: f32,
) -> Result<Image, Refusal> {
    let pixels = check_image(source, "source")?;
    if m.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("m"));
    }
    if o.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("o"));
    }
    if !w_m.is_finite() || !w_o.is_finite() {
        return Err(Refusal::ValueOutOfRange("weights"));
    }
    if m.iter().chain(o).any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("model_plane"));
    }

    let mut out = source.clone();
    for (index, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        for channel in 0..3 {
            let offset = index * 3 + channel;
            let src = px[channel] as f32;
            let value = src + 255.0 * (w_m * m[offset] * (src / 255.0) + w_o * o[offset]);
            px[channel] = value.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

const FULL_FACE_INPUTS: &[&str] = &[
    "native_degree", "owner_masks", "offset_rgb", "dual_lut_768", "model_tensor",
];
const FOREHEAD_INPUTS: &[&str] = &["degree", "flatten", "brighten", "model", "mask", "roi"];
const MID_FACE_INPUTS: &[&str] = &[
    "degree", "smooth", "brighten", "owner_mask", "model", "weights", "roi",
];
const CHEEK_INPUTS: &[&str] = &["native_degree", "mask_a", "owner_mask_b", "model", "roi"];
const NASAL_INPUTS: &[&str] = &[
    "native_degree", "model_mask", "protect_planes", "displacement_xy", "roi",
];
const BROW_INPUTS: &[&str] = &["native_degree", "model_field", "mask_a", "owner_mask_b", "luts"];
const CHIN_INPUTS: &[&str] = &[
    "native_degree", "chin_point", "displacement", "owner_mask", "roi",
];
const MOUTH_INPUTS: &[&str] = &[
    "native_degree", "mouth_angle_mask", "displacement_xy", "source_rgba", "model_output",
];
const WHITEN_INPUTS: &[&str] = &["source", "mask_r", "lut_rgb512", "mix", "roi"];
const RUDDY_INPUTS: &[&str] = &["source", "mask_r", "lut_rgb512", "mix", "roi"];
const TONE_BODY_INPUTS: &[&str] = &["source", "model_m", "model_o", "w_m", "w_o"];
pub const META: &[EffectMeta] = &[

    EffectMeta { pf: 30210, key: "fullFacePlump", label: "全脸丰盈", access: Access::GpuPlanned, inputs: FULL_FACE_INPUTS },
    EffectMeta { pf: 30191, key: "foreheadPlump", label: "额头程度", access: Access::GpuPlanned, inputs: FOREHEAD_INPUTS },
    EffectMeta { pf: 30194, key: "midFacePlump", label: "面中程度", access: Access::GpuPlanned, inputs: MID_FACE_INPUTS },
    EffectMeta { pf: 30197, key: "cheekPlump", label: "面颊程度", access: Access::GpuPlanned, inputs: CHEEK_INPUTS },
    EffectMeta { pf: 30198, key: "nasalBasePlump", label: "鼻基底程度", access: Access::GpuPlanned, inputs: NASAL_INPUTS },
    EffectMeta { pf: 30203, key: "geisonPlump", label: "眉弓程度", access: Access::GpuPlanned, inputs: BROW_INPUTS },
    EffectMeta { pf: 30204, key: "chinPlump", label: "下巴程度", access: Access::GpuPlanned, inputs: CHIN_INPUTS },
    EffectMeta { pf: 30207, key: "mouthAngulusPlump", label: "嘴角上扬", access: Access::GpuPlanned, inputs: MOUTH_INPUTS },
    EffectMeta { pf: 90005, key: "skinWhitening", label: "皮肤美白", access: Access::Cpu, inputs: WHITEN_INPUTS },
    EffectMeta { pf: 90007, key: "skinRosy", label: "皮肤红润", access: Access::Cpu, inputs: RUDDY_INPUTS },
    EffectMeta { pf: 90086, key: "skinToneUnifyBody", label: "肤色统一（身体）", access: Access::Cpu, inputs: TONE_BODY_INPUTS },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel() -> Image {
        Image::solid(1, 1, [100, 150, 200, 77])
    }

    fn flat_lut(rgb: [u8; 3]) -> Vec<u8> {
        rgb.iter().copied().cycle().take(LUT_BYTES).collect()
    }

    #[test]
    fn meta_covers_the_seventeen_remaining_pfs() {
        // 8068-8070 live in `skin_models`; 30059/90085 moved to `tone_unify`.
        assert_eq!(META.len(), 11);
        assert!(META.iter().all(|meta| !matches!(meta.pf, 8068 | 8069 | 8070 | 30059 | 90085 | 90004 | 90002 | 90010)));
        for (index, meta) in META.iter().enumerate() {
            assert_eq!(Row::from_pf(meta.pf).map(Row::key), Some(meta.key), "row {index}");
            assert!(META[index + 1..].iter().all(|other| other.pf != meta.pf));
        }
    }

    #[test]
    fn ui_bounds_and_finiteness_are_enforced() {
        let live = Live::default();
        assert_eq!(bind(Row::MoistFace, -1.0, &live), Err(Refusal::ValueOutOfRange("ui")));
        assert_eq!(bind(Row::SkinGloss, 101.0, &live), Err(Refusal::ValueOutOfRange("ui")));
        assert_eq!(bind(Row::SkinTexture, f32::NAN, &live), Err(Refusal::ValueOutOfRange("ui")));
        assert_eq!(
            bind(Row::FullFacePlump, 2.5, &live),
            Err(Refusal::ValueOutOfRange("ui_level"))
        );
    }

    #[test]
    fn rows_51_54_degree_laws() {
        let pivot_gain = Live { pivot: Some(0.0), gain: Some(1.0), ..Live::default() };
        let bound = bind(Row::MoistFace, 50.0, &pivot_gain).unwrap();
        assert!(bound.active);
        assert_eq!(bound.degree, 0.5);
        assert_eq!(bind(Row::MoistFace, 0.0, &Live::default()).unwrap(), identity());
        assert_eq!(
            bind(Row::MoistFace, 50.0, &Live::default()),
            Err(Refusal::MissingInput("pivot"))
        );

        let body = Live { captured_degree: Some(0.37), ..Live::default() };
        assert_eq!(bind(Row::MoistBody, 50.0, &body).unwrap().degree, 0.37);
        assert_eq!(
            bind(Row::MoistBody, 50.0, &Live { captured_degree: Some(1.1), ..Live::default() }),
            Err(Refusal::ValueOutOfRange("captured_degree"))
        );

        let collarbone = bind(Row::Collarbone, 50.0, &Live::default()).unwrap();
        assert!(collarbone.active);
        assert_eq!(collarbone.degree, 0.5);

        let plump = Live { native_degree: Some(0.5), ..Live::default() };
        let bound = bind(Row::FullFacePlump, 2.0, &plump).unwrap();
        assert_eq!((bound.display, bound.degree), (2.0, 0.5));
    }

    #[test]
    fn rows_55_56_effective_scales() {
        let forehead = Live {
            degree: Some(0.5),
            flatten: Some(0.8),
            brighten: Some(0.6),
            ..Live::default()
        };
        let bound = bind(Row::Forehead, 50.0, &forehead).unwrap();
        assert_eq!(bound.second, Some(0.4));
        assert!((bound.brighten.unwrap() - 0.3).abs() < 1e-6);

        let mid = Live {
            degree: Some(0.5),
            smooth: Some(0.8),
            brighten: Some(0.6),
            ..Live::default()
        };
        let bound = bind(Row::MidFace, 50.0, &mid).unwrap();
        assert!((bound.second.unwrap() - 0.4).abs() < 1e-6);
        assert!((bound.brighten.unwrap() - 0.3).abs() < 1e-6);
        assert_eq!(
            bind(Row::MidFace, 50.0, &Live::default()),
            Err(Refusal::MissingInput("degree"))
        );
    }

    #[test]
    fn rows_57_to_60_native_gates() {
        let native = Live { native_degree: Some(0.5), ..Live::default() };
        for row in [Row::Cheek, Row::BrowRidge, Row::Chin] {
            assert!(bind(row, 50.0, &native).unwrap().active);
            assert_eq!(
                bind(row, 50.0, &Live::default()),
                Err(Refusal::MissingInput("native_degree"))
            );
        }
        let signed = Live { native_degree: Some(-0.5), ..Live::default() };
        let bound = bind(Row::NasalBase, 50.0, &signed).unwrap();
        assert!(bound.active);
        assert_eq!(bound.degree, -0.5);
        let edge = Live { native_degree: Some(0.01), ..Live::default() };
        assert!(!bind(Row::NasalBase, 50.0, &edge).unwrap().active);
    }

    #[test]
    fn rows_61_and_62_branch_laws() {
        let mouth = Live { native_degree: Some(-0.25), ..Live::default() };
        assert_eq!(bind(Row::MouthCorner, 50.0, &mouth).unwrap().degree, -0.25);
        assert_eq!(
            bind(Row::MouthCorner, 50.0, &Live::default()),
            Err(Refusal::MissingInput("native_degree"))
        );

        let gloss = Live { native_percent: Some(75.0), ..Live::default() };
        let bound = bind(Row::SkinGloss, 50.0, &gloss).unwrap();
        assert_eq!((bound.degree, bound.second), (0.5, Some(0.0)));
        let negative = Live { native_percent: Some(25.0), ..Live::default() };
        let bound = bind(Row::SkinGloss, -50.0, &negative).unwrap();
        assert_eq!((bound.degree, bound.second), (0.0, Some(0.5)));
        assert!(!bind(Row::SkinGloss, 0.0, &Live::default()).unwrap().active);
        assert_eq!(
            bind(Row::SkinGloss, 50.0, &Live::default()),
            Err(Refusal::MissingInput("native_percent"))
        );
    }

    #[test]
    fn rows_63_to_67_degree_laws() {
        let bound = bind(Row::SkinTexture, 50.0, &Live::default()).unwrap();
        assert_eq!((bound.degree, bound.second), (0.5, Some(0.0)));
        let bound = bind(Row::SkinTexture, -100.0, &Live::default()).unwrap();
        assert_eq!(bound.second, Some(0.49));

        assert_eq!(bind(Row::SkinClarity, 50.0, &Live::default()).unwrap().degree, 0.5);

        let mean = Live { value_mean: Some(100.0), ..Live::default() };
        assert_eq!(bind(Row::SkinRadiance, 50.5, &mean).unwrap().degree, 51.0);
        assert_eq!(
            bind(Row::SkinRadiance, 50.0, &Live::default()),
            Err(Refusal::MissingInput("value_mean"))
        );

        assert_eq!(bind(Row::SkinWhitening, 50.0, &Live::default()).unwrap().degree, 0.5);
        assert_eq!(bind(Row::SkinRosy, 50.0, &Live::default()).unwrap().degree, 0.5);
    }

    #[test]
    fn rows_68_to_70_epsilon_gates() {
        for row in [Row::SkinToneMulti, Row::SkinToneFace, Row::SkinToneBody] {
            assert!(!bind(row, 0.1, &Live::default()).unwrap().active);
            assert!(bind(row, 0.11, &Live::default()).unwrap().active);
        }
        assert_eq!(bind(Row::SkinToneMulti, 50.0, &Live::default()).unwrap().degree, 0.5);
    }

    #[test]
    fn lut_blend_matches_exact_shader_quantization() {
        let src = pixel();
        let lut = flat_lut([10, 20, 30]);
        let out = skin_adjust_lut(&src, &[255], &lut, 1.0).unwrap();
        assert_eq!(out.pixel(0, 0), [10, 20, 30, 77]);

        let out = skin_adjust_lut(&src, &[255], &lut, 0.0).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn lut_blend_gate_is_strict_and_alpha_is_preserved() {
        let src = pixel();
        let lut = flat_lut([0, 0, 0]);
        let below = skin_adjust_lut(&src, &[1], &lut, 0.2).unwrap();
        assert_eq!(below, src);
        let above = skin_adjust_lut(&src, &[255], &lut, 0.5).unwrap();
        assert_eq!(above.pixel(0, 0)[3], 77);
        assert!(above.pixel(0, 0)[0] < 100);
    }

    #[test]
    fn lut_blend_refuses_bad_material() {
        let src = pixel();
        assert_eq!(
            skin_adjust_lut(&src, &[], &flat_lut([0, 0, 0]), 1.0),
            Err(Refusal::ShapeMismatch("mask_r"))
        );
        assert_eq!(
            skin_adjust_lut(&src, &[0], &[], 1.0),
            Err(Refusal::ShapeMismatch("lut_rgb512"))
        );
        assert_eq!(
            skin_adjust_lut(&src, &[0], &flat_lut([0, 0, 0]), f32::NAN),
            Err(Refusal::ValueOutOfRange("mix"))
        );
    }

    #[test]
    fn body_tone_unify_pixel_law() {
        let out = body_tone_unify(
            &pixel(),
            &[0.5, -0.5, 0.0],
            &[0.0, 0.0, 0.1],
            1.0,
            1.0,
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [150, 75, 225, 77]);
    }

    #[test]
    fn public_image_and_model_planes_are_validated() {
        let packed = Image { width: 2, height: 1, pixels: vec![0; 7] };
        assert_eq!(
            skin_adjust_lut(&packed, &[0, 0], &flat_lut([0, 0, 0]), 0.0),
            Err(Refusal::ShapeMismatch("source"))
        );
        assert_eq!(
            body_tone_unify(&pixel(), &[0.0], &[0.0, 0.0, 0.0], 1.0, 1.0),
            Err(Refusal::ShapeMismatch("m"))
        );
        assert_eq!(
            body_tone_unify(&pixel(), &[0.0, 0.0, f32::NAN], &[0.0; 3], 1.0, 1.0),
            Err(Refusal::ValueOutOfRange("model_plane"))
        );
    }
}
