use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use aurora_portrait::kernels::{face_detail, makeup, skin_adjust};
use aurora_portrait::{Image, Refusal};
use skin_adjust::{Roi, WrinkleUniforms};

use super::sha256_32;

pub type HistoricalSkinInputs = BTreeMap<u32, HistoricalSkinInput>;

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalSkinInput {
    pub ticket: u64,
    pub size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub control: f32,
    pub data: HistoricalSkinData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalSkinWrinkle {
    pub base: Image,
    pub residual: Image,
    pub mask_rgba: Vec<u8>,
    pub skin_mask: Vec<u8>,
    pub roi: Roi,
    pub uniforms: WrinkleUniforms,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalSkinPass {
    pub mask: Vec<u8>,
    pub effect: Image,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HistoricalSkinData {
    Glabella(HistoricalSkinWrinkle),
    Forehead(HistoricalSkinWrinkle),
    BodySkin(Vec<HistoricalSkinPass>),
    StretchMark {
        model: Image,
        mask: Vec<u8>,
        roi: Roi,
        alternate: bool,
    },
    BodyTone {
        m: Vec<f32>,
        o: Vec<f32>,
        w_m: f32,
        w_o: f32,
    },
}

pub(super) const PFS: &[u32] = &[21024, 21105, 90008, 90024, 90086];

pub(super) fn open_reason(pf: u32) -> Option<&'static str> {
    Some(match pf {
        0 => "PF0 final repair RGB blend and composer C binding are unresolved",
        21104 => "PF21104 final placement/composite of the native RGB and A-derived mask pair is unresolved",
        90003 => "PF90003 native Trad curve/LUT and final writeback are unresolved",
        90085 => "PF90085 native GammaMap and HSL palette transform are unresolved in this tree",
        90130 => "PF90130 native residual producer, A/B bridge, and cross-face writeback order are unresolved",
        _ => return None,
    })
}

impl HistoricalSkinData {
    fn pf(&self) -> u32 {
        match self {
            Self::Glabella(_) => 21024,
            Self::BodySkin(_) => 21105,
            Self::StretchMark { .. } => 90008,
            Self::Forehead(_) => 90024,
            Self::BodyTone { .. } => 90086,
        }
    }
}

pub(super) fn apply(
    source: &Image,
    ticket: Option<u64>,
    pf: u32,
    control: f32,
    inputs: Option<&HistoricalSkinInputs>,
) -> Result<Option<Image>, Refusal> {
    if !(0.0..=1.0).contains(&control) {
        return Err(Refusal::ValueOutOfRange("historical skin control"));
    }
    let ui = control * 100.0;
    if control == 0.0
        || (matches!(pf, 21024 | 90024) && control <= 0.001)
        || (pf == 90008 && !skin_adjust::stretch_mark_degree(ui)?.1)
    {
        return Ok(None);
    }
    let input = inputs
        .and_then(|inputs| inputs.get(&pf))
        .ok_or(Refusal::MissingInput("historical skin same-job payload"))?;
    if input.data.pf() != pf {
        return Err(Refusal::ValueOutOfRange("historical skin payload PF"));
    }
    if input.ticket == 0 || Some(input.ticket) != ticket {
        return Err(Refusal::ValueOutOfRange("historical skin ticket"));
    }
    if input.size != [source.width, source.height] {
        return Err(Refusal::ShapeMismatch("historical skin frame size"));
    }
    if input.control != control {
        return Err(Refusal::ValueOutOfRange("historical skin payload control"));
    }
    if input.source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("historical skin frame digest"));
    }

    let output = match &input.data {
        HistoricalSkinData::Glabella(input) | HistoricalSkinData::Forehead(input) => {
            skin_adjust::wrinkle_apply(
                source, &input.base, &input.residual, &input.mask_rgba,
                &input.skin_mask, input.roi, control, input.uniforms,
            )?
        }
        HistoricalSkinData::BodySkin(passes) => {
            if passes.iter().any(|pass| {
                (pass.effect.width, pass.effect.height) != (source.width, source.height)
            }) {
                return Err(Refusal::ShapeMismatch("historical skin pass frame"));
            }
            let passes: Vec<_> = passes.iter().map(|pass| face_detail::SkinUnifyPass {
                mask: &pass.mask,
                effect: &pass.effect,
            }).collect();
            face_detail::body_skin_unify(source, ui, &passes)?.output
        }
        HistoricalSkinData::StretchMark { model, mask, roi, alternate } => {
            skin_adjust::masked_model_output(source, model, *roi, mask, *alternate)?
        }
        HistoricalSkinData::BodyTone { m, o, w_m, w_o } => {
            makeup::body_tone_unify(source, m, o, *w_m, *w_o)?
        }
    };
    Ok((output != *source).then_some(output))
}

pub(super) fn fingerprint(inputs: &HistoricalSkinInputs) -> u64 {
    let mut hash = DefaultHasher::new();
    let image = |value: &Image, hash: &mut DefaultHasher| {
        (value.width, value.height, &value.pixels).hash(hash);
    };
    let roi = |value: &Roi, hash: &mut DefaultHasher| {
        (value.x, value.y, value.width, value.height).hash(hash);
    };
    for (pf, input) in inputs {
        (pf, input.ticket, input.size, input.source_sha256, input.control.to_bits(), input.data.pf())
            .hash(&mut hash);
        match &input.data {
            HistoricalSkinData::Glabella(input) | HistoricalSkinData::Forehead(input) => {
                image(&input.base, &mut hash);
                image(&input.residual, &mut hash);
                (&input.mask_rgba, &input.skin_mask).hash(&mut hash);
                roi(&input.roi, &mut hash);
                let u = input.uniforms;
                [u.legal_alpha, u.dark_circle_alpha, u.diff_factor, u.diff_factor2,
                 u.color_threshold_factor, u.mix_alpha, u.diff_min_threshold, u.diff_max_threshold,
                 u.high_freq_min_threshold, u.high_freq_max_threshold].map(f32::to_bits).hash(&mut hash);
            }
            HistoricalSkinData::BodySkin(passes) => {
                passes.len().hash(&mut hash);
                for pass in passes {
                    pass.mask.hash(&mut hash);
                    image(&pass.effect, &mut hash);
                }
            }
            HistoricalSkinData::StretchMark { model, mask, roi: rect, alternate } => {
                image(model, &mut hash);
                roi(rect, &mut hash);
                (mask, alternate).hash(&mut hash);
            }
            HistoricalSkinData::BodyTone { m, o, w_m, w_o } => {
                [w_m.to_bits(), w_o.to_bits()].hash(&mut hash);
                for plane in [m, o] {
                    plane.len().hash(&mut hash);
                    for value in plane {
                        value.to_bits().hash(&mut hash);
                    }
                }
            }
        }
    }
    hash.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(source: &Image, data: HistoricalSkinData) -> HistoricalSkinInputs {
        BTreeMap::from([(data.pf(), HistoricalSkinInput {
            ticket: 7,
            size: [source.width, source.height],
            source_sha256: sha256_32(&source.pixels),
            control: 1.0,
            data,
        })])
    }

    #[test]
    fn routes_write_pixels_and_keep_their_native_roi_and_alpha_rules() {
        let source = Image::new(2, 1, vec![100, 150, 200, 77, 1, 2, 3, 17]).unwrap();
        let model = Image::new(2, 1, vec![200, 100, 50, 91, 4, 5, 6, 99]).unwrap();
        let roi = Roi { x: 0, y: 0, width: 1, height: 1 };
        let wrinkle = HistoricalSkinWrinkle {
            base: source.clone(),
            residual: model.clone(),
            mask_rgba: vec![255; 8],
            skin_mask: vec![255; 2],
            roi,
            uniforms: WrinkleUniforms {
                legal_alpha: 1.0, dark_circle_alpha: 1.0, diff_factor: 1.0,
                diff_factor2: 0.0, color_threshold_factor: 1.0, mix_alpha: 1.0,
                diff_min_threshold: -1.0, diff_max_threshold: 1.0,
                high_freq_min_threshold: -1.0, high_freq_max_threshold: 1.0,
            },
        };
        for data in [
            HistoricalSkinData::Glabella(wrinkle.clone()),
            HistoricalSkinData::Forehead(wrinkle),
            HistoricalSkinData::BodySkin(vec![HistoricalSkinPass {
                mask: vec![255, 0], effect: model.clone(),
            }]),
            HistoricalSkinData::StretchMark {
                model, mask: vec![255], roi, alternate: false,
            },
            HistoricalSkinData::BodyTone {
                m: vec![0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                o: vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                w_m: 1.0,
                w_o: 1.0,
            },
        ] {
            let pf = data.pf();
            let inputs = payload(&source, data);
            let applied = apply(&source, Some(7), pf, 1.0, Some(&inputs)).unwrap().unwrap();
            assert_eq!(&applied.pixels[4..], &source.pixels[4..], "PF{pf}");
            assert_ne!(&applied.pixels[..3], &source.pixels[..3], "PF{pf}");
            if pf == 90086 {
                assert_eq!(&applied.pixels[..4], &[150, 75, 225, 77]);
            }
            if pf != 90008 {
                assert_eq!(applied.pixels[3], 77, "PF{pf}");
            }
        }
    }

    #[test]
    fn receipt_and_preview_fingerprint_bind_payload() {
        let source = Image::new(1, 1, vec![100, 150, 200, 77]).unwrap();
        let mut inputs = payload(&source, HistoricalSkinData::BodyTone {
            m: vec![0.0; 3], o: vec![0.0; 3], w_m: 1.0, w_o: 1.0,
        });
        assert_eq!(apply(&source, None, 90086, 0.0, None), Ok(None));
        assert_eq!(apply(&source, Some(7), 90086, 1.0, None),
            Err(Refusal::MissingInput("historical skin same-job payload")));
        assert_eq!(apply(&source, Some(8), 90086, 1.0, Some(&inputs)),
            Err(Refusal::ValueOutOfRange("historical skin ticket")));
        assert_eq!(apply(&source, Some(7), 90086, 0.8, Some(&inputs)),
            Err(Refusal::ValueOutOfRange("historical skin payload control")));
        assert_eq!(apply(&source, Some(7), 90086, 1.0, Some(&inputs)), Ok(None));
        let before = fingerprint(&inputs);
        let input = inputs.get_mut(&90086).unwrap();
        let HistoricalSkinData::BodyTone { m, .. } = &mut input.data else { unreachable!() };
        m[0] = 0.25;
        assert_ne!(fingerprint(&inputs), before);
        let mut next = source.clone();
        next.pixels[0] += 1;
        assert_eq!(apply(&next, Some(7), 90086, 1.0, Some(&inputs)),
            Err(Refusal::ValueOutOfRange("historical skin frame digest")));
    }
}
