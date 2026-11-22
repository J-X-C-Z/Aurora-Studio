use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use aurora_portrait::kernels::{face_detail, hair_body, pore, skin_adjust, tone_unify};
use aurora_portrait::{Image, Refusal};

use super::sha256_32;

/// One payload per PF, prepared against that effect's immediate input frame.
pub type HistoricalToneInputs = BTreeMap<u32, HistoricalToneInput>;

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneInput {
    pub ticket: u64,
    pub size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub control: f32,
    /// PF30015 and PF30016 controls for PF30014, in normalized units.
    pub gray_3d_controls: [f32; 2],
    pub data: HistoricalToneData,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HistoricalToneData {
    HairColor { rgb: Vec<u8>, mask: Vec<f32> },
    FlatGray { original: Image, mask: Vec<u8>, roi: skin_adjust::Roi },
    /// The planes are aligned 1:1 with the exact 512×512 renderer frame.
    Gray3d { light: Vec<f32>, dark: Vec<f32>, alpha: Vec<f32>, mask: Vec<f32> },
    Collarbone {
        model: Image,
        mask: Vec<u8>,
        judge: Vec<u8>,
        points: Vec<pore::Control>,
        face_present: bool,
        front_view: bool,
        roi: pore::Roi,
    },
    HandVeins {
        percent: i32,
        mask: Vec<u8>,
        rgb: Vec<u8>,
        feather: u32,
        roi: (u32, u32, u32, u32),
    },
    MultiTone(Vec<HistoricalToneFace>),
    HandGray { mask: Vec<u8>, alpha: Vec<u8>, target: Image },
    HandShape {
        native_parameter: i32,
        dx: Vec<f32>,
        dy: Vec<f32>,
        mask: Vec<u8>,
        protect: Vec<u8>,
        scale: Option<f32>,
    },
    EyeRedVeins { rgb: Vec<u8>, alpha: Vec<u8>, strength: f32 },
    BreastRepair {
        model: Image,
        mask: Vec<u8>,
        roi: skin_adjust::Roi,
        roi_value: f32,
        has_body_skeleton: bool,
        has_roi_points: bool,
        alternate: bool,
    },
}

/// Owned producer planes for one PF30059 face. Matrix entries are (matrix, weight).
#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneFace {
    pub group_id: i32,
    pub face_id: i32,
    pub reference: Option<i32>,
    pub target: Option<[u8; 3]>,
    pub brightness: Vec<u8>,
    pub skin: Vec<u8>,
    pub face: Vec<u8>,
    pub tooth: Vec<u8>,
    pub instance: Vec<u8>,
    pub tooth_alpha: f32,
    pub brightness_x: Vec<f32>,
    pub brightness_y: Vec<f32>,
    pub matrices: Vec<([f32; 12], f32)>,
    pub lut: Option<(Vec<f32>, usize)>,
}

impl HistoricalToneData {
    fn pf(&self) -> u32 {
        match self {
            Self::HairColor { .. } => 30009,
            Self::FlatGray { .. } => 30013,
            Self::Gray3d { .. } => 30014,
            Self::Collarbone { .. } => 30017,
            Self::HandVeins { .. } => 30030,
            Self::MultiTone(_) => 30059,
            Self::HandGray { .. } => 30187,
            Self::HandShape { .. } => 30190,
            Self::EyeRedVeins { .. } => 100004,
            Self::BreastRepair { .. } => 100008,
        }
    }
}

pub(super) fn apply(
    source: &Image,
    ticket: Option<u64>,
    pf: u32,
    control: f32,
    inputs: Option<&HistoricalToneInputs>,
    gray_3d_controls: [f32; 2],
) -> Result<Option<Image>, Refusal> {
    let min = if matches!(pf, 30009 | 30059) { -1.0 } else { 0.0 };
    if !(min..=1.0).contains(&control) {
        return Err(Refusal::ValueOutOfRange("historical tone control"));
    }
    if control == 0.0 {
        return Ok(None);
    }
    let input = inputs.and_then(|inputs| inputs.get(&pf))
        .ok_or(Refusal::MissingInput("historical tone same-job payload"))?;
    if input.data.pf() != pf {
        return Err(Refusal::ValueOutOfRange("historical tone payload PF"));
    }
    if input.ticket == 0 || Some(input.ticket) != ticket {
        return Err(Refusal::ValueOutOfRange("historical tone ticket"));
    }
    if input.size != [source.width, source.height] {
        return Err(Refusal::ShapeMismatch("historical tone frame size"));
    }
    if input.control != control || (pf == 30014 && input.gray_3d_controls != gray_3d_controls) {
        return Err(Refusal::ValueOutOfRange("historical tone payload control"));
    }
    if input.source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("historical tone frame digest"));
    }

    let ui = control * 100.0;
    let output = match &input.data {
        HistoricalToneData::HairColor { rgb, mask } =>
            hair_body::hair_color_unify(source, ui, rgb, mask),
        HistoricalToneData::FlatGray { original, mask, roi } =>
            skin_adjust::skin_grey_apply(original, source, mask, *roi, control),
        HistoricalToneData::Gray3d { light, dark, alpha, mask } => {
            apply_gray_3d(source, light, dark, alpha, mask, ui, gray_3d_controls)
        }
        HistoricalToneData::Collarbone { model, mask, judge, points, face_present, front_view, roi } =>
            pore::apply(source, model, mask, judge, points, ui, *face_present, *front_view, *roi),
        HistoricalToneData::HandVeins { percent, mask, rgb, feather, roi } =>
            hair_body::hand_blue_vein(source, *percent, mask, rgb, *feather, *roi),
        HistoricalToneData::MultiTone(faces) => apply_multi_tone(source, faces, ui),
        HistoricalToneData::HandGray { mask, alpha, target } =>
            hair_body::hand_neutral_gray(source, ui, mask, alpha, hair_body::NeutralTarget::Bgra(target)),
        HistoricalToneData::HandShape { native_parameter, dx, dy, mask, protect, scale } =>
            hair_body::hand_shape(source, *native_parameter, dx, dy, mask, protect, *scale),
        HistoricalToneData::EyeRedVeins { rgb, alpha, strength } =>
            face_detail::red_vein(source, rgb, alpha, ui, *strength),
        HistoricalToneData::BreastRepair { model, mask, roi, roi_value, has_body_skeleton, has_roi_points, alternate } => {
            let (_, active) = skin_adjust::breast_degree(ui, Some(*roi_value), *has_body_skeleton, *has_roi_points)?;
            if !active {
                return Ok(None);
            }
            skin_adjust::masked_model_output(source, model, *roi, mask, *alternate)
        }
    }?;
    Ok((output != *source).then_some(output))
}

fn apply_gray_3d(
    source: &Image,
    light: &[f32],
    dark: &[f32],
    alpha: &[f32],
    mask: &[f32],
    ui: f32,
    companions: [f32; 2],
) -> Result<Image, Refusal> {
    let side = skin_adjust::NEUTRAL_GRAY_3D_SIDE as u32;
    if [source.width, source.height] != [side; 2] {
        return Err(Refusal::ShapeMismatch("PF30014 aligned 512x512 frame"));
    }
    let degrees = skin_adjust::neutral_gray_3d_degrees(ui, companions[0] * 100.0, companions[1] * 100.0)?;
    let bgr: Vec<_> = source.pixels.chunks_exact(4).flat_map(|px| [px[2], px[1], px[0]]).collect();
    let result = skin_adjust::neutral_gray_3d_apply(&bgr, light, dark, alpha, mask, degrees)?;
    let mut output = source.clone();
    for (dst, src) in output.pixels.chunks_exact_mut(4).zip(result.chunks_exact(3)) {
        dst[..3].copy_from_slice(&[src[2], src[1], src[0]]);
    }
    Ok(output)
}

fn apply_multi_tone(source: &Image, faces: &[HistoricalToneFace], ui: f32) -> Result<Image, Refusal> {
    if faces.is_empty() {
        return Err(Refusal::MissingInput("PF30059 face passes"));
    }
    let matrices: Vec<Vec<_>> = faces.iter().map(|face| {
        face.matrices.iter().map(|&(matrix, weight)| tone_unify::WeightedMatrix { matrix, weight }).collect()
    }).collect();
    let passes: Vec<_> = faces.iter().zip(&matrices).map(|(face, matrices)| tone_unify::Pass30059 {
        group_id: face.group_id,
        face_id: face.face_id,
        explicit_reference: face.reference,
        target: face.target,
        brightness: &face.brightness,
        skin_mask: &face.skin,
        face_mask: &face.face,
        tooth_mask: &face.tooth,
        instance_mask: &face.instance,
        tooth_alpha: face.tooth_alpha,
        brightness_lut_x: &face.brightness_x,
        brightness_lut_y: &face.brightness_y,
        matrices,
        lut: face.lut.as_ref().map(|(values, grid)| (values.as_slice(), *grid)),
    }).collect();
    tone_unify::pf30059_apply(source, &passes, ui)
}

pub(super) fn fingerprint(inputs: &HistoricalToneInputs) -> u64 {
    let mut hash = DefaultHasher::new();
    for (pf, input) in inputs {
        (*pf, input.ticket, input.size, input.source_sha256, input.control.to_bits(), input.gray_3d_controls.map(f32::to_bits), input.data.pf())
            .hash(&mut hash);
        use HistoricalToneData::*;
        match &input.data {
            HairColor { rgb, mask } => {
                rgb.hash(&mut hash);
                floats(mask, &mut hash);
            }
            FlatGray { original, mask, roi } => {
                image(original, &mut hash);
                (mask, roi.x, roi.y, roi.width, roi.height).hash(&mut hash);
            }
            Gray3d { light, dark, alpha, mask } => {
                for plane in [light, dark, alpha, mask] {
                    floats(plane, &mut hash);
                }
            }
            Collarbone { model, mask, judge, points, face_present, front_view, roi } => {
                image(model, &mut hash);
                (mask, judge, face_present, front_view, roi.x, roi.y, roi.width, roi.height).hash(&mut hash);
                points.len().hash(&mut hash);
                for point in points {
                    floats(&[point.source.x, point.source.y, point.target.x, point.target.y, point.radius], &mut hash);
                }
            }
            HandVeins { percent, mask, rgb, feather, roi } =>
                (percent, mask, rgb, feather, roi).hash(&mut hash),
            MultiTone(faces) => {
                faces.len().hash(&mut hash);
                for face in faces {
                    (face.group_id, face.face_id, face.reference, face.target, face.tooth_alpha.to_bits()).hash(&mut hash);
                    (&face.brightness, &face.skin, &face.face, &face.tooth, &face.instance).hash(&mut hash);
                    floats(&face.brightness_x, &mut hash);
                    floats(&face.brightness_y, &mut hash);
                    face.matrices.len().hash(&mut hash);
                    for (matrix, weight) in &face.matrices {
                        floats(matrix, &mut hash);
                        weight.to_bits().hash(&mut hash);
                    }
                    face.lut.as_ref().map(|(_, grid)| *grid).hash(&mut hash);
                    if let Some((values, _)) = &face.lut {
                        floats(values, &mut hash);
                    }
                }
            }
            HandGray { mask, alpha, target } => {
                image(target, &mut hash);
                (mask, alpha).hash(&mut hash);
            }
            HandShape { native_parameter, dx, dy, mask, protect, scale } => {
                (native_parameter, mask, protect, scale.map(f32::to_bits)).hash(&mut hash);
                floats(dx, &mut hash);
                floats(dy, &mut hash);
            }
            EyeRedVeins { rgb, alpha, strength } =>
                (rgb, alpha, strength.to_bits()).hash(&mut hash),
            BreastRepair { model, mask, roi, roi_value, has_body_skeleton, has_roi_points, alternate } => {
                image(model, &mut hash);
                (mask, roi.x, roi.y, roi.width, roi.height, roi_value.to_bits(), has_body_skeleton, has_roi_points, alternate)
                    .hash(&mut hash);
            }
        }
    }
    hash.finish()
}

fn image(image: &Image, hash: &mut DefaultHasher) {
    (image.width, image.height, &image.pixels).hash(hash);
}

fn floats(values: &[f32], hash: &mut DefaultHasher) {
    values.len().hash(hash);
    for value in values {
        value.to_bits().hash(hash);
    }
}

pub(super) fn supports(pf: u32) -> bool {
    matches!(pf, 30009 | 30013 | 30014 | 30017 | 30030 | 30059 | 30187 | 30190 | 100004 | 100008)
}

pub(super) fn open_reason(pf: u32) -> Option<&'static str> {
    Some(match pf {
        30073 | 30079 => "lower-eyelid Type1011: prepared 100-point geometry and EBO are known; exact frame draw binding is open",
        30080 | 30081 => "forehead width Type1012: vertex deltas are known; exact frame draw binding is open",
        30082 | 30083 => "forehead height Type1012: vertex deltas are known; exact frame draw binding is open",
        30092 | 30093 => "face width: 414-point part geometry is known; exact frame draw binding is open",
        30188 => "double-chin degree: control-specific model field and final ROI writeback are open",
        30189 => "double-chin shadow: control-specific model output and shadow compositing are open",
        30191 => "forehead plump: control gate is known; FacialPlumpingPipeline final blend is open",
        30194 => "mid-face plump: control gate is known; FacialPlumpingPipeline final blend is open",
        30197 => "cheek plump: final owner-mask/model ROI compositing is open",
        30198 => "nasal-base plump: final displacement/protect-mask ROI consumer is open",
        100005 => "eye white purify: 4+4 split and mixer tail are known; model-to-layer RGB/mask binding is open",
        _ => return None,
    })
}


#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(crate) fn fixture() -> (Image, HistoricalToneInputs) {
        let source = Image::new(2, 1, vec![20, 60, 90, 27, 160, 100, 70, 81]).unwrap();
        let model = Image::solid(2, 1, [150, 40, 110, 27]);
        let roi = skin_adjust::Roi { x: 0, y: 0, width: 1, height: 1 };
        let data = [
            HistoricalToneData::HairColor { rgb: vec![100; 6], mask: vec![1.0, 0.0] },
            HistoricalToneData::FlatGray { original: model.clone(), mask: vec![255, 0], roi },
            HistoricalToneData::Collarbone {
                model: model.clone(), mask: vec![255, 0], judge: vec![255, 0], points: vec![],
                face_present: true, front_view: true,
                roi: pore::Roi { x: 0, y: 0, width: 1, height: 1 },
            },
            HistoricalToneData::HandVeins {
                percent: 80, mask: vec![255, 0], rgb: vec![120; 6], feather: 0, roi: (0, 0, 1, 1),
            },
            HistoricalToneData::MultiTone(vec![HistoricalToneFace {
                group_id: 1, face_id: 1, reference: None, target: Some([200, 120, 100]),
                brightness: vec![80; 2], skin: vec![255, 0], face: vec![255, 0],
                tooth: vec![0; 2], instance: vec![255, 0], tooth_alpha: 0.0,
                brightness_x: vec![0.0, 255.0], brightness_y: vec![1.0, 1.0], matrices: vec![], lut: None,
            }]),
            HistoricalToneData::HandGray { mask: vec![255, 0], alpha: vec![255; 2], target: model.clone() },
            HistoricalToneData::HandShape {
                native_parameter: 100, dx: vec![1.0, 0.0], dy: vec![0.0; 2],
                mask: vec![255, 0], protect: vec![0; 2], scale: Some(1.0),
            },
            HistoricalToneData::EyeRedVeins { rgb: vec![140; 6], alpha: vec![255, 0], strength: 1.0 },
            HistoricalToneData::BreastRepair {
                model, mask: vec![255], roi, roi_value: 80.0,
                has_body_skeleton: true, has_roi_points: true, alternate: false,
            },
        ];
        let inputs = data.into_iter().map(|data| (data.pf(), HistoricalToneInput {
            ticket: 7, size: [2, 1], source_sha256: sha256_32(&source.pixels),
            control: 0.8, gray_3d_controls: [0.0; 2], data,
        })).collect();
        (source, inputs)
    }

    #[test]
    fn historical_tone_cpu_matrix() {
        let (source, inputs) = fixture();
        for &pf in inputs.keys() {
            assert!(supports(pf));
            let render = |ticket, control, inputs| apply(&source, ticket, pf, control, inputs, [0.0; 2]);
            let output = render(Some(7), 0.8, Some(&inputs)).unwrap().unwrap();
            assert_ne!(output.pixel(0, 0), source.pixel(0, 0), "PF{pf}");
            assert_eq!(output.pixel(0, 0)[3], source.pixel(0, 0)[3], "PF{pf}");
            assert_eq!(output.pixel(1, 0), source.pixel(1, 0), "PF{pf}");
            assert_eq!(render(Some(8), 0.8, Some(&inputs)), Err(Refusal::ValueOutOfRange("historical tone ticket")));
            assert_eq!(render(Some(7), 0.7, Some(&inputs)), Err(Refusal::ValueOutOfRange("historical tone payload control")));
            assert_eq!(render(Some(7), 0.8, None), Err(Refusal::MissingInput("historical tone same-job payload")));
            assert_eq!(render(Some(7), 0.0, None), Ok(None));

            let mut stale = inputs.clone();
            stale.get_mut(&pf).unwrap().source_sha256[0] ^= 1;
            assert_ne!(fingerprint(&inputs), fingerprint(&stale));
            assert_eq!(render(Some(7), 0.8, Some(&stale)), Err(Refusal::ValueOutOfRange("historical tone frame digest")));
        }
        let mut malformed = inputs.clone();
        if let HistoricalToneData::HairColor { rgb, .. } = &mut malformed.get_mut(&30009).unwrap().data {
            rgb.pop();
        }
        assert_ne!(fingerprint(&inputs), fingerprint(&malformed));
        assert_eq!(apply(&source, Some(7), 30009, 0.8, Some(&malformed), [0.0; 2]),
            Err(Refusal::ShapeMismatch("PF30009_MODEL_EFFECT_RGB")));
    }

    #[test]
    fn historical_tone_gray_3d_aligned_frame_and_companion_gate() {
        let side = skin_adjust::NEUTRAL_GRAY_3D_SIDE as u32;
        let source = Image::solid(side, side, [20, 60, 90, 27]);
        let count = source.pixel_count();
        let mut light = vec![0.0; count * 3];
        light[0] = 0.5;
        let inputs = BTreeMap::from([(30014, HistoricalToneInput {
            ticket: 7, size: [side; 2], source_sha256: sha256_32(&source.pixels),
            control: 0.8, gray_3d_controls: [0.5, 0.0],
            data: HistoricalToneData::Gray3d {
                light, dark: vec![0.0; count * 3], alpha: vec![1.0; count], mask: vec![1.0; count],
            },
        })]);
        let output = apply(&source, Some(7), 30014, 0.8, Some(&inputs), [0.5, 0.0]).unwrap().unwrap();
        assert_eq!(output.pixel(0, 0), [20, 60, 154, 27]);
        assert_eq!(output.pixel(1, 0), source.pixel(1, 0));
        assert_eq!(apply(&source, Some(7), 30014, 0.8, Some(&inputs), [0.0; 2]),
            Err(Refusal::ValueOutOfRange("historical tone payload control")));
        let source = Image::solid(1, 1, [20, 60, 90, 27]);
        let mut inputs = inputs;
        let input = inputs.get_mut(&30014).unwrap();
        input.size = [1; 2];
        input.source_sha256 = sha256_32(&source.pixels);
        assert_eq!(apply(&source, Some(7), 30014, 0.8, Some(&inputs), [0.5, 0.0]),
            Err(Refusal::ShapeMismatch("PF30014 aligned 512x512 frame")));
    }
}
