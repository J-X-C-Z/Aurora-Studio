use std::collections::{BTreeMap, BTreeSet};

use aurora_portrait::kernels::{pore, skin_adjust};
use aurora_portrait::{Image, Refusal};

use crate::portrait_stage::{
    HistoricalToneData, HistoricalToneFace, HistoricalToneInput, HistoricalToneInputs,
};

pub const HISTORICAL_TONE_PROVIDER_PFS: [u32; 10] = [
    30009, 30013, 30014, 30017, 30030, 30059, 30187, 30190, 100004, 100008,
];

/// One row's binding to the frame it consumes.
///
/// The portrait stage walks the ordered stack and re-derives the frame digest,
/// the frame size and the control for every row, so these fields cannot be
/// shared between rows. Each row is bound to the frame that is input when that
/// row runs, which is the frame produced by every earlier executed row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoricalToneRowJob {
    pub source_sha256: [u8; 32],
    pub size: [u32; 2],
    pub control: f32,
}

/// One external row: its own frame binding plus owned model/material payloads.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneProviderRow {
    pub job: HistoricalToneRowJob,
    pub data: HistoricalToneRowData,
}

/// External payload for one historical-tone producer job.
///
/// Rows may be sparse when an effect is inactive. Present rows are fully owned
/// and are validated before they enter the normal portrait stage. `ticket` and
/// `gray_3d_controls` are genuinely job-scoped: the stage compares the ticket
/// against the job id for every row, and reads PF30015/PF30016 once per run.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneProviderPayload {
    pub ticket: u64,
    /// PF30015 and PF30016 controls consumed by PF30014.
    pub gray_3d_controls: [f32; 2],
    pub rows: Vec<HistoricalToneProviderRow>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum HistoricalToneRowData {
    HairColor(HistoricalToneHairColorPayload),
    FlatGray(HistoricalToneFlatGrayPayload),
    Gray3d(HistoricalToneGray3dPayload),
    Collarbone(HistoricalToneCollarbonePayload),
    HandVeins(HistoricalToneHandVeinsPayload),
    MultiTone(HistoricalToneMultiTonePayload),
    HandGray(HistoricalToneHandGrayPayload),
    HandShape(HistoricalToneHandShapePayload),
    EyeRedVeins(HistoricalToneEyeRedVeinsPayload),
    BreastRepair(HistoricalToneBreastRepairPayload),
}

impl HistoricalToneRowData {
    pub fn pf(&self) -> u32 {
        match self {
            Self::HairColor(_) => 30009,
            Self::FlatGray(_) => 30013,
            Self::Gray3d(_) => 30014,
            Self::Collarbone(_) => 30017,
            Self::HandVeins(_) => 30030,
            Self::MultiTone(_) => 30059,
            Self::HandGray(_) => 30187,
            Self::HandShape(_) => 30190,
            Self::EyeRedVeins(_) => 100004,
            Self::BreastRepair(_) => 100008,
        }
    }
}

impl HistoricalToneProviderRow {
    pub fn pf(&self) -> u32 {
        self.data.pf()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneHairColorPayload {
    pub effect_rgb: Vec<u8>,
    pub hair_mask: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneFlatGrayPayload {
    pub original: Image,
    pub mask: Vec<u8>,
    pub roi: skin_adjust::Roi,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneGray3dPayload {
    pub light: Vec<f32>,
    pub dark: Vec<f32>,
    pub alpha: Vec<f32>,
    pub mask: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneCollarbonePayload {
    pub model: Image,
    pub mask: Vec<u8>,
    pub judge: Vec<u8>,
    pub points: Vec<pore::Control>,
    pub face_present: bool,
    pub front_view: bool,
    pub roi: pore::Roi,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneHandVeinsPayload {
    pub percent: i32,
    pub mask: Vec<u8>,
    pub correction_rgb: Vec<u8>,
    pub feather: u32,
    pub roi: (u32, u32, u32, u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneMultiTonePayload {
    pub faces: Vec<HistoricalToneFace>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneHandGrayPayload {
    pub mask: Vec<u8>,
    pub alpha: Vec<u8>,
    pub target: Image,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneHandShapePayload {
    pub native_parameter: i32,
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
    pub mask: Vec<u8>,
    pub protect: Vec<u8>,
    pub scale: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneEyeRedVeinsPayload {
    pub effect_rgb: Vec<u8>,
    pub alpha: Vec<u8>,
    pub strength: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalToneBreastRepairPayload {
    pub model: Image,
    pub roi_mask: Vec<u8>,
    pub roi: skin_adjust::Roi,
    pub roi_value: f32,
    pub has_body_skeleton: bool,
    pub has_roi_points: bool,
    pub alternate: bool,
}

/// Builds the owned row map for all present external rows.
///
/// This is the production entry. Missing fields are rejected by name; no row is
/// fabricated and no plane is defaulted. Every row carries its own frame binding
/// because the ordered stage re-derives the frame digest, frame size and control
/// per row, so two rows can never share one binding unless an earlier row was a
/// no-op.
pub fn build_historical_tone_inputs(
    payload: HistoricalToneProviderPayload,
) -> Result<HistoricalToneInputs, Refusal> {
    let HistoricalToneProviderPayload {
        ticket,
        gray_3d_controls,
        rows,
    } = payload;
    if ticket == 0 {
        return Err(Refusal::MissingInput("historical tone provider ticket"));
    }
    if rows.is_empty() {
        return Err(Refusal::MissingInput("historical tone provider rows"));
    }
    if gray_3d_controls.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(
            "historical tone provider PF30014 companion controls",
        ));
    }

    let mut inputs = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for row in rows {
        let pf = row.pf();
        if !HISTORICAL_TONE_PROVIDER_PFS.contains(&pf) {
            return Err(Refusal::UnknownEffect(pf));
        }
        if !seen.insert(pf) {
            return Err(Refusal::ValueOutOfRange(
                "historical tone provider duplicate PF row",
            ));
        }
        let HistoricalToneProviderRow { job, data } = row;
        let HistoricalToneRowJob {
            source_sha256,
            size,
            control,
        } = job;
        let names = row_names(pf);
        if source_sha256 == [0; 32] {
            return Err(Refusal::MissingInput(names.frame_sha));
        }
        let frame_pixels = checked_pixels(size, names.frame_size)?;
        if !control.is_finite() || !(-1.0..=1.0).contains(&control) {
            return Err(Refusal::ValueOutOfRange(names.control));
        }
        // The stage only admits a negative degree for the two signed rows.
        if control < 0.0 && !matches!(pf, 30009 | 30059) {
            return Err(Refusal::ValueOutOfRange(names.unsigned_control));
        }
        let data = build_row(pf, size, frame_pixels, data)?;
        inputs.insert(
            pf,
            HistoricalToneInput {
                ticket,
                size,
                source_sha256,
                control,
                gray_3d_controls,
                data,
            },
        );
    }
    Ok(inputs)
}

/// Complete variant used by a producer that must install all ten closed rows.
///
/// Each row still binds its own frame, so installing all ten only works when the
/// caller can name the frame each row consumes at its position in the ordered
/// stack.
pub fn build_complete_historical_tone_inputs(
    payload: HistoricalToneProviderPayload,
) -> Result<HistoricalToneInputs, Refusal> {
    let inputs = build_historical_tone_inputs(payload)?;
    for pf in HISTORICAL_TONE_PROVIDER_PFS {
        if !inputs.contains_key(&pf) {
            return Err(Refusal::MissingInput(missing_row(pf)));
        }
    }
    Ok(inputs)
}

fn build_row(
    pf: u32,
    size: [u32; 2],
    frame_pixels: usize,
    data: HistoricalToneRowData,
) -> Result<HistoricalToneData, Refusal> {
    match (pf, data) {
        (
            30009,
            HistoricalToneRowData::HairColor(HistoricalToneHairColorPayload {
                effect_rgb,
                hair_mask,
            }),
        ) => {
            require_nonempty(&effect_rgb, "historical tone provider PF30009 effect rgb")?;
            require_nonempty(&hair_mask, "historical tone provider PF30009 hair mask")?;
            require_len(
                &effect_rgb,
                frame_pixels * 3,
                "historical tone provider PF30009 effect rgb shape",
            )?;
            require_len(
                &hair_mask,
                frame_pixels,
                "historical tone provider PF30009 hair mask shape",
            )?;
            require_finite(&hair_mask, "historical tone provider PF30009 hair mask")?;
            Ok(HistoricalToneData::HairColor {
                rgb: effect_rgb,
                mask: hair_mask,
            })
        }
        (
            30013,
            HistoricalToneRowData::FlatGray(HistoricalToneFlatGrayPayload {
                original,
                mask,
                roi,
            }),
        ) => {
            require_image(&original, "historical tone provider PF30013 original")?;
            require_image_shape(
                &original,
                size,
                "historical tone provider PF30013 original shape",
            )?;
            require_nonempty(&mask, "historical tone provider PF30013 mask")?;
            require_len(
                &mask,
                frame_pixels,
                "historical tone provider PF30013 mask shape",
            )?;
            require_roi(
                (roi.x, roi.y, roi.width, roi.height),
                size,
                "historical tone provider PF30013 ROI",
            )?;
            Ok(HistoricalToneData::FlatGray {
                original,
                mask,
                roi,
            })
        }
        (
            30014,
            HistoricalToneRowData::Gray3d(HistoricalToneGray3dPayload {
                light,
                dark,
                alpha,
                mask,
            }),
        ) => {
            let side = skin_adjust::NEUTRAL_GRAY_3D_SIDE as u32;
            if size != [side; 2] {
                return Err(Refusal::ShapeMismatch(
                    "historical tone provider PF30014 aligned 512x512 frame",
                ));
            }
            let count = frame_pixels;
            for (plane, name) in [
                (&light, "historical tone provider PF30014 light"),
                (&dark, "historical tone provider PF30014 dark"),
            ] {
                require_nonempty(plane, name)?;
                require_len(plane, count * 3, name)?;
                require_finite(plane, name)?;
            }
            for (plane, name) in [
                (&alpha, "historical tone provider PF30014 alpha"),
                (&mask, "historical tone provider PF30014 mask"),
            ] {
                require_nonempty(plane, name)?;
                require_len(plane, count, name)?;
                require_finite(plane, name)?;
            }
            Ok(HistoricalToneData::Gray3d {
                light,
                dark,
                alpha,
                mask,
            })
        }
        (
            30017,
            HistoricalToneRowData::Collarbone(HistoricalToneCollarbonePayload {
                model,
                mask,
                judge,
                points,
                face_present,
                front_view,
                roi,
            }),
        ) => {
            require_image(&model, "historical tone provider PF30017 model")?;
            require_image_shape(&model, size, "historical tone provider PF30017 model shape")?;
            for (plane, name) in [
                (&mask, "historical tone provider PF30017 mask"),
                (&judge, "historical tone provider PF30017 judge"),
            ] {
                require_nonempty(plane, name)?;
                require_len(plane, frame_pixels, name)?;
            }
            require_pore_points(&points)?;
            require_roi(
                (roi.x, roi.y, roi.width, roi.height),
                size,
                "historical tone provider PF30017 ROI",
            )?;
            Ok(HistoricalToneData::Collarbone {
                model,
                mask,
                judge,
                points,
                face_present,
                front_view,
                roi,
            })
        }
        (
            30030,
            HistoricalToneRowData::HandVeins(HistoricalToneHandVeinsPayload {
                percent,
                mask,
                correction_rgb,
                feather,
                roi,
            }),
        ) => {
            if !(0..=100).contains(&percent) {
                return Err(Refusal::ValueOutOfRange(
                    "historical tone provider PF30030 percent",
                ));
            }
            require_nonempty(&mask, "historical tone provider PF30030 vein mask")?;
            require_len(
                &mask,
                frame_pixels,
                "historical tone provider PF30030 vein mask shape",
            )?;
            require_nonempty(
                &correction_rgb,
                "historical tone provider PF30030 correction rgb",
            )?;
            require_len(
                &correction_rgb,
                frame_pixels * 3,
                "historical tone provider PF30030 correction rgb shape",
            )?;
            require_roi(roi, size, "historical tone provider PF30030 ROI")?;
            Ok(HistoricalToneData::HandVeins {
                percent,
                mask,
                rgb: correction_rgb,
                feather,
                roi,
            })
        }
        (30059, HistoricalToneRowData::MultiTone(HistoricalToneMultiTonePayload { faces })) => {
            if faces.is_empty() {
                return Err(Refusal::MissingInput(
                    "historical tone provider PF30059 face rows",
                ));
            }
            for face in &faces {
                validate_multi_tone_face(face, frame_pixels)?;
            }
            Ok(HistoricalToneData::MultiTone(faces))
        }
        (
            30187,
            HistoricalToneRowData::HandGray(HistoricalToneHandGrayPayload {
                mask,
                alpha,
                target,
            }),
        ) => {
            for (plane, name) in [
                (&mask, "historical tone provider PF30187 hand mask"),
                (&alpha, "historical tone provider PF30187 model alpha"),
            ] {
                require_nonempty(plane, name)?;
                require_len(plane, frame_pixels, name)?;
            }
            require_image(&target, "historical tone provider PF30187 neutral target")?;
            require_image_shape(
                &target,
                size,
                "historical tone provider PF30187 neutral target shape",
            )?;
            Ok(HistoricalToneData::HandGray {
                mask,
                alpha,
                target,
            })
        }
        (
            30190,
            HistoricalToneRowData::HandShape(HistoricalToneHandShapePayload {
                native_parameter,
                dx,
                dy,
                mask,
                protect,
                scale,
            }),
        ) => {
            for (plane, name) in [
                (&dx, "historical tone provider PF30190 dx"),
                (&dy, "historical tone provider PF30190 dy"),
            ] {
                require_nonempty(plane, name)?;
                require_len(plane, frame_pixels, name)?;
                require_finite(plane, name)?;
            }
            for (plane, name) in [
                (&mask, "historical tone provider PF30190 hand mask"),
                (&protect, "historical tone provider PF30190 protect mask"),
            ] {
                require_nonempty(plane, name)?;
                require_len(plane, frame_pixels, name)?;
            }
            if scale.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                return Err(Refusal::ValueOutOfRange(
                    "historical tone provider PF30190 source scale",
                ));
            }
            Ok(HistoricalToneData::HandShape {
                native_parameter,
                dx,
                dy,
                mask,
                protect,
                scale,
            })
        }
        (
            100004,
            HistoricalToneRowData::EyeRedVeins(HistoricalToneEyeRedVeinsPayload {
                effect_rgb,
                alpha,
                strength,
            }),
        ) => {
            require_nonempty(&effect_rgb, "historical tone provider PF100004 effect rgb")?;
            require_len(
                &effect_rgb,
                frame_pixels * 3,
                "historical tone provider PF100004 effect rgb shape",
            )?;
            require_nonempty(&alpha, "historical tone provider PF100004 alpha")?;
            require_len(
                &alpha,
                frame_pixels,
                "historical tone provider PF100004 alpha shape",
            )?;
            if !strength.is_finite() {
                return Err(Refusal::ValueOutOfRange(
                    "historical tone provider PF100004 strength",
                ));
            }
            Ok(HistoricalToneData::EyeRedVeins {
                rgb: effect_rgb,
                alpha,
                strength,
            })
        }
        (
            100008,
            HistoricalToneRowData::BreastRepair(HistoricalToneBreastRepairPayload {
                model,
                roi_mask,
                roi,
                roi_value,
                has_body_skeleton,
                has_roi_points,
                alternate,
            }),
        ) => {
            require_image(&model, "historical tone provider PF100008 model")?;
            require_image_shape(
                &model,
                size,
                "historical tone provider PF100008 model shape",
            )?;
            require_roi(
                (roi.x, roi.y, roi.width, roi.height),
                size,
                "historical tone provider PF100008 ROI",
            )?;
            let roi_pixels = usize::try_from(u64::from(roi.width) * u64::from(roi.height))
                .map_err(|_| {
                    Refusal::ShapeMismatch("historical tone provider PF100008 ROI shape")
                })?;
            require_nonempty(&roi_mask, "historical tone provider PF100008 ROI mask")?;
            require_len(
                &roi_mask,
                roi_pixels,
                "historical tone provider PF100008 ROI mask shape",
            )?;
            if !roi_value.is_finite() {
                return Err(Refusal::ValueOutOfRange(
                    "historical tone provider PF100008 ROI value",
                ));
            }
            Ok(HistoricalToneData::BreastRepair {
                model,
                mask: roi_mask,
                roi,
                roi_value,
                has_body_skeleton,
                has_roi_points,
                alternate,
            })
        }
        _ => Err(Refusal::ValueOutOfRange(
            "historical tone provider row/data PF",
        )),
    }
}

fn validate_multi_tone_face(face: &HistoricalToneFace, pixels: usize) -> Result<(), Refusal> {
    for (plane, name) in [
        (
            &face.brightness,
            "historical tone provider PF30059 brightness",
        ),
        (&face.skin, "historical tone provider PF30059 skin mask"),
        (&face.face, "historical tone provider PF30059 face mask"),
        (&face.tooth, "historical tone provider PF30059 tooth mask"),
        (
            &face.instance,
            "historical tone provider PF30059 instance mask",
        ),
    ] {
        require_nonempty(plane, name)?;
        require_len(plane, pixels, name)?;
    }
    if !face.tooth_alpha.is_finite() || !(0.0..=1.0).contains(&face.tooth_alpha) {
        return Err(Refusal::ValueOutOfRange(
            "historical tone provider PF30059 tooth alpha",
        ));
    }
    if face.brightness_x.is_empty() != face.brightness_y.is_empty() {
        return Err(Refusal::MissingInput(
            "historical tone provider PF30059 brightness LUT x or y",
        ));
    }
    if face.brightness_x.len() != face.brightness_y.len() {
        return Err(Refusal::ShapeMismatch(
            "historical tone provider PF30059 brightness LUT shape",
        ));
    }
    require_finite(
        &face.brightness_x,
        "historical tone provider PF30059 brightness LUT x",
    )?;
    require_finite(
        &face.brightness_y,
        "historical tone provider PF30059 brightness LUT y",
    )?;

    let mut weight_sum = 0.0;
    for (matrix, weight) in &face.matrices {
        if !weight.is_finite() || *weight < 0.0 || matrix.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange(
                "historical tone provider PF30059 matrix",
            ));
        }
        weight_sum += *weight;
    }
    if !face.matrices.is_empty() && weight_sum <= 0.0 {
        return Err(Refusal::ValueOutOfRange(
            "historical tone provider PF30059 matrix weight sum",
        ));
    }

    if let Some((values, grid)) = &face.lut {
        if *grid < 2 {
            return Err(Refusal::ValueOutOfRange(
                "historical tone provider PF30059 LUT grid",
            ));
        }
        let expected = grid
            .checked_mul(*grid)
            .and_then(|value| value.checked_mul(*grid))
            .and_then(|value| value.checked_mul(3))
            .ok_or(Refusal::ShapeMismatch(
                "historical tone provider PF30059 LUT shape",
            ))?;
        require_len(
            values,
            expected,
            "historical tone provider PF30059 LUT shape",
        )?;
        require_finite(values, "historical tone provider PF30059 LUT")?;
    }
    Ok(())
}

fn checked_pixels(size: [u32; 2], name: &'static str) -> Result<usize, Refusal> {
    if size[0] == 0 || size[1] == 0 {
        return Err(Refusal::MissingInput(name));
    }
    usize::try_from(u64::from(size[0]) * u64::from(size[1]))
        .map_err(|_| Refusal::ShapeMismatch(name))
}

fn require_nonempty<T>(values: &[T], name: &'static str) -> Result<(), Refusal> {
    if values.is_empty() {
        Err(Refusal::MissingInput(name))
    } else {
        Ok(())
    }
}

fn require_len<T>(values: &[T], expected: usize, name: &'static str) -> Result<(), Refusal> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(name))
    }
}

fn require_finite(values: &[f32], name: &'static str) -> Result<(), Refusal> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn require_image(image: &Image, name: &'static str) -> Result<(), Refusal> {
    if image.width == 0 || image.height == 0 || image.pixels.is_empty() {
        Err(Refusal::MissingInput(name))
    } else {
        Ok(())
    }
}

fn require_image_shape(image: &Image, size: [u32; 2], name: &'static str) -> Result<(), Refusal> {
    let expected = checked_pixels(size, name)?
        .checked_mul(4)
        .ok_or(Refusal::ShapeMismatch(name))?;
    if [image.width, image.height] == size && image.pixels.len() == expected {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(name))
    }
}

fn require_roi(
    roi: (u32, u32, u32, u32),
    size: [u32; 2],
    name: &'static str,
) -> Result<(), Refusal> {
    let (x, y, width, height) = roi;
    let inside = width > 0
        && height > 0
        && u64::from(x) + u64::from(width) <= u64::from(size[0])
        && u64::from(y) + u64::from(height) <= u64::from(size[1]);
    if inside {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn require_pore_points(points: &[pore::Control]) -> Result<(), Refusal> {
    for point in points {
        let values = [
            point.source.x,
            point.source.y,
            point.target.x,
            point.target.y,
            point.radius,
        ];
        if values.iter().any(|value| !value.is_finite()) || point.radius < 0.0 {
            return Err(Refusal::ValueOutOfRange(
                "historical tone provider PF30017 control point",
            ));
        }
    }
    Ok(())
}

/// Per-row refusal names, so a rejected frame binding names its owning PF.
#[derive(Clone, Copy, Debug)]
struct RowNames {
    frame_sha: &'static str,
    frame_size: &'static str,
    control: &'static str,
    unsigned_control: &'static str,
}

fn row_names(pf: u32) -> RowNames {
    match pf {
        30009 => RowNames {
            frame_sha: "historical tone provider PF30009 frame SHA-256",
            frame_size: "historical tone provider PF30009 frame size",
            control: "historical tone provider PF30009 control",
            unsigned_control: "historical tone provider PF30009 control for unsigned row",
        },
        30013 => RowNames {
            frame_sha: "historical tone provider PF30013 frame SHA-256",
            frame_size: "historical tone provider PF30013 frame size",
            control: "historical tone provider PF30013 control",
            unsigned_control: "historical tone provider PF30013 control for unsigned row",
        },
        30014 => RowNames {
            frame_sha: "historical tone provider PF30014 frame SHA-256",
            frame_size: "historical tone provider PF30014 frame size",
            control: "historical tone provider PF30014 control",
            unsigned_control: "historical tone provider PF30014 control for unsigned row",
        },
        30017 => RowNames {
            frame_sha: "historical tone provider PF30017 frame SHA-256",
            frame_size: "historical tone provider PF30017 frame size",
            control: "historical tone provider PF30017 control",
            unsigned_control: "historical tone provider PF30017 control for unsigned row",
        },
        30030 => RowNames {
            frame_sha: "historical tone provider PF30030 frame SHA-256",
            frame_size: "historical tone provider PF30030 frame size",
            control: "historical tone provider PF30030 control",
            unsigned_control: "historical tone provider PF30030 control for unsigned row",
        },
        30059 => RowNames {
            frame_sha: "historical tone provider PF30059 frame SHA-256",
            frame_size: "historical tone provider PF30059 frame size",
            control: "historical tone provider PF30059 control",
            unsigned_control: "historical tone provider PF30059 control for unsigned row",
        },
        30187 => RowNames {
            frame_sha: "historical tone provider PF30187 frame SHA-256",
            frame_size: "historical tone provider PF30187 frame size",
            control: "historical tone provider PF30187 control",
            unsigned_control: "historical tone provider PF30187 control for unsigned row",
        },
        30190 => RowNames {
            frame_sha: "historical tone provider PF30190 frame SHA-256",
            frame_size: "historical tone provider PF30190 frame size",
            control: "historical tone provider PF30190 control",
            unsigned_control: "historical tone provider PF30190 control for unsigned row",
        },
        100004 => RowNames {
            frame_sha: "historical tone provider PF100004 frame SHA-256",
            frame_size: "historical tone provider PF100004 frame size",
            control: "historical tone provider PF100004 control",
            unsigned_control: "historical tone provider PF100004 control for unsigned row",
        },
        _ => RowNames {
            frame_sha: "historical tone provider PF100008 frame SHA-256",
            frame_size: "historical tone provider PF100008 frame size",
            control: "historical tone provider PF100008 control",
            unsigned_control: "historical tone provider PF100008 control for unsigned row",
        },
    }
}

fn missing_row(pf: u32) -> &'static str {
    match pf {
        30009 => "historical tone provider missing PF30009 row",
        30013 => "historical tone provider missing PF30013 row",
        30014 => "historical tone provider missing PF30014 row",
        30017 => "historical tone provider missing PF30017 row",
        30030 => "historical tone provider missing PF30030 row",
        30059 => "historical tone provider missing PF30059 row",
        30187 => "historical tone provider missing PF30187 row",
        30190 => "historical tone provider missing PF30190 row",
        100004 => "historical tone provider missing PF100004 row",
        100008 => "historical tone provider missing PF100008 row",
        _ => "historical tone provider missing row",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: [u8; 32] = [7; 32];

    fn image(size: [u32; 2]) -> Image {
        Image::solid(size[0], size[1], [40, 80, 120, 255])
    }

    fn row_data(pf: u32, size: [u32; 2]) -> HistoricalToneRowData {
        let pixels = (size[0] * size[1]) as usize;
        let rgb = vec![90; pixels * 3];
        let mask = vec![255; pixels];
        let float_mask = vec![1.0; pixels];
        let roi = skin_adjust::Roi {
            x: 0,
            y: 0,
            width: size[0].min(1),
            height: size[1].min(1),
        };
        match pf {
            30009 => HistoricalToneRowData::HairColor(HistoricalToneHairColorPayload {
                effect_rgb: rgb,
                hair_mask: float_mask,
            }),
            30013 => HistoricalToneRowData::FlatGray(HistoricalToneFlatGrayPayload {
                original: image(size),
                mask,
                roi,
            }),
            30014 => HistoricalToneRowData::Gray3d(HistoricalToneGray3dPayload {
                light: vec![0.0; pixels * 3],
                dark: vec![0.0; pixels * 3],
                alpha: vec![1.0; pixels],
                mask: float_mask,
            }),
            30017 => HistoricalToneRowData::Collarbone(HistoricalToneCollarbonePayload {
                model: image(size),
                mask,
                judge: vec![255; pixels],
                points: Vec::new(),
                face_present: true,
                front_view: true,
                roi: pore::Roi {
                    x: 0,
                    y: 0,
                    width: size[0].min(1),
                    height: size[1].min(1),
                },
            }),
            30030 => HistoricalToneRowData::HandVeins(HistoricalToneHandVeinsPayload {
                percent: 80,
                mask,
                correction_rgb: rgb,
                feather: 0,
                roi: (0, 0, size[0], size[1]),
            }),
            30059 => HistoricalToneRowData::MultiTone(HistoricalToneMultiTonePayload {
                faces: vec![HistoricalToneFace {
                    group_id: 1,
                    face_id: 1,
                    reference: None,
                    target: Some([200, 120, 100]),
                    brightness: vec![80; pixels],
                    skin: vec![255; pixels],
                    face: vec![255; pixels],
                    tooth: vec![0; pixels],
                    instance: vec![255; pixels],
                    tooth_alpha: 0.0,
                    brightness_x: vec![0.0, 255.0],
                    brightness_y: vec![1.0, 1.0],
                    matrices: Vec::new(),
                    lut: None,
                }],
            }),
            30187 => HistoricalToneRowData::HandGray(HistoricalToneHandGrayPayload {
                mask,
                alpha: vec![255; pixels],
                target: image(size),
            }),
            30190 => HistoricalToneRowData::HandShape(HistoricalToneHandShapePayload {
                native_parameter: 50,
                dx: vec![1.0; pixels],
                dy: vec![0.0; pixels],
                mask,
                protect: vec![0; pixels],
                scale: Some(1.0),
            }),
            100004 => HistoricalToneRowData::EyeRedVeins(HistoricalToneEyeRedVeinsPayload {
                effect_rgb: rgb,
                alpha: vec![255; pixels],
                strength: 1.0,
            }),
            100008 => HistoricalToneRowData::BreastRepair(HistoricalToneBreastRepairPayload {
                model: image(size),
                roi_mask: vec![255],
                roi,
                roi_value: 80.0,
                has_body_skeleton: true,
                has_roi_points: true,
                alternate: false,
            }),
            _ => panic!("unknown test PF"),
        }
    }

    fn job(sha: [u8; 32], size: [u32; 2], control: f32) -> HistoricalToneRowJob {
        HistoricalToneRowJob {
            source_sha256: sha,
            size,
            control,
        }
    }

    fn row(pf: u32, sha: [u8; 32], size: [u32; 2], control: f32) -> HistoricalToneProviderRow {
        HistoricalToneProviderRow {
            job: job(sha, size, control),
            data: row_data(pf, size),
        }
    }

    fn row_at(pf: u32, size: [u32; 2]) -> HistoricalToneProviderRow {
        row(pf, SHA, size, 0.8)
    }

    fn payload(rows: Vec<HistoricalToneProviderRow>) -> HistoricalToneProviderPayload {
        HistoricalToneProviderPayload {
            ticket: 17,
            gray_3d_controls: [0.0, 0.0],
            rows,
        }
    }

    fn row_size(pf: u32) -> [u32; 2] {
        if pf == 30014 {
            [skin_adjust::NEUTRAL_GRAY_3D_SIDE as u32; 2]
        } else {
            [2, 1]
        }
    }

    #[test]
    fn each_of_the_ten_external_rows_binds_job_fields_and_owned_data() {
        for pf in HISTORICAL_TONE_PROVIDER_PFS {
            let size = row_size(pf);
            let inputs = build_historical_tone_inputs(payload(vec![row_at(pf, size)])).unwrap();
            let input = inputs.get(&pf).expect("built row");
            assert_eq!(input.ticket, 17);
            assert_eq!(input.size, size);
            assert_eq!(input.source_sha256, SHA);
            assert_eq!(input.control, 0.8);
            assert_eq!(input.data.pf(), pf);
        }
    }

    #[test]
    fn rows_carry_independent_frame_bindings() {
        // Two ordered rows: the second consumes the frame the first produces, so
        // digest, size and control all differ and must survive the builder.
        let second_sha = [9; 32];
        let inputs = build_historical_tone_inputs(payload(vec![
            row(30009, SHA, [2, 1], 0.8),
            row(30059, second_sha, [1, 2], -0.4),
        ]))
        .unwrap();
        let first = inputs.get(&30009).expect("first row");
        let second = inputs.get(&30059).expect("second row");
        assert_eq!(first.source_sha256, SHA);
        assert_eq!(first.size, [2, 1]);
        assert_eq!(first.control, 0.8);
        assert_eq!(second.source_sha256, second_sha);
        assert_eq!(second.size, [1, 2]);
        assert_eq!(second.control, -0.4);
        assert_ne!(first.source_sha256, second.source_sha256);
    }

    #[test]
    fn complete_builder_names_each_absent_row() {
        let size = [2, 1];
        let error =
            build_complete_historical_tone_inputs(payload(vec![row_at(30009, size)])).unwrap_err();
        assert_eq!(
            error,
            Refusal::MissingInput("historical tone provider missing PF30013 row")
        );
    }

    #[test]
    fn job_fields_and_missing_material_are_named_refusals() {
        let size = [2, 1];
        for (mut payload, expected) in [
            (
                payload(vec![row_at(30009, size)]),
                Refusal::MissingInput("historical tone provider ticket"),
            ),
            (
                payload(Vec::new()),
                Refusal::MissingInput("historical tone provider rows"),
            ),
            (
                payload(vec![row_at(30009, size)]),
                Refusal::ValueOutOfRange("historical tone provider PF30014 companion controls"),
            ),
        ] {
            match &expected {
                Refusal::MissingInput("historical tone provider ticket") => payload.ticket = 0,
                Refusal::MissingInput("historical tone provider rows") => payload.rows.clear(),
                Refusal::ValueOutOfRange("historical tone provider PF30014 companion controls") => {
                    payload.gray_3d_controls = [f32::NAN, 0.0]
                }
                _ => unreachable!(),
            }
            assert_eq!(build_historical_tone_inputs(payload), Err(expected));
        }

        let empty_rgb = HistoricalToneProviderRow {
            job: job(SHA, size, 0.8),
            data: HistoricalToneRowData::HairColor(HistoricalToneHairColorPayload {
                effect_rgb: Vec::new(),
                hair_mask: vec![1.0; 2],
            }),
        };
        assert_eq!(
            build_historical_tone_inputs(payload(vec![empty_rgb])),
            Err(Refusal::MissingInput(
                "historical tone provider PF30009 effect rgb"
            ))
        );
    }

    #[test]
    fn each_row_frame_binding_is_validated_by_name() {
        let size = [2, 1];
        // A zero digest is a missing binding, named with the owning PF.
        assert_eq!(
            build_historical_tone_inputs(payload(vec![row(30013, [0; 32], size, 0.8)])),
            Err(Refusal::MissingInput(
                "historical tone provider PF30013 frame SHA-256"
            ))
        );
        // A zero frame size is refused and names the owning PF.
        assert_eq!(
            build_historical_tone_inputs(payload(vec![row(30030, SHA, [0, 0], 0.8)])).unwrap_err(),
            Refusal::MissingInput("historical tone provider PF30030 frame size")
        );
        // Per-row control range is checked, not shared across rows.
        assert_eq!(
            build_historical_tone_inputs(payload(vec![row(30017, SHA, size, 1.5)])).unwrap_err(),
            Refusal::ValueOutOfRange("historical tone provider PF30017 control")
        );
        // Only PF30009 and PF30059 accept a negative (signed) control.
        assert_eq!(
            build_historical_tone_inputs(payload(vec![row(30187, SHA, size, -0.5)])).unwrap_err(),
            Refusal::ValueOutOfRange("historical tone provider PF30187 control for unsigned row")
        );
        assert!(build_historical_tone_inputs(payload(vec![row(30009, SHA, size, -0.5)])).is_ok());
        assert!(build_historical_tone_inputs(payload(vec![row(30059, SHA, size, -0.5)])).is_ok());
    }

    #[test]
    fn duplicate_rows_and_malformed_pf30059_lut_are_refused() {
        let size = [2, 1];
        assert_eq!(
            build_historical_tone_inputs(payload(vec![row_at(30009, size), row_at(30009, size),])),
            Err(Refusal::ValueOutOfRange(
                "historical tone provider duplicate PF row"
            ))
        );

        let mut row = row_at(30059, size);
        let HistoricalToneRowData::MultiTone(multi) = &mut row.data else {
            unreachable!()
        };
        multi.faces[0].lut = Some((vec![0.0; 12], 2));
        assert_eq!(
            build_historical_tone_inputs(payload(vec![row])),
            Err(Refusal::ShapeMismatch(
                "historical tone provider PF30059 LUT shape"
            ))
        );
        // grid below 2 and non-finite LUT values stay named refusals.
        let mut grid_row = row_at(30059, size);
        let HistoricalToneRowData::MultiTone(multi) = &mut grid_row.data else {
            unreachable!()
        };
        multi.faces[0].lut = Some((vec![0.0; 3], 1));
        assert_eq!(
            build_historical_tone_inputs(payload(vec![grid_row])),
            Err(Refusal::ValueOutOfRange(
                "historical tone provider PF30059 LUT grid"
            ))
        );

        let mut half_lut = row_at(30059, size);
        let HistoricalToneRowData::MultiTone(multi) = &mut half_lut.data else {
            unreachable!()
        };
        multi.faces[0].brightness_y = Vec::new();
        assert_eq!(
            build_historical_tone_inputs(payload(vec![half_lut])),
            Err(Refusal::MissingInput(
                "historical tone provider PF30059 brightness LUT x or y"
            ))
        );
    }
}
