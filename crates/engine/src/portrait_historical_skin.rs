use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use aurora_portrait::kernels::{face_detail, makeup, skin_adjust};
use aurora_portrait::kernels::stretch_v3::StretchV3StageTicket;
use aurora_portrait::{Image, Refusal};
use skin_adjust::{Roi, WrinkleUniforms};

use super::sha256_32;
use crate::face_parsing::{FaceParsingProducerInputs, project_face_parsing_skin_mask};

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
    EyeLines(HistoricalSkinWrinkle),
    BodySkin(Vec<HistoricalSkinPass>),
    StretchMark(StretchV3StageTicket),
    BodyTone {
        m: Vec<f32>,
        o: Vec<f32>,
        w_m: f32,
        w_o: f32,
    },
}

pub const HISTORICAL_SKIN_EXTERNAL_PFS: [u32; 6] = [21024, 21105, 90008, 90023, 90024, 90086];
pub const GPU_PROVEN_HISTORICAL_SKIN_PFS: [u32; 3] = [21024, 90023, 90024];
pub(super) const PFS: &[u32] = &HISTORICAL_SKIN_EXTERNAL_PFS;

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
            Self::StretchMark(_) => 90008,
            Self::EyeLines(_) => 90023,
            Self::Forehead(_) => 90024,
            Self::BodyTone { .. } => 90086,
        }
    }
}

pub const HISTORICAL_SKIN_EXTERNAL_SCHEMA: &str = "aurora.portrait.historical-skin.external.v1";

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalSkinExternalRow {
    pub pf: u32,
    pub frame: Image,
    pub frame_sha256: [u8; 32],
    pub ticket: u64,
    pub control: f32,
    pub data: HistoricalSkinData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalSkinExternalInputs {
    pub ticket: u64,
    pub rows: Vec<HistoricalSkinExternalRow>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalSkinImmediateRow {
    pub pf: u32,
    pub ticket: u64,
    pub control: f32,
    pub data: HistoricalSkinData,
}

/// Same-job planes whose source digest is bound at each PF's execution point.
/// Wrinkle skin masks are projected from the canonical fbe class 1.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoricalSkinImmediateInputs {
    pub ticket: u64,
    pub rows: Vec<HistoricalSkinImmediateRow>,
}

impl HistoricalSkinExternalInputs {
    pub fn build(self) -> Result<HistoricalSkinInputs, Refusal> {
        build_historical_skin_inputs(Some(self))
    }
}

pub fn build_immediate_historical_skin_inputs(
    frame: &Image,
    parsing: &FaceParsingProducerInputs,
    inputs: HistoricalSkinImmediateInputs,
) -> Result<HistoricalSkinInputs, Refusal> {
    if inputs.ticket == 0 {
        return Err(Refusal::MissingInput("historical skin immediate ticket"));
    }
    if inputs.rows.is_empty() {
        return Err(Refusal::MissingInput("historical skin immediate rows"));
    }
    frame.validate()?;
    let size = [frame.width, frame.height];
    let parsed_skin = project_face_parsing_skin_mask(parsing, frame.width, frame.height)?;
    let mut built = BTreeMap::new();
    for mut row in inputs.rows {
        let pf = row.pf;
        if !PFS.contains(&pf) {
            return Err(Refusal::UnknownEffect(pf));
        }
        if built.contains_key(&pf) {
            return Err(Refusal::ValueOutOfRange("historical skin immediate duplicate PF"));
        }
        if row.data.pf() != pf {
            return Err(Refusal::ValueOutOfRange(
                "historical skin immediate row PF/data",
            ));
        }
        if row.ticket == 0 || row.ticket != inputs.ticket {
            return Err(Refusal::MissingInput("historical skin immediate row ticket"));
        }
        if !row.control.is_finite() || !(0.0..=1.0).contains(&row.control) {
            return Err(Refusal::ValueOutOfRange(external_control_invalid(pf)));
        }
        match &mut row.data {
            HistoricalSkinData::Glabella(input)
            | HistoricalSkinData::Forehead(input)
            | HistoricalSkinData::EyeLines(input)
                if input.skin_mask.is_empty() =>
            {
                input.skin_mask.clone_from(&parsed_skin);
            }
            _ => {}
        }
        validate_external_data(pf, size, &row.data)?;
        built.insert(
            pf,
            HistoricalSkinInput {
                ticket: row.ticket,
                size,
                source_sha256: [0; 32],
                control: row.control,
                data: row.data,
            },
        );
    }
    Ok(built)
}

/// Builds the owned map from an external producer payload. Every row carries
/// the immediate frame and its caller-supplied digest; the builder rejects any
/// digest, ticket, control, plane, ROI or uniform that is not bound to that row.
pub fn build_historical_skin_inputs(
    inputs: Option<HistoricalSkinExternalInputs>,
) -> Result<HistoricalSkinInputs, Refusal> {
    let inputs = inputs.ok_or(Refusal::MissingInput("historical skin external inputs"))?;
    if inputs.ticket == 0 {
        return Err(Refusal::MissingInput("historical skin external ticket"));
    }
    if inputs.rows.is_empty() {
        return Err(Refusal::MissingInput("historical skin external rows"));
    }
    let mut built = BTreeMap::new();
    for row in inputs.rows {
        let pf = row.pf;
        if !PFS.contains(&pf) {
            return Err(Refusal::UnknownEffect(pf));
        }
        if built.contains_key(&pf) {
            return Err(Refusal::ValueOutOfRange("historical skin external duplicate PF"));
        }
        if row.data.pf() != pf {
            return Err(Refusal::ValueOutOfRange(
                "historical skin external row PF/data",
            ));
        }
        if row.ticket == 0 {
            return Err(Refusal::MissingInput(external_ticket_missing(pf)));
        }
        if row.ticket != inputs.ticket {
            return Err(Refusal::ValueOutOfRange("historical skin external row ticket"));
        }
        if row.frame.width == 0 || row.frame.height == 0 {
            return Err(Refusal::MissingInput(external_frame_missing(pf)));
        }
        let size = [row.frame.width, row.frame.height];
        let pixels = (size[0] as usize)
            .checked_mul(size[1] as usize)
            .ok_or(Refusal::ShapeMismatch("historical skin external frame plane"))?;
        if row.frame.pixels.len() != pixels * 4 {
            return Err(Refusal::ShapeMismatch("historical skin external frame plane"));
        }
        if row.frame_sha256 == [0; 32] {
            return Err(Refusal::MissingInput(external_frame_sha_missing(pf)));
        }
        if row.frame_sha256 != super::sha256_32(&row.frame.pixels) {
            return Err(Refusal::ValueOutOfRange(external_frame_sha_mismatch(pf)));
        }
        if !row.control.is_finite() || !(0.0..=1.0).contains(&row.control) {
            return Err(Refusal::ValueOutOfRange(external_control_invalid(pf)));
        }
        validate_external_data(pf, size, &row.data)?;
        built.insert(
            pf,
            HistoricalSkinInput {
                ticket: row.ticket,
                size,
                source_sha256: row.frame_sha256,
                control: row.control,
                data: row.data,
            },
        );
    }
    Ok(built)
}

fn external_ticket_missing(pf: u32) -> &'static str {
    match pf {
        21024 => "historical skin external PF21024 ticket",
        21105 => "historical skin external PF21105 ticket",
        90008 => "historical skin external PF90008 ticket",
        90023 => "historical skin external PF90023 wrinkle ticket",
        90024 => "historical skin external PF90024 ticket",
        90086 => "historical skin external PF90086 ticket",
        _ => "historical skin external row ticket",
    }
}

fn external_frame_missing(pf: u32) -> &'static str {
    match pf {
        21024 => "historical skin external PF21024 frame",
        21105 => "historical skin external PF21105 frame",
        90008 => "historical skin external PF90008 frame",
        90023 => "historical skin external PF90023 wrinkle frame",
        90024 => "historical skin external PF90024 frame",
        90086 => "historical skin external PF90086 frame",
        _ => "historical skin external row frame",
    }
}

fn external_frame_sha_missing(pf: u32) -> &'static str {
    match pf {
        21024 => "historical skin external PF21024 frame SHA-256",
        21105 => "historical skin external PF21105 frame SHA-256",
        90008 => "historical skin external PF90008 frame SHA-256",
        90023 => "historical skin external PF90023 wrinkle frame SHA-256",
        90024 => "historical skin external PF90024 frame SHA-256",
        90086 => "historical skin external PF90086 frame SHA-256",
        _ => "historical skin external row frame SHA-256",
    }
}

fn external_frame_sha_mismatch(pf: u32) -> &'static str {
    match pf {
        21024 => "historical skin external PF21024 frame SHA-256 mismatch",
        21105 => "historical skin external PF21105 frame SHA-256 mismatch",
        90008 => "historical skin external PF90008 frame SHA-256 mismatch",
        90023 => "historical skin external PF90023 wrinkle frame SHA-256 mismatch",
        90024 => "historical skin external PF90024 frame SHA-256 mismatch",
        90086 => "historical skin external PF90086 frame SHA-256 mismatch",
        _ => "historical skin external row frame SHA-256 mismatch",
    }
}

fn external_control_invalid(pf: u32) -> &'static str {
    match pf {
        21024 => "historical skin external PF21024 control",
        21105 => "historical skin external PF21105 control",
        90008 => "historical skin external PF90008 control",
        90023 => "historical skin external PF90023 wrinkle control",
        90024 => "historical skin external PF90024 control",
        90086 => "historical skin external PF90086 control",
        _ => "historical skin external row control",
    }
}

fn require_len<T>(values: &[T], expected: usize, field: &'static str) -> Result<(), Refusal> {
    if values.is_empty() {
        return Err(Refusal::MissingInput(field));
    }
    if values.len() != expected {
        return Err(Refusal::ShapeMismatch(field));
    }
    Ok(())
}

fn require_image(image: &Image, size: [u32; 2], field: &'static str) -> Result<(), Refusal> {
    if image.width == 0 || image.height == 0 || image.pixels.is_empty() {
        return Err(Refusal::MissingInput(field));
    }
    if [image.width, image.height] != size {
        return Err(Refusal::ShapeMismatch(field));
    }
    let pixels = (size[0] as usize)
        .checked_mul(size[1] as usize)
        .ok_or(Refusal::ShapeMismatch(field))?;
    if image.pixels.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch(field));
    }
    Ok(())
}

fn require_roi(roi: Roi, size: [u32; 2], field: &'static str) -> Result<(), Refusal> {
    if roi.width == 0 || roi.height == 0 {
        return Err(Refusal::MissingInput(field));
    }
    if roi.x.checked_add(roi.width).is_none_or(|right| right > size[0])
        || roi.y.checked_add(roi.height).is_none_or(|bottom| bottom > size[1])
    {
        return Err(Refusal::ShapeMismatch(field));
    }
    Ok(())
}

fn validate_external_data(
    pf: u32,
    size: [u32; 2],
    data: &HistoricalSkinData,
) -> Result<(), Refusal> {
    let pixels = (size[0] as usize)
        .checked_mul(size[1] as usize)
        .ok_or(Refusal::ShapeMismatch("historical skin external frame plane"))?;
    match (pf, data) {
        (21024, HistoricalSkinData::Glabella(input))
        | (90023, HistoricalSkinData::EyeLines(input))
        | (90024, HistoricalSkinData::Forehead(input)) => {
            require_image(&input.base, size, "historical skin external wrinkle base")?;
            require_image(&input.residual, size, "historical skin external wrinkle residual")?;
            require_len(
                input.mask_rgba.as_slice(),
                pixels * 4,
                "historical skin external wrinkle rgba mask",
            )?;
            require_len(
                input.skin_mask.as_slice(),
                pixels,
                "historical skin external wrinkle skin mask",
            )?;
            require_roi(input.roi, size, "historical skin external wrinkle roi")?;
            let uniforms = input.uniforms;
            let values = [
                uniforms.legal_alpha,
                uniforms.dark_circle_alpha,
                uniforms.diff_factor,
                uniforms.diff_factor2,
                uniforms.color_threshold_factor,
                uniforms.mix_alpha,
                uniforms.diff_min_threshold,
                uniforms.diff_max_threshold,
                uniforms.high_freq_min_threshold,
                uniforms.high_freq_max_threshold,
            ];
            if values.iter().any(|value| !value.is_finite())
                || uniforms.diff_min_threshold > uniforms.diff_max_threshold
                || uniforms.high_freq_min_threshold > uniforms.high_freq_max_threshold
            {
                return Err(Refusal::ValueOutOfRange(
                    "historical skin external wrinkle uniforms",
                ));
            }
        }
        (21105, HistoricalSkinData::BodySkin(passes)) => {
            if passes.is_empty() {
                return Err(Refusal::MissingInput(
                    "historical skin external PF21105 passes",
                ));
            }
            for pass in passes {
                require_len(
                    pass.mask.as_slice(),
                    pixels,
                    "historical skin external PF21105 pass mask",
                )?;
                require_image(
                    &pass.effect,
                    size,
                    "historical skin external PF21105 pass effect",
                )?;
            }
        }
        (90008, HistoricalSkinData::StretchMark(ticket)) => {
            ticket.validate_for_size(size)?;
        }
        (90086, HistoricalSkinData::BodyTone { m, o, w_m, w_o }) => {
            require_len(
                m.as_slice(),
                pixels * 3,
                "historical skin external PF90086 m plane",
            )?;
            require_len(
                o.as_slice(),
                pixels * 3,
                "historical skin external PF90086 o plane",
            )?;
            if m.iter().chain(o).any(|value| !value.is_finite())
                || !w_m.is_finite()
                || !w_o.is_finite()
            {
                return Err(Refusal::ValueOutOfRange(
                    "historical skin external PF90086 body tone",
                ));
            }
        }
        _ => return Err(Refusal::ValueOutOfRange("historical skin external row data")),
    }
    Ok(())
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
        || (matches!(pf, 21024 | 90023 | 90024) && control <= 0.001)
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
    if input.source_sha256 != [0; 32] && input.source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("historical skin frame digest"));
    }

    let output = match &input.data {
        HistoricalSkinData::Glabella(input)
        | HistoricalSkinData::Forehead(input)
        | HistoricalSkinData::EyeLines(input) => {
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
        HistoricalSkinData::StretchMark(stage) => stage.execute_image(source, ticket, ui)?,
        HistoricalSkinData::BodyTone { m, o, w_m, w_o } => {
            makeup::body_tone_unify(source, m, o, *w_m, *w_o)?
        }
    };
    Ok((output != *source).then_some(output))
}

pub(super) fn gpu_bound_input<'a>(
    source: &Image,
    ticket: Option<u64>,
    pf: u32,
    control: f32,
    inputs: Option<&'a HistoricalSkinInputs>,
) -> Result<Option<&'a HistoricalSkinInput>, Refusal> {
    if !(0.0..=1.0).contains(&control) {
        return Err(Refusal::ValueOutOfRange("historical skin control"));
    }
    if control == 0.0
        || (matches!(pf, 21024 | 90023 | 90024) && control <= 0.001)
        || (pf == 90008 && !skin_adjust::stretch_mark_degree(control * 100.0)?.1)
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
    if input.source_sha256 != [0; 32] && input.source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("historical skin frame digest"));
    }
    Ok(Some(input))
}

/// Backend that produced one historical-skin row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoricalSkinPath {
    Cpu,
    Gpu,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoricalSkinGpuPlan {
    pub backend: HistoricalSkinPath,
    pub kernel: Option<&'static str>,
    pub fallback_reason: Option<String>,
}

pub const PF21105_GPU_REFUSAL: &str =
    "PF21105 0x46E4D80 BGRA8 blend is CPU-executable, but CUDA is refused: the native source/mask strides and float effect slot have no byte-exact kernel/parity";
pub const PF90008_GPU_REFUSAL: &str =
    "PF90008 CUDA refused: StretchV3 nOrder 0x1E9DD78 model/mask/ROI/affine/output-texture ABI and blend law remain unresolved";
pub const PF90086_GPU_REFUSAL: &str =
    "PF90086 CUDA refused: M/O 3xf32 planes are partial evidence; stride/ROI/color-domain, degree multiplication and pass parity remain unresolved";

/// P15 has an exact native plane ABI for the three wrinkle rows. The remaining
/// rows keep the exact CPU oracle but name the first missing GPU-boundary fact.
pub fn plan_historical_skin_backend(pf: u32, data: &HistoricalSkinData) -> HistoricalSkinGpuPlan {
    let matched = matches!(
        (pf, data),
        (21024, HistoricalSkinData::Glabella(_))
            | (90023, HistoricalSkinData::EyeLines(_))
            | (90024, HistoricalSkinData::Forehead(_))
    );
    if matched && GPU_PROVEN_HISTORICAL_SKIN_PFS.contains(&pf) {
        return HistoricalSkinGpuPlan {
            backend: HistoricalSkinPath::Gpu,
            kernel: Some(aurora_portrait::cuda_wrinkle::U8_KERNEL),
            fallback_reason: None,
        };
    }
    let fallback_reason = match pf {
        21105 => PF21105_GPU_REFUSAL.to_owned(),
        90008 => PF90008_GPU_REFUSAL.to_owned(),
        90086 => PF90086_GPU_REFUSAL.to_owned(),
        _ => {
            let stage = match data {
        HistoricalSkinData::Glabella(_) => "glabella wrinkle base/residual/mask",
        HistoricalSkinData::BodySkin(_) => "body-skin unify mask/effect passes",
        HistoricalSkinData::StretchMark(_) => "stretch-mark model/mask/ROI",
        HistoricalSkinData::EyeLines(_) => "eye-line wrinkle base/residual/mask",
        HistoricalSkinData::Forehead(_) => "forehead wrinkle base/residual/mask",
        HistoricalSkinData::BodyTone { .. } => "body-tone M/O weighted planes",
            };
            format!(
                "PF{pf} {stage} refused on CUDA: no matching ABI/kernel/parity; CPU fallback"
            )
        }
    };
    HistoricalSkinGpuPlan {
        backend: HistoricalSkinPath::Cpu,
        kernel: None,
        fallback_reason: Some(fallback_reason),
    }
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
            HistoricalSkinData::Glabella(input)
            | HistoricalSkinData::Forehead(input)
            | HistoricalSkinData::EyeLines(input) => {
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
            HistoricalSkinData::StretchMark(ticket) => ticket.hash_into(&mut hash),
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
            HistoricalSkinData::Forehead(wrinkle.clone()),
            HistoricalSkinData::EyeLines(wrinkle),
            HistoricalSkinData::BodySkin(vec![HistoricalSkinPass {
                mask: vec![255, 0], effect: model.clone(),
            }]),
            HistoricalSkinData::StretchMark(StretchV3StageTicket::shared_oracle(
                model,
                vec![255],
                roi,
                false,
                source.width * 4,
            )),
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

    fn external_wrinkle(source: &Image, model: &Image) -> HistoricalSkinWrinkle {
        HistoricalSkinWrinkle {
            base: source.clone(),
            residual: model.clone(),
            mask_rgba: vec![255; source.pixels.len()],
            skin_mask: vec![255; (source.width * source.height) as usize],
            roi: Roi {
                x: 0,
                y: 0,
                width: source.width,
                height: source.height,
            },
            uniforms: WrinkleUniforms {
                legal_alpha: 1.0,
                dark_circle_alpha: 1.0,
                diff_factor: 1.0,
                diff_factor2: 0.0,
                color_threshold_factor: 1.0,
                mix_alpha: 1.0,
                diff_min_threshold: -1.0,
                diff_max_threshold: 1.0,
                high_freq_min_threshold: -1.0,
                high_freq_max_threshold: 1.0,
            },
        }
    }

    fn external_row(
        pf: u32,
        frame: &Image,
        data: HistoricalSkinData,
    ) -> HistoricalSkinExternalRow {
        HistoricalSkinExternalRow {
            pf,
            frame: frame.clone(),
            frame_sha256: sha256_32(&frame.pixels),
            ticket: 7,
            control: 1.0,
            data,
        }
    }

    #[test]
    fn external_builder_binds_all_six_rows_and_pf90023_wrinkle() {
        let source = Image::new(2, 1, vec![100, 150, 200, 77, 1, 2, 3, 17]).unwrap();
        let model = Image::new(2, 1, vec![200, 100, 50, 91, 4, 5, 6, 99]).unwrap();
        let wrinkle = external_wrinkle(&source, &model);
        let roi = Roi {
            x: 0,
            y: 0,
            width: 2,
            height: 1,
        };
        let rows = vec![
            external_row(21024, &source, HistoricalSkinData::Glabella(wrinkle.clone())),
            external_row(
                21105,
                &source,
                HistoricalSkinData::BodySkin(vec![HistoricalSkinPass {
                    mask: vec![255, 0],
                    effect: model.clone(),
                }]),
            ),
            external_row(
                90008,
                &source,
                HistoricalSkinData::StretchMark(StretchV3StageTicket::shared_oracle(
                    model.clone(),
                    vec![255, 255],
                    roi,
                    false,
                    source.width * 4,
                )),
            ),
            external_row(90023, &source, HistoricalSkinData::EyeLines(wrinkle.clone())),
            external_row(90024, &source, HistoricalSkinData::Forehead(wrinkle)),
            external_row(
                90086,
                &source,
                HistoricalSkinData::BodyTone {
                    m: vec![0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
                    o: vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    w_m: 1.0,
                    w_o: 1.0,
                },
            ),
        ];
        let built = build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
            ticket: 7,
            rows,
        }))
        .unwrap();
        assert_eq!(
            built.keys().copied().collect::<Vec<_>>(),
            vec![21024, 21105, 90008, 90023, 90024, 90086]
        );
        assert_eq!(built[&90023].source_sha256, sha256_32(&source.pixels));
        assert!(apply(&source, Some(7), 90023, 1.0, Some(&built)).is_ok());
        assert_eq!(fingerprint(&built), fingerprint(&built.clone()));
    }

    #[test]
    fn backend_plan_routes_only_the_three_p15_wrinkle_rows() {
        let source = Image::new(1, 1, vec![100, 150, 200, 77]).unwrap();
        let model = Image::new(1, 1, vec![200, 100, 50, 91]).unwrap();
        let wrinkle = external_wrinkle(&source, &model);
        for (pf, data) in [
            (21024, HistoricalSkinData::Glabella(wrinkle.clone())),
            (90023, HistoricalSkinData::EyeLines(wrinkle.clone())),
            (90024, HistoricalSkinData::Forehead(wrinkle.clone())),
        ] {
            let plan = plan_historical_skin_backend(pf, &data);
            assert_eq!(plan.backend, HistoricalSkinPath::Gpu, "PF{pf}");
            assert_eq!(
                plan.kernel,
                Some(aurora_portrait::cuda_wrinkle::U8_KERNEL),
                "PF{pf}"
            );
            assert_eq!(plan.fallback_reason, None, "PF{pf}");
        }
        for (pf, data) in [
            (
                21105,
                HistoricalSkinData::BodySkin(vec![HistoricalSkinPass {
                    mask: vec![255],
                    effect: model.clone(),
                }]),
            ),
            (
                90008,
                HistoricalSkinData::StretchMark(StretchV3StageTicket::shared_oracle(
                    model.clone(),
                    vec![255],
                    Roi {
                        x: 0,
                        y: 0,
                        width: 1,
                        height: 1,
                    },
                    false,
                    source.width * 4,
                )),
            ),
            (
                90086,
                HistoricalSkinData::BodyTone {
                    m: vec![0.0; 3],
                    o: vec![0.0; 3],
                    w_m: 1.0,
                    w_o: 1.0,
                },
            ),
        ] {
            let plan = plan_historical_skin_backend(pf, &data);
            assert_eq!(plan.backend, HistoricalSkinPath::Cpu, "PF{pf}");
            assert_eq!(plan.kernel, None, "PF{pf}");
            let expected = match pf {
                21105 => "0x46E4D80",
                90008 => "0x1E9DD78",
                90086 => "M/O 3xf32",
                _ => unreachable!(),
            };
            assert!(
                plan.fallback_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains(expected)),
                "PF{pf}"
            );
        }
    }

    #[test]
    fn external_builder_names_missing_ticket_sha_control_and_planes() {
        let source = Image::new(2, 1, vec![100, 150, 200, 77, 1, 2, 3, 17]).unwrap();
        let model = Image::new(2, 1, vec![200, 100, 50, 91, 4, 5, 6, 99]).unwrap();
        let row = external_row(
            21024,
            &source,
            HistoricalSkinData::Glabella(external_wrinkle(&source, &model)),
        );
        assert_eq!(
            build_historical_skin_inputs(None),
            Err(Refusal::MissingInput("historical skin external inputs"))
        );

        let mut zero_ticket = row.clone();
        zero_ticket.ticket = 0;
        assert_eq!(
            build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
                ticket: 7,
                rows: vec![zero_ticket],
            })),
            Err(Refusal::MissingInput(
                "historical skin external PF21024 ticket"
            ))
        );

        let mut zero_sha = row.clone();
        zero_sha.frame_sha256 = [0; 32];
        assert_eq!(
            build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
                ticket: 7,
                rows: vec![zero_sha],
            })),
            Err(Refusal::MissingInput(
                "historical skin external PF21024 frame SHA-256"
            ))
        );

        let mut wrong_sha = row.clone();
        wrong_sha.frame_sha256[0] ^= 1;
        assert_eq!(
            build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
                ticket: 7,
                rows: vec![wrong_sha],
            })),
            Err(Refusal::ValueOutOfRange(
                "historical skin external PF21024 frame SHA-256 mismatch"
            ))
        );

        let mut bad_control = row.clone();
        bad_control.control = f32::NAN;
        assert_eq!(
            build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
                ticket: 7,
                rows: vec![bad_control],
            })),
            Err(Refusal::ValueOutOfRange(
                "historical skin external PF21024 control"
            ))
        );

        let mut empty_mask = row;
        let HistoricalSkinData::Glabella(wrinkle) = &mut empty_mask.data else {
            unreachable!()
        };
        wrinkle.mask_rgba.clear();
        assert_eq!(
            build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
                ticket: 7,
                rows: vec![empty_mask],
            })),
            Err(Refusal::MissingInput(
                "historical skin external wrinkle rgba mask"
            ))
        );
    }

    #[test]
    fn external_builder_refuses_cross_job_duplicate_and_wrong_variant() {
        let source = Image::new(2, 1, vec![100, 150, 200, 77, 1, 2, 3, 17]).unwrap();
        let model = Image::new(2, 1, vec![200, 100, 50, 91, 4, 5, 6, 99]).unwrap();
        let wrinkle = external_wrinkle(&source, &model);
        let mut cross_job = external_row(
            90024,
            &source,
            HistoricalSkinData::Forehead(wrinkle.clone()),
        );
        cross_job.ticket = 8;
        assert_eq!(
            build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
                ticket: 7,
                rows: vec![cross_job],
            })),
            Err(Refusal::ValueOutOfRange(
                "historical skin external row ticket"
            ))
        );

        let duplicate = vec![
            external_row(90024, &source, HistoricalSkinData::Forehead(wrinkle.clone())),
            external_row(90024, &source, HistoricalSkinData::Forehead(wrinkle.clone())),
        ];
        assert_eq!(
            build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
                ticket: 7,
                rows: duplicate,
            })),
            Err(Refusal::ValueOutOfRange(
                "historical skin external duplicate PF"
            ))
        );

        let mismatch = external_row(90024, &source, HistoricalSkinData::Glabella(wrinkle));
        assert_eq!(
            build_historical_skin_inputs(Some(HistoricalSkinExternalInputs {
                ticket: 7,
                rows: vec![mismatch],
            })),
            Err(Refusal::ValueOutOfRange(
                "historical skin external row PF/data"
            ))
        );
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
