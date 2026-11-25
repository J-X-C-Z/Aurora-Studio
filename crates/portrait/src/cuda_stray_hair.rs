//! PF90088 stray-hair removal on NVIDIA.
//!
//! Same pixel law as `kernels::hair_body::stray_hair`: the raw stray-hair mask,
//! native degree and max(person, background) area gate produce the alpha, then
//! only RGB is mixed and source alpha is copied. The two same-job producers are
//! required when the effect is active and are reported as distinct refusals.

use crate::{Image, Refusal};

#[cfg(windows)]
use crate::cuda::{CudaError, Dispatch, Nvidia};

pub const KERNEL: &str = "pf90088_stray_hair_kernel";
const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/stray_hair.ptx");

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub degree: f32,
    pub person_area: f32,
    pub background_area: f32,
    pub has_person: i32,
    pub has_background: i32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub ui: f32,
    pub native_degree: f32,
    pub native_person_area: f32,
    pub native_background_area: f32,
}

struct Checked<'a> {
    repair: &'a Image,
    raw_mask: &'a [u8],
    person_mask: Option<&'a [u8]>,
    background_mask: Option<&'a [u8]>,
}

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn percent(value: f32, name: &'static str) -> Result<f32, Refusal> {
    if !value.is_finite() || !(0.0..=100.0).contains(&value) {
        Err(Refusal::ValueOutOfRange(name))
    } else {
        Ok(value)
    }
}

fn unit(value: f32, name: &'static str) -> Result<f32, Refusal> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(Refusal::ValueOutOfRange(name))
    } else {
        Ok(value)
    }
}

fn check<'a>(
    source: &Image,
    params: Params,
    repair: Option<&'a Image>,
    raw_mask: Option<&'a [u8]>,
    person_mask: Option<&'a [u8]>,
    background_mask: Option<&'a [u8]>,
) -> Result<Option<Checked<'a>>, Refusal> {
    source.validate()?;
    if percent(params.ui, "PF90088_UI")? == 0.0 {
        return Ok(None);
    }
    let degree = unit(params.native_degree, "PF90088_NATIVE_DEGREE")?;
    // Same recovered gate as the CPU consumer: active only when `d > 0.001`
    // (`0x8BCBA40`, doubled into `xmm9`; `ucomiss` + `seta`).
    if degree <= crate::kernels::hair_body::STRAY_HAIR_ACTIVE_GATE {
        return Ok(None);
    }
    // PF90113 / PF90114 only matter once the main slider is active.
    unit(params.native_person_area, "PF90113_NATIVE_PERSON_AREA")?;
    unit(
        params.native_background_area,
        "PF90114_NATIVE_BACKGROUND_AREA",
    )?;

    let repair = repair.ok_or(Refusal::MissingInput("PF90088_HAIRFLY_V1"))?;
    repair
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("PF90088_HAIRFLY_V1"))?;
    if (source.width, source.height) != (repair.width, repair.height) {
        return Err(Refusal::ShapeMismatch("PF90088_HAIRFLY_V1"));
    }

    let raw_mask = raw_mask.ok_or(Refusal::MissingInput("PF90088_RAW_STRAY_HAIR_MASK"))?;
    let pixels = source.pixel_count();
    if raw_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF90088_RAW_STRAY_HAIR_MASK"));
    }
    if let Some(mask) = person_mask {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF90088_PERSON_MASK"));
        }
    }
    if let Some(mask) = background_mask {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF90088_BACKGROUND_MASK"));
        }
    }
    if raw_mask.iter().all(|&value| value == 0) {
        return Ok(None);
    }

    Ok(Some(Checked {
        repair,
        raw_mask,
        person_mask,
        background_mask,
    }))
}

#[cfg(windows)]
pub struct Blend {
    gpu: Nvidia,
}

#[cfg(windows)]
impl Blend {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: Nvidia::open_ptx(PTX)?,
        })
    }

    pub fn apply(
        &self,
        source: &Image,
        params: Params,
        repair: Option<&Image>,
        raw_mask: Option<&[u8]>,
        person_mask: Option<&[u8]>,
        background_mask: Option<&[u8]>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let Some(checked) = check(
            source,
            params,
            repair,
            raw_mask,
            person_mask,
            background_mask,
        )
        .map_err(CudaError::Refused)?
        else {
            return Ok((source.clone(), Vec::new()));
        };

        let uniforms = Uniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            degree: params.native_degree,
            person_area: params.native_person_area,
            background_area: params.native_background_area,
            has_person: checked.person_mask.is_some() as i32,
            has_background: checked.background_mask.is_some() as i32,
        };
        let absent = [0u8; 1];
        let person = checked.person_mask.unwrap_or(&absent);
        let background = checked.background_mask.unwrap_or(&absent);
        let grid = (
            source.width.div_ceil(BLOCK).max(1),
            source.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8(
            KERNEL,
            grid,
            &[
                &source.pixels,
                &checked.repair.pixels,
                checked.raw_mask,
                person,
                background,
            ],
            as_bytes(&uniforms),
        )?;
        let image = Image::new(source.width, source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params {
            ui: 100.0,
            native_degree: 0.5,
            native_person_area: 1.0,
            native_background_area: 0.5,
        }
    }

    #[test]
    fn missing_producers_are_distinct_typed_refusals() {
        let source = Image::solid(2, 1, [1, 2, 3, 4]);
        let repair = Image::solid(2, 1, [5, 6, 7, 8]);
        let mask = [255u8; 2];

        assert_eq!(
            check(&source, params(), None, Some(&mask), None, None).err(),
            Some(Refusal::MissingInput("PF90088_HAIRFLY_V1"))
        );
        assert_eq!(
            check(&source, params(), Some(&repair), None, None, None).err(),
            Some(Refusal::MissingInput("PF90088_RAW_STRAY_HAIR_MASK"))
        );
    }

    #[test]
    fn inactive_effects_need_no_producers() {
        let source = Image::solid(2, 1, [1, 2, 3, 4]);
        let mut ui_off = params();
        ui_off.ui = 0.0;
        assert!(check(&source, ui_off, None, None, None, None).unwrap().is_none());
        let mut degree_off = params();
        degree_off.native_degree = 0.0;
        assert!(
            check(&source, degree_off, None, None, None, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn malformed_planes_keep_their_producer_names() {
        let source = Image::solid(2, 1, [1, 2, 3, 4]);
        let repair = Image::solid(2, 1, [5, 6, 7, 8]);
        assert_eq!(
            check(&source, params(), Some(&repair), Some(&[1]), None, None).err(),
            Some(Refusal::ShapeMismatch("PF90088_RAW_STRAY_HAIR_MASK"))
        );
        assert_eq!(
            check(
                &source,
                params(),
                Some(&repair),
                Some(&[1, 2]),
                Some(&[1]),
                None
            )
            .err(),
            Some(Refusal::ShapeMismatch("PF90088_PERSON_MASK"))
        );
    }

    #[test]
    fn uniform_block_is_thirty_two_bytes() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
    }

    #[test]
    fn native_gate_matches_the_cpu_consumer() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        let repair = Image::solid(1, 1, [5, 6, 7, 8]);
        let mask = [255u8; 1];

        let mut at_gate = params();
        at_gate.native_degree = crate::kernels::hair_body::STRAY_HAIR_ACTIVE_GATE;
        assert!(
            check(&source, at_gate, None, None, None, None)
                .unwrap()
                .is_none()
        );

        let mut active = params();
        active.native_degree = 0.0011;
        assert!(
            check(&source, active, Some(&repair), Some(&mask), None, None)
                .unwrap()
                .is_some()
        );
    }
}
