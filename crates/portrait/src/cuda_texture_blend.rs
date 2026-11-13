//! PF30009 `texture_mask_blend` on NVIDIA.
//!
//! Same law and same input contract as `kernels::texture_mask_blend::apply`:
//! `mix(original.rgb, effect.rgb, mask/255 * degree)`, original alpha kept, and
//! the effect texture sampled through its own UV stream with clamp-to-edge
//! bilinear taps. There is no CPU fallback: an unavailable driver is reported
//! as an unavailable-driver error, never silently replaced by the CPU path.

#![cfg_attr(not(windows), allow(dead_code))]

use crate::kernels::texture_mask_blend::NormalBlend;
use crate::{Image, Refusal};

#[cfg(windows)]
use crate::cuda::CudaError;
#[cfg(windows)]
use crate::cuda::Nvidia;

/// Recovered kernel entry point.
pub const KERNEL: &str = "texture_mask_blend_kernel";
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/texture_mask_blend.ptx");

/// 32-byte uniform block, matching `native/cuda/texture_mask_blend.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: i32,
    pub height: i32,
    pub effect_width: i32,
    pub effect_height: i32,
    pub degree: f32,
    pub reverse_alpha: i32,
    pub has_uv: i32,
    pub reserved: i32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn uv_bytes(points: &[[f32; 2]]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(points.as_ptr() as *const u8, std::mem::size_of_val(points))
    }
}

/// Entry checks shared with the CPU lane so both refuse identically.
fn check(
    original: &Image,
    effect: &Image,
    mask: &[u8],
    effect_uv: Option<&[[f32; 2]]>,
    params: NormalBlend,
) -> Result<usize, Refusal> {
    original
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("original"))?;
    effect
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("effect"))?;
    let pixels = original.pixel_count();
    if mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("s_maskTexture"));
    }
    if !params.degree.is_finite() || !(0.0..=1.0).contains(&params.degree) {
        return Err(Refusal::ValueOutOfRange("u_blendDegree"));
    }
    if let Some(uv) = effect_uv {
        if uv.len() != pixels {
            return Err(Refusal::ShapeMismatch("v_bgmasktexcoordOut"));
        }
        if uv.iter().any(|point| !point[0].is_finite() || !point[1].is_finite()) {
            return Err(Refusal::ValueOutOfRange("v_bgmasktexcoordOut"));
        }
    }
    Ok(pixels)
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

    /// Full-frame blend. Byte-identical to the CPU reference across the tested
    /// degree / reverse / mask / effect-UV combinations.
    pub fn apply(
        &self,
        original: &Image,
        effect: &Image,
        mask: &[u8],
        effect_uv: Option<&[[f32; 2]]>,
        params: NormalBlend,
    ) -> Result<Image, CudaError> {
        check(original, effect, mask, effect_uv, params)?;
        let uniforms = Uniforms {
            width: original.width as i32,
            height: original.height as i32,
            effect_width: effect.width as i32,
            effect_height: effect.height as i32,
            degree: params.degree,
            reverse_alpha: params.reverse_alpha as i32,
            has_uv: effect_uv.is_some() as i32,
            reserved: 0,
        };
        let dummy = [0u8; 8];
        let uv_plane: &[u8] = match effect_uv {
            Some(points) => uv_bytes(points),
            None => &dummy,
        };
        let grid = (
            original.width.div_ceil(BLOCK).max(1),
            original.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, _dispatch) = self.gpu.run_u8(
            KERNEL,
            grid,
            &[&original.pixels, &effect.pixels, mask, uv_plane],
            as_bytes(&uniforms),
        )?;
        Image::new(original.width, original.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        let signature = PTX.split(&format!(".entry {KERNEL}(")).nth(1)
            .expect("PTX kernel entry").split(')').next().unwrap();
        let params: Vec<_> = signature.lines().filter(|line| line.contains(".param")).collect();
        assert_eq!(params.len(), 6);
        assert!(params[5].contains(".b8") && params[5].contains("[32]"),
            "run_u8 passes uniforms by value, not a device pointer");
    }

    #[test]
    fn entry_checks_are_the_cpu_refusals() {
        let original = Image::solid(2, 2, [1, 2, 3, 4]);
        let effect = Image::solid(2, 2, [5, 6, 7, 8]);
        let mask = [255u8; 4];
        assert_eq!(check(&original, &effect, &mask, None, NormalBlend::PF30009), Ok(4));
        assert_eq!(
            check(&original, &effect, &mask[..3], None, NormalBlend::PF30009),
            Err(Refusal::ShapeMismatch("s_maskTexture"))
        );
        assert_eq!(
            check(
                &original,
                &effect,
                &mask,
                None,
                NormalBlend {
                    degree: 1.5,
                    reverse_alpha: false
                }
            ),
            Err(Refusal::ValueOutOfRange("u_blendDegree"))
        );
        // Length is checked first, so a short plane reports the shape refusal.
        assert_eq!(
            check(
                &original,
                &effect,
                &mask,
                Some(&[[f32::NAN, 0.0]]),
                NormalBlend::PF30009
            ),
            Err(Refusal::ShapeMismatch("v_bgmasktexcoordOut"))
        );
        // A full-length plane with a non-finite entry is a value refusal.
        assert_eq!(
            check(
                &original,
                &effect,
                &mask,
                Some(&[[f32::NAN, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                NormalBlend::PF30009
            ),
            Err(Refusal::ValueOutOfRange("v_bgmasktexcoordOut"))
        );
    }
}
