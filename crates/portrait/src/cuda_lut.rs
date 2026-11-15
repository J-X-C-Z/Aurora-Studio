//! PF90005 SKIN_WHITEN / PF90007 SKIN_ROSY / PF21015 SKIN_TINT on NVIDIA.
//!
//! Same pixel law as `kernels::makeup::skin_adjust_lut`: packed 512x512x3 atlas,
//! 8x8 tiles of 64x64 selected by blue, 8-bit fixed-point bilinear tap,
//! `(q*257+128)>>8` normalisation, mul-then-add blend, 12-bit quantised store,
//! and the strict `mask/255*mix > 0.001` gate. Alpha is copied, never computed.
//!
//! Buffers are RGBA8 like [`Image`]; the recovered Metal reference reads BGRA,
//! and this port keeps the Rust lane's packing, which the CPU oracle defines.
//!
//! The driver, its failure guards and the 222-only PTX retry live in
//! [`crate::cuda`]; this module only validates inputs, builds the ABI block and
//! names the kernel.
//!
//! PF21015 is the fixed-point contract: the host selects one of the two tint
//! atlases and passes `mix = abs(ui/100)`. It is not byte-identical to
//! `kernels::face_detail::skin_tint`, which uses f64 sampling and half-even.

use crate::cuda::{CudaError, Dispatch};
use crate::{Image, Refusal};

/// Recovered kernel entry point.
pub const KERNEL: &str = "skin_adjust_lut_kernel";
/// Strict weight gate; pixels at or below it are copied.
pub const GATE: f32 = 0.001;
/// Packed atlas size: 512 x 512 x RGB8.
pub const LUT_BYTES: usize = 512 * 512 * 3;
/// PF21015 SKIN_TINT with TintGreen/TintMagenta selection.
pub const PF_TINT: u32 = 21015;
/// Kernel launch shape, shared with the P15 stages.
const BLOCK: (u32, u32) = (16, 16);

const PTX: &str = include_str!("../native/cuda/skin_lut.ptx");

/// 32-byte kernel argument block, matching `native/cuda/skin_lut.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub mask_row_bytes: u32,
    pub mix: f32,
    pub gate: f32,
    pub lut_width: u32,
    pub lut_height: u32,
}

// The launch passes this block by value; the PTX signature fixes its size.
const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

impl Uniforms {
    pub fn new(width: u32, height: u32, mix: f32) -> Self {
        Self {
            width,
            height,
            stride: width * 4,
            mask_row_bytes: width,
            mix,
            gate: GATE,
            lut_width: 512,
            lut_height: 512,
        }
    }
}

/// Refusals shared with the CPU lane, checked before any driver call.
fn check(
    source: &Image,
    mask_r: &[u8],
    lut_rgb512: &[u8],
    mix: f32,
    pf: u32,
) -> Result<(), Refusal> {
    source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    if mask_r.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    if lut_rgb512.len() != LUT_BYTES {
        return Err(Refusal::ShapeMismatch("lut_rgb512"));
    }
    if !mix.is_finite() || !(0.0..=1.0).contains(&mix) {
        return Err(Refusal::ValueOutOfRange("mix"));
    }
    if pf != 90005 && pf != 90007 && pf != PF_TINT {
        return Err(Refusal::UnknownEffect(pf));
    }
    Ok(())
}

/// Maps the signed UI to `(use_green, |signed|)`.
fn tint_plan(ui: f32) -> Result<(bool, f32), Refusal> {
    if !ui.is_finite() {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let signed = ui.clamp(-100.0, 100.0) / 100.0;
    Ok((signed <= 0.0, signed.abs()))
}

fn as_bytes(value: &Uniforms) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            value as *const Uniforms as *const u8,
            std::mem::size_of::<Uniforms>(),
        )
    }
}

/// The LUT stage's only piece of state: an open driver with this image loaded.
#[cfg(windows)]
pub struct SkinLut(crate::cuda::Nvidia);

#[cfg(windows)]
impl SkinLut {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self(crate::cuda::Nvidia::open_ptx(PTX)?))
    }

    /// GPU twin of `kernels::makeup::skin_adjust_lut`.
    pub fn skin_adjust(
        &self,
        source: &Image,
        mask_r: &[u8],
        lut_rgb512: &[u8],
        mix: f32,
        pf: u32,
    ) -> Result<(Image, Dispatch), CudaError> {
        check(source, mask_r, lut_rgb512, mix, pf)?;
        let uniforms = Uniforms::new(source.width, source.height, mix);
        let grid = (
            source.width.div_ceil(BLOCK.0).max(1),
            source.height.div_ceil(BLOCK.1).max(1),
        );
        let (pixels, dispatch) = self.0.run_u8(
            KERNEL,
            grid,
            &[&source.pixels, mask_r, lut_rgb512],
            as_bytes(&uniforms),
        )?;
        Ok((
            Image {
                width: source.width,
                height: source.height,
                pixels,
            },
            dispatch,
        ))
    }

    /// PF21015 fixed-point tint: host selects TintGreen/TintMagenta and passes
    /// `|ui|/100` as the kernel's mix. The 32-byte ABI and kernel stay shared.
    pub fn tint(
        &self,
        source: &Image,
        mask_r: &[u8],
        tint_green: &[u8],
        tint_magenta: &[u8],
        ui: f32,
    ) -> Result<(Image, Dispatch), CudaError> {
        if tint_green.len() != LUT_BYTES {
            return Err(Refusal::ShapeMismatch("tint_green").into());
        }
        if tint_magenta.len() != LUT_BYTES {
            return Err(Refusal::ShapeMismatch("tint_magenta").into());
        }
        let (use_green, magnitude) = tint_plan(ui)?;
        let lut = if use_green { tint_green } else { tint_magenta };
        self.skin_adjust(source, mask_r, lut, magnitude, PF_TINT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        let uniforms = Uniforms::new(37, 19, 0.5);
        assert_eq!((uniforms.stride, uniforms.mask_row_bytes), (37 * 4, 37));
        assert_eq!((uniforms.lut_width, uniforms.lut_height), (512, 512));
        assert_eq!(uniforms.gate, GATE);
        assert_eq!(as_bytes(&uniforms).len(), 32);
    }

    #[test]
    fn gate_and_atlas_size_are_the_recovered_constants() {
        assert_eq!(GATE, 0.001);
        assert_eq!(LUT_BYTES, 512 * 512 * 3);
        assert_eq!(PF_TINT, 21015);
    }

    #[test]
    fn tint_plan_is_sign_and_limit_exact() {
        assert_eq!(tint_plan(-100.0).unwrap(), (true, 1.0));
        assert_eq!(tint_plan(-0.1).unwrap(), (true, 0.001));
        assert_eq!(tint_plan(-0.0).unwrap(), (true, 0.0));
        assert_eq!(tint_plan(0.0).unwrap(), (true, 0.0));
        assert_eq!(tint_plan(0.22).unwrap(), (false, 0.0022));
        assert_eq!(tint_plan(100.0).unwrap(), (false, 1.0));
        assert_eq!(tint_plan(101.0).unwrap(), (false, 1.0));
        assert_eq!(tint_plan(-101.0).unwrap(), (true, 1.0));
    }

    #[test]
    fn tint_plan_refuses_non_finite_ui() {
        for ui in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(tint_plan(ui), Err(Refusal::ValueOutOfRange("ui")));
        }
    }
}
