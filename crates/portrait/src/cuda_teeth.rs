//! PF108-PF110 teeth whitening on NVIDIA.
//!
//! Same law as `kernels::face_detail::teeth_whitening`: a per-pixel weight from
//! `mask_g` times `dz = common*deyellow` and `ea = common*bright`, a whiten pass
//! then a bright pass over the 512x512x3 atlases, and the recovered
//! threshold-table store rule. A pixel whose two weights are both `<= 0.001`
//! keeps its source bytes; alpha is copied.
//!
//! Driver, context, allocation and the 222-only PTX retry live in
//! [`crate::cuda`]. A [`Session`] keeps the two 786,432-byte atlases resident
//! and reuses every buffer, so a repeated call pays no allocation.

use crate::cuda::{CudaError, Dispatch};
use crate::{Image, Refusal};

/// Recovered kernel entry point.
pub const KERNEL: &str = "teeth_whiten_bright_kernel";
/// Packed atlas size: 512 x 512 x RGB8.
pub const LUT_BYTES: usize = 512 * 512 * 3;
/// Kernel launch shape, shared with the P15 stages.
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/teeth_lut.ptx");

/// 32-byte kernel argument block, matching `native/cuda/teeth_lut.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub mask_row_bytes: u32,
    pub dz: f32,
    pub ea: f32,
    pub reserved0: u32,
    pub reserved1: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

impl Uniforms {
    pub fn new(width: u32, height: u32, dz: f32, ea: f32) -> Self {
        Self {
            width,
            height,
            stride: width * 4,
            mask_row_bytes: width,
            dz,
            ea,
            reserved0: 0,
            reserved1: 0,
        }
    }
}

#[cfg(test)]
fn as_bytes(value: &Uniforms) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            value as *const Uniforms as *const u8,
            std::mem::size_of::<Uniforms>(),
        )
    }
}

/// Refusals shared with the CPU lane, checked before any driver call.
fn check(
    source: &Image,
    mask_g: &[u8],
    whiten_lut: &[u8],
    bright_lut: &[u8],
    common_ui: f32,
    bright_ui: f32,
    deyellow_ui: f32,
) -> Result<(f32, f32), Refusal> {
    source.validate().map_err(|_| Refusal::ShapeMismatch("source"))?;
    if mask_g.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("mask_g"));
    }
    if whiten_lut.len() != LUT_BYTES {
        return Err(Refusal::ShapeMismatch("whiten_lut"));
    }
    if bright_lut.len() != LUT_BYTES {
        return Err(Refusal::ShapeMismatch("bright_lut"));
    }
    for (name, value) in [
        ("common_ui", common_ui),
        ("bright_ui", bright_ui),
        ("deyellow_ui", deyellow_ui),
    ] {
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
            return Err(Refusal::ValueOutOfRange(name));
        }
    }
    let common = common_ui / 100.0;
    Ok((common * deyellow_ui / 100.0, common * bright_ui / 100.0))
}

#[cfg(windows)]
pub struct Teeth {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl Teeth {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: crate::cuda::Nvidia::open_ptx(PTX)? })
    }

    /// Buffers and atlases for one image shape, reused across calls.
    pub fn session(&self, width: u32, height: u32, whiten: &[u8], bright: &[u8]) -> Result<Session<'_>, CudaError> {
        if whiten.len() != LUT_BYTES || bright.len() != LUT_BYTES {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("lut")));
        }
        let pixels = width as usize * height as usize;
        let set = self.gpu.alloc_set(&[pixels * 4, pixels, LUT_BYTES, LUT_BYTES, pixels * 4])?;
        self.gpu.set_upload(&set, 2, whiten)?;
        self.gpu.set_upload(&set, 3, bright)?;
        Ok(Session { gpu: &self.gpu, set, width, height })
    }

    /// One-shot call: opens a session, then applies.
    pub fn apply(
        &self,
        source: &Image,
        mask_g: &[u8],
        whiten_lut: &[u8],
        bright_lut: &[u8],
        common_ui: f32,
        bright_ui: f32,
        deyellow_ui: f32,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check(source, mask_g, whiten_lut, bright_lut, common_ui, bright_ui, deyellow_ui)?;
        let session = self.session(source.width, source.height, whiten_lut, bright_lut)?;
        session.apply(source, mask_g, common_ui, bright_ui, deyellow_ui)
    }
}

#[cfg(windows)]
pub struct Session<'a> {
    gpu: &'a crate::cuda::Nvidia,
    set: crate::cuda::DeviceSet<'a>,
    width: u32,
    height: u32,
}

#[cfg(windows)]
impl Session<'_> {
    pub fn apply(
        &self,
        source: &Image,
        mask_g: &[u8],
        common_ui: f32,
        bright_ui: f32,
        deyellow_ui: f32,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        source
            .validate()
            .map_err(|_| CudaError::Refused(Refusal::ShapeMismatch("source")))?;
        if (source.width, source.height) != (self.width, self.height) {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("source")));
        }
        let pixels = source.pixel_count();
        if mask_g.len() != pixels {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("mask_g")));
        }
        for (name, value) in [
            ("common_ui", common_ui),
            ("bright_ui", bright_ui),
            ("deyellow_ui", deyellow_ui),
        ] {
            if !value.is_finite() || !(0.0..=100.0).contains(&value) {
                return Err(CudaError::Refused(Refusal::ValueOutOfRange(name)));
            }
        }
        let common = common_ui / 100.0;
        let dz = common * deyellow_ui / 100.0;
        let ea = common * bright_ui / 100.0;
        if !(dz > 0.0 || ea > 0.0) {
            return Ok((source.clone(), Vec::new()));
        }

        self.gpu.set_upload(&self.set, 0, &source.pixels)?;
        self.gpu.set_upload(&self.set, 1, mask_g)?;
        let uniforms = Uniforms::new(self.width, self.height, dz, ea);
        let mut source_ptr = self.set.handle(0);
        let mut mask_ptr = self.set.handle(1);
        let mut whiten_ptr = self.set.handle(2);
        let mut bright_ptr = self.set.handle(3);
        let mut out_ptr = self.set.handle(4);
        let mut uniforms_arg = uniforms;
        let mut params: [*mut std::ffi::c_void; 6] = [
            &mut source_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut mask_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut whiten_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut bright_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut uniforms_arg as *mut Uniforms as *mut std::ffi::c_void,
        ];
        let grid = (self.width.div_ceil(BLOCK).max(1), self.height.div_ceil(BLOCK).max(1));
        let dispatch = self.gpu.launch_raw(KERNEL, grid, (BLOCK, BLOCK), &mut params)?;
        let mut pixels_out = vec![0u8; pixels * 4];
        self.gpu.set_download(&self.set, 4, &mut pixels_out)?;
        let image = Image::new(self.width, self.height, pixels_out)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        let uniforms = Uniforms::new(17, 13, 0.5, 0.25);
        assert_eq!((uniforms.stride, uniforms.mask_row_bytes), (17 * 4, 17));
        assert_eq!(as_bytes(&uniforms).len(), 32);
    }

    #[test]
    fn weights_follow_common_times_each_ui() {
        let source = Image::solid(2, 1, [10, 20, 30, 40]);
        let lut = vec![0u8; LUT_BYTES];
        let (dz, ea) = check(&source, &[0, 0], &lut, &lut, 50.0, 80.0, 60.0).unwrap();
        assert!((dz - 0.30).abs() < 1e-6);
        assert!((ea - 0.40).abs() < 1e-6);
        assert_eq!(check(&source, &[0], &lut, &lut, 0.0, 0.0, 0.0), Err(Refusal::ShapeMismatch("mask_g")));
        assert_eq!(
            check(&source, &[0, 0], &lut[..3], &lut, 0.0, 0.0, 0.0),
            Err(Refusal::ShapeMismatch("whiten_lut"))
        );
        assert_eq!(
            check(&source, &[0, 0], &lut, &lut, 101.0, 0.0, 0.0),
            Err(Refusal::ValueOutOfRange("common_ui"))
        );
    }
}
