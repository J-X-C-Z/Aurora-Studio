//! PF407 type-143 UV-map apply on NVIDIA, plus the direct-pixel raw-byte mode.
//!
//! Same law as `kernels::uvmap_color::apply`: at each pixel centre the field
//! gives a signed offset, the target UV is `uv + offset`, and the colour is a
//! bilinear tap or the 9-tap Catmull-Rom sample of the source.
//!
//! The field travels as floats, never as packed bytes, so negative offsets
//! survive: four floats per pixel, `.xy` for the RG selector and `.zw` for BA.
//!
//! Mode 1 keeps the same entry point but reads one direct field pixel. Its
//! `.xy` values are source-pixel offsets and zero `.w` copies the source pixel.
//! The CPU hand-shape path and the three body-remap rows need raw-u8 bilinear,
//! not the normalized PF407 tap, so mode 1 selects their exact quantizer and
//! alpha policy without changing the 32-byte ABI.
//!
//! Driver, context and the 222-only PTX retry live in [`crate::cuda`]; a
//! [`Session`] reuses every buffer for a fixed image shape.

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::hair_body;
use crate::kernels::uvmap_color::Channel;
use crate::{Image, Refusal};

/// Recovered kernel entry point.
pub const KERNEL: &str = "uvmap_color_kernel";
/// Kernel launch shape, shared with the P15 stages.
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/uvmap.ptx");

/// 32-byte kernel argument block, matching `native/cuda/uvmap.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub channel: u32,
    pub catmull: u32,
    pub catmull_w: f32,
    pub catmull_h: f32,
    pub mode: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

impl Uniforms {
    pub fn new(width: u32, height: u32, channel: Channel, catmull: Option<(f32, f32)>) -> Self {
        Self {
            width,
            height,
            stride: width * 4,
            channel: match channel {
                Channel::Rg => 0,
                Channel::Ba => 1,
            },
            catmull: u32::from(catmull.is_some()),
            catmull_w: catmull.map_or(0.0, |size| size.0),
            catmull_h: catmull.map_or(0.0, |size| size.1),
            mode: 0,
        }
    }

    fn direct(width: u32, height: u32, policy: DirectPolicy) -> Self {
        Self {
            width,
            height,
            stride: width * 4,
            channel: u32::from(policy == DirectPolicy::Pf400),
            catmull: u32::from(policy == DirectPolicy::Pf30190),
            catmull_w: 0.0,
            catmull_h: 0.0,
            mode: 1,
        }
    }
}

/// Exact storage policy for a direct-pixel raw-byte remap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectPolicy {
    /// PF30190: round half-up, destination alpha, zero `.w` hard gate.
    Pf30190,
    /// PF400: round ties-to-even, sampled alpha.
    Pf400,
    /// PF404/PF416: round half-up, sampled alpha.
    Pf404Or416,
}

fn as_bytes(value: &Uniforms) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            value as *const Uniforms as *const u8,
            std::mem::size_of::<Uniforms>(),
        )
    }
}

fn check(source: &Image, field: &[[f32; 4]], catmull: Option<(f32, f32)>) -> Result<(), Refusal> {
    source.validate().map_err(|_| Refusal::ShapeMismatch("source"))?;
    if field.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("uv field"));
    }
    if let Some((width, height)) = catmull {
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(Refusal::ValueOutOfRange("catmull size"));
        }
    }
    Ok(())
}

fn check_direct(source: &Image, field: &[[f32; 4]]) -> Result<(), Refusal> {
    source.validate().map_err(|_| Refusal::ShapeMismatch("source"))?;
    if field.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("direct uv field"));
    }
    if field
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
        || field.iter().any(|pixel| pixel[3] != 0.0 && pixel[3] != 1.0)
    {
        return Err(Refusal::ValueOutOfRange("direct uv field"));
    }
    Ok(())
}

fn field_bytes(field: &[[f32; 4]]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(field.as_ptr() as *const u8, std::mem::size_of_val(field))
    }
}

#[cfg(windows)]
pub struct Uvmap {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl Uvmap {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: crate::cuda::Nvidia::open_ptx(PTX)? })
    }

    /// Buffers for one image shape, reused across calls.
    pub fn session(&self, width: u32, height: u32) -> Result<Session<'_>, CudaError> {
        let pixels = width as usize * height as usize;
        let set = self.gpu.alloc_set(&[pixels * 4, pixels * 16, pixels * 4])?;
        Ok(Session { gpu: &self.gpu, set, width, height })
    }

    /// One-shot call: opens a session, then applies.
    pub fn apply(
        &self,
        source: &Image,
        field: &[[f32; 4]],
        channel: Channel,
        catmull: Option<(f32, f32)>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check(source, field, catmull)?;
        let session = self.session(source.width, source.height)?;
        session.apply(source, field, channel, catmull)
    }

    /// One-shot direct-pixel raw-byte remap for the PF30190/body-remap rows.
    pub fn apply_direct(
        &self,
        source: &Image,
        field: &[[f32; 4]],
        policy: DirectPolicy,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check_direct(source, field)?;
        let session = self.session(source.width, source.height)?;
        session.apply_direct(source, field, policy)
    }

    /// Builds the PF30190 field from the same-job shift/protect inputs.
    ///
    /// A missing shift or mask is a typed refusal, never a fabricated default.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_hand_shape(
        &self,
        source: &Image,
        native_parameter: i32,
        shift_x: Option<&[f32]>,
        shift_y: Option<&[f32]>,
        hand_mask: Option<&[u8]>,
        protect_mask: Option<&[u8]>,
        source_scale: Option<f32>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let field = hair_body::hand_shape_field(
            source,
            native_parameter,
            shift_x,
            shift_y,
            hand_mask,
            protect_mask,
            source_scale,
        )?;
        match field {
            Some(field) => self.apply_direct(source, &field, DirectPolicy::Pf30190),
            None => Ok((source.clone(), Vec::new())),
        }
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
        field: &[[f32; 4]],
        channel: Channel,
        catmull: Option<(f32, f32)>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check(source, field, catmull)?;
        let uniforms = Uniforms::new(self.width, self.height, channel, catmull);
        self.run(source, field, uniforms)
    }

    pub fn apply_direct(
        &self,
        source: &Image,
        field: &[[f32; 4]],
        policy: DirectPolicy,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check_direct(source, field)?;
        let uniforms = Uniforms::direct(self.width, self.height, policy);
        self.run(source, field, uniforms)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_hand_shape(
        &self,
        source: &Image,
        native_parameter: i32,
        shift_x: Option<&[f32]>,
        shift_y: Option<&[f32]>,
        hand_mask: Option<&[u8]>,
        protect_mask: Option<&[u8]>,
        source_scale: Option<f32>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let field = hair_body::hand_shape_field(
            source,
            native_parameter,
            shift_x,
            shift_y,
            hand_mask,
            protect_mask,
            source_scale,
        )?;
        match field {
            Some(field) => self.apply_direct(source, &field, DirectPolicy::Pf30190),
            None => Ok((source.clone(), Vec::new())),
        }
    }

    fn run(
        &self,
        source: &Image,
        field: &[[f32; 4]],
        uniforms: Uniforms,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        if (source.width, source.height) != (self.width, self.height) {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("source")));
        }
        self.gpu.set_upload(&self.set, 0, &source.pixels)?;
        self.gpu.set_upload(&self.set, 1, field_bytes(field))?;
        let mut source_ptr = self.set.handle(0);
        let mut field_ptr = self.set.handle(1);
        let mut out_ptr = self.set.handle(2);
        let mut uniforms_arg = uniforms;
        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut source_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut field_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut uniforms_arg as *mut Uniforms as *mut std::ffi::c_void,
        ];
        let grid = (self.width.div_ceil(BLOCK).max(1), self.height.div_ceil(BLOCK).max(1));
        let dispatch = self.gpu.launch_raw(KERNEL, grid, (BLOCK, BLOCK), &mut params)?;
        let mut pixels = vec![0u8; source.pixels.len()];
        self.gpu.set_download(&self.set, 2, &mut pixels)?;
        let image = Image::new(self.width, self.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        let _ = as_bytes(&uniforms);
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
        let plain = Uniforms::new(17, 13, Channel::Ba, None);
        assert_eq!((plain.channel, plain.catmull), (1, 0));
        assert_eq!((plain.stride, plain.catmull_w), (17 * 4, 0.0));
        let catmull = Uniforms::new(5, 3, Channel::Rg, Some((10.0, 6.0)));
        assert_eq!((catmull.catmull, catmull.catmull_w, catmull.catmull_h), (1, 10.0, 6.0));
        assert_eq!(plain.mode, 0);

        let gated = Uniforms::direct(7, 5, DirectPolicy::Pf30190);
        assert_eq!((gated.mode, gated.channel, gated.catmull), (1, 0, 1));
        let even = Uniforms::direct(7, 5, DirectPolicy::Pf400);
        assert_eq!((even.mode, even.channel, even.catmull), (1, 1, 0));
        let half_up = Uniforms::direct(7, 5, DirectPolicy::Pf404Or416);
        assert_eq!((half_up.mode, half_up.channel, half_up.catmull), (1, 0, 0));
    }

    #[test]
    fn field_and_size_refusals_match_the_cpu_lane() {
        let source = Image::solid(2, 1, [10, 20, 30, 40]);
        let field = [[0.0f32; 4]; 2];
        assert_eq!(check(&source, &field, None), Ok(()));
        assert_eq!(check(&source, &field[..1], None), Err(Refusal::ShapeMismatch("uv field")));
        assert_eq!(
            check(&source, &field, Some((0.0, 4.0))),
            Err(Refusal::ValueOutOfRange("catmull size"))
        );
        assert_eq!(check_direct(&source, &field), Ok(()));
        assert_eq!(
            check_direct(&source, &field[..1]),
            Err(Refusal::ShapeMismatch("direct uv field"))
        );
        let mut bad = field;
        bad[0][3] = 0.5;
        assert_eq!(
            check_direct(&source, &bad),
            Err(Refusal::ValueOutOfRange("direct uv field"))
        );
    }
}
