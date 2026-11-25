//! First real multi-kernel Metal chain (M2): box blur -> masked 3D-LUT whiten.
//!
//! This module is a *chain harness*, not a product stage. It exists to prove
//! that the runtime can run two dependent kernels in one command buffer and
//! read the result back once, with a CPU twin to compare against.
//!
//! Kernel provenance:
//! * `o3_masked_lut_whiten.metal` is a byte-identical copy of
//!   `crates/portrait/native/metal/o3b_masked_lut_whiten.metal` (verified with
//!   `cmp`). It is embedded below with `include_str!`.
//! * `skin_whiten_prep.metal` is the engine's 3x3 box-blur prep kernel. Its
//!   CPU twin [`cpu_box_blur_bgra`] uses identical integer arithmetic.
//!
//! The masked-whiten CPU reference is the portrait crate's finished
//! `aurora_portrait::kernels::skin_color::masked_lut_blend`, so this chain does
//! not invent a new oracle.

use aurora_portrait::Image as PortraitImage;
use aurora_portrait::kernels::skin_color::masked_lut_blend;

use crate::metal_runtime::{
    Binding, Grid, MetalError, MetalRuntime, PipelineKey, PipelineLimits,
};

/// Byte-identical copy of the portrait-crate kernel; see module docs.
const WHITEN_MSL: &str = include_str!("../../native/metal/o3b_masked_lut_whiten.metal");
const PREP_MSL: &str = include_str!("../../native/metal/skin_whiten_prep.metal");

pub const WHITEN_ENTRY: &str = "o3_masked_lut_whiten";
pub const BLUR_ENTRY: &str = "skin_whiten_box_blur";

/// 3D LUT atlas layout shared with the portrait kernel: 8x8 tiles of 64x64,
/// RGBA float4, 512x512.
pub const LUT_ATLAS_WIDTH: usize = 512;
pub const LUT_ATLAS_FLOATS: usize = LUT_ATLAS_WIDTH * LUT_ATLAS_WIDTH * 4;

pub const BLUR_KEY: PipelineKey = PipelineKey::new(PREP_MSL, BLUR_ENTRY);
pub const WHITEN_KEY: PipelineKey = PipelineKey::new(WHITEN_MSL, WHITEN_ENTRY);

/// Inputs for one chain run. The alpha channel of `bgra` is part of the
/// packed BGRA surface; `virtual_mask` is a `degree` scalar in 0..=1.
pub struct SkinWhitenInput<'a> {
    pub width: u32,
    pub height: u32,
    /// Packed BGRA8, `width * height * 4` bytes.
    pub bgra: &'a [u8],
    /// One byte per pixel, `width * height` bytes.
    pub mask: &'a [u8],
    /// RGBA float4 atlas, [`LUT_ATLAS_FLOATS`] values.
    pub lut: &'a [f32],
    /// Blend degree, `0.0 ..= 1.0`.
    pub degree: f32,
}

pub fn validate(input: &SkinWhitenInput<'_>) -> Result<usize, MetalError> {
    if input.width == 0 || input.height == 0 {
        return Err(MetalError::Invalid("chain width/height must be non-zero"));
    }
    let pixels = (input.width as usize)
        .checked_mul(input.height as usize)
        .ok_or(MetalError::Invalid("chain dimensions overflow"))?;
    if pixels > u32::MAX as usize {
        return Err(MetalError::Invalid("pixel count exceeds u32 ABI"));
    }
    if input.bgra.len() != pixels * 4 {
        return Err(MetalError::Invalid("bgra length must be width*height*4"));
    }
    if input.mask.len() != pixels {
        return Err(MetalError::Invalid("mask length must be width*height"));
    }
    if input.lut.len() != LUT_ATLAS_FLOATS {
        return Err(MetalError::Invalid("LUT atlas must be 512x512 RGBA float4"));
    }
    if !input.degree.is_finite() || !(0.0..=1.0).contains(&input.degree) {
        return Err(MetalError::Invalid("degree must be finite and in 0..=1"));
    }
    Ok(pixels)
}

// ABI structs, byte-for-byte what the MSL `static_assert`s require.
#[repr(C)]
#[derive(Clone, Copy)]
struct O3MaskedWhitenParams {
    surface_width: u32,
    surface_height: u32,
    mask_x: u32,
    mask_y: u32,
    mask_width: u32,
    mask_height: u32,
    pixel_count: u32,
    flags: u32,
    alpha: f32,
    reserved0: f32,
    reserved1: f32,
    reserved2: f32,
}
const _: () = assert!(std::mem::size_of::<O3MaskedWhitenParams>() == 48);

#[repr(C)]
#[derive(Clone, Copy)]
struct SkinWhitenBlurParams {
    width: u32,
    height: u32,
    reserved0: u32,
    reserved1: u32,
}
const _: () = assert!(std::mem::size_of::<SkinWhitenBlurParams>() == 16);

/// `repr(C)` POD bytes. Callers only pass initialized structs.
fn as_bytes<T>(value: &T) -> &[u8] {
    // SAFETY: `value` is an initialized `repr(C)` POD; reading it as bytes is
    // sound and the borrow is tied to `value`.
    unsafe {
        std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
    }
}

fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is initialized and the byte view
    // keeps the same lifetime.
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            std::mem::size_of_val(values),
        )
    }
}

/// Device output plus the frame's generation stamp and the real pipeline
/// limits/grids used (evidence that 16x16 was not assumed).
pub struct SkinWhitenResult {
    pub pixels: Vec<u8>,
    pub generation: u64,
    pub blur_limits: PipelineLimits,
    pub whiten_limits: PipelineLimits,
    pub blur_grid: Grid,
    pub whiten_grid: Grid,
}

/// Runs the two-kernel chain on the device runtime and reads the result back.
///
/// One `MTLCommandBuffer`, two compute encoders, one readback. Pipeline grid
/// sizes come from each pipeline's reported `threadExecutionWidth` /
/// `maxTotalThreadsPerThreadgroup`.
pub fn run_skin_whiten(
    runtime: &'static MetalRuntime,
    input: &SkinWhitenInput<'_>,
) -> Result<SkinWhitenResult, MetalError> {
    let pixels = validate(input)?;
    let width = input.width;
    let height = input.height;
    let byte_len = pixels * 4;

    let mut workspace = runtime.begin()?;

    let source = workspace.upload(input.bgra)?;
    let mask = workspace.upload(input.mask)?;
    let lut = workspace.upload(f32_slice_as_bytes(input.lut))?;
    let blurred = workspace.alloc(byte_len)?;
    let output = workspace.alloc(byte_len)?;

    let blur_params = SkinWhitenBlurParams {
        width,
        height,
        reserved0: 0,
        reserved1: 0,
    };
    let blur_params = workspace.upload(as_bytes(&blur_params))?;

    let whiten_params = O3MaskedWhitenParams {
        surface_width: width,
        surface_height: height,
        mask_x: 0,
        mask_y: 0,
        mask_width: width,
        mask_height: height,
        pixel_count: pixels as u32,
        flags: 0,
        alpha: input.degree,
        reserved0: 0.0,
        reserved1: 0.0,
        reserved2: 0.0,
    };
    let whiten_params = workspace.upload(as_bytes(&whiten_params))?;

    let blur_limits = workspace.pipeline_limits(BLUR_KEY)?;
    let blur_grid = Grid::for_2d(width, height, blur_limits);
    workspace.dispatch(
        BLUR_KEY,
        &[
            Binding::Buffer(source),
            Binding::Buffer(blurred),
            Binding::Buffer(blur_params),
        ],
        blur_grid,
    )?;

    let whiten_limits = workspace.pipeline_limits(WHITEN_KEY)?;
    let whiten_grid = Grid::for_2d(width, height, whiten_limits);
    workspace.dispatch(
        WHITEN_KEY,
        &[
            Binding::Buffer(blurred),
            Binding::Buffer(output),
            Binding::Buffer(mask),
            Binding::Buffer(lut),
            Binding::Buffer(whiten_params),
        ],
        whiten_grid,
    )?;

    let frame = workspace.commit(output)?;
    let generation = frame.generation();
    let mut dst = vec![0u8; byte_len];
    frame.read_into(&mut dst)?;
    Ok(SkinWhitenResult {
        pixels: dst,
        generation,
        blur_limits,
        whiten_limits,
        blur_grid,
        whiten_grid,
    })
}

/// CPU twin of `skin_whiten_box_blur`: clamped-edge 3x3 box blur, every channel
/// including alpha, `(sum + 4) / 9` in integer arithmetic (identical bytes).
pub fn cpu_box_blur_bgra(source: &[u8], width: usize, height: usize) -> Vec<u8> {
    if width == 0 || height == 0 || source.len() < width * height * 4 {
        return Vec::new();
    }
    let mut out = vec![0u8; width * height * 4];
    let last_x = width as i32 - 1;
    let last_y = height as i32 - 1;
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) * 4;
            for channel in 0..4 {
                let mut sum = 0u32;
                for dy in -1..=1 {
                    let sy = (y as i32 + dy).clamp(0, last_y) as usize;
                    let row = sy * width;
                    for dx in -1..=1 {
                        let sx = (x as i32 + dx).clamp(0, last_x) as usize;
                        sum += u32::from(source[(row + sx) * 4 + channel]);
                    }
                }
                out[index + channel] = ((sum + 4) / 9) as u8;
            }
        }
    }
    out
}

/// Full CPU reference: box blur, then the portrait crate's masked-LUT blend.
///
/// The MSL kernel packs BGRA (R at byte 2), while
/// `masked_lut_blend`/`Image` are RGBA (R at byte 0). The blur is
/// channel-order-agnostic, so only the handoff to and from the portrait
/// reference swaps the R/B bytes. The returned bytes are BGRA, matching the
/// device output.
pub fn cpu_reference(input: &SkinWhitenInput<'_>) -> Result<Vec<u8>, MetalError> {
    validate(input)?;
    let blurred = cpu_box_blur_bgra(input.bgra, input.width as usize, input.height as usize);
    let mut rgba = blurred;
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    let image = PortraitImage::new(input.width, input.height, rgba)
        .ok_or(MetalError::Invalid("CPU reference image dimensions"))?;
    let blended = masked_lut_blend(&image, input.mask, input.lut, input.degree)
        .map_err(|refusal| {
            MetalError::Compile(format!("masked_lut_blend refused the input: {refusal:?}"))
        })?;
    let mut bgra = blended.pixels;
    for pixel in bgra.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
    Ok(bgra)
}

/// Deterministic synthetic LUT for tests and local probes.
///
/// Not a product asset: each R/G/B cell maps through a mild per-channel linear
/// transform in `[0.075, 0.925]`, with alpha 1.0. The same atlas is fed to the
/// GPU and the CPU reference, so it tests the chain rather than a vendor LUT.
pub fn synthetic_test_lut() -> Vec<f32> {
    let mut atlas = vec![0.0f32; LUT_ATLAS_FLOATS];
    for b in 0..64usize {
        for g in 0..64usize {
            for r in 0..64usize {
                let y = (b >> 3) * 64 + g;
                let x = (b & 7) * 64 + r;
                let index = (y * LUT_ATLAS_WIDTH + x) * 4;
                let rf = r as f32 / 63.0;
                let gf = g as f32 / 63.0;
                let bf = b as f32 / 63.0;
                atlas[index] = rf * 0.82 + 0.09;
                atlas[index + 1] = gf * 0.86 + 0.07;
                atlas[index + 2] = bf * 0.78 + 0.11;
                atlas[index + 3] = 1.0;
            }
        }
    }
    atlas
}
