//! Model preprocessing conversions shared by every ONNX head.
//!
//! Three conversions, each with a CPU reference here and a bit-identical CUDA
//! twin in `native/cuda/model_preprocess.cu`:
//!
//! * RGBA8 (stride-padded, ROI, bilinear resize) -> planar NCHW f32
//! * NHWC f32 -> channel-planar f32 (C=3, or C=4 for the alpha plane)
//! * interleaved f32 -> interleaved u8 through the Rust `byte_away` byte law
//!
//! The CPU references are the acceptance oracle: they are what the engine's
//! per-model preprocesses do today (`engine/src/aimask`, `autorotate`,
//! `matchlook`, `closedeyes`), lifted into one shared shape. They are not a
//! vendor-parity claim: a model consumer adopts this lane only after its own
//! same-job golden says the layout, mean/std and byte rule match.

use crate::Refusal;

#[cfg(windows)]
use crate::cuda::{CudaError, Dispatch, Nvidia};

pub const NCHW_KERNEL: &str = "mpp_rgba8_to_nchw_kernel";
pub const PLANAR_KERNEL: &str = "mpp_nhwc4_to_planar_kernel";
pub const QUANT_KERNEL: &str = "mpp_f32_to_u8_kernel";

const BLOCK: u32 = 128;
const PTX: &str = include_str!("../native/cuda/model_preprocess.ptx");

/// 80-byte argument block, matching `native/cuda/model_preprocess.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NchwUniforms {
    pub src_width: u32,
    pub src_height: u32,
    /// Pixels per source row; must be at least `src_width`.
    pub src_stride: u32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_w: u32,
    pub roi_h: u32,
    pub out_w: u32,
    pub out_h: u32,
    /// 3 or 4.
    pub channels: u32,
    /// Per-sample divide before the lerp, normally 255.0.
    pub divisor: f32,
    pub mean: [f32; 4],
    pub std_dev: [f32; 4],
    /// Non-zero reads bytes b,g,r for channels 0..2.
    pub swap_rb: u32,
}

/// 16-byte argument block, matching `native/cuda/model_preprocess.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanarUniforms {
    pub width: u32,
    pub height: u32,
    /// Pixels per source row; must be at least `width`.
    pub stride: u32,
    /// 1..=4.
    pub channels: u32,
}

/// 32-byte argument block, matching `native/cuda/model_preprocess.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantUniforms {
    pub width: u32,
    pub height: u32,
    /// Pixels per source row; must be at least `width`.
    pub stride: u32,
    /// 1..=4.
    pub channels: u32,
    pub scale: f32,
    pub offset: f32,
    pub pad0: u32,
    pub pad1: u32,
}

const _: () = assert!(std::mem::size_of::<NchwUniforms>() == 80);
const _: () = assert!(std::mem::size_of::<PlanarUniforms>() == 16);
const _: () = assert!(std::mem::size_of::<QuantUniforms>() == 32);

/// Midpoint-away-from-zero byte rule used across the recovered pixel stages.
fn byte_away(value: f32) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    let adjusted = if value >= 0.0 {
        value + 0.5
    } else {
        value - 0.5
    };
    (adjusted as i32).clamp(0, 255) as u8
}

/// Half-pixel-centred bilinear axis, shared by both CPU references.
///
/// `source`/`target` are the extents of the sampling window and the output;
/// the multiply precedes the divide so the CPU and CUDA rounding agree.
fn axis(index: u32, source: u32, target: u32) -> (u32, u32, f32) {
    let scaled = ((index as f32 + 0.5) * source as f32 / target as f32) - 0.5;
    let last = source - 1;
    let low = scaled.max(0.0).min(last as f32);
    let i0 = low as u32;
    let i1 = (i0 + 1).min(last);
    let frac = (scaled - i0 as f32).clamp(0.0, 1.0);
    (i0, i1, frac)
}

fn check_nchw(source: &[u8], uniforms: &NchwUniforms) -> Result<usize, Refusal> {
    let n = uniforms;
    if n.src_width == 0 || n.src_height == 0 || n.out_w == 0 || n.out_h == 0 {
        return Err(Refusal::ShapeMismatch("preprocess extent"));
    }
    if n.channels < 3 || n.channels > 4 {
        return Err(Refusal::ShapeMismatch("preprocess channels"));
    }
    if n.src_stride < n.src_width {
        return Err(Refusal::ShapeMismatch("preprocess stride"));
    }
    if n.roi_w == 0
        || n.roi_h == 0
        || n.roi_x
            .checked_add(n.roi_w)
            .ok_or(Refusal::ValueOutOfRange("roi"))?
            > n.src_width
        || n.roi_y
            .checked_add(n.roi_h)
            .ok_or(Refusal::ValueOutOfRange("roi"))?
            > n.src_height
    {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    let row_bytes = (n.src_stride as usize)
        .checked_mul(4)
        .ok_or(Refusal::ShapeMismatch("preprocess stride"))?;
    let expected = (n.src_height as usize)
        .checked_mul(row_bytes)
        .ok_or(Refusal::ShapeMismatch("preprocess source"))?;
    if source.len() != expected {
        return Err(Refusal::ShapeMismatch("preprocess source"));
    }
    if !n.divisor.is_finite() || n.divisor == 0.0 {
        return Err(Refusal::ValueOutOfRange("divisor"));
    }
    for channel in 0..n.channels as usize {
        let std_dev = n.std_dev[channel];
        if !n.mean[channel].is_finite() || !std_dev.is_finite() || std_dev == 0.0 {
            return Err(Refusal::ValueOutOfRange("mean/std_dev"));
        }
    }
    Ok((n.out_w as usize) * (n.out_h as usize))
}

/// CPU reference for `NCHW_KERNEL`: ROI bilinear resize into planar f32.
///
/// Samples are divided by `divisor` before the lerp, then normalized per
/// output channel. Alpha is channel 3 and is never byte-swapped.
pub fn rgba8_to_nchw_cpu(source: &[u8], uniforms: &NchwUniforms) -> Result<Vec<f32>, Refusal> {
    let plane = check_nchw(source, uniforms)?;
    let n = uniforms;
    let mut out = vec![0.0f32; plane * n.channels as usize];
    for y in 0..n.out_h {
        let (y0, y1, fy) = axis(y, n.roi_h, n.out_h);
        let row0 = ((y0 + n.roi_y) as usize * n.src_stride as usize) * 4;
        let row1 = ((y1 + n.roi_y) as usize * n.src_stride as usize) * 4;
        for x in 0..n.out_w {
            let (x0, x1, fx) = axis(x, n.roi_w, n.out_w);
            let (x0, x1) = (x0 + n.roi_x, x1 + n.roi_x);
            let index = (y * n.out_w + x) as usize;
            for channel in 0..n.channels as usize {
                let byte = if channel > 2 {
                    3
                } else if n.swap_rb != 0 {
                    [2, 1, 0][channel]
                } else {
                    channel
                };
                let at =
                    |px: u32, row: usize| source[row + (px as usize) * 4 + byte] as f32 / n.divisor;
                let top = at(x0, row0) * (1.0 - fx) + at(x1, row0) * fx;
                let bottom = at(x0, row1) * (1.0 - fx) + at(x1, row1) * fx;
                let sample = top * (1.0 - fy) + bottom * fy;
                out[channel * plane + index] = (sample - n.mean[channel]) / n.std_dev[channel];
            }
        }
    }
    Ok(out)
}

fn check_planar(source: &[f32], uniforms: &PlanarUniforms) -> Result<usize, Refusal> {
    if uniforms.width == 0 || uniforms.height == 0 {
        return Err(Refusal::ShapeMismatch("planar extent"));
    }
    if uniforms.channels == 0 || uniforms.channels > 4 {
        return Err(Refusal::ShapeMismatch("planar channels"));
    }
    if uniforms.stride < uniforms.width {
        return Err(Refusal::ShapeMismatch("planar stride"));
    }
    let expected = uniforms.height as usize * uniforms.stride as usize * 4;
    if source.len() != expected {
        return Err(Refusal::ShapeMismatch("planar source"));
    }
    if source.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("planar source"));
    }
    Ok((uniforms.width as usize) * (uniforms.height as usize))
}

/// CPU reference for `PLANAR_KERNEL`: NHWC f32 with padded rows into `C`
/// tight f32 planes, skipping the `stride - width` padding columns.
///
/// C=3 gives the three colour planes an ONNX head consumes; C=4 additionally
/// emits the alpha/mask plane, which is why the source is always 4-wide.
pub fn nhwc4_to_planar_cpu(source: &[f32], uniforms: &PlanarUniforms) -> Result<Vec<f32>, Refusal> {
    let plane = check_planar(source, uniforms)?;
    let mut out = vec![0.0f32; plane * uniforms.channels as usize];
    for y in 0..uniforms.height {
        let row = (y as usize * uniforms.stride as usize) * 4;
        for x in 0..uniforms.width {
            let base = row + (x as usize) * 4;
            let index = (y * uniforms.width + x) as usize;
            for channel in 0..uniforms.channels as usize {
                out[channel * plane + index] = source[base + channel];
            }
        }
    }
    Ok(out)
}

fn check_quant(source: &[f32], uniforms: &QuantUniforms) -> Result<usize, Refusal> {
    if uniforms.width == 0 || uniforms.height == 0 {
        return Err(Refusal::ShapeMismatch("quant extent"));
    }
    if uniforms.channels == 0 || uniforms.channels > 4 {
        return Err(Refusal::ShapeMismatch("quant channels"));
    }
    if uniforms.stride < uniforms.width {
        return Err(Refusal::ShapeMismatch("quant stride"));
    }
    let row = uniforms.stride as usize * uniforms.channels as usize;
    let expected = uniforms.height as usize * row;
    if source.len() != expected {
        return Err(Refusal::ShapeMismatch("quant source"));
    }
    if !uniforms.scale.is_finite() || !uniforms.offset.is_finite() {
        return Err(Refusal::ValueOutOfRange("quant scale/offset"));
    }
    Ok((uniforms.width as usize) * (uniforms.height as usize))
}

/// CPU reference for `QUANT_KERNEL`: padded interleaved f32 to tight u8.
///
/// Model logits arrive unbounded, so the byte rule must saturate rather than
/// wrap: this reuses the same `byte_away` law as the pixel stages.
pub fn f32_to_u8_cpu(source: &[f32], uniforms: &QuantUniforms) -> Result<Vec<u8>, Refusal> {
    let pixels = check_quant(source, uniforms)?;
    let channels = uniforms.channels as usize;
    let mut out = vec![0u8; pixels * channels];
    for y in 0..uniforms.height {
        let row = y as usize * uniforms.stride as usize * channels;
        for x in 0..uniforms.width {
            let base = row + (x as usize) * channels;
            let dest = ((y * uniforms.width + x) as usize) * channels;
            for channel in 0..channels {
                out[dest + channel] =
                    byte_away(source[base + channel] * uniforms.scale + uniforms.offset);
            }
        }
    }
    Ok(out)
}

#[cfg(windows)]
pub struct ModelPreprocess {
    gpu: Nvidia,
}

#[cfg(windows)]
fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

#[cfg(windows)]
fn words_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte word")))
        .collect()
}

#[cfg(windows)]
impl ModelPreprocess {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: Nvidia::open_ptx(PTX)?,
        })
    }

    /// Runs one 1-D kernel over `pixels` threads: `(input, output, uniforms)`.
    ///
    /// Every kernel here maps one thread to one pixel and loops the channels,
    /// so `pixels` is the pixel count and `output_bytes` the full element size.
    ///
    /// Input and output lengths differ for all three conversions, so this uses
    /// the explicit buffer path instead of the shared staging helper, whose
    /// output is sized from `inputs[0]`.
    fn run_twoway(
        &self,
        kernel: &'static str,
        pixels: usize,
        input: &[u8],
        output_bytes: usize,
        uniforms_ptr: *mut std::ffi::c_void,
    ) -> Result<(Vec<u8>, Dispatch), CudaError> {
        self.gpu.with_reusable_buffers(
            "model-preprocess.twoway",
            &[input.len(), output_bytes],
            |buffers| {
                buffers.upload(0, input)?;
                let mut input_arg = buffers.handle(0);
                let mut output_arg = buffers.handle(1);
                let mut params: [*mut std::ffi::c_void; 3] = [
                    &mut input_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut output_arg as *mut u64 as *mut std::ffi::c_void,
                    uniforms_ptr,
                ];
                let dispatch = buffers.launch(
                    kernel,
                    (pixels.div_ceil(BLOCK as usize) as u32, 1),
                    (BLOCK, 1),
                    &mut params,
                )?;
                let mut output = vec![0u8; output_bytes];
                buffers.download(1, &mut output)?;
                buffers.mark_consumed();
                Ok((output, dispatch))
            },
        )
    }

    /// GPU twin of `rgba8_to_nchw_cpu`.
    pub fn rgba8_to_nchw(
        &self,
        source: &[u8],
        uniforms: &NchwUniforms,
    ) -> Result<(Vec<f32>, Dispatch), CudaError> {
        let plane = check_nchw(source, uniforms).map_err(CudaError::from)?;
        let elements = plane * uniforms.channels as usize;
        let mut block = *uniforms;
        let (bytes, dispatch) = self.run_twoway(
            NCHW_KERNEL,
            plane,
            source,
            elements * 4,
            &mut block as *mut NchwUniforms as *mut std::ffi::c_void,
        )?;
        Ok((words_to_f32(&bytes), dispatch))
    }

    /// GPU twin of `nhwc4_to_planar_cpu`.
    pub fn nhwc4_to_planar(
        &self,
        source: &[f32],
        uniforms: &PlanarUniforms,
    ) -> Result<(Vec<f32>, Dispatch), CudaError> {
        let plane = check_planar(source, uniforms).map_err(CudaError::from)?;
        let elements = plane * uniforms.channels as usize;
        let mut block = *uniforms;
        let (bytes, dispatch) = self.run_twoway(
            PLANAR_KERNEL,
            plane,
            f32_bytes(source),
            elements * 4,
            &mut block as *mut PlanarUniforms as *mut std::ffi::c_void,
        )?;
        Ok((words_to_f32(&bytes), dispatch))
    }

    /// GPU twin of `f32_to_u8_cpu`.
    pub fn f32_to_u8(
        &self,
        source: &[f32],
        uniforms: &QuantUniforms,
    ) -> Result<(Vec<u8>, Dispatch), CudaError> {
        let pixels = check_quant(source, uniforms).map_err(CudaError::from)?;
        let elements = pixels * uniforms.channels as usize;
        let mut block = *uniforms;
        let (bytes, dispatch) = self.run_twoway(
            QUANT_KERNEL,
            pixels,
            f32_bytes(source),
            elements,
            &mut block as *mut QuantUniforms as *mut std::ffi::c_void,
        )?;
        Ok((bytes, dispatch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGENET_MEAN: [f32; 4] = [0.485, 0.456, 0.406, 0.0];
    const IMAGENET_STD: [f32; 4] = [0.229, 0.224, 0.225, 1.0];

    fn nchw_uniforms(src_w: u32, src_h: u32, stride: u32, out: u32) -> NchwUniforms {
        NchwUniforms {
            src_width: src_w,
            src_height: src_h,
            src_stride: stride,
            roi_x: 0,
            roi_y: 0,
            roi_w: src_w,
            roi_h: src_h,
            out_w: out,
            out_h: out,
            channels: 3,
            divisor: 255.0,
            mean: IMAGENET_MEAN,
            std_dev: IMAGENET_STD,
            swap_rb: 0,
        }
    }

    fn planar_uniforms(w: u32, h: u32, stride: u32, channels: u32) -> PlanarUniforms {
        PlanarUniforms {
            width: w,
            height: h,
            stride,
            channels,
        }
    }

    fn quant_uniforms(w: u32, h: u32, stride: u32, channels: u32) -> QuantUniforms {
        QuantUniforms {
            width: w,
            height: h,
            stride,
            channels,
            scale: 255.0,
            offset: 0.0,
            pad0: 0,
            pad1: 0,
        }
    }

    /// Deterministic pseudo-random bytes so CPU and GPU see identical inputs.
    fn noise(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223)
                    .rotate_left(5);
                (state >> 24) as u8
            })
            .collect()
    }

    fn noise_f32(len: usize, seed: u32) -> Vec<f32> {
        noise(len * 4, seed)
            .chunks_exact(4)
            .map(|word| {
                let raw = u32::from_le_bytes(word.try_into().unwrap());
                // Map to [-1.5, 1.5] so the quantizer sees both saturations.
                ((raw % 300_000) as f32 / 100_000.0) - 1.5
            })
            .collect()
    }

    fn assert_f32_bits_eq(cpu: &[f32], gpu: &[f32], context: &str) {
        assert_eq!(cpu.len(), gpu.len(), "{context}: output length differs");
        if let Some((index, (cpu_value, gpu_value))) = cpu
            .iter()
            .zip(gpu)
            .enumerate()
            .find(|(_, (cpu_value, gpu_value))| cpu_value.to_bits() != gpu_value.to_bits())
        {
            panic!(
                "{context}: f32 bits differ at {index}: cpu={:#010x} gpu={:#010x}",
                cpu_value.to_bits(),
                gpu_value.to_bits()
            );
        }
    }

    #[test]
    fn argument_blocks_match_the_c_headers() {
        assert_eq!(std::mem::size_of::<NchwUniforms>(), 80);
        assert_eq!(std::mem::size_of::<PlanarUniforms>(), 16);
        assert_eq!(std::mem::size_of::<QuantUniforms>(), 32);
        assert_eq!(std::mem::align_of::<NchwUniforms>(), 4);
    }

    #[test]
    fn constant_image_maps_to_the_exact_normalized_plane() {
        let uniforms = nchw_uniforms(4, 4, 4, 2);
        let source = vec![128u8; 4 * 4 * 4];
        let out = rgba8_to_nchw_cpu(&source, &uniforms).unwrap();
        assert_eq!(out.len(), 3 * 4);
        for channel in 0..3 {
            let expected = (128.0f32 / 255.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            for value in &out[channel * 4..channel * 4 + 4] {
                assert_eq!(*value, expected);
            }
        }
    }

    #[test]
    fn roi_stride_and_byte_order_are_honoured() {
        // Two rows of three pixels with a one-pixel padded tail; the ROI is the
        // right-hand 2x2, read as BGR into channel order 0..2.
        let stride = 4;
        let mut source = vec![0u8; 2 * stride * 4];
        for (y, row) in source.chunks_mut(stride * 4).enumerate() {
            for (x, pixel) in row.chunks_mut(4).enumerate() {
                pixel.copy_from_slice(&[(x * 20 + y) as u8, (x * 30) as u8, (y * 40) as u8, 9]);
            }
        }
        let uniforms = NchwUniforms {
            roi_x: 1,
            roi_y: 0,
            roi_w: 2,
            roi_h: 2,
            swap_rb: 1,
            mean: [0.0; 4],
            std_dev: [1.0; 4],
            ..nchw_uniforms(3, 2, stride as u32, 2)
        };
        let out = rgba8_to_nchw_cpu(&source, &uniforms).unwrap();
        // Output 2x2 at 1:1 scale lands exactly on the ROI corners. Pixel
        // bytes are [x*20 + y, x*30, y*40, 9], and swap_rb maps channel 0..2
        // to bytes 2,1,0.
        let read = |channel: usize, x: usize, y: usize| out[channel * 4 + y * 2 + x];
        assert_eq!(read(0, 0, 0), 0.0 / 255.0); // ROI (1,0): byte 2 = 0*40
        assert_eq!(read(1, 0, 0), 30.0 / 255.0); // byte 1 = 1*30
        assert_eq!(read(2, 0, 0), 20.0 / 255.0); // byte 0 = 1*20 + 0
        assert_eq!(read(0, 1, 1), 40.0 / 255.0); // ROI (2,1): byte 2 = 1*40
        assert_eq!(read(2, 1, 1), 41.0 / 255.0); // byte 0 = 2*20 + 1
    }

    #[test]
    fn planar_split_skips_padding_and_keeps_planes_tight() {
        let uniforms = planar_uniforms(2, 2, 3, 3);
        let mut source = vec![-9.0f32; 2 * 3 * 4];
        for y in 0..2u32 {
            for x in 0..2u32 {
                let base = (y as usize * 3 + x as usize) * 4;
                for c in 0..4 {
                    source[base + c] = (y * 100 + x * 10 + c as u32) as f32;
                }
            }
        }
        let out = nhwc4_to_planar_cpu(&source, &uniforms).unwrap();
        assert_eq!(out.len(), 3 * 4);
        assert_eq!(&out[0..4], &[0.0, 10.0, 100.0, 110.0]);
        assert_eq!(&out[4..8], &[1.0, 11.0, 101.0, 111.0]);
        assert_eq!(&out[8..12], &[2.0, 12.0, 102.0, 112.0]);
    }

    #[test]
    fn quantizer_saturates_and_follows_the_byte_law() {
        let uniforms = QuantUniforms {
            scale: 255.0,
            offset: 0.0,
            ..quant_uniforms(1, 1, 1, 4)
        };
        let out = f32_to_u8_cpu(&[-0.4, 0.5, 0.5019608, 1.7], &uniforms).unwrap();
        assert_eq!(out, [0, 128, 128, 255]);
        // Midpoint away from zero, then clamp: 0.5019608 * 255 = 128.0, 1.7 saturates.
        let offset = QuantUniforms {
            scale: 0.0,
            offset: -3.0,
            ..uniforms
        };
        assert_eq!(f32_to_u8_cpu(&[0.0; 4], &offset).unwrap(), [0; 4]);
    }

    #[test]
    fn shape_gates_refuse_before_any_work() {
        assert_eq!(
            rgba8_to_nchw_cpu(&[0; 16], &nchw_uniforms(4, 4, 4, 2)),
            Err(Refusal::ShapeMismatch("preprocess source"))
        );
        assert_eq!(
            rgba8_to_nchw_cpu(&[0; 64], &nchw_uniforms(4, 4, 3, 2)),
            Err(Refusal::ShapeMismatch("preprocess stride"))
        );
        let mut roi = nchw_uniforms(4, 4, 4, 2);
        roi.roi_w = 5;
        assert_eq!(
            rgba8_to_nchw_cpu(&[0; 64], &roi),
            Err(Refusal::ValueOutOfRange("roi"))
        );
        let mut zero_std = nchw_uniforms(4, 4, 4, 2);
        zero_std.std_dev[1] = 0.0;
        assert_eq!(
            rgba8_to_nchw_cpu(&[0; 64], &zero_std),
            Err(Refusal::ValueOutOfRange("mean/std_dev"))
        );
        assert_eq!(
            nhwc4_to_planar_cpu(&[0.0; 8], &planar_uniforms(2, 1, 2, 5)),
            Err(Refusal::ShapeMismatch("planar channels"))
        );
        assert_eq!(
            f32_to_u8_cpu(&[0.0; 3], &quant_uniforms(1, 1, 1, 4)),
            Err(Refusal::ShapeMismatch("quant source"))
        );
    }

    #[cfg(windows)]
    fn gpu() -> Option<ModelPreprocess> {
        match ModelPreprocess::open() {
            Ok(gpu) => Some(gpu),
            Err(error) => {
                if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
                    panic!("CUDA required but unavailable: {error:?}");
                }
                eprintln!("SKIPPED (no NVIDIA execution path): {error:?}");
                None
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_nchw_matches_the_cpu_bit_for_bit() {
        let Some(gpu) = gpu() else { return };
        for (src_w, src_h, stride, out) in [(5, 4, 5, 3), (6, 6, 9, 8), (17, 3, 17, 1)] {
            let mut uniforms = nchw_uniforms(src_w, src_h, stride, out);
            let source = noise((src_h * stride * 4) as usize, 7);
            for swap in [0, 1] {
                uniforms.swap_rb = swap;
                uniforms.channels = if swap == 0 { 4 } else { 3 };
                let cpu = rgba8_to_nchw_cpu(&source, &uniforms).unwrap();
                let (gpu_out, _) = gpu.rgba8_to_nchw(&source, &uniforms).unwrap();
                eprintln!(
                    "WL46_NCHW_SHAPE {src_w}x{src_h} stride={stride} out={out} swap={swap} channels={} bytes={}",
                    uniforms.channels,
                    cpu.len() * std::mem::size_of::<f32>()
                );
                assert_f32_bits_eq(
                    &cpu,
                    &gpu_out,
                    &format!("nchw {src_w}x{src_h} stride {stride} swap {swap}"),
                );
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_planar_and_quant_match_the_cpu_bit_for_bit() {
        let Some(gpu) = gpu() else { return };
        for (w, h, stride) in [(4, 4, 4), (5, 3, 8), (1, 1, 1)] {
            let source = noise_f32((h * stride * 4) as usize, 11);
            for channels in [1u32, 3, 4] {
                let uniforms = planar_uniforms(w, h, stride, channels);
                let cpu = nhwc4_to_planar_cpu(&source, &uniforms).unwrap();
                let (gpu_out, _) = gpu.nhwc4_to_planar(&source, &uniforms).unwrap();
                eprintln!(
                    "WL46_PLANAR_SHAPE {w}x{h} stride={stride} channels={channels} bytes={}",
                    cpu.len() * std::mem::size_of::<f32>()
                );
                assert_f32_bits_eq(&cpu, &gpu_out, &format!("planar {w}x{h} c{channels}"));
            }
            for channels in [1u32, 3, 4] {
                let quant_source = noise_f32((h * stride * channels) as usize, 13);
                let uniforms = quant_uniforms(w, h, stride, channels);
                let cpu = f32_to_u8_cpu(&quant_source, &uniforms).unwrap();
                let (gpu_out, _) = gpu.f32_to_u8(&quant_source, &uniforms).unwrap();
                eprintln!(
                    "WL46_QUANT_SHAPE {w}x{h} stride={stride} channels={channels} bytes={}",
                    cpu.len()
                );
                assert_eq!(cpu, gpu_out, "quant {w}x{h} c{channels}");
            }
        }
    }
}
