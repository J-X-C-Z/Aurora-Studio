//! CUDA twin of the C1 blemish gain-map consumer.
//!
//! The kernel in `native/cuda/blemish_gainmap.cu` is one thread per output
//! pixel: it bilinearly samples the 450-long-side RGBA32F map with the same
//! expression the Rust scalar oracle uses, applies the amount/confidence curve
//! and the optional protection mask, then multiplies each channel.
//!
//! It is compiled with `-fmad=false` and every arithmetic step is an explicit
//! `__f*_rn` intrinsic, so the GPU performs the same sequence of IEEE-754
//! binary32 roundings as the CPU. `AURORA_BLEMISH_GPU=0` keeps the CPU bytes.

use crate::blemish_gainmap::{validate_gainmap_input, GainMap, SpotlessGainmapInputs};
use aurora_portrait::cuda::{CudaError, Dispatch, Nvidia, PinnedBuffer};
use aurora_portrait::Refusal;
use std::slice;

const PTX: &str = include_str!("../native/cuda/blemish_gainmap.ptx");
const KERNEL_16U: &str = "blemish_gainmap_16u_kernel";
const KERNEL_8U: &str = "blemish_gainmap_8u_kernel";
const BLOCK: (u32, u32) = (16, 16);

/// Kernel argument block; field-for-field with `BlemishParams` in the .cu.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Params {
    width: u32,
    height: u32,
    map_width: u32,
    map_height: u32,
    sx: f32,
    sy: f32,
    amount_scale: f32,
    curve_t: f32,
    has_curve: u32,
    has_mask: u32,
}

const _: () = assert!(std::mem::size_of::<Params>() == 40);

/// Opens the bundled module; the shared driver caches one module per image.
pub fn open() -> Result<Nvidia, CudaError> {
    Nvidia::open_ptx(PTX)
}

/// Allocates a device buffer and uploads `bytes` into it.
pub fn alloc_and_upload<'a>(
    gpu: &'a Nvidia,
    bytes: &[u8],
) -> Result<aurora_portrait::cuda::DeviceBuffer<'a>, CudaError> {
    let buffer = gpu.alloc_buffer(bytes.len())?;
    gpu.upload_buffer(&buffer, bytes)?;
    Ok(buffer)
}

impl Params {
    fn new(
        width: u32,
        height: u32,
        map: &GainMap,
        amount: f32,
        has_mask: bool,
    ) -> Self {
        let curve_t = if amount > 0.25 {
            (4.0 * (amount - 0.75)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            width,
            height,
            map_width: map.width,
            map_height: map.height,
            sx: map.width as f32 / width as f32,
            sy: map.height as f32 / height as f32,
            amount_scale: 1.0 / (1.0 - amount).max(0.01),
            curve_t,
            has_curve: u32::from(amount > 0.25),
            has_mask: u32::from(has_mask),
        }
    }
}

fn grid(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(BLOCK.0).max(1), height.div_ceil(BLOCK.1).max(1))
}

unsafe fn bytes_of<T>(values: &[T]) -> &[u8] {
    unsafe {
        slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

unsafe fn bytes_of_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, std::mem::size_of_val(values))
    }
}

/// Runs the 16-bit consumer, including H2D and D2H, and returns the dispatch
/// record that proves which kernel and device produced the pixels.
pub fn apply_16u(
    input: &SpotlessGainmapInputs<'_>,
) -> Result<(Vec<u16>, Dispatch), CudaError> {
    let (w, h) = validate_gainmap_input(input)?;
    if input.amount == 0.0 {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount").into());
    }
    let gpu = Nvidia::open_ptx(PTX)?;
    let map = input.gain_map;
    let params = Params::new(w as u32, h as u32, map, input.amount, input.protection_mask.is_some());
    let src_bytes = unsafe { bytes_of(input.rgb16) };
    let map_bytes = unsafe { bytes_of(&map.rgba) };
    let mask_bytes = input
        .protection_mask
        .map(|mask| unsafe { bytes_of(mask) });

    let mut out = vec![0u16; w * h * 3];
    let out_bytes = w * h * 3 * 2;
    let has_mask = mask_bytes.is_some();
    let out_index = if has_mask { 3 } else { 2 };
    let mut sizes = vec![src_bytes.len(), map_bytes.len()];
    if let Some(bytes) = mask_bytes {
        sizes.push(bytes.len());
    }
    sizes.push(out_bytes);
    let pool = if has_mask { "blemish-16u-mask" } else { "blemish-16u" };

    let dispatch = gpu.with_reusable_buffers(pool, &sizes, |session| {
        session.upload(0, src_bytes)?;
        session.upload(1, map_bytes)?;
        if let (true, Some(bytes)) = (has_mask, mask_bytes) {
            session.upload(2, bytes)?;
        }
        let mut src_ptr = session.handle(0);
        let mut map_ptr = session.handle(1);
        let mut mask_ptr = if has_mask { session.handle(2) } else { 0 };
        let mut out_ptr = session.handle(out_index);
        let mut params_arg = params;
        let mut args: [*mut std::ffi::c_void; 5] = [
            &mut src_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut map_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut mask_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut params_arg as *mut Params as *mut std::ffi::c_void,
        ];
        let dispatch = session.launch(
            KERNEL_16U,
            grid(w as u32, h as u32),
            BLOCK,
            &mut args,
        )?;
        session.download(out_index, unsafe { bytes_of_mut(&mut out) })?;
        session.mark_consumed();
        Ok(dispatch)
    })?;
    Ok((out, dispatch))
}

/// Runs the 8-bit consumer with the same law as the 16-bit path:
/// `out = round(src/255 * (gain*w + 1-w) * 255)`.
pub fn apply_8u(
    rgb8: &[u8],
    width: u32,
    height: u32,
    map: &GainMap,
    amount: f32,
    protection_mask: Option<&[u8]>,
) -> Result<(Vec<u8>, Dispatch), CudaError> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || rgb8.len() != w * h * 3 {
        return Err(Refusal::ShapeMismatch("blemish gain map frame").into());
    }
    if !amount.is_finite() || !(0.0..=1.0).contains(&amount) {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount").into());
    }
    if amount == 0.0 {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount").into());
    }
    if map.width == 0
        || map.height == 0
        || map.rgba.len() != map.width as usize * map.height as usize * 4
    {
        return Err(Refusal::ShapeMismatch("blemish gain map").into());
    }
    if let Some(mask) = protection_mask {
        if mask.len() != w * h {
            return Err(Refusal::ShapeMismatch("blemish protection mask").into());
        }
    }

    let gpu = Nvidia::open_ptx(PTX)?;
    let params = Params::new(width, height, map, amount, protection_mask.is_some());
    let map_bytes = unsafe { bytes_of(&map.rgba) };
    let mask_bytes = protection_mask.map(|mask| unsafe { bytes_of(mask) });

    let mut out = vec![0u8; w * h * 3];
    let has_mask = mask_bytes.is_some();
    let out_index = if has_mask { 3 } else { 2 };
    let mut sizes = vec![rgb8.len(), map_bytes.len()];
    if let Some(bytes) = mask_bytes {
        sizes.push(bytes.len());
    }
    sizes.push(out.len());
    let pool = if has_mask { "blemish-8u-mask" } else { "blemish-8u" };

    let dispatch = gpu.with_reusable_buffers(pool, &sizes, |session| {
        session.upload(0, unsafe { bytes_of(rgb8) })?;
        session.upload(1, map_bytes)?;
        if let (true, Some(bytes)) = (has_mask, mask_bytes) {
            session.upload(2, bytes)?;
        }
        let mut src_ptr = session.handle(0);
        let mut map_ptr = session.handle(1);
        let mut mask_ptr = if has_mask { session.handle(2) } else { 0 };
        let mut out_ptr = session.handle(out_index);
        let mut params_arg = params;
        let mut args: [*mut std::ffi::c_void; 5] = [
            &mut src_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut map_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut mask_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut params_arg as *mut Params as *mut std::ffi::c_void,
        ];
        let dispatch =
            session.launch(KERNEL_8U, grid(width, height), BLOCK, &mut args)?;
        session.download(out_index, &mut out)?;
        session.mark_consumed();
        Ok(dispatch)
    })?;
    Ok((out, dispatch))
}

/// 16-bit consumer for caller-owned pinned host buffers: H2D + kernel + D2H
/// without the pageable staging penalty. The frame must already be in
/// `src_pinned`; the result lands in `out_pinned`.
pub fn apply_16u_pinned(
    input: &SpotlessGainmapInputs<'_>,
    src_pinned: &PinnedBuffer<'_>,
    out_pinned: &mut PinnedBuffer<'_>,
) -> Result<Dispatch, CudaError> {
    let (w, h) = validate_gainmap_input(input)?;
    if input.amount == 0.0 {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount").into());
    }
    let src_bytes = unsafe { bytes_of(input.rgb16) };
    let map_bytes = unsafe { bytes_of(&input.gain_map.rgba) };
    let out_bytes = w * h * 3 * 2;
    if src_pinned.bytes() < src_bytes.len() || out_pinned.bytes() < out_bytes {
        return Err(Refusal::ShapeMismatch("pinned blemish frame").into());
    }
    let gpu = open()?;
    let params = Params::new(
        w as u32,
        h as u32,
        input.gain_map,
        input.amount,
        input.protection_mask.is_some(),
    );
    let mask_bytes = input
        .protection_mask
        .map(|mask| unsafe { bytes_of(mask) });
    let has_mask = mask_bytes.is_some();
    let sizes = if has_mask {
        vec![src_bytes.len(), map_bytes.len(), mask_bytes.unwrap().len(), out_bytes]
    } else {
        vec![src_bytes.len(), map_bytes.len(), out_bytes]
    };
    let out_index = if has_mask { 3 } else { 2 };
    let pool = if has_mask { "blemish-16u-pinned-mask" } else { "blemish-16u-pinned" };
    gpu.with_reusable_buffers(pool, &sizes, |session| {
        session.upload_pinned(0, src_pinned)?;
        session.upload(1, map_bytes)?;
        if let Some(bytes) = mask_bytes {
            session.upload(2, bytes)?;
        }
        let mut src_ptr = session.handle(0);
        let mut map_ptr = session.handle(1);
        let mut mask_ptr = if has_mask { session.handle(2) } else { 0 };
        let mut out_ptr = session.handle(out_index);
        let mut params_arg = params;
        let mut args: [*mut std::ffi::c_void; 5] = [
            &mut src_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut map_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut mask_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut params_arg as *mut Params as *mut std::ffi::c_void,
        ];
        let dispatch =
            session.launch(KERNEL_16U, grid(w as u32, h as u32), BLOCK, &mut args)?;
        session.download_pinned(out_index, out_pinned)?;
        session.mark_consumed();
        Ok(dispatch)
    })
}

/// 8-bit twin of [`apply_16u_pinned`].
pub fn apply_8u_pinned(
    rgb8: &[u8],
    width: u32,
    height: u32,
    map: &GainMap,
    amount: f32,
    src_pinned: &PinnedBuffer<'_>,
    out_pinned: &mut PinnedBuffer<'_>,
) -> Result<Dispatch, CudaError> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || rgb8.len() != w * h * 3 {
        return Err(Refusal::ShapeMismatch("blemish gain map frame").into());
    }
    if !amount.is_finite() || !(0.0..=1.0).contains(&amount) || amount == 0.0 {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount").into());
    }
    let map_bytes = unsafe { bytes_of(&map.rgba) };
    let out_bytes = w * h * 3;
    if src_pinned.bytes() < rgb8.len() || out_pinned.bytes() < out_bytes {
        return Err(Refusal::ShapeMismatch("pinned blemish frame").into());
    }
    let gpu = open()?;
    let params = Params::new(width, height, map, amount, false);
    let sizes = vec![rgb8.len(), map_bytes.len(), out_bytes];
    gpu.with_reusable_buffers("blemish-8u-pinned", &sizes, |session| {
        session.upload_pinned(0, src_pinned)?;
        session.upload(1, map_bytes)?;
        let mut src_ptr = session.handle(0);
        let mut map_ptr = session.handle(1);
        let mut mask_ptr: u64 = 0;
        let mut out_ptr = session.handle(2);
        let mut params_arg = params;
        let mut args: [*mut std::ffi::c_void; 5] = [
            &mut src_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut map_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut mask_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
            &mut params_arg as *mut Params as *mut std::ffi::c_void,
        ];
        let dispatch = session.launch(KERNEL_8U, grid(width, height), BLOCK, &mut args)?;
        session.download_pinned(2, out_pinned)?;
        session.mark_consumed();
        Ok(dispatch)
    })
}

/// Device-resident entry point: no frame H2D/D2H, only the kernel. `src` and
/// `out` are existing device pointers for `width*height*3` u16 samples; `map`
/// is the device copy of `gain_map.rgba`.
pub fn apply_16u_device(
    src: u64,
    map: u64,
    mask: u64,
    out: u64,
    width: u32,
    height: u32,
    gain_map: &GainMap,
    amount: f32,
    has_mask: bool,
) -> Result<Dispatch, CudaError> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("blemish gain map frame").into());
    }
    if !amount.is_finite() || !(0.0..=1.0).contains(&amount) || amount == 0.0 {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount").into());
    }
    let gpu = Nvidia::open_ptx(PTX)?;
    let mut params = Params::new(width, height, gain_map, amount, has_mask);
    let mut src_ptr = src;
    let mut map_ptr = map;
    let mut mask_ptr = mask;
    let mut out_ptr = out;
    let mut args: [*mut std::ffi::c_void; 5] = [
        &mut src_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut map_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut mask_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut params as *mut Params as *mut std::ffi::c_void,
    ];
    gpu.launch_raw(KERNEL_16U, grid(width, height), BLOCK, &mut args)
}

/// Device-resident 8-bit entry point, same contract as [`apply_16u_device`].
pub fn apply_8u_device(
    src: u64,
    map: u64,
    mask: u64,
    out: u64,
    width: u32,
    height: u32,
    gain_map: &GainMap,
    amount: f32,
    has_mask: bool,
) -> Result<Dispatch, CudaError> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("blemish gain map frame").into());
    }
    if !amount.is_finite() || !(0.0..=1.0).contains(&amount) || amount == 0.0 {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount").into());
    }
    let gpu = open()?;
    let mut params = Params::new(width, height, gain_map, amount, has_mask);
    let mut src_ptr = src;
    let mut map_ptr = map;
    let mut mask_ptr = mask;
    let mut out_ptr = out;
    let mut args: [*mut std::ffi::c_void; 5] = [
        &mut src_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut map_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut mask_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut out_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut params as *mut Params as *mut std::ffi::c_void,
    ];
    gpu.launch_raw(KERNEL_8U, grid(width, height), BLOCK, &mut args)
}
