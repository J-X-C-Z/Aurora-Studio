//! Native CUDA GPU-resident reconstruction pipeline for Nikon Z6III HE RAW.
//!
//! Single H2D upload (raw 14-bit Bayer) -> 7 GPU-resident stages on the shared
//! CUDA context (zero intermediate CPU readbacks) -> single D2H download
//! (oriented RGBA16). Embedded PTX is JIT-loaded via system `nvcuda.dll`
//! through `aurora_portrait::cuda::Nvidia` (zero external DLLs).

use super::{NikonCrop, Stage1Params};
use aurora_portrait::cuda::{CudaError, Nvidia};
use aurora_portrait::Refusal;
use std::ffi::c_void;
use std::slice;

const PTX: &str = include_str!("../../../../native/cuda/z6iii_raw.ptx");

const BLOCK_2D: (u32, u32) = (16, 16);
const BLOCK_1D: (u32, u32) = (256, 1);
const HALO: usize = 20;

/// Returns `false` when `AURORA_Z6III_CUDA` is explicitly disabled (`0`/`false`/`off`/`no`).
pub(in crate::aurora_raw::z6iii) fn enabled() -> bool {
    match std::env::var("AURORA_Z6III_CUDA") {
        Ok(val) => !matches!(
            val.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

/// Whether the CUDA Z6III pipeline is enabled and can open the bundled PTX on this host.
pub(in crate::aurora_raw::z6iii) fn available() -> bool {
    enabled() && Nvidia::open_ptx(PTX).is_ok()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HighlightParams {
    p: [f32; 16],
}

pub(in crate::aurora_raw::z6iii) struct CudaPipelineInput<'a> {
    pub full_w: usize,
    pub full_h: usize,
    pub crop: NikonCrop,
    pub samples: &'a [u16],
    pub stage1: Stage1Params,
    pub black: u32,
    pub wb: [f32; 3],
    pub offset: f32,
    pub noise0: f32,
    pub recovery: [f32; 16],
    pub aa_params: [f32; 2],
    pub ca_warp: Option<&'a crate::mock::lens_profile::LensWarp>,
    pub orientation: u16,
}

#[inline]
fn grid_2d(width: usize, height: usize) -> (u32, u32) {
    (
        (width as u32).div_ceil(BLOCK_2D.0).max(1),
        (height as u32).div_ceil(BLOCK_2D.1).max(1),
    )
}

#[inline]
fn grid_1d(count: usize) -> (u32, u32) {
    ((count as u32).div_ceil(BLOCK_1D.0).max(1), 1)
}

unsafe fn bytes_of<T>(values: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) }
}

unsafe fn bytes_of_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

/// Runs the full Z6III reconstruction pipeline entirely in VRAM using two
/// ping-pong buffers (`buf_a`, `buf_b`), performing a single H2D upload of the
/// raw Bayer frame and a single D2H readback of the oriented RGBA16 output.
pub(in crate::aurora_raw::z6iii) fn reconstruct_rgba16(
    input: &CudaPipelineInput<'_>,
) -> Result<(usize, usize, Vec<u16>), CudaError> {
    if !enabled() {
        return Err(CudaError::Unavailable("AURORA_Z6III_CUDA disabled"));
    }

    let active = input.crop;
    if active.width < 2 * HALO
        || active.width % 2 != 0
        || active.height < HALO
        || active.cfa_phase() != (0, 0)
        || input.samples.len() != input.full_w.saturating_mul(input.full_h)
        || active.left + active.width > input.full_w
        || active.top + active.height > input.full_h
    {
        return Err(Refusal::ShapeMismatch("z6iii cuda bayer crop").into());
    }
    if !(0.0..=65535.0).contains(&input.offset)
        || !(0.0..=65535.0).contains(&(input.black as f32))
        || !input.wb.into_iter().all(|g| g.is_finite() && g > 0.0)
    {
        return Err(Refusal::ValueOutOfRange("z6iii cuda params").into());
    }

    let gpu = Nvidia::open_ptx(PTX)?;

    let halo_w = active.width + 2 * HALO;
    let halo_h = active.height + 2 * HALO;
    let support_left = HALO - 8; // 12
    let support_top = HALO - 8; // 12
    let support_w = active.width + 16; // 6064
    let support_h = active.height + 16; // 4048
    let gg_w = active.width + 4; // 6052
    let gg_h = active.height + 4; // 4036
    let out_w = active.width; // 6048
    let out_h = active.height; // 4032

    let raw_bytes = unsafe { bytes_of(input.samples) };
    let f4_bytes = halo_w * halo_h * std::mem::size_of::<[f32; 4]>();
    let out_rgba_bytes = out_w * out_h * 4 * std::mem::size_of::<u16>();
    let ping_bytes = f4_bytes.max(raw_bytes.len()).max(out_rgba_bytes);

    // Two ping-pong VRAM buffers carry all intermediate stages with zero overlap.
    let buf_a = gpu.alloc_buffer(ping_bytes)?;
    let buf_b = gpu.alloc_buffer(ping_bytes)?;

    let has_ca = input
        .ca_warp
        .is_some_and(|w| !w.red.is_empty() || !w.blue.is_empty());
    let ca_lut_buf = if let Some(warp) = input.ca_warp.filter(|_| has_ca) {
        let total_nodes = warp.red.len() + warp.blue.len();
        let mut packed = Vec::<[f32; 2]>::with_capacity(total_nodes);
        packed.extend(warp.red.iter().map(|&(r, v)| [r, v]));
        packed.extend(warp.blue.iter().map(|&(r, v)| [r, v]));
        let lut_bytes = unsafe { bytes_of(&packed) };
        let dev = gpu.alloc_buffer(lut_bytes.len())?;
        gpu.upload_buffer(&dev, lut_bytes)?;
        Some((dev, warp.red.len() as i32, warp.blue.len() as i32))
    } else {
        None
    };

    let use_adaptive_gg = input.offset > 0.0;

    // 1. Single H2D upload of raw 14-bit Bayer plane.
    //    If AdaptiveGG runs: upload -> buf_b, mirror -> buf_a, adaptive_gg -> buf_b.
    //    If AdaptiveGG skipped: upload -> buf_a, mirror -> buf_b.
    let raw_upload_dst = if use_adaptive_gg { &buf_b } else { &buf_a };
    gpu.upload_buffer(raw_upload_dst, raw_bytes)?;

    // 2. Stage 0: mirror_bayer_halo_shift2
    {
        let mut src_ptr = raw_upload_dst.handle();
        let mut dst_ptr = if use_adaptive_gg {
            buf_a.handle()
        } else {
            buf_b.handle()
        };
        let mut full_w_i = input.full_w as i32;
        let mut active_left_i = active.left as i32;
        let mut active_top_i = active.top as i32;
        let mut active_w_i = active.width as i32;
        let mut active_h_i = active.height as i32;
        let mut halo_i = HALO as i32;
        let mut args: [*mut c_void; 8] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut full_w_i as *mut i32).cast(),
            (&mut active_left_i as *mut i32).cast(),
            (&mut active_top_i as *mut i32).cast(),
            (&mut active_w_i as *mut i32).cast(),
            (&mut active_h_i as *mut i32).cast(),
            (&mut halo_i as *mut i32).cast(),
        ];
        gpu.launch_raw(
            "mirror_bayer_halo_shift2",
            grid_2d(halo_w, halo_h),
            BLOCK_2D,
            &mut args,
        )?;
    }

    // 3. Stage 0b: adaptive_gg (buf_a -> buf_b) when threshold > 0.0
    if use_adaptive_gg {
        let mut src_ptr = buf_a.handle();
        let mut dst_ptr = buf_b.handle();
        let mut w_i = halo_w as i32;
        let mut h_i = halo_h as i32;
        let mut threshold_f = input.offset;
        let mut s1_f = input.offset * 0.1;
        let mut black_f = input.black as f32;
        let mut args: [*mut c_void; 7] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut w_i as *mut i32).cast(),
            (&mut h_i as *mut i32).cast(),
            (&mut threshold_f as *mut f32).cast(),
            (&mut s1_f as *mut f32).cast(),
            (&mut black_f as *mut f32).cast(),
        ];
        gpu.launch_raw(
            "adaptive_gg",
            grid_2d(halo_w / 2, halo_h),
            BLOCK_2D,
            &mut args,
        )?;
    }

    // 4. Stage 1: bayer_stage1 (buf_b -> buf_a)
    {
        let mut src_ptr = buf_b.handle();
        let mut dst_ptr = buf_a.handle();
        let mut roi_x = 8i32;
        let mut roi_y = 8i32;
        let mut roi_w = (halo_w - 16) as i32;
        let mut roi_h = (halo_h - 16) as i32;
        let mut stride = halo_w as i32;
        let mut red_gain = input.stage1.red_gain;
        let mut blue_gain = input.stage1.blue_gain;
        let mut floor_code = input.stage1.noise_floor;
        let mut black_pivot = input.stage1.black_pivot;
        let mut args: [*mut c_void; 11] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut roi_x as *mut i32).cast(),
            (&mut roi_y as *mut i32).cast(),
            (&mut roi_w as *mut i32).cast(),
            (&mut roi_h as *mut i32).cast(),
            (&mut stride as *mut i32).cast(),
            (&mut red_gain as *mut f32).cast(),
            (&mut blue_gain as *mut f32).cast(),
            (&mut floor_code as *mut f32).cast(),
            (&mut black_pivot as *mut f32).cast(),
        ];
        gpu.launch_raw(
            "bayer_stage1",
            grid_2d((halo_w - 16) / 2, halo_h - 16),
            BLOCK_2D,
            &mut args,
        )?;
    }

    // 5. Stage 2: stage2_refine_green_and_scale (buf_a -> buf_b)
    {
        let mut src_ptr = buf_a.handle();
        let mut dst_ptr = buf_b.handle();
        let mut halo_w_i = halo_w as i32;
        let mut sup_left_i = support_left as i32;
        let mut sup_top_i = support_top as i32;
        let mut sup_w_i = support_w as i32;
        let mut sup_h_i = support_h as i32;
        let mut neg_black = -(input.black as i32 as f32);
        let mut wb_r = input.wb[0];
        let mut wb_g = input.wb[1];
        let mut wb_b = input.wb[2];
        let mut args: [*mut c_void; 11] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut halo_w_i as *mut i32).cast(),
            (&mut sup_left_i as *mut i32).cast(),
            (&mut sup_top_i as *mut i32).cast(),
            (&mut sup_w_i as *mut i32).cast(),
            (&mut sup_h_i as *mut i32).cast(),
            (&mut neg_black as *mut f32).cast(),
            (&mut wb_r as *mut f32).cast(),
            (&mut wb_g as *mut f32).cast(),
            (&mut wb_b as *mut f32).cast(),
        ];
        gpu.launch_raw(
            "stage2_refine_green_and_scale",
            grid_2d(support_w, support_h),
            BLOCK_2D,
            &mut args,
        )?;
    }

    // 6. Stage 3a: reconstruct_diagonal (buf_b -> buf_a)
    {
        let mut src_ptr = buf_b.handle();
        let mut dst_ptr = buf_a.handle();
        let mut w_i = support_w as i32;
        let mut h_i = support_h as i32;
        let mut floor_f = input.offset;
        let mut args: [*mut c_void; 5] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut w_i as *mut i32).cast(),
            (&mut h_i as *mut i32).cast(),
            (&mut floor_f as *mut f32).cast(),
        ];
        gpu.launch_raw(
            "reconstruct_diagonal",
            grid_2d((support_w - 4) / 2, support_h - 4),
            BLOCK_2D,
            &mut args,
        )?;
    }

    // 7. Stage 3b: reconstruct_green (buf_a -> buf_b)
    {
        let mut src_ptr = buf_a.handle();
        let mut dst_ptr = buf_b.handle();
        let mut w_i = support_w as i32;
        let mut h_i = support_h as i32;
        let mut floor_f = input.offset;
        let mut args: [*mut c_void; 5] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut w_i as *mut i32).cast(),
            (&mut h_i as *mut i32).cast(),
            (&mut floor_f as *mut f32).cast(),
        ];
        gpu.launch_raw(
            "reconstruct_green",
            grid_2d((support_w - 8) / 2, support_h - 8),
            BLOCK_2D,
            &mut args,
        )?;
    }

    // 8. Stage 4: gg_compensate (buf_b -> buf_a)
    {
        let [inv_r, inv_g, inv_b] = input.wb.map(f32::recip);
        let mut src_ptr = buf_b.handle();
        let mut dst_ptr = buf_a.handle();
        let mut w_i = support_w as i32;
        let mut margin_i = 6i32;
        let mut out_w_i = gg_w as i32;
        let mut out_h_i = gg_h as i32;
        let mut phase_i = 0i32;
        let mut wx = inv_r;
        let mut wy = inv_g;
        let mut wz = inv_b;
        let mut ww = 1.0f32;
        let mut p_noise = input.noise0;
        let mut p_thresh = input.offset;
        let mut args: [*mut c_void; 13] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut w_i as *mut i32).cast(),
            (&mut margin_i as *mut i32).cast(),
            (&mut out_w_i as *mut i32).cast(),
            (&mut out_h_i as *mut i32).cast(),
            (&mut phase_i as *mut i32).cast(),
            (&mut wx as *mut f32).cast(),
            (&mut wy as *mut f32).cast(),
            (&mut wz as *mut f32).cast(),
            (&mut ww as *mut f32).cast(),
            (&mut p_noise as *mut f32).cast(),
            (&mut p_thresh as *mut f32).cast(),
        ];
        gpu.launch_raw(
            "gg_compensate",
            grid_2d(gg_w, gg_h),
            BLOCK_2D,
            &mut args,
        )?;
    }

    // Track which ping-pong buffer holds the live float4 plane.
    let mut cur_is_a = true;

    // 9. Stage 5 (optional): linear_ca_warp (buf_a -> buf_b)
    if let Some((ref lut_dev, red_count, blue_count)) = ca_lut_buf {
        let mut src_ptr = buf_a.handle();
        let mut dst_ptr = buf_b.handle();
        let mut tile_w_i = gg_w as i32;
        let mut tile_h_i = gg_h as i32;
        let mut origin_x_i = -2i32;
        let mut origin_y_i = -2i32;
        let mut full_w_i = out_w as i32;
        let mut full_h_i = out_h as i32;
        let mut red_ptr = lut_dev.handle();
        let mut red_cnt = red_count;
        let mut blue_ptr = lut_dev.handle() + (red_count as u64) * 8;
        let mut blue_cnt = blue_count;
        let mut args: [*mut c_void; 12] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut tile_w_i as *mut i32).cast(),
            (&mut tile_h_i as *mut i32).cast(),
            (&mut origin_x_i as *mut i32).cast(),
            (&mut origin_y_i as *mut i32).cast(),
            (&mut full_w_i as *mut i32).cast(),
            (&mut full_h_i as *mut i32).cast(),
            (&mut red_ptr as *mut u64).cast(),
            (&mut red_cnt as *mut i32).cast(),
            (&mut blue_ptr as *mut u64).cast(),
            (&mut blue_cnt as *mut i32).cast(),
        ];
        gpu.launch_raw(
            "linear_ca_warp",
            grid_2d(gg_w, gg_h),
            BLOCK_2D,
            &mut args,
        )?;
        cur_is_a = false;
    }

    // 10. Stage 6: highlight_recover (cur -> other)
    {
        let mut src_ptr = if cur_is_a { buf_a.handle() } else { buf_b.handle() };
        let mut dst_ptr = if cur_is_a { buf_b.handle() } else { buf_a.handle() };
        let total = gg_w * gg_h;
        let mut total_i = total as i32;
        let mut hl_params = HighlightParams { p: input.recovery };
        let mut args: [*mut c_void; 4] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut total_i as *mut i32).cast(),
            (&mut hl_params as *mut HighlightParams).cast(),
        ];
        gpu.launch_raw(
            "highlight_recover",
            grid_1d(total),
            BLOCK_1D,
            &mut args,
        )?;
        cur_is_a = !cur_is_a;
    }

    // 11. Stage 7: anti_alias_and_orient_rgba16 (cur -> other)
    let out_buf = if cur_is_a { &buf_b } else { &buf_a };
    {
        let mut src_ptr = if cur_is_a { buf_a.handle() } else { buf_b.handle() };
        let mut dst_ptr = out_buf.handle();
        let mut halo_w_i = gg_w as i32;
        let mut out_w_i = out_w as i32;
        let mut out_h_i = out_h as i32;
        let mut px = input.aa_params[0];
        let mut py = input.aa_params[1];
        let mut orient_i = input.orientation as i32;
        let mut args: [*mut c_void; 8] = [
            (&mut src_ptr as *mut u64).cast(),
            (&mut dst_ptr as *mut u64).cast(),
            (&mut halo_w_i as *mut i32).cast(),
            (&mut out_w_i as *mut i32).cast(),
            (&mut out_h_i as *mut i32).cast(),
            (&mut px as *mut f32).cast(),
            (&mut py as *mut f32).cast(),
            (&mut orient_i as *mut i32).cast(),
        ];
        gpu.launch_raw(
            "anti_alias_and_orient_rgba16",
            grid_2d(out_w, out_h),
            BLOCK_2D,
            &mut args,
        )?;
    }

    // 12. Single D2H readback of final oriented RGBA16 frame.
    let mut out = vec![0u16; out_w * out_h * 4];
    gpu.download_buffer(out_buf, unsafe { bytes_of_mut(&mut out) })?;

    let (final_w, final_h) = match input.orientation {
        6 | 8 => (out_h, out_w),
        _ => (out_w, out_h),
    };
    Ok((final_w, final_h, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a_u16(samples: &[u16]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        for &v in samples {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    #[test]
    fn cuda_rejects_mismatched_shape() {
        let crop = NikonCrop {
            left: 0,
            top: 0,
            width: 64,
            height: 64,
        };
        let input = CudaPipelineInput {
            full_w: 64,
            full_h: 64,
            crop,
            samples: &[],
            stage1: Stage1Params::prepare([1.5, 1.0, 1.4], [12.8, 0.0], 4032, false),
            black: 4032,
            wb: [1.5, 1.0, 1.4],
            offset: 128.0,
            noise0: 12.8,
            recovery: [0.0; 16],
            aa_params: [0.05_f32.sqrt(), 0.5],
            ca_warp: None,
            orientation: 1,
        };
        let err = reconstruct_rgba16(&input).unwrap_err();
        assert!(matches!(
            err,
            CudaError::Refused(Refusal::ShapeMismatch(_)) | CudaError::Unavailable(_)
        ));
    }

    #[test]
    #[ignore = "requires real Z6III NEF on Desktop and NVIDIA GPU"]
    fn cuda_vs_opencl_bit_exact_and_bench_real_nef() {
        use std::time::Instant;

        let default_path = std::env::var("USERPROFILE")
            .map(|p| std::path::PathBuf::from(p).join("Desktop").join("_DSC5673.NEF"))
            .unwrap();
        let path = std::env::var("AURORA_TEST_RAW")
            .map(std::path::PathBuf::from)
            .unwrap_or(default_path);
        let bytes = std::fs::read(&path).expect("read NEF");
        let exif = crate::exif::parse_exif(&bytes);
        let orientation = crate::raw::exif_orientation_bytes(&bytes).unwrap_or(1);
        let ca_warp = crate::mock::lens_profile::LensShot::from_exif(&exif)
            .and_then(|shot| crate::mock::lens_profile::auto_warp(&shot))
            .filter(|warp| !warp.red.is_empty() || !warp.blue.is_empty());
        println!(
            "Sample: {} | orientation={orientation} | has_ca={} (red={}, blue={})",
            path.display(),
            ca_warp.is_some(),
            ca_warp.as_ref().map_or(0, |w| w.red.len()),
            ca_warp.as_ref().map_or(0, |w| w.blue.len()),
        );

        // 1. Decode HE Bayer once so we can measure HE decode separately and also run end-to-end.
        let t_he = Instant::now();
        let he_frame = super::super::decode_frame_cropped(&bytes)
            .expect("HE decode")
            .expect("Z6III HE frame");
        let he_ms = t_he.elapsed().as_secs_f64() * 1000.0;
        println!(
            "HE decode alone: {he_ms:.1} ms ({}x{}, crop {:?})",
            he_frame.full_width(),
            he_frame.full_height(),
            he_frame.crop()
        );

        // 2. OpenCL baseline (AURORA_Z6III_CUDA=0)
        unsafe { std::env::set_var("AURORA_Z6III_CUDA", "0") };
        let t_ocl = Instant::now();
        let mut timings_ocl = crate::raw::RawStageTimings::default();
        let (ow, oh, ocl_px, ocl_route) =
            crate::aurora_raw::decode_he_rgba16_with(&bytes, &exif, None, &mut timings_ocl)
                .expect("OpenCL decode");
        let ocl_ms = t_ocl.elapsed().as_secs_f64() * 1000.0;
        let ocl_hash = fnv1a_u16(&ocl_px);
        println!(
            "OpenCL baseline: {ow}x{oh} route={ocl_route:?} total={ocl_ms:.1} ms hash={ocl_hash:016x} samples={}",
            ocl_px.len()
        );

        // 3. CUDA cold (AURORA_Z6III_CUDA=1)
        unsafe { std::env::set_var("AURORA_Z6III_CUDA", "1") };
        let t_cuda_cold = Instant::now();
        let mut timings_cuda = crate::raw::RawStageTimings::default();
        let (cw, ch, cuda_px, cuda_route) =
            crate::aurora_raw::decode_he_rgba16_with(&bytes, &exif, None, &mut timings_cuda)
                .expect("CUDA decode");
        let cuda_cold_ms = t_cuda_cold.elapsed().as_secs_f64() * 1000.0;
        let cuda_hash = fnv1a_u16(&cuda_px);

        // 4. CUDA warm (module already JIT-cached)
        let t_cuda_warm = Instant::now();
        let mut timings_warm = crate::raw::RawStageTimings::default();
        let (_ww, _wh, _warm_px, _warm_route) =
            crate::aurora_raw::decode_he_rgba16_with(&bytes, &exif, None, &mut timings_warm)
                .expect("CUDA warm decode");
        let cuda_warm_ms = t_cuda_warm.elapsed().as_secs_f64() * 1000.0;

        // 5. CUDA GPU-only (pre-decoded Bayer)
        let predecoded = (
            he_frame.full_width(),
            he_frame.full_height(),
            he_frame.full_samples().to_vec(),
        );
        let t_gpu_only = Instant::now();
        let (gpu_frame, _route, _oriented) =
            crate::aurora_raw::z6iii::render_linear_rgba16_oriented_traced(
                &bytes,
                Some(predecoded),
                ca_warp.as_ref(),
                orientation,
            );
        let gpu_only_ms = t_gpu_only.elapsed().as_secs_f64() * 1000.0;
        assert!(gpu_frame.is_some());

        println!(
            "CUDA pipeline:   {cw}x{ch} route={cuda_route:?} cold={cuda_cold_ms:.1} ms warm={cuda_warm_ms:.1} ms gpu_only(H2D+Compute+D2H)={gpu_only_ms:.1} ms hash={cuda_hash:016x}"
        );

        assert_eq!((cw, ch), (ow, oh));
        assert_eq!(cuda_px.len(), ocl_px.len());

        let mut diff_samples = 0usize;
        let mut diff_pixels = 0usize;
        let mut max_abs_diff = 0u16;
        let mut first_diff = None;
        for (px_idx, (a, b)) in cuda_px.chunks_exact(4).zip(ocl_px.chunks_exact(4)).enumerate() {
            let mut px_diff = false;
            for ch_idx in 0..4 {
                let d = a[ch_idx].abs_diff(b[ch_idx]);
                if d > 0 {
                    diff_samples += 1;
                    px_diff = true;
                    if d > max_abs_diff {
                        max_abs_diff = d;
                    }
                    if first_diff.is_none() {
                        let x = px_idx % (cw as usize);
                        let y = px_idx / (cw as usize);
                        first_diff = Some((x, y, ch_idx, a[ch_idx], b[ch_idx]));
                    }
                }
            }
            if px_diff {
                diff_pixels += 1;
            }
        }
        println!(
            "PARITY (with default CA): diff_pixels={diff_pixels} diff_samples={diff_samples}/{} max_abs_diff={max_abs_diff} first_diff={first_diff:?}",
            ocl_px.len()
        );

        // 6. Also compare without CA (orientation=1 and real orientation).
        unsafe { std::env::set_var("AURORA_Z6III_CUDA", "0") };
        let (n_ocl, _, _) = crate::aurora_raw::z6iii::render_linear_rgba16_oriented_traced(&bytes, None, None, 1);
        let (_, _, n_ocl_px) = n_ocl.unwrap();
        unsafe { std::env::set_var("AURORA_Z6III_CUDA", "1") };
        let (n_cuda, _, _) = crate::aurora_raw::z6iii::render_linear_rgba16_oriented_traced(&bytes, None, None, 1);
        let (_, _, n_cuda_px) = n_cuda.unwrap();
        let n_ocl_hash = fnv1a_u16(&n_ocl_px);
        let n_cuda_hash = fnv1a_u16(&n_cuda_px);
        let mut n_diff = 0usize;
        let mut n_max = 0u16;
        for (&a, &b) in n_cuda_px.iter().zip(n_ocl_px.iter()) {
            let d = a.abs_diff(b);
            if d > 0 {
                n_diff += 1;
                n_max = n_max.max(d);
            }
        }
        println!(
            "PARITY (no CA, orient=1): ocl_hash={n_ocl_hash:016x} cuda_hash={n_cuda_hash:016x} diff_samples={n_diff} max_abs_diff={n_max}"
        );

        assert_eq!(max_abs_diff, 0, "CUDA vs OpenCL mismatch: diff_pixels={diff_pixels}, first={first_diff:?}");
        assert_eq!(n_max, 0, "CUDA vs OpenCL (no CA) mismatch: diff_samples={n_diff}");
    }

    #[test]
    #[ignore = "requires real Z6III NEF on Desktop and NVIDIA GPU"]
    fn cuda_vs_opencl_bisect_stages() {
        let default_path = std::env::var("USERPROFILE")
            .map(|p| std::path::PathBuf::from(p).join("Desktop").join("_DSC5673.NEF"))
            .unwrap();
        let bytes = std::fs::read(&default_path).expect("read NEF");
        let iso = super::super::nikon_exif_sensitivity(&bytes).unwrap().unwrap();
        let noise = crate::aurora_raw::z6iii::color::z6iii_noise::z6iii_noise_for_iso(iso).unwrap();
        let offset = super::super::Z6iiiStage3::offset_for_noise(noise, (50, 0.0));
        let wb = super::super::nikon_as_shot_rgb(&bytes)
            .unwrap()
            .map(crate::aurora_raw::z6iii::color::prepare_z6iii_global_white_balance)
            .unwrap()
            .map(|gain| ((f64::from(gain) * 1000.0).round_ties_even() / 1000.0) as f32);
        let black = super::super::nikon_black_levels(&bytes).unwrap().unwrap().capture_one_scalar();
        let stage1 = Stage1Params::prepare(wb, noise, black, false);
        let raw_frame = super::super::decode_frame_cropped(&bytes).unwrap().unwrap();
        let active = raw_frame.crop();
        let halo_w = active.width + 2 * HALO;
        let halo_h = active.height + 2 * HALO;

        // OpenCL Stage 0b + Stage 1
        let ocl_frame = super::super::NikonHeCropFrame::new(
            active,
            raw_frame.full_width(),
            raw_frame.full_height(),
            raw_frame.full_samples().to_vec(),
        )
        .unwrap()
        .with_control_bayer_halo(offset, black as f32)
        .unwrap()
        .with_stage1_opencl(stage1)
        .unwrap();

        // Run CUDA Stage 0 + 0b + 1 + 2 and download intermediate buffers to compare
        let gpu = Nvidia::open_ptx(PTX).unwrap();
        let ping_bytes = halo_w * halo_h * 16;
        let buf_a = gpu.alloc_buffer(ping_bytes).unwrap();
        let buf_b = gpu.alloc_buffer(ping_bytes).unwrap();
        gpu.upload_buffer(&buf_b, unsafe { bytes_of(raw_frame.full_samples()) }).unwrap();

        // Stage 0
        {
            let mut src_ptr = buf_b.handle();
            let mut dst_ptr = buf_a.handle();
            let mut full_w_i = raw_frame.full_width() as i32;
            let mut active_left_i = active.left as i32;
            let mut active_top_i = active.top as i32;
            let mut active_w_i = active.width as i32;
            let mut active_h_i = active.height as i32;
            let mut halo_i = HALO as i32;
            let mut args: [*mut c_void; 8] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut dst_ptr as *mut u64).cast(),
                (&mut full_w_i as *mut i32).cast(),
                (&mut active_left_i as *mut i32).cast(),
                (&mut active_top_i as *mut i32).cast(),
                (&mut active_w_i as *mut i32).cast(),
                (&mut active_h_i as *mut i32).cast(),
                (&mut halo_i as *mut i32).cast(),
            ];
            gpu.launch_raw("mirror_bayer_halo_shift2", grid_2d(halo_w, halo_h), BLOCK_2D, &mut args).unwrap();
        }
        // Check Stage 0 (mirror + shift2)
        let mut s0_out = vec![0u16; halo_w * halo_h];
        gpu.download_buffer(&buf_a, unsafe { bytes_of_mut(&mut s0_out) }).unwrap();
        let ocl_s0 = ocl_frame.full_samples();
        let mut s0_diffs = 0usize;
        let mut s0_first = None;
        for (i, (&a, &b)) in s0_out.iter().zip(ocl_s0.iter()).enumerate() {
            let exp = b << 2;
            if a != exp {
                s0_diffs += 1;
                if s0_first.is_none() {
                    s0_first = Some((i % halo_w, i / halo_w, a, exp));
                }
            }
        }
        println!("Stage 0 (mirror+shift2): diffs={s0_diffs} first={s0_first:?}");

        // Stage 0b
        {
            let mut src_ptr = buf_a.handle();
            let mut dst_ptr = buf_b.handle();
            let mut w_i = halo_w as i32;
            let mut h_i = halo_h as i32;
            let mut threshold_f = offset;
            let mut s1_f = offset * 0.1;
            let mut black_f = black as f32;
            let mut args: [*mut c_void; 7] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut dst_ptr as *mut u64).cast(),
                (&mut w_i as *mut i32).cast(),
                (&mut h_i as *mut i32).cast(),
                (&mut threshold_f as *mut f32).cast(),
                (&mut s1_f as *mut f32).cast(),
                (&mut black_f as *mut f32).cast(),
            ];
            gpu.launch_raw("adaptive_gg", grid_2d(halo_w / 2, halo_h), BLOCK_2D, &mut args).unwrap();
        }
        let mut s0b_out = vec![0u16; halo_w * halo_h];
        gpu.download_buffer(&buf_b, unsafe { bytes_of_mut(&mut s0b_out) }).unwrap();
        let ocl_s0b = super::super::c1_adaptive_opencl::prepare(
            ocl_frame.full_samples(),
            halo_w,
            halo_h,
            NikonCrop { left: 2, top: 2, width: halo_w - 4, height: halo_h - 4 },
            offset,
            black as f32,
        )
        .unwrap();
        let mut s0b_diffs = 0usize;
        let mut s0b_first = None;
        for (i, (&a, &b)) in s0b_out.iter().zip(ocl_s0b.iter()).enumerate() {
            if a != b {
                s0b_diffs += 1;
                if s0b_first.is_none() {
                    s0b_first = Some((i % halo_w, i / halo_w, a, b));
                }
            }
        }
        println!("Stage 0b (adaptive_gg): diffs={s0b_diffs} first={s0b_first:?}");

        // Stage 1
        {
            let mut src_ptr = buf_b.handle();
            let mut dst_ptr = buf_a.handle();
            let mut roi_x = 8i32;
            let mut roi_y = 8i32;
            let mut roi_w = (halo_w - 16) as i32;
            let mut roi_h = (halo_h - 16) as i32;
            let mut stride = halo_w as i32;
            let mut red_gain = stage1.red_gain;
            let mut blue_gain = stage1.blue_gain;
            let mut floor_code = stage1.noise_floor;
            let mut black_pivot = stage1.black_pivot;
            let mut args: [*mut c_void; 11] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut dst_ptr as *mut u64).cast(),
                (&mut roi_x as *mut i32).cast(),
                (&mut roi_y as *mut i32).cast(),
                (&mut roi_w as *mut i32).cast(),
                (&mut roi_h as *mut i32).cast(),
                (&mut stride as *mut i32).cast(),
                (&mut red_gain as *mut f32).cast(),
                (&mut blue_gain as *mut f32).cast(),
                (&mut floor_code as *mut f32).cast(),
                (&mut black_pivot as *mut f32).cast(),
            ];
            gpu.launch_raw("bayer_stage1", grid_2d((halo_w - 16) / 2, halo_h - 16), BLOCK_2D, &mut args).unwrap();
        }
        let mut s1_out = vec![[0.0f32; 4]; halo_w * halo_h];
        gpu.download_buffer(&buf_a, unsafe { bytes_of_mut(&mut s1_out) }).unwrap();
        let ocl_s1 = super::super::c1_adaptive_opencl::stage1(&ocl_s0b, halo_w, halo_h, stage1).unwrap();
        let mut s1_diffs = 0usize;
        let mut s1_first = None;
        for y in 8..halo_h - 8 {
            for x in 8..halo_w - 8 {
                let i = y * halo_w + x;
                for c in 0..4 {
                    if s1_out[i][c].to_bits() != ocl_s1[i][c].to_bits() {
                        s1_diffs += 1;
                        if s1_first.is_none() {
                            s1_first = Some((x, y, c, s1_out[i][c], ocl_s1[i][c]));
                        }
                    }
                }
            }
        }
        println!("Stage 1 (bayer_stage1): diffs={s1_diffs} first={s1_first:?}");

        // Stage 2
        let support_w = active.width + 16;
        let support_h = active.height + 16;
        {
            let mut src_ptr = buf_a.handle();
            let mut dst_ptr = buf_b.handle();
            let mut halo_w_i = halo_w as i32;
            let mut sup_left_i = (HALO - 8) as i32;
            let mut sup_top_i = (HALO - 8) as i32;
            let mut sup_w_i = support_w as i32;
            let mut sup_h_i = support_h as i32;
            let mut neg_black = -(black as i32 as f32);
            let mut wb_r = wb[0];
            let mut wb_g = wb[1];
            let mut wb_b = wb[2];
            let mut args: [*mut c_void; 11] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut dst_ptr as *mut u64).cast(),
                (&mut halo_w_i as *mut i32).cast(),
                (&mut sup_left_i as *mut i32).cast(),
                (&mut sup_top_i as *mut i32).cast(),
                (&mut sup_w_i as *mut i32).cast(),
                (&mut sup_h_i as *mut i32).cast(),
                (&mut neg_black as *mut f32).cast(),
                (&mut wb_r as *mut f32).cast(),
                (&mut wb_g as *mut f32).cast(),
                (&mut wb_b as *mut f32).cast(),
            ];
            gpu.launch_raw("stage2_refine_green_and_scale", grid_2d(support_w, support_h), BLOCK_2D, &mut args).unwrap();
        }
        let mut s2_out = vec![[0.0f32; 4]; support_w * support_h];
        gpu.download_buffer(&buf_b, unsafe { bytes_of_mut(&mut s2_out) }).unwrap();
        let ocl_s2 = super::super::reconstruct_stage2_scaled_f4(
            &ocl_frame,
            NikonCrop { left: HALO - 2, top: HALO - 2, width: active.width + 4, height: active.height + 4 },
            stage1,
            black as i32,
            wb,
        )
        .unwrap();
        let mut s2_diffs = 0usize;
        let mut s2_first = None;
        for i in 0..support_w * support_h {
            for c in 0..4 {
                if s2_out[i][c].to_bits() != ocl_s2[i][c].to_bits() {
                    s2_diffs += 1;
                    if s2_first.is_none() {
                        s2_first = Some((i % support_w, i / support_w, c, s2_out[i][c], ocl_s2[i][c]));
                    }
                }
            }
        }
        println!("Stage 2 (refine_green+scale): diffs={s2_diffs} first={s2_first:?}");

        // Stage 3a + 3b
        {
            let mut src_ptr = buf_b.handle();
            let mut dst_ptr = buf_a.handle();
            let mut w_i = support_w as i32;
            let mut h_i = support_h as i32;
            let mut floor_f = offset;
            let mut args: [*mut c_void; 5] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut dst_ptr as *mut u64).cast(),
                (&mut w_i as *mut i32).cast(),
                (&mut h_i as *mut i32).cast(),
                (&mut floor_f as *mut f32).cast(),
            ];
            gpu.launch_raw("reconstruct_diagonal", grid_2d((support_w - 4) / 2, support_h - 4), BLOCK_2D, &mut args).unwrap();
        }
        {
            let mut src_ptr = buf_a.handle();
            let mut dst_ptr = buf_b.handle();
            let mut w_i = support_w as i32;
            let mut h_i = support_h as i32;
            let mut floor_f = offset;
            let mut args: [*mut c_void; 5] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut dst_ptr as *mut u64).cast(),
                (&mut w_i as *mut i32).cast(),
                (&mut h_i as *mut i32).cast(),
                (&mut floor_f as *mut f32).cast(),
            ];
            gpu.launch_raw("reconstruct_green", grid_2d((support_w - 8) / 2, support_h - 8), BLOCK_2D, &mut args).unwrap();
        }
        let mut s3_out = vec![[0.0f32; 4]; support_w * support_h];
        gpu.download_buffer(&buf_b, unsafe { bytes_of_mut(&mut s3_out) }).unwrap();
        let s2_unscaled: Vec<[f32; 3]> = super::super::reconstruct_basic_roi(
            &ocl_frame,
            NikonCrop { left: HALO - 8, top: HALO - 8, width: support_w, height: support_h },
            stage1,
            Some(black as i32),
        )
        .unwrap();
        let ocl_s3 = super::super::c1_adaptive_opencl::stage3(&s2_unscaled, support_w, support_h, wb, offset).unwrap();
        let mut s3_diffs = 0usize;
        let mut s3_first = None;
        for y in 4..support_h - 4 {
            for x in 4..support_w - 4 {
                let i = y * support_w + x;
                for c in 0..3 {
                    if s3_out[i][c].to_bits() != ocl_s3[i][c].to_bits() {
                        s3_diffs += 1;
                        if s3_first.is_none() {
                            s3_first = Some((x, y, c, s3_out[i][c], ocl_s3[i][c]));
                        }
                    }
                }
            }
        }
        println!("Stage 3 (diag+green): diffs={s3_diffs} first={s3_first:?}");

        // Stage 4: gg_compensate
        let gg_w = active.width + 4;
        let gg_h = active.height + 4;
        {
            let [inv_r, inv_g, inv_b] = wb.map(f32::recip);
            let mut src_ptr = buf_b.handle();
            let mut dst_ptr = buf_a.handle();
            let mut w_i = support_w as i32;
            let mut margin_i = 6i32;
            let mut out_w_i = gg_w as i32;
            let mut out_h_i = gg_h as i32;
            let mut phase_i = 0i32;
            let mut wx = inv_r;
            let mut wy = inv_g;
            let mut wz = inv_b;
            let mut ww = 1.0f32;
            let mut p_noise = noise[0];
            let mut p_thresh = offset;
            let mut args: [*mut c_void; 13] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut dst_ptr as *mut u64).cast(),
                (&mut w_i as *mut i32).cast(),
                (&mut margin_i as *mut i32).cast(),
                (&mut out_w_i as *mut i32).cast(),
                (&mut out_h_i as *mut i32).cast(),
                (&mut phase_i as *mut i32).cast(),
                (&mut wx as *mut f32).cast(),
                (&mut wy as *mut f32).cast(),
                (&mut wz as *mut f32).cast(),
                (&mut ww as *mut f32).cast(),
                (&mut p_noise as *mut f32).cast(),
                (&mut p_thresh as *mut f32).cast(),
            ];
            gpu.launch_raw("gg_compensate", grid_2d(gg_w, gg_h), BLOCK_2D, &mut args).unwrap();
        }
        let mut s4_out = vec![[0.0f32; 4]; gg_w * gg_h];
        gpu.download_buffer(&buf_a, unsafe { bytes_of_mut(&mut s4_out) }).unwrap();
        let ocl_s4 = super::super::reconstruct_planar_green_halo_roi(
            &ocl_frame,
            NikonCrop { left: HALO - 2, top: HALO - 2, width: gg_w, height: gg_h },
            stage1,
            black as i32,
            wb,
            offset,
            noise[0],
        )
        .unwrap();
        let mut s4_diffs = 0usize;
        let mut s4_first = None;
        for i in 0..gg_w * gg_h {
            for c in 0..3 {
                if s4_out[i][c].to_bits() != ocl_s4[i][c].to_bits() {
                    s4_diffs += 1;
                    if s4_first.is_none() {
                        s4_first = Some((i % gg_w, i / gg_w, c, s4_out[i][c], ocl_s4[i][c]));
                    }
                }
            }
        }
        println!("Stage 3+4 (gg_compensate): diffs={s4_diffs} first={s4_first:?}");

        // Stage 6: highlight
        let recovery = {
            let mut gains = wb;
            let range = (65535 - black) as f32;
            let threshold = range * (-0.05_f32).exp2();
            gains[1] /= 1.5;
            let mut settings = [0.0; 16];
            settings[..3].copy_from_slice(&gains.map(|gain| gain * threshold));
            settings[3] = (2.0 * (-0.3_f32).exp2() / threshold) * threshold;
            settings[4..7].copy_from_slice(&gains.map(|gain| 2.0 * (range - threshold) * gain));
            settings[7..10].copy_from_slice(&gains.map(|gain| 1.0 / gain));
            settings[10] = (0.125 / threshold) * (1.0 / 3.0);
            settings
        };
        {
            let mut src_ptr = buf_a.handle();
            let mut dst_ptr = buf_b.handle();
            let mut total_i = (gg_w * gg_h) as i32;
            let mut hl_params = HighlightParams { p: recovery };
            let mut args: [*mut c_void; 4] = [
                (&mut src_ptr as *mut u64).cast(),
                (&mut dst_ptr as *mut u64).cast(),
                (&mut total_i as *mut i32).cast(),
                (&mut hl_params as *mut HighlightParams).cast(),
            ];
            gpu.launch_raw("highlight_recover", grid_1d(gg_w * gg_h), BLOCK_1D, &mut args).unwrap();
        }
        let mut s6_out = vec![[0.0f32; 4]; gg_w * gg_h];
        gpu.download_buffer(&buf_b, unsafe { bytes_of_mut(&mut s6_out) }).unwrap();
        let mut ocl_s6 = ocl_s4;
        super::super::recover_highlights(&mut ocl_s6, recovery).unwrap();
        let mut s6_diffs = 0usize;
        let mut s6_first = None;
        for i in 0..gg_w * gg_h {
            for c in 0..3 {
                if s6_out[i][c].to_bits() != ocl_s6[i][c].to_bits() {
                    s6_diffs += 1;
                    if s6_first.is_none() {
                        s6_first = Some((i % gg_w, i / gg_w, c, s6_out[i][c], ocl_s6[i][c]));
                    }
                }
            }
        }
        println!("Stage 6 (highlight): diffs={s6_diffs} first={s6_first:?}");
    }
}
