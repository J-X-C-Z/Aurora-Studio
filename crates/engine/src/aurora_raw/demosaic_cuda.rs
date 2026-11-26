//! Native zero-DLL embedded-PTX CUDA demosaicing for:
//! - Generic 2x2 Bayer RAW (Sony ARW, Canon CR3, classic Nikon NEF, DNG, ORF, RW2, etc.),
//!   1:1 aligned with `aurora_raw::c1_raw_he` Stage 1/2/3 + active-crop mirror boundary.
//! - Fujifilm X-Trans 6x6 CFA (`filters == 9`),
//!   1:1 aligned with `aurora_raw::xtrans::demosaic_xtrans`.
//!
//! All stages execute device-resident in reusable VRAM buffers with a single H2D
//! upload of raw `u16` sensor samples, zero intermediate readbacks, and a single
//! D2H download of tightly packed `[f32; 3]` linear RGB.
//! Set `AURORA_DEMOSAIC_CUDA=0` to disable CUDA and fall back to CPU.

/// Returns `false` when `AURORA_DEMOSAIC_CUDA` is explicitly disabled
/// (`0`, `false`, or `off`).
pub(crate) fn demosaic_cuda_enabled() -> bool {
    parse_enabled_env(std::env::var("AURORA_DEMOSAIC_CUDA").ok().as_deref())
}

fn parse_enabled_env(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            let trimmed = v.trim();
            !(trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off"))
        }
        None => true,
    }
}

/// Whether the fused demosaic -> linear-finish chain may run.
///
/// Set `AURORA_RAW_FUSED_CUDA=0` (`false`/`off`) to force the historical
/// two-segment path (`demosaic_*_cuda` + `linear_post_cuda::linear_finish_cuda`)
/// so a frame can be compared between the two, or to isolate a regression to
/// one of the segments.
pub(crate) fn fused_cuda_enabled() -> bool {
    parse_enabled_env(std::env::var("AURORA_RAW_FUSED_CUDA").ok().as_deref())
}

/// Test-only accounting for the fused demosaic -> linear-finish entries: one
/// increment per host<->device copy they perform. One fused call must leave it
/// at exactly 2 (the raw window up, the finished frame down); the two-segment
/// path it replaces performs 4 for the same frame.
#[cfg(test)]
pub(crate) static FUSED_HOST_COPIES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Evaluates LibRaw's 2x2 Bayer filter bitmask at `(x, y)` (0=R, 1/3=G, 2=B).
#[inline]
fn bayer_color_at(filters: u32, x: usize, y: usize) -> u8 {
    ((filters >> ((((y << 1) & 14) + (x & 1)) << 1)) & 3) as u8
}

/// Finds the `(dx, dy)` in `{0, 1} x {0, 1}` that aligns `(left_margin + dx, top_margin + dy)`
/// to an RGGB origin (R at (0,0), G at (1,0) & (0,1), B at (1,1)).
fn find_rggb_offset(filters: u32, left_margin: usize, top_margin: usize) -> Option<(usize, usize)> {
    let is_green = |c: u8| c == 1 || c == 3;
    for dy in 0..=1 {
        for dx in 0..=1 {
            let x = left_margin + dx;
            let y = top_margin + dy;
            if bayer_color_at(filters, x, y) == 0
                && is_green(bayer_color_at(filters, x + 1, y))
                && is_green(bayer_color_at(filters, x, y + 1))
                && bayer_color_at(filters, x + 1, y + 1) == 2
            {
                return Some((dx, dy));
            }
        }
    }
    None
}

/// Computes `(shift_left, shift_right)` to normalize sensor samples of `bits`
/// precision into the 14-bit domain expected by C1 Stage 1/2/3 and X-Trans.
#[inline]
fn bit_shifts_to_14(bits: u8) -> (u32, u32) {
    if bits == 14 || bits == 0 {
        (0, 0)
    } else if bits < 14 {
        (14 - u32::from(bits), 0)
    } else {
        (0, u32::from(bits) - 14)
    }
}

#[cfg(windows)]
mod imp {
    use super::{bit_shifts_to_14, demosaic_cuda_enabled, find_rggb_offset};
    use crate::aurora_raw::c1_raw_he::{NikonHeCropFrame, Stage1Params};
    use crate::aurora_raw::linear_post_cuda::{launch_finish_chain, linear_post_ptx_image};
    use aurora_portrait::cuda::{BufferSession, CudaError, Nvidia};
    use std::ffi::c_void;
    use std::slice;

    const PTX: &str = include_str!("../../native/cuda/raw_demosaic.ptx");

    /// The embedded PTX image, for the startup warmup in
    /// [`crate::aurora_raw::prewarm`]. Handing out the same text (not a second
    /// `include_str!`) is what makes the warmup hit the shared domain's module
    /// cache instead of JIT-compiling a duplicate.
    pub(crate) fn ptx_image() -> &'static str {
        PTX
    }

    const KERNEL_BAYER_STAGE1: &str = "bayer_stage1_kernel";
    const KERNEL_BAYER_STAGE2_SCALE: &str = "bayer_stage2_scale_kernel";
    const KERNEL_BAYER_STAGE3_DIAG: &str = "bayer_stage3_diagonal_kernel";
    const KERNEL_BAYER_STAGE3_GREEN: &str = "bayer_stage3_green_pack_kernel";

    const KERNEL_XTRANS_NORM: &str = "xtrans_normalize_kernel";
    const KERNEL_XTRANS_GREEN: &str = "xtrans_green_interp_kernel";
    const KERNEL_XTRANS_FIX_GREEN: &str = "xtrans_fix_green_kernel";
    const KERNEL_XTRANS_CHROMA: &str = "xtrans_chroma_rebuild_kernel";

    const BLOCK_2D: (u32, u32) = (16, 16);

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct BayerDemosaicParams {
        raw_stride: i32,
        src_left: i32,
        src_top: i32,
        width: i32,
        height: i32,
        shift_left: i32,
        shift_right: i32,
        rgb_black: i32,
        red_gain: f32,
        blue_gain: f32,
        noise_floor: f32,
        black_pivot: f32,
        wb_r: f32,
        wb_g: f32,
        wb_b: f32,
        stage3_offset: f32,
    }

    const _: () = assert!(std::mem::size_of::<BayerDemosaicParams>() == 64);

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct XTransDemosaicParams {
        raw_stride: i32,
        left: i32,
        top: i32,
        width: i32,
        height: i32,
        shift_left: i32,
        shift_right: i32,
        black: f32,
        wb_r: f32,
        wb_g: f32,
        wb_b: f32,
    }

    const _: () = assert!(std::mem::size_of::<XTransDemosaicParams>() == 44);

    fn grid_2d(width: u32, height: u32) -> (u32, u32) {
        (
            width.div_ceil(BLOCK_2D.0).max(1),
            height.div_ceil(BLOCK_2D.1).max(1),
        )
    }

    unsafe fn bytes_of<T>(values: &[T]) -> &[u8] {
        unsafe {
            slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        }
    }

    unsafe fn bytes_of_mut<T>(values: &mut [T]) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                values.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(values),
            )
        }
    }

    /// The four C1 Stage 1/2/3 kernels over an already-uploaded raw window.
    ///
    /// `image` is the PTX image that declares them, so the same chain can run in
    /// a session opened on another image (the fused demosaic -> linear-finish
    /// chain) without re-uploading anything.
    #[allow(clippy::too_many_arguments)]
    fn launch_bayer_chain(
        session: &mut BufferSession<'_>,
        image: &'static str,
        params: &BayerDemosaicParams,
        buf_raw: u64,
        buf_s1: u64,
        buf_s2: u64,
        buf_out: u64,
        width: usize,
        height: usize,
    ) -> Result<(), CudaError> {
        // Stage 1: directional green + black offset
        {
            let mut src = buf_raw;
            let mut dst = buf_s1;
            let mut p = *params;
            let mut args: [*mut c_void; 3] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut p as *mut BayerDemosaicParams as *mut c_void,
            ];
            session.launch_from_image(
                image,
                KERNEL_BAYER_STAGE1,
                grid_2d((width + 16) as u32, (height + 16) as u32),
                BLOCK_2D,
                &mut args,
            )?;
        }

        // Stage 2: basic green refinement + WB channel scale
        {
            let mut src = buf_s1;
            let mut dst = buf_s2;
            let mut p = *params;
            let mut args: [*mut c_void; 3] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut p as *mut BayerDemosaicParams as *mut c_void,
            ];
            session.launch_from_image(
                image,
                KERNEL_BAYER_STAGE2_SCALE,
                grid_2d((width + 12) as u32, (height + 12) as u32),
                BLOCK_2D,
                &mut args,
            )?;
        }

        // Stage 3a: diagonal R/B reconstruction (in-place on buf_s2)
        {
            let mut buf = buf_s2;
            let mut p = *params;
            let mut args: [*mut c_void; 2] = [
                &mut buf as *mut u64 as *mut c_void,
                &mut p as *mut BayerDemosaicParams as *mut c_void,
            ];
            session.launch_from_image(
                image,
                KERNEL_BAYER_STAGE3_DIAG,
                grid_2d((width + 8) as u32, (height + 8) as u32),
                BLOCK_2D,
                &mut args,
            )?;
        }

        // Stage 3b: cross R/B on G sites + pack tightly to [f32; 3]
        {
            let mut src = buf_s2;
            let mut dst = buf_out;
            let mut p = *params;
            let mut args: [*mut c_void; 3] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut p as *mut BayerDemosaicParams as *mut c_void,
            ];
            session.launch_from_image(
                image,
                KERNEL_BAYER_STAGE3_GREEN,
                grid_2d(width as u32, height as u32),
                BLOCK_2D,
                &mut args,
            )?;
        }
        Ok(())
    }

    /// The four X-Trans kernels over an already-uploaded active window.
    fn launch_xtrans_chain(
        session: &mut BufferSession<'_>,
        image: &'static str,
        params: &XTransDemosaicParams,
        buf_raw: u64,
        buf_plane: u64,
        buf_green: u64,
        buf_green_fixed: u64,
        buf_out: u64,
        width: usize,
        height: usize,
    ) -> Result<(), CudaError> {
        let p = *params;
        let w = params.width;
        let h = params.height;
        let g2 = grid_2d(width as u32, height as u32);

        // Stage 0: normalize
        {
            let mut src = buf_raw;
            let mut dst = buf_plane;
            let mut params = p;
            let mut args: [*mut c_void; 3] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut params as *mut XTransDemosaicParams as *mut c_void,
            ];
            session.launch_from_image(image, KERNEL_XTRANS_NORM, g2, BLOCK_2D, &mut args)?;
        }

        // Stage 1: directional green interpolation
        {
            let mut src = buf_plane;
            let mut dst = buf_green;
            let mut w = w;
            let mut h = h;
            let mut args: [*mut c_void; 4] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut w as *mut i32 as *mut c_void,
                &mut h as *mut i32 as *mut c_void,
            ];
            session.launch_from_image(image, KERNEL_XTRANS_GREEN, g2, BLOCK_2D, &mut args)?;
        }

        // Stage 2: fixGreen cross clamp
        {
            let mut src = buf_green;
            let mut dst = buf_green_fixed;
            let mut w = w;
            let mut h = h;
            let mut args: [*mut c_void; 4] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut w as *mut i32 as *mut c_void,
                &mut h as *mut i32 as *mut c_void,
            ];
            session.launch_from_image(image, KERNEL_XTRANS_FIX_GREEN, g2, BLOCK_2D, &mut args)?;
        }

        // Stage 3: 5x5 R-G / B-G chroma rebuild -> [f32; 3]
        {
            let mut src_plane = buf_plane;
            let mut src_gf = buf_green_fixed;
            let mut dst = buf_out;
            let mut w = w;
            let mut h = h;
            let mut args: [*mut c_void; 5] = [
                &mut src_plane as *mut u64 as *mut c_void,
                &mut src_gf as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut w as *mut i32 as *mut c_void,
                &mut h as *mut i32 as *mut c_void,
            ];
            session.launch_from_image(image, KERNEL_XTRANS_CHROMA, g2, BLOCK_2D, &mut args)?;
        }
        Ok(())
    }

    /// Validates an RGGB-aligned Bayer request and returns the kernel params
    /// plus the contiguous raw row window that is uploaded.
    ///
    /// Shared by the standalone demosaic path and the fused
    /// demosaic -> linear-finish path so both validate identically.
    #[allow(clippy::too_many_arguments)]
    fn bayer_stage_params<'a>(
        raw: &'a [u16],
        raw_stride: usize,
        src_left: usize,
        src_top: usize,
        width: usize,
        height: usize,
        shift_left: u32,
        shift_right: u32,
        black_14: u32,
        wb: [f32; 3],
        noise: [f32; 2],
    ) -> Result<(BayerDemosaicParams, &'a [u16]), String> {
        if width < 16 || height < 16 {
            return Err(format!("Bayer active dimensions too small: {width}x{height}"));
        }
        if !wb.into_iter().all(|g| g.is_finite() && g > 0.0) {
            return Err(format!("invalid Bayer white balance: {wb:?}"));
        }
        let needed_rows = src_top
            .checked_add(height)
            .ok_or_else(|| "Bayer row overflow".to_string())?;
        let needed_cols = src_left
            .checked_add(width)
            .ok_or_else(|| "Bayer col overflow".to_string())?;
        if needed_cols > raw_stride {
            return Err(format!(
                "Bayer crop ({src_left}+{width}) exceeds raw_stride {raw_stride}"
            ));
        }
        let min_len = needed_rows
            .checked_mul(raw_stride)
            .ok_or_else(|| "Bayer buffer length overflow".to_string())?;
        if raw.len() < min_len {
            return Err(format!(
                "Bayer raw slice too short: {} < {min_len}",
                raw.len()
            ));
        }

        let s1 = Stage1Params::prepare(wb, noise, black_14, false);
        if ![s1.red_gain, s1.blue_gain, s1.noise_floor]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
            || !s1.black_pivot.is_finite()
            || s1.black_pivot < 0.0
        {
            return Err(format!("invalid Stage1Params: {s1:?}"));
        }

        let w_i32 = i32::try_from(width).map_err(|_| "width exceeds i32::MAX".to_string())?;
        let h_i32 = i32::try_from(height).map_err(|_| "height exceeds i32::MAX".to_string())?;
        let stride_i32 =
            i32::try_from(raw_stride).map_err(|_| "raw_stride exceeds i32::MAX".to_string())?;
        let left_i32 =
            i32::try_from(src_left).map_err(|_| "src_left exceeds i32::MAX".to_string())?;
        let top_i32 =
            i32::try_from(src_top).map_err(|_| "src_top exceeds i32::MAX".to_string())?;
        let black_i32 =
            i32::try_from(black_14).map_err(|_| "black exceeds i32::MAX".to_string())?;

        let params = BayerDemosaicParams {
            raw_stride: stride_i32,
            src_left: left_i32,
            src_top: top_i32,
            width: w_i32,
            height: h_i32,
            shift_left: shift_left as i32,
            shift_right: shift_right as i32,
            rgb_black: black_i32,
            red_gain: s1.red_gain,
            blue_gain: s1.blue_gain,
            noise_floor: s1.noise_floor,
            black_pivot: s1.black_pivot,
            wb_r: wb[0],
            wb_g: wb[1],
            wb_b: wb[2],
            stage3_offset: 128.0,
        };

        // Upload only the contiguous row slice [src_top .. src_top + height) to avoid
        // copying unused top/bottom optical black rows over PCIe.
        let slice_start = src_top * raw_stride;
        let slice_end = (src_top + height) * raw_stride;
        Ok((params, &raw[slice_start..slice_end]))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_bayer_rggb_cuda_inner(
        raw: &[u16],
        raw_stride: usize,
        src_left: usize,
        src_top: usize,
        width: usize,
        height: usize,
        shift_left: u32,
        shift_right: u32,
        black_14: u32,
        wb: [f32; 3],
        noise: [f32; 2],
    ) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
        if !demosaic_cuda_enabled() {
            return Err("disabled via AURORA_DEMOSAIC_CUDA=0".to_string());
        }
        let (mut params, raw_window) = bayer_stage_params(
            raw, raw_stride, src_left, src_top, width, height, shift_left, shift_right, black_14,
            wb, noise,
        )?;
        params.src_top = 0;

        let stage1_pixels = (width + 16) * (height + 16);
        let stage2_pixels = (width + 12) * (height + 12);
        let out_pixels = width * height;

        let raw_bytes = raw_window.len() * std::mem::size_of::<u16>();
        let stage1_bytes = stage1_pixels * 16; // float4
        let stage2_bytes = stage2_pixels * 16; // float4
        let out_bytes = out_pixels * 12; // [f32; 3]

        let gpu = Nvidia::open_ptx(PTX).map_err(|e| format!("CUDA open_ptx: {e:?}"))?;
        let sizes = [raw_bytes, stage1_bytes, stage2_bytes, out_bytes];
        let mut out = vec![[0.0f32; 3]; out_pixels];

        gpu.with_reusable_buffers("raw-demosaic-bayer", &sizes, |session| {
            session.upload(0, unsafe { bytes_of(raw_window) })?;
            let buf_raw = session.handle(0);
            let buf_s1 = session.handle(1);
            let buf_s2 = session.handle(2);
            let buf_out = session.handle(3);

            launch_bayer_chain(
                session, PTX, &params, buf_raw, buf_s1, buf_s2, buf_out, width, height,
            )?;

            session.download(3, unsafe { bytes_of_mut(&mut out) })?;
            session.mark_consumed();
            Ok(())
        })
        .map_err(|e| format!("CUDA demosaic_bayer: {e:?}"))?;

        Ok((width, height, out))
    }

    /// Demosaics any standard 2x2 Bayer RAW frame (RGGB/GRBG/GBRG/BGGR) through
    /// the C1 Stage 1/2/3 pipeline entirely in VRAM.
    ///
    /// Automatically aligns the active crop to an even RGGB origin and normalizes
    /// bit depth to 14-bit identically to `aurora_raw::generic_bayer_c1_stage_input`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn demosaic_bayer_cuda(
        raw: &[u16],
        raw_width: usize,
        raw_height: usize,
        left_margin: usize,
        top_margin: usize,
        width: usize,
        height: usize,
        bits: u8,
        filters: u32,
        black: u32,
        wb: [f32; 3],
        noise: [f32; 2],
    ) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
        if filters == 0 || filters == 9 {
            return Err(format!("unsupported Bayer filters mask: {filters:#x}"));
        }
        let (dx, dy) = find_rggb_offset(filters, left_margin, top_margin)
            .ok_or_else(|| format!("cannot find RGGB origin for filters={filters:#x}"))?;
        let src_left = left_margin + dx;
        let src_top = top_margin + dy;
        let out_w = width.saturating_sub(dx) & !1;
        let out_h = height.saturating_sub(dy) & !1;
        if out_w < 16 || out_h < 16 || src_left + out_w > raw_width || src_top + out_h > raw_height {
            return Err(format!(
                "invalid RGGB-aligned Bayer crop: ({src_left},{src_top}) {out_w}x{out_h} in {raw_width}x{raw_height}"
            ));
        }

        let (shift_left, shift_right) = bit_shifts_to_14(bits);
        let black_14 = if shift_left > 0 {
            black << shift_left
        } else if shift_right > 0 {
            black >> shift_right
        } else {
            black
        };

        run_bayer_rggb_cuda_inner(
            raw,
            raw_width,
            src_left,
            src_top,
            out_w,
            out_h,
            shift_left,
            shift_right,
            black_14,
            wb,
            noise,
        )
    }

    /// Demosaics an already RGGB-aligned 14-bit `NikonHeCropFrame` (e.g. from
    /// `nikon_c1_stage_input` or `generic_bayer_c1_stage_input`) on CUDA.
    pub(crate) fn demosaic_rggb_crop_cuda(
        frame: &NikonHeCropFrame,
        wb: [f32; 3],
        black: u32,
        noise: [f32; 2],
    ) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
        let crop = frame.crop();
        if (crop.left | crop.top) & 1 != 0 {
            return Err(format!(
                "NikonHeCropFrame crop origin ({}, {}) is not even RGGB-aligned",
                crop.left, crop.top
            ));
        }
        run_bayer_rggb_cuda_inner(
            frame.full_samples(),
            frame.full_width(),
            crop.left,
            crop.top,
            crop.width,
            crop.height,
            0,
            0,
            black,
            wb,
            noise,
        )
    }

    /// Validates an X-Trans active window and returns the kernel params plus the
    /// uploaded row window. Shared by the standalone and the fused paths.
    #[allow(clippy::too_many_arguments)]
    fn xtrans_stage_params<'a>(
        raw: &'a [u16],
        raw_width: usize,
        left: usize,
        top: usize,
        width: usize,
        height: usize,
        bits: u8,
        black: f32,
        wb: [f32; 3],
    ) -> Result<(XTransDemosaicParams, &'a [u16]), String> {
        if width < 8 || height < 8 {
            return Err(format!("X-Trans dimensions too small: {width}x{height}"));
        }
        let needed_rows = top
            .checked_add(height)
            .ok_or_else(|| "X-Trans row overflow".to_string())?;
        let needed_cols = left
            .checked_add(width)
            .ok_or_else(|| "X-Trans col overflow".to_string())?;
        if needed_cols > raw_width {
            return Err(format!(
                "X-Trans crop ({left}+{width}) exceeds raw_width {raw_width}"
            ));
        }
        let min_len = needed_rows
            .checked_mul(raw_width)
            .ok_or_else(|| "X-Trans buffer length overflow".to_string())?;
        if raw.len() < min_len {
            return Err(format!(
                "X-Trans raw slice too short: {} < {min_len}",
                raw.len()
            ));
        }

        let (shift_left, shift_right) = bit_shifts_to_14(bits);
        let black_14 = if shift_left > 0 {
            black * (1u32 << shift_left) as f32
        } else if shift_right > 0 {
            black / (1u32 << shift_right) as f32
        } else {
            black
        };

        let w_i32 = i32::try_from(width).map_err(|_| "width exceeds i32::MAX".to_string())?;
        let h_i32 = i32::try_from(height).map_err(|_| "height exceeds i32::MAX".to_string())?;
        let stride_i32 =
            i32::try_from(raw_width).map_err(|_| "raw_width exceeds i32::MAX".to_string())?;
        let left_i32 = i32::try_from(left).map_err(|_| "left exceeds i32::MAX".to_string())?;

        let params = XTransDemosaicParams {
            raw_stride: stride_i32,
            left: left_i32,
            top: 0,
            width: w_i32,
            height: h_i32,
            shift_left: shift_left as i32,
            shift_right: shift_right as i32,
            black: black_14,
            wb_r: wb[0],
            wb_g: wb[1],
            wb_b: wb[2],
        };

        // Upload only the active rows [top .. top + height).
        let slice_start = top * raw_width;
        let slice_end = (top + height) * raw_width;
        Ok((params, &raw[slice_start..slice_end]))
    }

    /// Demosaics a Fujifilm X-Trans 6x6 CFA active plane into linear 16-bit camera
    /// RGB (`0.0..=65535.0`) on CUDA, 1:1 aligned with `xtrans::demosaic_xtrans`.
    ///
    /// Pass `bits = 14` (or `0`) when `raw` and `black` are already in 14-bit units;
    /// for other bit depths (`bits < 14` or `bits > 14`), both sample values and
    /// `black` are shifted into 14-bit identically to `aurora_raw::decode_xtrans_linear`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn demosaic_xtrans_cuda(
        raw: &[u16],
        raw_width: usize,
        left: usize,
        top: usize,
        width: usize,
        height: usize,
        bits: u8,
        black: f32,
        wb: [f32; 3],
    ) -> Result<Vec<[f32; 3]>, String> {
        if !demosaic_cuda_enabled() {
            return Err("disabled via AURORA_DEMOSAIC_CUDA=0".to_string());
        }
        let (params, raw_window) =
            xtrans_stage_params(raw, raw_width, left, top, width, height, bits, black, wb)?;

        let pixels = width * height;
        let raw_bytes = raw_window.len() * std::mem::size_of::<u16>();
        let plane_bytes = pixels * std::mem::size_of::<f32>();
        let out_bytes = pixels * 12; // [f32; 3]


        let gpu = Nvidia::open_ptx(PTX).map_err(|e| format!("CUDA open_ptx: {e:?}"))?;
        let sizes = [raw_bytes, plane_bytes, plane_bytes, plane_bytes, out_bytes];
        let mut out = vec![[0.0f32; 3]; pixels];

        gpu.with_reusable_buffers("raw-demosaic-xtrans", &sizes, |session| {
            session.upload(0, unsafe { bytes_of(raw_window) })?;
            let buf_raw = session.handle(0);
            let buf_plane = session.handle(1);
            let buf_green = session.handle(2);
            let buf_green_fixed = session.handle(3);
            let buf_out = session.handle(4);

            launch_xtrans_chain(
                session,
                PTX,
                &params,
                buf_raw,
                buf_plane,
                buf_green,
                buf_green_fixed,
                buf_out,
                width,
                height,
            )?;

            session.download(4, unsafe { bytes_of_mut(&mut out) })?;
            session.mark_consumed();
            Ok(())
        })
        .map_err(|e| format!("CUDA demosaic_xtrans: {e:?}"))?;

        Ok(out)
    }

    // ---------------------------------------------------------------------
    // Fused demosaic -> linear finish (task 3.2)
    //
    // The historical shape is two CUDA calls with a host round trip in the
    // middle: `demosaic_*_cuda` downloads the whole frame as `[f32; 3]`, then
    // `linear_post_cuda::linear_finish_cuda` uploads the same bytes again.
    // Both stages already run in the process-wide CUDA domain, so the tail can
    // read the demosaic's `[f32; 3]` buffer directly. These entries keep one
    // session open across both kernel sets, which removes exactly two host
    // copies per frame of `pixels * 12` bytes each (one D2H + one H2D),
    // leaving one H2D of the raw window and one D2H of the finished frame.
    //
    // The device-side pack (RGB3 -> float4) still runs: the ColorNoise2012 and
    // anti-alias kernels are float4-based, and it now reads the demosaic
    // output in place instead of a freshly uploaded copy.
    // ---------------------------------------------------------------------

    /// The black point handed to highlight recovery, in the same `0..65535`
    /// domain `linear_post_cuda::linear_finish_cuda` expects, or `None` when the
    /// frame/black point leaves nothing to recover. Mirrors that function's own
    /// guard, so a fused frame recovers highlights exactly when the two-segment
    /// path would.
    pub(super) fn recovery_black(
        black_16: u32,
        width: usize,
        height: usize,
        wb: [f32; 3],
    ) -> Option<u32> {
        let usable = width >= 4
            && height >= 4
            && black_16 < 65_535
            && wb.into_iter().all(|gain| gain.is_finite() && gain > 0.0);
        usable.then_some(black_16)
    }

    /// The linear-finish tail over a `[f32; 3]` buffer that is already in VRAM.
    /// Delegates to the same chain the two-stage path runs, so fused and
    /// two-stage decodes share one implementation and cannot drift.
    #[allow(clippy::too_many_arguments)]
    fn launch_linear_finish_tail(
        session: &mut BufferSession<'_>,
        buf_rgb3: u64,
        buf_a: u64,
        buf_b: u64,
        width: usize,
        height: usize,
        recovery: Option<u32>,
        wb: [f32; 3],
        iso: u32,
        noise: [f32; 2],
    ) -> Result<(), CudaError> {
        launch_finish_chain(
            session,
            buf_rgb3,
            buf_a,
            buf_b,
            width as i32,
            height as i32,
            (width * height) as i32,
            recovery.map(|black| (black, wb)),
            iso,
            noise,
        )
    }

    /// Bayer Stage 1/2/3 + linear finish over one VRAM allocation set.
    #[allow(clippy::too_many_arguments)]
    fn run_bayer_rggb_fused_inner(
        raw: &[u16],
        raw_stride: usize,
        src_left: usize,
        src_top: usize,
        width: usize,
        height: usize,
        shift_left: u32,
        shift_right: u32,
        black_14: u32,
        finish_black: u32,
        wb: [f32; 3],
        noise: [f32; 2],
        iso: u32,
    ) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
        if !super::fused_cuda_enabled() {
            return Err("disabled via AURORA_RAW_FUSED_CUDA=0".to_string());
        }
        if !demosaic_cuda_enabled() {
            return Err("disabled via AURORA_DEMOSAIC_CUDA=0".to_string());
        }
        if !crate::aurora_raw::linear_post_cuda::linear_post_cuda_enabled() {
            return Err("disabled via AURORA_LINEAR_POST_CUDA=0".to_string());
        }
        let (mut params, raw_window) = bayer_stage_params(
            raw, raw_stride, src_left, src_top, width, height, shift_left, shift_right, black_14,
            wb, noise,
        )?;
        params.src_top = 0;

        let stage1_pixels = (width + 16) * (height + 16);
        let stage2_pixels = (width + 12) * (height + 12);
        let out_pixels = width * height;

        let raw_bytes = raw_window.len() * std::mem::size_of::<u16>();
        let stage1_bytes = stage1_pixels * 16; // float4
        let stage2_bytes = stage2_pixels * 16; // float4
        let out_bytes = out_pixels * 12; // [f32; 3]
        let f4_bytes = out_pixels * 16; // float4 scratch for the linear tail

        // The session is opened on the linear-post image because the tail has
        // more kernels than the head; the four demosaic kernels are reached
        // through `launch_from_image`, which resolves the same cached module.
        let gpu = Nvidia::open_ptx(linear_post_ptx_image())
            .map_err(|e| format!("CUDA open_ptx: {e:?}"))?;
        let sizes = [
            raw_bytes, stage1_bytes, stage2_bytes, out_bytes, f4_bytes, f4_bytes,
        ];
        let mut out = vec![[0.0f32; 3]; out_pixels];
        let recovery = recovery_black(finish_black, width, height, wb);

        gpu.with_reusable_buffers("raw-demosaic-finish-bayer", &sizes, |session| {
            session.upload(0, unsafe { bytes_of(raw_window) })?;
            #[cfg(test)]
            super::FUSED_HOST_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let buf_raw = session.handle(0);
            let buf_s1 = session.handle(1);
            let buf_s2 = session.handle(2);
            let buf_rgb3 = session.handle(3);
            let buf_a = session.handle(4);
            let buf_b = session.handle(5);

            launch_bayer_chain(
                session, PTX, &params, buf_raw, buf_s1, buf_s2, buf_rgb3, width, height,
            )?;
            launch_linear_finish_tail(
                session, buf_rgb3, buf_a, buf_b, width, height, recovery, wb, iso, noise,
            )?;

            session.download(3, unsafe { bytes_of_mut(&mut out) })?;
            #[cfg(test)]
            super::FUSED_HOST_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            session.mark_consumed();
            Ok(())
        })
        .map_err(|e| format!("CUDA demosaic_finish_bayer: {e:?}"))?;

        Ok((width, height, out))
    }

    /// Demosaics a generic 2x2 Bayer frame and runs `linear_finish` without a
    /// host round trip. Same arguments and result as calling
    /// [`demosaic_bayer_cuda`] and then
    /// `linear_post_cuda::linear_finish_cuda(..., Some((black_14 << 2, wb)), iso, noise)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn demosaic_and_finish_bayer_cuda(
        raw: &[u16],
        raw_width: usize,
        raw_height: usize,
        left_margin: usize,
        top_margin: usize,
        width: usize,
        height: usize,
        bits: u8,
        filters: u32,
        black: u32,
        wb: [f32; 3],
        noise: [f32; 2],
        iso: u32,
    ) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
        if filters == 0 || filters == 9 {
            return Err(format!("unsupported Bayer filters mask: {filters:#x}"));
        }
        let (dx, dy) = find_rggb_offset(filters, left_margin, top_margin)
            .ok_or_else(|| format!("cannot find RGGB origin for filters={filters:#x}"))?;
        let src_left = left_margin + dx;
        let src_top = top_margin + dy;
        let out_w = width.saturating_sub(dx) & !1;
        let out_h = height.saturating_sub(dy) & !1;
        if out_w < 16 || out_h < 16 || src_left + out_w > raw_width || src_top + out_h > raw_height {
            return Err(format!(
                "invalid RGGB-aligned Bayer crop: ({src_left},{src_top}) {out_w}x{out_h} in {raw_width}x{raw_height}"
            ));
        }

        let (shift_left, shift_right) = bit_shifts_to_14(bits);
        let black_14 = if shift_left > 0 {
            black << shift_left
        } else if shift_right > 0 {
            black >> shift_right
        } else {
            black
        };

        run_bayer_rggb_fused_inner(
            raw,
            raw_width,
            src_left,
            src_top,
            out_w,
            out_h,
            shift_left,
            shift_right,
            black_14,
            black_14 << 2, // generic Bayer: match segmented 16-bit recovery input
            wb,
            noise,
            iso,
        )
    }

    /// Fused variant over an already RGGB-aligned 14-bit `NikonHeCropFrame`.
    /// The recovery black is the input `black` itself (already 16-bit domain),
    /// matching what the two-segment caller passes to `linear_finish_cuda`.
    pub(crate) fn demosaic_and_finish_rggb_crop_cuda(
        frame: &NikonHeCropFrame,
        wb: [f32; 3],
        black: u32,
        noise: [f32; 2],
        iso: u32,
    ) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
        let crop = frame.crop();
        if (crop.left | crop.top) & 1 != 0 {
            return Err(format!(
                "NikonHeCropFrame crop origin ({}, {}) is not even RGGB-aligned",
                crop.left, crop.top
            ));
        }
        run_bayer_rggb_fused_inner(
            frame.full_samples(),
            frame.full_width(),
            crop.left,
            crop.top,
            crop.width,
            crop.height,
            0,
            0,
            black,
            black, // Nikon crop: MakerNote black is passed unchanged to recovery
            wb,
            noise,
            iso,
        )
    }

    /// Demosaics an X-Trans active plane and runs the linear finish in the same
    /// session. Equivalent to [`demosaic_xtrans_cuda`] followed by
    /// `linear_finish_cuda(rgb, w, h, None, iso, noise)`; there is no Bayer
    /// highlight-recovery stage on this sensor, so `recovery` is `None`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn demosaic_and_finish_xtrans_cuda(
        raw: &[u16],
        raw_width: usize,
        left: usize,
        top: usize,
        width: usize,
        height: usize,
        bits: u8,
        black: f32,
        wb: [f32; 3],
        noise: [f32; 2],
        iso: u32,
    ) -> Result<Vec<[f32; 3]>, String> {
        if !super::fused_cuda_enabled() {
            return Err("disabled via AURORA_RAW_FUSED_CUDA=0".to_string());
        }
        if !demosaic_cuda_enabled() {
            return Err("disabled via AURORA_DEMOSAIC_CUDA=0".to_string());
        }
        if !crate::aurora_raw::linear_post_cuda::linear_post_cuda_enabled() {
            return Err("disabled via AURORA_LINEAR_POST_CUDA=0".to_string());
        }
        let (params, raw_window) =
            xtrans_stage_params(raw, raw_width, left, top, width, height, bits, black, wb)?;

        let pixels = width * height;
        let raw_bytes = raw_window.len() * std::mem::size_of::<u16>();
        let plane_bytes = pixels * std::mem::size_of::<f32>();
        let out_bytes = pixels * 12; // [f32; 3]
        let f4_bytes = pixels * 16; // float4 scratch for the linear tail

        let gpu = Nvidia::open_ptx(linear_post_ptx_image())
            .map_err(|e| format!("CUDA open_ptx: {e:?}"))?;
        let sizes = [
            raw_bytes, plane_bytes, plane_bytes, plane_bytes, out_bytes, f4_bytes, f4_bytes,
        ];
        let mut out = vec![[0.0f32; 3]; pixels];

        gpu.with_reusable_buffers("raw-demosaic-finish-xtrans", &sizes, |session| {
            session.upload(0, unsafe { bytes_of(raw_window) })?;
            #[cfg(test)]
            super::FUSED_HOST_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let buf_raw = session.handle(0);
            let buf_plane = session.handle(1);
            let buf_green = session.handle(2);
            let buf_green_fixed = session.handle(3);
            let buf_rgb3 = session.handle(4);
            let buf_a = session.handle(5);
            let buf_b = session.handle(6);

            launch_xtrans_chain(
                session,
                PTX,
                &params,
                buf_raw,
                buf_plane,
                buf_green,
                buf_green_fixed,
                buf_rgb3,
                width,
                height,
            )?;
            launch_linear_finish_tail(
                session, buf_rgb3, buf_a, buf_b, width, height, None, wb, iso, noise,
            )?;

            session.download(4, unsafe { bytes_of_mut(&mut out) })?;
            #[cfg(test)]
            super::FUSED_HOST_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            session.mark_consumed();
            Ok(())
        })
        .map_err(|e| format!("CUDA demosaic_finish_xtrans: {e:?}"))?;

        Ok(out)
    }
}

#[cfg(windows)]
pub(crate) use imp::{
    demosaic_and_finish_bayer_cuda, demosaic_and_finish_rggb_crop_cuda,
    demosaic_and_finish_xtrans_cuda, demosaic_bayer_cuda, demosaic_rggb_crop_cuda,
    demosaic_xtrans_cuda, ptx_image as demosaic_ptx_image,
};

#[cfg(not(windows))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn demosaic_bayer_cuda(
    _raw: &[u16],
    _raw_width: usize,
    _raw_height: usize,
    _left_margin: usize,
    _top_margin: usize,
    _width: usize,
    _height: usize,
    _bits: u8,
    _filters: u32,
    _black: u32,
    _wb: [f32; 3],
    _noise: [f32; 2],
) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
    Err("demosaic_cuda is only available on Windows".to_string())
}

#[cfg(not(windows))]
pub(crate) fn demosaic_rggb_crop_cuda(
    _frame: &super::c1_raw_he::NikonHeCropFrame,
    _wb: [f32; 3],
    _black: u32,
    _noise: [f32; 2],
) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
    Err("demosaic_cuda is only available on Windows".to_string())
}

#[cfg(not(windows))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn demosaic_xtrans_cuda(
    _raw: &[u16],
    _raw_width: usize,
    _left: usize,
    _top: usize,
    _width: usize,
    _height: usize,
    _bits: u8,
    _black: f32,
    _wb: [f32; 3],
) -> Result<Vec<[f32; 3]>, String> {
    Err("demosaic_cuda is only available on Windows".to_string())
}

#[cfg(not(windows))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn demosaic_and_finish_bayer_cuda(
    _raw: &[u16],
    _raw_width: usize,
    _raw_height: usize,
    _left_margin: usize,
    _top_margin: usize,
    _width: usize,
    _height: usize,
    _bits: u8,
    _filters: u32,
    _black: u32,
    _wb: [f32; 3],
    _noise: [f32; 2],
    _iso: u32,
) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
    Err("demosaic_cuda is only available on Windows".to_string())
}

#[cfg(not(windows))]
pub(crate) fn demosaic_and_finish_rggb_crop_cuda(
    _frame: &super::c1_raw_he::NikonHeCropFrame,
    _wb: [f32; 3],
    _black: u32,
    _noise: [f32; 2],
    _iso: u32,
) -> Result<(usize, usize, Vec<[f32; 3]>), String> {
    Err("demosaic_cuda is only available on Windows".to_string())
}

#[cfg(not(windows))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn demosaic_and_finish_xtrans_cuda(
    _raw: &[u16],
    _raw_width: usize,
    _left: usize,
    _top: usize,
    _width: usize,
    _height: usize,
    _bits: u8,
    _black: f32,
    _wb: [f32; 3],
    _noise: [f32; 2],
    _iso: u32,
) -> Result<Vec<[f32; 3]>, String> {
    Err("demosaic_cuda is only available on Windows".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_toggle_parses_disable_values() {
        assert!(parse_enabled_env(None));
        assert!(parse_enabled_env(Some("1")));
        assert!(parse_enabled_env(Some("true")));
        assert!(!parse_enabled_env(Some("0")));
        assert!(!parse_enabled_env(Some(" 0 ")));
        assert!(!parse_enabled_env(Some("false")));
        assert!(!parse_enabled_env(Some("OFF")));
    }

    #[cfg(windows)]
    fn max_abs_diff(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .flat_map(|(pa, pb)| {
                [
                    (pa[0] - pb[0]).abs(),
                    (pa[1] - pb[1]).abs(),
                    (pa[2] - pb[2]).abs(),
                ]
            })
            .fold(0.0f32, f32::max)
    }

    #[cfg(windows)]
    #[test]
    fn xtrans_demosaic_cuda_matches_cpu() {
        const XTRANS_STD: [[u8; 6]; 6] = [
            [1, 2, 1, 1, 0, 1],
            [0, 1, 0, 2, 1, 2],
            [1, 2, 1, 1, 0, 1],
            [1, 0, 1, 1, 2, 1],
            [2, 1, 2, 0, 1, 0],
            [1, 0, 1, 1, 2, 1],
        ];
        let (raw_w, raw_h) = (96usize, 84usize);
        let (left, top, w, h) = (7usize, 5usize, 73usize, 61usize);
        let mut raw = vec![0u16; raw_w * raw_h];
        for y in 0..h {
            for x in 0..w {
                let c = XTRANS_STD[y % 6][x % 6];
                // Rich synthetic scene: diagonal edge, fine stripes, darks, highlights.
                let edge = if x * 3 + y * 2 > 140 { 6200i32 } else { 1800i32 };
                let stripe = if ((x + y) % 5) == 0 { 950i32 } else { -300i32 };
                let grad = (x as i32 * 47 + y as i32 * 31) % 2400;
                let ch_bias = match c {
                    0 => 800,
                    1 => 1400,
                    _ => 450,
                };
                let val = (edge + stripe + grad + ch_bias).clamp(200, 16383) as u16;
                raw[(top + y) * raw_w + left + x] = val;
            }
        }
        let black = 1024.0f32;
        let wb = [1.85f32, 1.0, 1.62];

        let cpu = super::super::xtrans::demosaic_xtrans(&raw, raw_w, left, top, w, h, black, wb)
            .expect("CPU X-Trans demosaic");
        match demosaic_xtrans_cuda(&raw, raw_w, left, top, w, h, 14, black, wb) {
            Ok(gpu) => {
                let diff = max_abs_diff(&cpu, &gpu);
                println!("X-Trans CUDA vs CPU ({w}x{h}): max_abs_diff={diff:.6e}");
                assert!(
                    diff < 1e-3,
                    "X-Trans CUDA vs CPU max_abs_diff={diff} exceeds 1e-3"
                );
            }
            Err(err) => {
                eprintln!("skipping CUDA X-Trans parity check (no CUDA device): {err}");
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn bayer_demosaic_cuda_matches_cpu_all_cfa_phases() {
        // LibRaw filter masks for RGGB, GRBG, GBRG, BGGR.
        let patterns = [
            ("RGGB", 0x9494_9494u32),
            ("GRBG", 0x6161_6161u32),
            ("GBRG", 0x4949_4949u32),
            ("BGGR", 0x1616_1616u32),
        ];
        let wb = [1.92f32, 1.0, 1.54];
        let black = 1008u32;
        let noise = [12.8f32, 16.0];

        for (name, filters) in patterns {
            let (raw_w, raw_h) = (192usize, 160usize);
            let (left_margin, top_margin, w, h) = (7usize, 9usize, 144usize, 136usize);
            let mut raw = vec![0u16; raw_w * raw_h];
            for y in 0..raw_h {
                for x in 0..raw_w {
                    let c = bayer_color_at(filters, x, y);
                    let diag = if (x as i32 - y as i32) > 12 { 7200i32 } else { 2200i32 };
                    let ripple = (((x * 7 + y * 11) % 19) as i32) * 140;
                    let high = if x < 40 && y < 40 { 5200 } else { 0 };
                    let ch = match c {
                        0 => 900,
                        2 => 550,
                        _ => 1250,
                    };
                    raw[y * raw_w + x] = (diag + ripple + high + ch).clamp(400, 16383) as u16;
                }
            }

            let bayer = crate::raw::BayerFrame {
                raw_width: raw_w,
                raw_height: raw_h,
                width: w,
                height: h,
                top_margin,
                left_margin,
                stride: raw_w * 2,
                bits: 14,
                data: raw.clone(),
                filters,
                colors: 3,
                black,
                maximum: 16383,
                cam_mul: [wb[0] * 1024.0, 1024.0, wb[2] * 1024.0, 1024.0],
                rgb_cam: [[0.0; 4]; 3],
            };

            let (frame, cpu_wb, cpu_black) =
                super::super::generic_bayer_c1_stage_input(&bayer).expect("CPU RGGB align");
            let (cpu_w, cpu_h, cpu_rgb) =
                super::super::run_c1_stage123_tiles_f32(&frame, cpu_wb, cpu_black, noise)
                    .expect("CPU C1 Stage 1/2/3");

            match demosaic_bayer_cuda(
                &raw,
                raw_w,
                raw_h,
                left_margin,
                top_margin,
                w,
                h,
                14,
                filters,
                black,
                wb,
                noise,
            ) {
                Ok((gpu_w, gpu_h, gpu_rgb)) => {
                    assert_eq!((gpu_w, gpu_h), (cpu_w, cpu_h));
                    let diff = max_abs_diff(&cpu_rgb, &gpu_rgb);
                    println!(
                        "Bayer {name} CUDA vs CPU ({gpu_w}x{gpu_h}): max_abs_diff={diff:.6e}"
                    );
                    assert!(
                        diff < 1e-2,
                        "Bayer {name} CUDA vs CPU max_abs_diff={diff} exceeds 1e-2"
                    );
                }
                Err(err) => {
                    eprintln!("skipping CUDA Bayer parity check (no CUDA device): {err}");
                    return;
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Fused demosaic -> linear finish (task 3.2)
    // ---------------------------------------------------------------

    // Nikon MakerNote and generic Bayer use different recovery black domains.
    // Comparing only generic CFA phases did not expose this on a real Nikon Z8.
    #[cfg(windows)]
    #[test]
    fn fused_nikon_crop_uses_segmented_recovery_black() {
        use crate::aurora_raw::c1_raw_he::{NikonCrop, NikonHeCropFrame};
        use crate::aurora_raw::linear_post_cuda::linear_finish_cuda;
        let (w, h) = (48, 40);
        let samples = (0..w*h).map(|i| ((i * 157 + i / w * 197) % 16_384) as u16).collect();
        let frame = NikonHeCropFrame::new(NikonCrop { left: 0, top: 0, width: w, height: h }, w, h, samples).unwrap();
        let (wb, black, noise) = ([1.87, 1., 1.51], 1024, [10.5, 22.]);
        let Ok((width, height, mut reference)) = demosaic_rggb_crop_cuda(&frame, wb, black, noise) else { eprintln!("skip Nikon GPU check: device unavailable"); return };
        linear_finish_cuda(&mut reference, width, height, Some((black, wb)), 800, noise).unwrap();
        let (_, _, fused) = demosaic_and_finish_rggb_crop_cuda(&frame, wb, black, noise, 800).unwrap();
        assert_eq!(max_abs_diff(&reference, &fused), 0.0, "Nikon recovery black diverged");
        eprintln!("Nikon crop GPU: every RGB float matches the segmented reference");
    }

    /// Small-image parity for the fused chain: the frame produced by
    /// `demosaic_and_finish_bayer_cuda` must equal the one the historical
    /// two-segment path produces for the same input, at every ISO branch the
    /// linear tail has (stage 1 only, ISO>=400 wide stage, ISO>=1600 repeat,
    /// ISO>=3200 anti-alias gain).
    #[cfg(windows)]
    #[test]
    fn fused_bayer_cuda_matches_two_segment() {
        use crate::aurora_raw::linear_post_cuda::linear_finish_cuda;

        let patterns = [
            ("RGGB", 0x9494_9494u32),
            ("GRBG", 0x6161_6161u32),
            ("GBRG", 0x4949_4949u32),
            ("BGGR", 0x1616_1616u32),
        ];
        let wb = [1.87f32, 1.0, 1.51];
        let black = 1024u32;
        let noise = [10.5f32, 22.0];
        let (raw_w, raw_h) = (64usize, 56usize);
        let (left_margin, top_margin, w, h) = (5usize, 7usize, 48usize, 40usize);

        for (name, filters) in patterns {
            let mut raw = vec![0u16; raw_w * raw_h];
            for y in 0..raw_h {
                for x in 0..raw_w {
                    let c = bayer_color_at(filters, x, y);
                    let diag = if (x as i32 - y as i32) > 4 { 7300i32 } else { 2100i32 };
                    let ripple = (((x * 5 + y * 3) % 17) as i32) * 120;
                    let high = if x < 12 && y < 12 { 5400 } else { 0 };
                    let ch = match c {
                        0 => 820,
                        2 => 500,
                        _ => 1180,
                    };
                    raw[y * raw_w + x] = (diag + ripple + high + ch).clamp(300, 16383) as u16;
                }
            }

            for iso in [100u32, 800, 1600, 3200] {
                let (tw, th, mut two) = match demosaic_bayer_cuda(
                    &raw, raw_w, raw_h, left_margin, top_margin, w, h, 14, filters, black, wb,
                    noise,
                ) {
                    Ok(value) => value,
                    Err(err) => {
                        eprintln!("skipping fused Bayer parity check (no CUDA device): {err}");
                        return;
                    }
                };
                linear_finish_cuda(&mut two, tw, th, Some((black << 2, wb)), iso, noise)
                    .expect("two-segment linear_finish_cuda");

                let (fw, fh, fused) = demosaic_and_finish_bayer_cuda(
                    &raw, raw_w, raw_h, left_margin, top_margin, w, h, 14, filters, black, wb,
                    noise, iso,
                )
                .expect("fused Bayer chain");
                assert_eq!((fw, fh), (tw, th), "{name} ISO {iso}: dimension mismatch");
                let diff = max_abs_diff(&two, &fused);
                println!(
                    "fused vs two-segment Bayer {name} ISO {iso} ({fw}x{fh}): max_abs_diff={diff:.6e}"
                );
                assert!(
                    diff < 1e-4,
                    "{name} ISO {iso}: fused vs two-segment max_abs_diff={diff}"
                );
            }
        }
    }

    /// The same parity check on the X-Trans path (no highlight recovery there,
    /// so `recovery = None`), at the low and the ISO>=1600 double-pass settings.
    #[cfg(windows)]
    #[test]
    fn fused_xtrans_cuda_matches_two_segment() {
        use crate::aurora_raw::linear_post_cuda::linear_finish_cuda;

        const XTRANS_STD: [[u8; 6]; 6] = [
            [1, 2, 1, 1, 0, 1],
            [0, 1, 0, 2, 1, 2],
            [1, 2, 1, 1, 0, 1],
            [1, 0, 1, 1, 2, 1],
            [2, 1, 2, 0, 1, 0],
            [1, 0, 1, 1, 2, 1],
        ];
        let (raw_w, raw_h) = (48usize, 42usize);
        let (left, top, w, h) = (5usize, 3usize, 40usize, 34usize);
        let black = 1024.0f32;
        let wb = [1.85f32, 1.0, 1.62];
        let noise = [14.0f32, 120.0];

        let mut raw = vec![0u16; raw_w * raw_h];
        for y in 0..h {
            for x in 0..w {
                let c = XTRANS_STD[y % 6][x % 6];
                let edge = if x * 3 + y * 2 > 90 { 6100i32 } else { 1900i32 };
                let stripe = if ((x + y) % 5) == 0 { 900i32 } else { -320i32 };
                let grad = (x as i32 * 41 + y as i32 * 29) % 2300;
                let ch_bias = match c {
                    0 => 780,
                    1 => 1350,
                    _ => 430,
                };
                raw[(top + y) * raw_w + left + x] =
                    (edge + stripe + grad + ch_bias).clamp(200, 16383) as u16;
            }
        }

        for iso in [100u32, 1600] {
            let mut two = match demosaic_xtrans_cuda(&raw, raw_w, left, top, w, h, 14, black, wb)
            {
                Ok(value) => value,
                Err(err) => {
                    eprintln!("skipping fused X-Trans parity check (no CUDA device): {err}");
                    return;
                }
            };
            linear_finish_cuda(&mut two, w, h, None, iso, noise).expect("two-segment finish");

            let fused = demosaic_and_finish_xtrans_cuda(
                &raw, raw_w, left, top, w, h, 14, black, wb, noise, iso,
            )
            .expect("fused X-Trans chain");
            assert_eq!(fused.len(), two.len());
            let diff = max_abs_diff(&two, &fused);
            println!("fused vs two-segment X-Trans ISO {iso}: max_abs_diff={diff:.6e}");
            assert!(diff < 1e-4, "X-Trans ISO {iso}: max_abs_diff={diff}");
        }
    }

    /// The fused frame must leave the device exactly twice: the raw window up
    /// and the finished frame down. The two-segment path it replaces does four
    /// copies of the same frame (down for the demosaic result, up again for the
    /// linear stage), which is the whole point of the fusion.
    #[cfg(windows)]
    #[test]
    fn fused_chain_performs_exactly_two_host_copies() {
        use std::sync::atomic::Ordering;

        let (raw_w, raw_h) = (64usize, 56usize);
        let raw = vec![1800u16; raw_w * raw_h];
        let before = FUSED_HOST_COPIES.load(Ordering::Relaxed);
        let result = demosaic_and_finish_bayer_cuda(
            &raw, raw_w, raw_h, 4, 4, 48, 40, 14, 0x9494_9494, 1024, [1.9, 1.0, 1.5],
            [10.0, 20.0], 800,
        );
        if let Err(err) = &result {
            eprintln!("skipping host-copy accounting (no CUDA device): {err}");
            return;
        }
        let copies = FUSED_HOST_COPIES.load(Ordering::Relaxed) - before;
        assert_eq!(
            copies, 2,
            "one fused call must upload once and download once, counted {copies}"
        );
    }

    /// Boundaries: the fused entries refuse exactly the frames the standalone
    /// entries refuse, so a caller can fall back on the same error shape.
    #[cfg(windows)]
    #[test]
    fn fused_entries_share_the_standalone_dimension_guards() {
        let raw = vec![1500u16; 64 * 64];
        let wb = [1.9f32, 1.0, 1.5];
        let noise = [10.0f32, 20.0];

        for filters in [0x9494_9494u32, 0x6161_6161u32, 0x4949_4949u32, 0x1616_1616u32] {
            // 15x15 aligns to a 14x14 active area, below the 16x16 kernel floor.
            let two = demosaic_bayer_cuda(&raw, 64, 64, 0, 0, 15, 15, 14, filters, 1024, wb, noise);
            let fused = demosaic_and_finish_bayer_cuda(
                &raw, 64, 64, 0, 0, 15, 15, 14, filters, 1024, wb, noise, 800,
            );
            assert!(two.is_err(), "{filters:#x}: standalone must refuse 15x15");
            assert!(fused.is_err(), "{filters:#x}: fused must refuse 15x15");
        }

        // X-Trans floor is 8x8.
        assert!(demosaic_xtrans_cuda(&raw, 64, 0, 0, 7, 7, 14, 1024.0, wb).is_err());
        assert!(
            demosaic_and_finish_xtrans_cuda(&raw, 64, 0, 0, 7, 7, 14, 1024.0, wb, noise, 800)
                .is_err()
        );

        // A CFA mask that is not 2x2 Bayer and not X-Trans is refused up front.
        assert!(demosaic_and_finish_bayer_cuda(
            &raw, 64, 64, 0, 0, 32, 32, 14, 0, 1024, wb, noise, 800
        )
        .is_err());
    }

    /// The recovery stage runs on the fused frame under exactly the conditions
    /// `linear_finish_cuda` applies to its `recovery` argument.
    #[cfg(windows)]
    #[test]
    fn fused_recovery_black_matches_linear_finish_guard() {
        let wb = [1.9f32, 1.0, 1.5];
        assert_eq!(super::imp::recovery_black(4096, 32, 32, wb), Some(4096));
        // Too small for the 5x5 anti-alias pass.
        assert_eq!(super::imp::recovery_black(4096, 3, 32, wb), None);
        assert_eq!(super::imp::recovery_black(4096, 32, 3, wb), None);
        // No headroom left above the black point.
        assert_eq!(super::imp::recovery_black(65_535, 32, 32, wb), None);
        // White balance the highlight kernel cannot scale by.
        assert_eq!(super::imp::recovery_black(4096, 32, 32, [0.0, 1.0, 1.5]), None);
        assert_eq!(super::imp::recovery_black(4096, 32, 32, [1.9, f32::NAN, 1.5]), None);
    }

    /// On a host without the Windows CUDA chain the fused entries report the
    /// same unavailability the standalone ones do.
    #[cfg(not(windows))]
    #[test]
    fn fused_entries_report_platform_unavailability() {
        assert!(fused_cuda_enabled());
        let raw = [1000u16; 64];
        assert!(demosaic_and_finish_bayer_cuda(
            &raw, 8, 8, 0, 0, 8, 8, 14, 0x9494_9494, 0, [1.0; 3], [0.0; 2], 100
        )
        .is_err());
        assert!(demosaic_and_finish_xtrans_cuda(
            &raw, 8, 0, 0, 8, 8, 14, 0.0, [1.0; 3], [0.0; 2], 100
        )
        .is_err());
    }
}
