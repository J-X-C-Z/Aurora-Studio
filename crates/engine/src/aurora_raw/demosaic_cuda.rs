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
    use aurora_portrait::cuda::Nvidia;
    use std::ffi::c_void;
    use std::slice;

    const PTX: &str = include_str!("../../native/cuda/raw_demosaic.ptx");

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

        let mut params = BayerDemosaicParams {
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
        let raw_window = &raw[slice_start..slice_end];
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

            // Stage 1: directional green + black offset
            {
                let mut src = buf_raw;
                let mut dst = buf_s1;
                let mut p = params;
                let mut args: [*mut c_void; 3] = [
                    &mut src as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut p as *mut BayerDemosaicParams as *mut c_void,
                ];
                session.launch(
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
                let mut p = params;
                let mut args: [*mut c_void; 3] = [
                    &mut src as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut p as *mut BayerDemosaicParams as *mut c_void,
                ];
                session.launch(
                    KERNEL_BAYER_STAGE2_SCALE,
                    grid_2d((width + 12) as u32, (height + 12) as u32),
                    BLOCK_2D,
                    &mut args,
                )?;
            }

            // Stage 3a: diagonal R/B reconstruction (in-place on buf_s2)
            {
                let mut buf = buf_s2;
                let mut p = params;
                let mut args: [*mut c_void; 2] = [
                    &mut buf as *mut u64 as *mut c_void,
                    &mut p as *mut BayerDemosaicParams as *mut c_void,
                ];
                session.launch(
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
                let mut p = params;
                let mut args: [*mut c_void; 3] = [
                    &mut src as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut p as *mut BayerDemosaicParams as *mut c_void,
                ];
                session.launch(
                    KERNEL_BAYER_STAGE3_GREEN,
                    grid_2d(width as u32, height as u32),
                    BLOCK_2D,
                    &mut args,
                )?;
            }

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

        // Upload only the active rows [top .. top + height).
        let slice_start = top * raw_width;
        let slice_end = (top + height) * raw_width;
        let raw_window = &raw[slice_start..slice_end];

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

        let pixels = width * height;
        let raw_bytes = raw_window.len() * std::mem::size_of::<u16>();
        let plane_bytes = pixels * std::mem::size_of::<f32>();
        let out_bytes = pixels * 12; // [f32; 3]

        let gpu = Nvidia::open_ptx(PTX).map_err(|e| format!("CUDA open_ptx: {e:?}"))?;
        let sizes = [raw_bytes, plane_bytes, plane_bytes, plane_bytes, out_bytes];
        let mut out = vec![[0.0f32; 3]; pixels];
        let g2 = grid_2d(width as u32, height as u32);

        gpu.with_reusable_buffers("raw-demosaic-xtrans", &sizes, |session| {
            session.upload(0, unsafe { bytes_of(raw_window) })?;
            let buf_raw = session.handle(0);
            let buf_plane = session.handle(1);
            let buf_green = session.handle(2);
            let buf_green_fixed = session.handle(3);
            let buf_out = session.handle(4);

            // Stage 0: normalize
            {
                let mut src = buf_raw;
                let mut dst = buf_plane;
                let mut p = params;
                let mut args: [*mut c_void; 3] = [
                    &mut src as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut p as *mut XTransDemosaicParams as *mut c_void,
                ];
                session.launch(KERNEL_XTRANS_NORM, g2, BLOCK_2D, &mut args)?;
            }

            // Stage 1: directional green interpolation
            {
                let mut src = buf_plane;
                let mut dst = buf_green;
                let mut w = w_i32;
                let mut h = h_i32;
                let mut args: [*mut c_void; 4] = [
                    &mut src as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut w as *mut i32 as *mut c_void,
                    &mut h as *mut i32 as *mut c_void,
                ];
                session.launch(KERNEL_XTRANS_GREEN, g2, BLOCK_2D, &mut args)?;
            }

            // Stage 2: fixGreen cross clamp
            {
                let mut src = buf_green;
                let mut dst = buf_green_fixed;
                let mut w = w_i32;
                let mut h = h_i32;
                let mut args: [*mut c_void; 4] = [
                    &mut src as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut w as *mut i32 as *mut c_void,
                    &mut h as *mut i32 as *mut c_void,
                ];
                session.launch(KERNEL_XTRANS_FIX_GREEN, g2, BLOCK_2D, &mut args)?;
            }

            // Stage 3: 5x5 R-G / B-G chroma rebuild -> [f32; 3]
            {
                let mut src_plane = buf_plane;
                let mut src_gf = buf_green_fixed;
                let mut dst = buf_out;
                let mut w = w_i32;
                let mut h = h_i32;
                let mut args: [*mut c_void; 5] = [
                    &mut src_plane as *mut u64 as *mut c_void,
                    &mut src_gf as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut w as *mut i32 as *mut c_void,
                    &mut h as *mut i32 as *mut c_void,
                ];
                session.launch(KERNEL_XTRANS_CHROMA, g2, BLOCK_2D, &mut args)?;
            }

            session.download(4, unsafe { bytes_of_mut(&mut out) })?;
            session.mark_consumed();
            Ok(())
        })
        .map_err(|e| format!("CUDA demosaic_xtrans: {e:?}"))?;

        Ok(out)
    }
}

#[cfg(windows)]
pub(crate) use imp::{demosaic_bayer_cuda, demosaic_rggb_crop_cuda, demosaic_xtrans_cuda};

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
}
