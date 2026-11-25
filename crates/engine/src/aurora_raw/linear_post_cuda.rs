//! Native zero-DLL embedded-PTX CUDA linear RAW post-processing for generic
//! Bayer and Fujifilm X-Trans frames.
//!
//! Mirrors:
//! - `z6iii::render::recover_and_antialias_frame` (`highlight.cl` + `reconstruct.cl` anti_alias)
//! - `z6iii::render::apply_linear_color_noise_2012` (1-stage & ISO>=400 2-stage LUV range filter)
//!
//! All stages execute device-resident in reusable VRAM ping-pong buffers with
//! a single H2D upload, zero intermediate readbacks, and a single D2H download.
//! Set `AURORA_LINEAR_POST_CUDA=0` to disable CUDA and fall back to CPU/OpenCL.

/// Returns `false` when `AURORA_LINEAR_POST_CUDA` is explicitly disabled
/// (`0`, `false`, or `off`).
pub(crate) fn linear_post_cuda_enabled() -> bool {
    parse_enabled_env(std::env::var("AURORA_LINEAR_POST_CUDA").ok().as_deref())
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

#[cfg(windows)]
mod imp {
    use super::linear_post_cuda_enabled;
    use aurora_portrait::cuda::{CudaError, Nvidia, BufferSession};
    use std::ffi::c_void;
    use std::slice;

    const PTX: &str = include_str!("../../native/cuda/raw_linear_post.ptx");

    const KERNEL_PACK: &str = "pack_rgb3_to_f4_kernel";
    const KERNEL_UNPACK: &str = "unpack_f4_to_rgb3_kernel";
    const KERNEL_HIGHLIGHT: &str = "highlight_recovery_kernel";
    const KERNEL_AA: &str = "anti_alias_clamped_kernel";
    const KERNEL_CN_ENCODE: &str = "color_noise_2012_encode_kernel";
    const KERNEL_CN_PASS_H: &str = "color_noise_2012_pass_h_kernel";
    const KERNEL_CN_PASS_V: &str = "color_noise_2012_pass_v_kernel";
    const KERNEL_CN_DECODE: &str = "color_noise_2012_decode_kernel";

    const BLOCK_1D: (u32, u32) = (256, 1);
    const BLOCK_2D: (u32, u32) = (16, 16);

    const PZ_BASE: f32 = 0.401_832_58;
    const PW_BASE: f32 = 2.968_700_4;
    const CN_STEP: i32 = 4;
    const CN_RADIUS: i32 = 28;
    const CN_HI_STEP: i32 = 6;
    const CN_HI_RADIUS: i32 = 42;

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct HighlightParams {
        p: [f32; 16],
    }

    const _: () = assert!(std::mem::size_of::<HighlightParams>() == 64);

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct ColorNoisePassParams {
        width: i32,
        height: i32,
        step: i32,
        radius: i32,
        pz_luma: f32,
        pw_luma: f32,
        pz_chroma: f32,
        pw_chroma: f32,
        blend: f32,
        use_blend: i32,
    }

    const _: () = assert!(std::mem::size_of::<ColorNoisePassParams>() == 40);

    fn grid_1d(total: u32) -> (u32, u32) {
        (total.div_ceil(BLOCK_1D.0).max(1), 1)
    }

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

    fn highlight_parameters(black: u32, mut gains: [f32; 3]) -> HighlightParams {
        let range = (65535 - black) as f32;
        let threshold = range * (-0.05_f32).exp2();
        gains[1] /= 1.5;
        let mut settings = [0.0f32; 16];
        settings[..3].copy_from_slice(&gains.map(|gain| gain * threshold));
        settings[3] = (2.0 * (-0.3_f32).exp2() / threshold) * threshold;
        settings[4..7].copy_from_slice(&gains.map(|gain| 2.0 * (range - threshold) * gain));
        settings[7..10].copy_from_slice(&gains.map(|gain| 1.0 / gain));
        settings[10] = (0.125 / threshold) * (1.0 / 3.0);
        HighlightParams { p: settings }
    }

    fn launch_pack(
        session: &mut BufferSession<'_>,
        src_rgb3: u64,
        dst_f4: u64,
        total_pixels: i32,
    ) -> Result<(), CudaError> {
        let mut src = src_rgb3;
        let mut dst = dst_f4;
        let mut n = total_pixels;
        let mut args: [*mut c_void; 3] = [
            &mut src as *mut u64 as *mut c_void,
            &mut dst as *mut u64 as *mut c_void,
            &mut n as *mut i32 as *mut c_void,
        ];
        session.launch(KERNEL_PACK, grid_1d(total_pixels as u32), BLOCK_1D, &mut args)?;
        Ok(())
    }

    fn launch_unpack(
        session: &mut BufferSession<'_>,
        src_f4: u64,
        dst_rgb3: u64,
        total_pixels: i32,
    ) -> Result<(), CudaError> {
        let mut src = src_f4;
        let mut dst = dst_rgb3;
        let mut n = total_pixels;
        let mut args: [*mut c_void; 3] = [
            &mut src as *mut u64 as *mut c_void,
            &mut dst as *mut u64 as *mut c_void,
            &mut n as *mut i32 as *mut c_void,
        ];
        session.launch(KERNEL_UNPACK, grid_1d(total_pixels as u32), BLOCK_1D, &mut args)?;
        Ok(())
    }

    /// Runs highlight recovery (`buf_a -> buf_b`) followed by clamped 5x5
    /// anti-aliasing (`buf_b -> buf_a`) entirely in VRAM.
    fn launch_recover_and_aa_in_place(
        session: &mut BufferSession<'_>,
        buf_a: u64,
        buf_b: u64,
        width: i32,
        height: i32,
        black: u32,
        wb: [f32; 3],
        iso: u32,
    ) -> Result<(), CudaError> {
        let total_pixels = width * height;
        let mut hl_params = highlight_parameters(black, wb);
        {
            let mut src = buf_a;
            let mut dst = buf_b;
            let mut n = total_pixels;
            let mut args: [*mut c_void; 4] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
                &mut hl_params as *mut HighlightParams as *mut c_void,
            ];
            session.launch(
                KERNEL_HIGHLIGHT,
                grid_1d(total_pixels as u32),
                BLOCK_1D,
                &mut args,
            )?;
        }

        let aca_gain = if iso >= 3200 {
            1.0f32
        } else if iso >= 1600 {
            0.75f32
        } else {
            0.5f32
        };
        let mut param_x = 0.05_f32.sqrt();
        let mut param_y = aca_gain;
        {
            let mut src = buf_b;
            let mut dst = buf_a;
            let mut w = width;
            let mut h = height;
            let mut args: [*mut c_void; 6] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut w as *mut i32 as *mut c_void,
                &mut h as *mut i32 as *mut c_void,
                &mut param_x as *mut f32 as *mut c_void,
                &mut param_y as *mut f32 as *mut c_void,
            ];
            session.launch(
                KERNEL_AA,
                grid_2d(width as u32, height as u32),
                BLOCK_2D,
                &mut args,
            )?;
        }
        Ok(())
    }

    /// Runs one full `ColorNoise2012` pass on `buf_a` (reading `buf_a`, ping-ponging
    /// through `buf_b`, writing decoded linear RGB back to `buf_a`) with zero readbacks.
    fn launch_color_noise_2012_in_place(
        session: &mut BufferSession<'_>,
        buf_a: u64,
        buf_b: u64,
        width: i32,
        height: i32,
        iso: u32,
        noise: [f32; 2],
    ) -> Result<(), CudaError> {
        let total_pixels = width * height;
        let g1 = grid_1d(total_pixels as u32);
        let g2 = grid_2d(width as u32, height as u32);

        let iso_var = (iso as f32 / 100.0).max(1.0);
        let pz_luma = (noise[1] * 12.0).max(PZ_BASE * iso_var);
        let pw_luma = (noise[0] * 4.5).max(PW_BASE * iso_var);
        let pz_chroma = (noise[1] * 60.0 + 1800.0 * iso_var).max(4000.0 * iso_var);
        let pw_chroma = (noise[0] * 16.0).max(18.0 * iso_var);

        // 1. Encode: buf_a (RGB) -> buf_b (LUV)
        {
            let mut src = buf_a;
            let mut dst = buf_b;
            let mut n = total_pixels;
            let mut args: [*mut c_void; 3] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];
            session.launch(KERNEL_CN_ENCODE, g1, BLOCK_1D, &mut args)?;
        }

        // 2. Stage 1 H (buf_b -> buf_a) & V (buf_a -> buf_b)
        let mut p1 = ColorNoisePassParams {
            width,
            height,
            step: CN_STEP,
            radius: CN_RADIUS,
            pz_luma,
            pw_luma,
            pz_chroma,
            pw_chroma,
            blend: 1.0,
            use_blend: 0,
        };
        {
            let mut src = buf_b;
            let mut dst = buf_a;
            let mut args: [*mut c_void; 3] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut p1 as *mut ColorNoisePassParams as *mut c_void,
            ];
            session.launch(KERNEL_CN_PASS_H, g2, BLOCK_2D, &mut args)?;
        }
        {
            let mut src = buf_a;
            let mut dst = buf_b;
            let mut args: [*mut c_void; 3] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut p1 as *mut ColorNoisePassParams as *mut c_void,
            ];
            session.launch(KERNEL_CN_PASS_V, g2, BLOCK_2D, &mut args)?;
        }

        // 3. Wide-scale stage 2 for ISO >= 400: H (buf_b -> buf_a) & V (buf_a -> buf_b)
        if iso >= 400 {
            let hi_blend =
                (((iso as f32).log2() - 400.0_f32.log2()) / 3.0).clamp(0.35, 1.0);
            let mut p2 = ColorNoisePassParams {
                width,
                height,
                step: CN_HI_STEP,
                radius: CN_HI_RADIUS,
                pz_luma,
                pw_luma,
                pz_chroma: pz_chroma * 2.5,
                pw_chroma: pw_chroma * 1.5,
                blend: hi_blend,
                use_blend: 1,
            };
            {
                let mut src = buf_b;
                let mut dst = buf_a;
                let mut args: [*mut c_void; 3] = [
                    &mut src as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut p2 as *mut ColorNoisePassParams as *mut c_void,
                ];
                session.launch(KERNEL_CN_PASS_H, g2, BLOCK_2D, &mut args)?;
            }
            {
                let mut src = buf_a;
                let mut dst = buf_b;
                let mut args: [*mut c_void; 3] = [
                    &mut src as *mut u64 as *mut c_void,
                    &mut dst as *mut u64 as *mut c_void,
                    &mut p2 as *mut ColorNoisePassParams as *mut c_void,
                ];
                session.launch(KERNEL_CN_PASS_V, g2, BLOCK_2D, &mut args)?;
            }
        }

        // 4. Decode: buf_b (LUV) -> buf_a (RGB)
        {
            let mut src = buf_b;
            let mut dst = buf_a;
            let mut n = total_pixels;
            let mut args: [*mut c_void; 3] = [
                &mut src as *mut u64 as *mut c_void,
                &mut dst as *mut u64 as *mut c_void,
                &mut n as *mut i32 as *mut c_void,
            ];
            session.launch(KERNEL_CN_DECODE, g1, BLOCK_1D, &mut args)?;
        }
        Ok(())
    }

    fn validate_dims(rgb: &[[f32; 3]], width: usize, height: usize) -> Result<(i32, i32, i32), String> {
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| "frame dimension overflow".to_string())?;
        if rgb.len() != pixels {
            return Err(format!(
                "frame buffer length {} does not match {}x{}",
                rgb.len(),
                width,
                height
            ));
        }
        let w = i32::try_from(width).map_err(|_| "width exceeds i32::MAX".to_string())?;
        let h = i32::try_from(height).map_err(|_| "height exceeds i32::MAX".to_string())?;
        let n = i32::try_from(pixels).map_err(|_| "pixel count exceeds i32::MAX".to_string())?;
        Ok((w, h, n))
    }

    /// Runs highlight recovery + clamped 5x5 anti-aliasing on CUDA in a single
    /// H2D/D2H roundtrip (no 512x512 tiling and no CPU halo copy).
    pub(crate) fn recover_and_antialias_cuda(
        rgb: &mut [[f32; 3]],
        width: usize,
        height: usize,
        black: u32,
        wb: [f32; 3],
        iso: u32,
    ) -> Result<(), String> {
        if !linear_post_cuda_enabled() {
            return Err("disabled via AURORA_LINEAR_POST_CUDA=0".to_string());
        }
        let (w, h, n) = validate_dims(rgb, width, height)?;
        if width < 4 || height < 4 {
            return Ok(());
        }
        if black >= 65_535 || !wb.into_iter().all(|g| g.is_finite() && g > 0.0) {
            return Ok(());
        }

        let gpu = Nvidia::open_ptx(PTX).map_err(|e| format!("CUDA open_ptx: {e:?}"))?;
        let rgb3_bytes = rgb.len() * 12;
        let f4_bytes = rgb.len() * 16;
        let sizes = [rgb3_bytes, f4_bytes, f4_bytes];

        gpu.with_reusable_buffers("raw-linear-post", &sizes, |session| {
            session.upload(0, unsafe { bytes_of(rgb) })?;
            let buf_rgb3 = session.handle(0);
            let buf_a = session.handle(1);
            let buf_b = session.handle(2);

            launch_pack(session, buf_rgb3, buf_a, n)?;
            launch_recover_and_aa_in_place(session, buf_a, buf_b, w, h, black, wb, iso)?;
            launch_unpack(session, buf_a, buf_rgb3, n)?;

            session.download(0, unsafe { bytes_of_mut(rgb) })?;
            session.mark_consumed();
            Ok(())
        })
        .map_err(|e| format!("CUDA recover_and_antialias: {e:?}"))
    }

    /// Runs `ColorNoise2012` (stage 1 + optional ISO>=400 wide-scale stage 2)
    /// on CUDA in a single H2D/D2H roundtrip with zero intermediate readbacks.
    pub(crate) fn color_noise_2012_cuda(
        rgb: &mut [[f32; 3]],
        width: usize,
        height: usize,
        iso: u32,
        noise: [f32; 2],
    ) -> Result<(), String> {
        if !linear_post_cuda_enabled() {
            return Err("disabled via AURORA_LINEAR_POST_CUDA=0".to_string());
        }
        if width == 0 || height == 0 {
            return Ok(());
        }
        let (w, h, n) = validate_dims(rgb, width, height)?;

        let gpu = Nvidia::open_ptx(PTX).map_err(|e| format!("CUDA open_ptx: {e:?}"))?;
        let rgb3_bytes = rgb.len() * 12;
        let f4_bytes = rgb.len() * 16;
        let sizes = [rgb3_bytes, f4_bytes, f4_bytes];

        gpu.with_reusable_buffers("raw-linear-post", &sizes, |session| {
            session.upload(0, unsafe { bytes_of(rgb) })?;
            let buf_rgb3 = session.handle(0);
            let buf_a = session.handle(1);
            let buf_b = session.handle(2);

            launch_pack(session, buf_rgb3, buf_a, n)?;
            launch_color_noise_2012_in_place(session, buf_a, buf_b, w, h, iso, noise)?;
            launch_unpack(session, buf_a, buf_rgb3, n)?;

            session.download(0, unsafe { bytes_of_mut(rgb) })?;
            session.mark_consumed();
            Ok(())
        })
        .map_err(|e| format!("CUDA color_noise_2012: {e:?}"))
    }

    /// One-stop device-resident linear post-processing for generic Bayer and
    /// X-Trans frames in a single H2D + single D2H:
    /// 1. Optional highlight recovery + clamped 5x5 anti-aliasing when
    ///    `recovery = Some((black, wb))` (generic Bayer path; pass `None` for X-Trans).
    /// 2. `ColorNoise2012(iso, noise)`.
    /// 3. When `iso >= 1600`, a second `ColorNoise2012(iso, [noise[0]*1.6, noise[1]*2.0])`
    ///    matching `aurora_raw::finish_linear_stages`.
    pub(crate) fn linear_finish_cuda(
        rgb: &mut [[f32; 3]],
        width: usize,
        height: usize,
        recovery: Option<(u32, [f32; 3])>,
        iso: u32,
        noise: [f32; 2],
    ) -> Result<(), String> {
        if !linear_post_cuda_enabled() {
            return Err("disabled via AURORA_LINEAR_POST_CUDA=0".to_string());
        }
        if width == 0 || height == 0 {
            return Ok(());
        }
        let (w, h, n) = validate_dims(rgb, width, height)?;

        let run_recovery = recovery.filter(|&(black, wb)| {
            width >= 4
                && height >= 4
                && black < 65_535
                && wb.into_iter().all(|g| g.is_finite() && g > 0.0)
        });

        let gpu = Nvidia::open_ptx(PTX).map_err(|e| format!("CUDA open_ptx: {e:?}"))?;
        let rgb3_bytes = rgb.len() * 12;
        let f4_bytes = rgb.len() * 16;
        let sizes = [rgb3_bytes, f4_bytes, f4_bytes];

        gpu.with_reusable_buffers("raw-linear-post", &sizes, |session| {
            session.upload(0, unsafe { bytes_of(rgb) })?;
            let buf_rgb3 = session.handle(0);
            let buf_a = session.handle(1);
            let buf_b = session.handle(2);

            launch_pack(session, buf_rgb3, buf_a, n)?;
            if let Some((black, wb)) = run_recovery {
                launch_recover_and_aa_in_place(session, buf_a, buf_b, w, h, black, wb, iso)?;
            }
            launch_color_noise_2012_in_place(session, buf_a, buf_b, w, h, iso, noise)?;
            if iso >= 1600 {
                launch_color_noise_2012_in_place(
                    session,
                    buf_a,
                    buf_b,
                    w,
                    h,
                    iso,
                    [noise[0] * 1.6, noise[1] * 2.0],
                )?;
            }
            launch_unpack(session, buf_a, buf_rgb3, n)?;

            session.download(0, unsafe { bytes_of_mut(rgb) })?;
            session.mark_consumed();
            Ok(())
        })
        .map_err(|e| format!("CUDA linear_finish: {e:?}"))
    }
}

#[cfg(windows)]
pub(crate) use imp::{color_noise_2012_cuda, linear_finish_cuda, recover_and_antialias_cuda};

#[cfg(not(windows))]
pub(crate) fn recover_and_antialias_cuda(
    _rgb: &mut [[f32; 3]],
    _width: usize,
    _height: usize,
    _black: u32,
    _wb: [f32; 3],
    _iso: u32,
) -> Result<(), String> {
    Err("linear_post_cuda is only available on Windows".to_string())
}

#[cfg(not(windows))]
pub(crate) fn color_noise_2012_cuda(
    _rgb: &mut [[f32; 3]],
    _width: usize,
    _height: usize,
    _iso: u32,
    _noise: [f32; 2],
) -> Result<(), String> {
    Err("linear_post_cuda is only available on Windows".to_string())
}

#[cfg(not(windows))]
pub(crate) fn linear_finish_cuda(
    _rgb: &mut [[f32; 3]],
    _width: usize,
    _height: usize,
    _recovery: Option<(u32, [f32; 3])>,
    _iso: u32,
    _noise: [f32; 2],
) -> Result<(), String> {
    Err("linear_post_cuda is only available on Windows".to_string())
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

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    fn synthetic_linear_frame(w: usize, h: usize) -> Vec<[f32; 3]> {
        let mut rgb = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                // Mix of highlights (>60000), midtones, shadows, sharp edges, and 8px chroma blotches.
                let edge = if x >= w / 2 { 14000.0 } else { 4000.0 };
                let highlight = if x < 24 && y < 24 { 58000.0 } else { 0.0 };
                let blotch = if ((x / 8) + (y / 8)) % 2 == 0 { 1.0 } else { -1.0 };
                let grain = if (x + y) % 2 == 0 { 320.0 } else { -320.0 };
                let base = edge + highlight + grain;
                let r = (base + blotch * 1100.0 + (x % 7) as f32 * 90.0).max(0.0);
                let g = (base - blotch * 500.0 + (y % 5) as f32 * 60.0).max(0.0);
                let b = (base + blotch * 700.0 + ((x + y) % 9) as f32 * 80.0).max(0.0);
                rgb[y * w + x] = [r, g, b];
            }
        }
        rgb
    }

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
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

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    #[test]
    fn color_noise_2012_cuda_matches_cpu_low_and_high_iso() {
        let (w, h) = (97, 83);
        let src = synthetic_linear_frame(w, h);

        for (iso, noise) in [(100u32, [4.0f32, 16.0]), (1000u32, [18.0f32, 200.0])] {
            let mut cpu = src.clone();
            let mut gpu = src.clone();
            super::super::z6iii::apply_linear_color_noise_2012(&mut cpu, w, h, iso, noise);
            match color_noise_2012_cuda(&mut gpu, w, h, iso, noise) {
                Ok(()) => {
                    let diff = max_abs_diff(&cpu, &gpu);
                    assert!(
                        diff < 1e-3,
                        "ColorNoise2012 CUDA vs CPU diff too large at ISO {iso}: max_abs_diff={diff}"
                    );
                }
                Err(err) => {
                    eprintln!("skipping CUDA ColorNoise2012 parity check (no CUDA device): {err}");
                    return;
                }
            }
        }
    }

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    #[test]
    fn linear_finish_cuda_matches_cpu_full_chain_at_high_iso() {
        let (w, h) = (80, 64);
        let src = synthetic_linear_frame(w, h);
        let black = 1008u32;
        let wb = [1.85f32, 1.0, 1.55];
        let iso = 1600u32;
        let noise = [24.0f32, 260.0];

        // X-Trans mode (recovery = None): pure ColorNoise2012 x2 parity against CPU.
        {
            let mut cpu = src.clone();
            let mut gpu = src.clone();
            super::super::z6iii::apply_linear_color_noise_2012(&mut cpu, w, h, iso, noise);
            super::super::z6iii::apply_linear_color_noise_2012(
                &mut cpu,
                w,
                h,
                iso,
                [noise[0] * 1.6, noise[1] * 2.0],
            );
            match linear_finish_cuda(&mut gpu, w, h, None, iso, noise) {
                Ok(()) => {
                    let diff = max_abs_diff(&cpu, &gpu);
                    assert!(
                        diff < 1e-3,
                        "linear_finish_cuda (X-Trans ISO 1600) max_abs_diff={diff}"
                    );
                }
                Err(err) => {
                    eprintln!("skipping CUDA linear_finish parity check (no CUDA device): {err}");
                    return;
                }
            }
        }

        // Generic Bayer mode (recovery = Some((black, wb))): compare against OpenCL+CPU if OpenCL is present.
        if super::super::z6iii::opencl_available() {
            let mut cpu = src.clone();
            let mut gpu = src.clone();
            super::super::z6iii::recover_and_antialias_frame(&mut cpu, w, h, black, wb, iso);
            super::super::z6iii::apply_linear_color_noise_2012(&mut cpu, w, h, iso, noise);
            super::super::z6iii::apply_linear_color_noise_2012(
                &mut cpu,
                w,
                h,
                iso,
                [noise[0] * 1.6, noise[1] * 2.0],
            );
            linear_finish_cuda(&mut gpu, w, h, Some((black, wb)), iso, noise)
                .expect("CUDA linear_finish with recovery");
            let diff = max_abs_diff(&cpu, &gpu);
            eprintln!("linear_finish_cuda (Bayer ISO 1600, 0..65535 domain) max_abs_diff={diff}");
            // In the 0..65535 linear domain, OpenCL's default FMA/approx-div
            // vs CUDA's --fmad=false --prec-div=true differs by ~0.03 (~4e-7 rel).
            assert!(
                diff < 0.1,
                "linear_finish_cuda (Bayer ISO 1600) max_abs_diff={diff}"
            );
        }
    }
}
