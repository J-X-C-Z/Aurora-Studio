use aurora_portrait::wgpu_backend::{
    adapter_info, available, execute_clarity, execute_region_box_smooth,
    execute_rgba8_stage_fusion, execute_skin_lut_fixed, region_box_smooth_cpu,
    rgba8_stage_fusion_cpu, vulkan_available, Rgba8FusionParams,
};
use aurora_portrait::Image;

fn synthetic_image(w: u32, h: u32) -> (Image, Vec<u8>) {
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    let mut mask = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 7 + y * 13 + 11) & 0xff) as u8;
            let g = (((x * 31) ^ (y * 17) ^ 0x5a) & 0xff) as u8;
            let b = (((x | y) * 5 + 3) & 0xff) as u8;
            let a = match (x + y) % 5 {
                0 => 0,
                1 => 128,
                2 => 1,
                _ => 255,
            };
            pixels.extend_from_slice(&[r, g, b, a]);
            mask.push((((x * 19 + y * 29) & 0xff) as u8).max(16));
        }
    }
    (Image::new(w, h, pixels).unwrap(), mask)
}

fn synthetic_lut512() -> Vec<u8> {
    let mut lut = vec![0u8; 512 * 512 * 3];
    for y in 0..512usize {
        for x in 0..512usize {
            let qx = x / 64;
            let qy = y / 64;
            let r = ((x % 64) * 255 / 63) as u8;
            let g = ((y % 64) * 255 / 63) as u8;
            let b = ((qy * 8 + qx) * 255 / 63) as u8;
            let i = (y * 512 + x) * 3;
            lut[i] = r.saturating_add(12);
            lut[i + 1] = g.saturating_add(6);
            lut[i + 2] = b.saturating_sub(4);
        }
    }
    lut
}

fn max_channel_diff(a: &Image, b: &Image) -> u8 {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    a.pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(&x, &y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

#[test]
fn probe_reports_real_adapter_without_panicking() {
    let avail = available();
    let vk_avail = vulkan_available();
    let info = adapter_info();
    eprintln!("wgpu available={avail} vulkan_available={vk_avail} info={info:?}");
    if vk_avail {
        assert!(avail);
    }
    assert_eq!(avail, info.is_some());
}

#[test]
fn skin_lut_fixed_matches_cpu_within_1_ulp() {
    if !available() {
        eprintln!("skipping skin_lut_fixed GPU parity test: no WGPU adapter");
        return;
    }
    let lut = synthetic_lut512();
    for &(w, h) in &[(1u32, 1u32), (37, 23), (64, 64)] {
        let (src, mask) = synthetic_image(w, h);
        for &mix in &[0.0f32, 0.35, 1.0] {
            let cpu =
                aurora_portrait::kernels::makeup::skin_adjust_lut(&src, &mask, &lut, mix).unwrap();
            let (gpu, receipt) = execute_skin_lut_fixed(&src, &mask, &lut, mix, false).unwrap();
            assert_eq!(receipt.kernel, "skin_lut_fixed");
            assert_eq!(receipt.dispatch_count, 1);
            let diff = max_channel_diff(&cpu, &gpu);
            assert!(
                diff <= 1,
                "{w}x{h} mix={mix}: max_diff={diff} exceeds 1 ULP"
            );
            for (c, g) in cpu.pixels.chunks_exact(4).zip(gpu.pixels.chunks_exact(4)) {
                assert_eq!(c[3], g[3], "alpha must be preserved");
            }
        }
    }
}

#[test]
fn clarity_matches_cpu_within_1_ulp() {
    if !available() {
        eprintln!("skipping clarity GPU parity test: no WGPU adapter");
        return;
    }
    let (src, full_mask) = synthetic_image(37, 23);
    let roi = (5, 3, 19, 13);
    let roi_pixels = (roi.2 * roi.3) as usize;
    let base = Image::new(roi.2, roi.3, src.pixels[..roi_pixels * 4].to_vec()).unwrap();
    let mut blur = base.clone();
    for px in blur.pixels.chunks_exact_mut(4) {
        px[0] = px[0].saturating_sub(8);
        px[1] = px[1].saturating_add(6);
        px[2] = px[2].saturating_sub(4);
    }
    let roi_mask = &full_mask[..roi_pixels];

    let cpu = aurora_portrait::kernels::skin_models::clarity_apply(
        &src,
        &base,
        &blur,
        roi_mask,
        roi,
        0.75,
        0.02,
        4.68,
    )
    .unwrap();
    let (gpu, receipt) =
        execute_clarity(&src, &base, &blur, roi_mask, roi, 0.75, 0.02, 4.68, false).unwrap();
    assert_eq!(receipt.kernel, "clarity_apply");
    let diff = max_channel_diff(&cpu, &gpu);
    assert!(diff <= 1, "clarity max_diff={diff} exceeds 1 ULP");
}

#[test]
fn region_box_smooth_matches_cpu_within_1_ulp() {
    if !available() {
        eprintln!("skipping region_box_smooth GPU parity test: no WGPU adapter");
        return;
    }
    let (src, mask) = synthetic_image(37, 23);
    let roi = (4, 2, 21, 15);
    let cpu = region_box_smooth_cpu(&src, &mask, roi, 2, 0.8).unwrap();
    let (gpu, receipt) = execute_region_box_smooth(&src, &mask, roi, 2, 0.8, false).unwrap();
    assert_eq!(receipt.kernel, "region_box_smooth");
    let diff = max_channel_diff(&cpu, &gpu);
    assert!(diff <= 1, "region_box_smooth max_diff={diff} exceeds 1 ULP");
}

#[test]
fn rgba8_stage_fusion_is_bit_exact_against_cpu() {
    if !available() {
        eprintln!("skipping rgba8_stage_fusion GPU parity test: no WGPU adapter");
        return;
    }
    let params = Rgba8FusionParams {
        gain: [281, 256, 230],
        off: [4, 0, -3],
        contrast: 320,
        strength_percent: 100,
        alpha_policy: 1,
    };
    for &(w, h) in &[(1u32, 1u32), (37, 23), (63, 65)] {
        let (src, _) = synthetic_image(w, h);
        let cpu = rgba8_stage_fusion_cpu(&src, params).unwrap();
        let (gpu, receipt) = execute_rgba8_stage_fusion(&src, params, false).unwrap();
        assert_eq!(receipt.kernel, "op_ab");
        assert_eq!(cpu.pixels, gpu.pixels, "{w}x{h} Q8 fusion must be bit-exact");
    }
}
