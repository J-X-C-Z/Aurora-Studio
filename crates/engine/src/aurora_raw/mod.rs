//! Aurora Raw: Nikon HE RAW decoding and reconstruction pipeline, ported from the parse lanes.
//!
//! Decoding is byte-identical to the bundled LibRaw on the validated Z6III
//! samples; the reason to run it in the product is the demosaic that follows,
//! which LibRaw replaces with its own. Everything here is opt-in through
//! `AURORA_C1_RAW=1` in `crate::raw` until the demosaic output has a visual
//! acceptance of its own.

pub mod c1_raw_he;
pub mod xtrans;
#[cfg(all(windows, feature = "z6iii-raw-export"))]
pub mod z6iii;

pub use c1_raw_he::decode_frame_cropped;

use std::path::Path;

/// Full-frame RGB16 through the C1 chain:
/// 1. Z6III HE NEF -> `z6iii::render_linear_rgba16` (OpenCL full reconstruct + linear CA +
///    highlight recovery + anti-alias);
/// 2. Fujifilm X-Trans (`filters == 9`) -> `xtrans::demosaic_xtrans` + linear CA + linear denoise;
/// 3. Any standard 2x2 Bayer RAW (Sony ARW, Canon CR3, classic Nikon NEF, DNG, etc.) ->
///    C1 Stage 1/2/3 over 128-pixel tiles with camera-specific ISO noise + linear CA + linear denoise.
pub fn decode_he_rgba16(path: &Path) -> Option<(u32, u32, Vec<u16>)> {
    let bytes = std::fs::read(path).ok()?;
    let orientation = crate::raw::exif_orientation_bytes(&bytes).unwrap_or(1);
    let libraw_frame = crate::raw::decode_bayer16(path).ok();
    let libraw_bayer = libraw_frame
        .as_ref()
        .map(|lib| (lib.raw_width, lib.raw_height, lib.data.clone()));

    let exif = crate::exif::read_exif(path);
    let ca_warp = crate::mock::lens_profile::LensShot::from_exif(&exif)
        .and_then(|shot| crate::mock::lens_profile::auto_warp(&shot))
        .filter(|warp| !warp.red.is_empty() || !warp.blue.is_empty());

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    if let Some((width, height, rgba)) =
        z6iii::render_linear_rgba16(&bytes, libraw_bayer, ca_warp.as_ref())
    {
        return Some(orient_rgba16(rgba, width, height, orientation));
    }

    let iso = exif.iso.unwrap_or(100);
    let noise = lookup_camera_noise(exif.camera_model.as_deref(), iso);

    // Fujifilm X-Trans 6x6 sensor path (`filters == 9`).
    if let Some(bayer) = libraw_frame.as_ref() {
        if bayer.filters == 9 && bayer.colors == 3 {
            let (width, height, mut rgb) = decode_xtrans_linear(bayer)?;
            #[cfg(all(windows, feature = "z6iii-raw-export"))]
            finish_linear_stages(&mut rgb, width, height, iso, noise, ca_warp.as_ref());
            let out = pack_rgba16(&rgb);
            return Some(orient_rgba16(out, width, height, orientation));
        }
    }

    // Try Nikon MakerNote metadata first when present (exact Nikon crop/WB/black),
    // otherwise construct the C1 Stage 1/2/3 input directly from LibRaw's generic
    // BayerFrame (covers Sony ARW, Canon CR3, classic Nikon NEF, DNG, etc.).
    let (frame, wb, black) = if let Some((frame, wb, black)) =
        nikon_c1_stage_input(&bytes, libraw_frame.as_ref())
    {
        (frame, wb, black)
    } else if let Some(bayer) = libraw_frame.as_ref() {
        generic_bayer_c1_stage_input(bayer)?
    } else {
        return None;
    };

    let (width, height, mut rgb) = run_c1_stage123_tiles_f32(&frame, wb, black, noise)?;
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    {
        if let Some(warp) = ca_warp.as_ref() {
            z6iii::apply_linear_chromatic_aberration(&mut rgb, width, height, 0, 0, width, height, warp);
        }
        z6iii::recover_and_antialias_frame(&mut rgb, width, height, black, wb, iso);
        finish_linear_stages(&mut rgb, width, height, iso, noise, None);
    }
    let out = pack_rgba16(&rgb);
    Some(orient_rgba16(out, width, height, orientation))
}

fn lookup_camera_noise(model: Option<&str>, iso: u32) -> [f32; 2] {
    if let Some(name) = model {
        if let Some(cam) = crate::camera_params::for_camera(name) {
            if let Some(row) = cam.noise_at(iso) {
                let slope = row.poisson.unwrap_or(12.8).max(0.5);
                let floor = row.floor.unwrap_or(0.0).max(0.0);
                return [slope, floor];
            }
        }
    }
    let iso_scale = (iso as f32 / 100.0).max(1.0);
    [2.5 * iso_scale, 8.0 * iso_scale]
}

fn decode_xtrans_linear(bayer: &crate::raw::BayerFrame) -> Option<(usize, usize, Vec<[f32; 3]>)> {
    let g = if bayer.cam_mul[1].is_finite() && bayer.cam_mul[1] > 0.0 {
        bayer.cam_mul[1]
    } else {
        1.0
    };
    let wb = [
        if bayer.cam_mul[0].is_finite() && bayer.cam_mul[0] > 0.0 { bayer.cam_mul[0] / g } else { 1.0 },
        1.0,
        if bayer.cam_mul[2].is_finite() && bayer.cam_mul[2] > 0.0 { bayer.cam_mul[2] / g } else { 1.0 },
    ];
    let (data, black) = if bayer.bits == 14 || bayer.bits == 0 {
        (bayer.data.clone(), bayer.black as f32)
    } else if bayer.bits < 14 {
        let s = 14 - u32::from(bayer.bits);
        (bayer.data.iter().map(|&v| ((u32::from(v) << s).min(16383)) as u16).collect(), (bayer.black << s) as f32)
    } else {
        let s = u32::from(bayer.bits) - 14;
        (bayer.data.iter().map(|&v| v >> s).collect(), (bayer.black >> s) as f32)
    };
    let rgb = xtrans::demosaic_xtrans(
        &data,
        bayer.raw_width,
        bayer.left_margin,
        bayer.top_margin,
        bayer.width,
        bayer.height,
        black,
        wb,
    )?;
    Some((bayer.width, bayer.height, rgb))
}

#[cfg(all(windows, feature = "z6iii-raw-export"))]
fn finish_linear_stages(
    rgb: &mut [[f32; 3]],
    width: usize,
    height: usize,
    iso: u32,
    noise: [f32; 2],
    ca_warp: Option<&crate::mock::lens_profile::LensWarp>,
) {
    if let Some(warp) = ca_warp {
        z6iii::apply_linear_chromatic_aberration(rgb, width, height, 0, 0, width, height, warp);
    }
    // Luminance NR is deferred to the full-precision detail stage so it is not
    // double-applied and stays 100% responsive to the UI slider.
    z6iii::apply_linear_color_noise_2012(rgb, width, height, iso, noise);
    if iso >= 1600 {
        z6iii::apply_linear_color_noise_2012(
            rgb,
            width,
            height,
            iso,
            [noise[0] * 1.6, noise[1] * 2.0],
        );
    }
}

fn pack_rgba16(rgb: &[[f32; 3]]) -> Vec<u16> {
    let mut out = vec![0u16; rgb.len() * 4];
    for (dst, px) in out.chunks_exact_mut(4).zip(rgb.iter()) {
        dst[0] = px[0].clamp(0.0, 65535.0) as u16;
        dst[1] = px[1].clamp(0.0, 65535.0) as u16;
        dst[2] = px[2].clamp(0.0, 65535.0) as u16;
        dst[3] = 65535;
    }
    out
}

fn nikon_c1_stage_input(
    bytes: &[u8],
    libraw_frame: Option<&crate::raw::BayerFrame>,
) -> Option<(c1_raw_he::NikonHeCropFrame, [f32; 3], u32)> {
    let fallback = || decode_frame_cropped(bytes).ok().flatten();
    let frame = match libraw_frame {
        Some(lib) => match c1_raw_he::nikon_crop(bytes) {
            Ok(Some(crop)) => c1_raw_he::NikonHeCropFrame::new(
                crop,
                lib.raw_width,
                lib.raw_height,
                lib.data.clone(),
            )
            .ok()
            .or_else(fallback)?,
            _ => fallback()?,
        },
        None => fallback()?,
    };
    let wb = c1_raw_he::nikon_as_shot_rgb(bytes).ok()??;
    let black = c1_raw_he::nikon_black_levels(bytes).ok()??.capture_one_scalar();
    Some((frame, wb, black))
}

/// Align any standard 2x2 Bayer frame (RGGB/GRBG/GBRG/BGGR) from LibRaw to an
/// RGGB-origin active crop in 14-bit domain for C1 Stage 1/2/3 reconstruction.
fn generic_bayer_c1_stage_input(
    bayer: &crate::raw::BayerFrame,
) -> Option<(c1_raw_he::NikonHeCropFrame, [f32; 3], u32)> {
    // Reject non-Bayer sensors (e.g. X-Trans uses filters == 9, monochrome == 0).
    if bayer.colors != 3 || bayer.filters == 0 || bayer.filters == 9 {
        return None;
    }
    let is_green = |c: u8| c == 1 || c == 3;
    let mut rggb_offset = None;
    for dy in 0..=1 {
        for dx in 0..=1 {
            let x = bayer.left_margin + dx;
            let y = bayer.top_margin + dy;
            if bayer.color_at(x, y) == 0
                && is_green(bayer.color_at(x + 1, y))
                && is_green(bayer.color_at(x, y + 1))
                && bayer.color_at(x + 1, y + 1) == 2
            {
                rggb_offset = Some((dx, dy));
                break;
            }
        }
        if rggb_offset.is_some() {
            break;
        }
    }
    let (dx, dy) = rggb_offset?;
    let src_left = bayer.left_margin + dx;
    let src_top = bayer.top_margin + dy;
    let width = bayer.width.saturating_sub(dx) & !1;
    let height = bayer.height.saturating_sub(dy) & !1;
    if width < 16 || height < 16 || src_left + width > bayer.raw_width || src_top + height > bayer.raw_height {
        return None;
    }
    // Extract the active RGGB-aligned plane with origin (0, 0) so `crop.left`
    // and `crop.top` are always even regardless of sensor CFA phase or margins.
    let (data, black) = {
        let mut plane = Vec::with_capacity(width * height);
        let (shift_left, shift_right, black) = if bayer.bits == 14 || bayer.bits == 0 {
            (0u32, 0u32, bayer.black)
        } else if bayer.bits < 14 {
            let s = 14 - u32::from(bayer.bits);
            (s, 0, bayer.black << s)
        } else {
            let s = u32::from(bayer.bits) - 14;
            (0, s, bayer.black >> s)
        };
        for y in 0..height {
            let row = (src_top + y) * bayer.raw_width + src_left;
            for &v in &bayer.data[row..row + width] {
                let sample = if shift_left > 0 {
                    ((u32::from(v) << shift_left).min(16383)) as u16
                } else if shift_right > 0 {
                    v >> shift_right
                } else {
                    v
                };
                plane.push(sample);
            }
        }
        (plane, black)
    };
    let g = if bayer.cam_mul[1].is_finite() && bayer.cam_mul[1] > 0.0 {
        bayer.cam_mul[1]
    } else {
        1.0
    };
    let wb = [
        if bayer.cam_mul[0].is_finite() && bayer.cam_mul[0] > 0.0 {
            bayer.cam_mul[0] / g
        } else {
            1.0
        },
        1.0,
        if bayer.cam_mul[2].is_finite() && bayer.cam_mul[2] > 0.0 {
            bayer.cam_mul[2] / g
        } else {
            1.0
        },
    ];
    let crop = c1_raw_he::NikonCrop { left: 0, top: 0, width, height };
    let frame = c1_raw_he::NikonHeCropFrame::new(crop, width, height, data).ok()?;
    Some((frame, wb, black))
}

fn run_c1_stage123_tiles_f32(
    frame: &c1_raw_he::NikonHeCropFrame,
    wb: [f32; 3],
    black: u32,
    noise: [f32; 2],
) -> Option<(usize, usize, Vec<[f32; 3]>)> {
    use rayon::prelude::*;

    let params = c1_raw_he::Stage1Params::prepare(wb, noise, black, false);
    let crop = frame.crop();
    let (width, height) = (crop.width, crop.height);
    if width == 0 || height == 0 {
        return None;
    }

    let mut coords = Vec::new();
    for top in (0..height).step_by(128) {
        for left in (0..width).step_by(128) {
            coords.push((left, top, (width - left).min(128), (height - top).min(128)));
        }
    }
    let tiles: Option<Vec<(usize, usize, usize, usize, Vec<[f32; 3]>)>> = coords
        .into_par_iter()
        .map(|(left, top, tile_width, tile_height)| {
            let area = c1_raw_he::NikonCrop {
                left: crop.left + left,
                top: crop.top + top,
                width: tile_width,
                height: tile_height,
            };
            let tile = c1_raw_he::reconstruct_planar_active_mirror_roi(
                frame,
                area,
                params,
                Some(black as i32),
                wb,
                128.0,
            )
            .ok()?;
            Some((left, top, tile_width, tile_height, tile))
        })
        .collect();
    let tiles = tiles?;

    let mut rgb = vec![[0.0f32; 3]; width * height];
    for (left, top, tile_width, tile_height, tile) in tiles {
        for y in 0..tile_height {
            let dst = (top + y) * width + left;
            let src = y * tile_width;
            rgb[dst..dst + tile_width].copy_from_slice(&tile[src..src + tile_width]);
        }
    }
    Some((width, height, rgb))
}

#[cfg(test)]
fn run_c1_stage123_tiles(
    frame: &c1_raw_he::NikonHeCropFrame,
    wb: [f32; 3],
    black: u32,
) -> Option<(usize, usize, Vec<u16>)> {
    let (w, h, rgb) = run_c1_stage123_tiles_f32(frame, wb, black, [12.8, 0.0])?;
    Some((w, h, pack_rgba16(&rgb)))
}

/// LibRaw stores sensor frames already turned by EXIF orientation. The C1 HE
/// decoder returns the stored landscape buffer, so apply the same turn here;
/// callers then keep `geometry.orientation` at zero for normal folder imports.
fn orient_rgba16(mut pixels: Vec<u16>, width: usize, height: usize, orientation: u16) -> (u32, u32, Vec<u16>) {
    match orientation {
        3 => {
            for index in 0..width * height / 2 {
                let other = width * height - 1 - index;
                for channel in 0..4 {
                    pixels.swap(index * 4 + channel, other * 4 + channel);
                }
            }
            (width as u32, height as u32, pixels)
        }
        6 | 8 => {
            let mut rotated = vec![0; pixels.len()];
            for (index, pixel) in pixels.chunks_exact(4).enumerate() {
                let (column, row) = (index % width, index / width);
                let dst = if orientation == 6 { column * height + height - 1 - row }
                    else { (width - 1 - column) * height + row };
                rotated[dst * 4..dst * 4 + 4].copy_from_slice(pixel);
            }
            (height as u32, width as u32, rotated)
        }
        _ => (width as u32, height as u32, pixels),
    }
}

#[cfg(test)]
mod tests {
    /// `AURORA_TEST_RAW=<nef> cargo test -p aurora-engine aurora_raw -- --ignored`
    #[test]
    #[ignore]
    fn decodes_a_real_nef_through_the_c1_chain() {
        let path = std::env::var("AURORA_TEST_RAW").expect("AURORA_TEST_RAW");
        let (width, height, pixels) = super::decode_he_rgba16(std::path::Path::new(&path))
            .expect("C1 chain rejected the sample");
        assert!(width.max(height) > 3000, "{width}x{height}");
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        let mut histogram = vec![0u64; 65536];
        for pixel in pixels.chunks_exact(4) {
            for &value in &pixel[..3] {
                histogram[value as usize] += 1;
            }
        }
        let total = u64::from(width) * u64::from(height) * 3;
        let at = |fraction: f64| {
            let target = (fraction * total as f64) as u64;
            let mut seen = 0;
            histogram
                .iter()
                .position(|&count| {
                    seen += count;
                    seen >= target
                })
                .unwrap() as u16
        };
        let clipped = histogram[65535];
        let max = histogram.iter().rposition(|&count| count > 0).unwrap();
        println!(
            "C1 RAW RGB16 {width}x{height}: min={} p50={} p90={} p99={} p99.9={} max={max} clipped={clipped}/{total} ({:.6}%)",
            at(0.0),
            at(0.50),
            at(0.90),
            at(0.99),
            at(0.999),
            clipped as f64 / total as f64 * 100.0
        );
        assert!(at(0.50) < 65535, "median channel clipped to white");
        assert!(clipped * 2 < total, "clipped channels {clipped}/{total}");
    }

    #[test]
    fn generic_bayer_aligns_all_cfa_phases_through_c1_stage123() {
        // LibRaw filter masks for RGGB, GRBG, GBRG, BGGR.
        let patterns = [0x9494_9494u32, 0x6161_6161, 0x4949_4949, 0x1616_1616];
        for filters in patterns {
            let (raw_w, raw_h) = (64usize, 64usize);
            let mut data = vec![0u16; raw_w * raw_h];
            for y in 0..raw_h {
                for x in 0..raw_w {
                    let c = ((filters >> ((((y << 1) & 14) + (x & 1)) << 1)) & 3) as u8;
                    data[y * raw_w + x] = match c {
                        0 => 4000, // R
                        2 => 3000, // B
                        _ => 3500, // G
                    };
                }
            }
            let bayer = crate::raw::BayerFrame {
                raw_width: raw_w,
                raw_height: raw_h,
                width: 48,
                height: 48,
                top_margin: 8,
                left_margin: 8,
                stride: raw_w * 2,
                bits: 14,
                data,
                filters,
                colors: 3,
                black: 1000,
                maximum: 16383,
                cam_mul: [2048.0, 1024.0, 1536.0, 1024.0],
                rgb_cam: [[0.0; 4]; 3],
            };
            let (frame, wb, black) = super::generic_bayer_c1_stage_input(&bayer)
                .expect("generic Bayer frame must align to RGGB");
            assert_eq!(black, 1000);
            assert!((wb[0] - 2.0).abs() < 1e-4 && (wb[1] - 1.0).abs() < 1e-4 && (wb[2] - 1.5).abs() < 1e-4);
            let (w, h, out) = super::run_c1_stage123_tiles(&frame, wb, black)
                .expect("C1 Stage 1/2/3 must succeed on generic Bayer");
            assert!(w >= 46 && h >= 46);
            assert_eq!(out.len(), w * h * 4);
            let center = ((h / 2) * w + (w / 2)) * 4;
            assert!(out[center] > 1000 && out[center + 1] > 1000 && out[center + 2] > 1000);
        }
    }
}
