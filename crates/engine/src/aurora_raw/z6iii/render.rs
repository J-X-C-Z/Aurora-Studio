use super::ExportAssets;
#[cfg(windows)]
use super::{color, reconstruct as raw};
use anyhow::{Result, ensure};
#[cfg(windows)]
use anyhow::{Context, anyhow};
#[cfg(windows)]
use color::output::OutputClut;

#[cfg(windows)]
const TILE: usize = 512;

/// Runs the Z6III OpenCL reconstruction + optional linear CA (stage 16) +
/// highlight recovery + anti-aliasing pipeline over 512x512 tiles, invoking
/// `on_tile(left, top, width, height, core, iso, noise)` for each tile.
#[cfg(windows)]
pub(super) fn render_linear_tiles(
    nef: &[u8],
    bayer: Option<(usize, usize, Vec<u16>)>,
    ca_warp: Option<&crate::mock::lens_profile::LensWarp>,
    mut on_tile: impl FnMut(usize, usize, usize, usize, &[[f32; 3]], u32, [f32; 2]) -> Result<()>,
) -> Result<(usize, usize)> {
    let iso = raw::nikon_exif_sensitivity(nef).map_err(|error| anyhow!("ISO: {error}"))?
        .context("missing NEF ISO")?;
    let noise = color::z6iii_noise::z6iii_noise_for_iso(iso).context("unsupported Z6III ISO")?;
    let offset = raw::Z6iiiStage3::offset_for_noise(noise, (50, 0.0));
    let wb = raw::nikon_as_shot_rgb(nef).map_err(|error| anyhow!("white balance: {error}"))?
        .map(color::prepare_z6iii_global_white_balance).context("missing as-shot white balance")?
        .map(|gain| ((f64::from(gain) * 1000.0).round_ties_even() / 1000.0) as f32);
    let black = raw::nikon_black_levels(nef).map_err(|error| anyhow!("black levels: {error}"))?
        .context("missing black levels")?.capture_one_scalar();
    ensure!(black < 65_535 && wb.into_iter().all(|gain| gain.is_finite() && gain > 0.0), "invalid NEF black level or white balance");
    let stage1 = raw::Stage1Params::prepare(wb, noise, black, false);
    let frame = match bayer {
        Some((full_w, full_h, samples)) => {
            let crop = raw::nikon_crop(nef).map_err(|error| anyhow!("crop: {error}"))?
                .context("missing Z6III crop")?;
            raw::NikonHeCropFrame::new(crop, full_w, full_h, samples)
                .map_err(|error| anyhow!("Bayer frame: {error}"))?
        }
        None => raw::decode_frame_cropped(nef).map_err(|error| anyhow!("HE decode: {error}"))?
            .context("unsupported Nikon HE RAW variant")?,
    };
    let frame = frame.with_control_bayer_halo(offset, black as f32)
        .map_err(|error| anyhow!("AdaptiveGG and Bayer halo: {error}"))?
        .with_stage1_opencl(stage1).map_err(|error| anyhow!("OpenCL Bayer stage 1: {error}"))?;
    let crop = frame.crop();
    ensure!((crop.width, crop.height) == (6048, 4032) && crop.left >= 2 && crop.top >= 2, "unsupported Z6III active crop");

    let recovery = highlight_parameters(black, wb);
    let iso = iso as u32;
    let aca_gain = if iso >= 3200 { 1.0 } else if iso >= 1600 { 0.75 } else { 0.5 };
    let aa_params = [0.05_f32.sqrt(), aca_gain];
    // Process full-width horizontal strips (4 strips for 4032 px height) instead
    // of 96 512x512 tiles, reducing GPU roundtrips by 24x while keeping VRAM
    // working set modest (~50 MB/strip) and pixels 100% bit-identical.
    const STRIP_H: usize = 1008;
    for top in (0..crop.height).step_by(STRIP_H) {
        let height = STRIP_H.min(crop.height - top);
        let width = crop.width;
        let area = raw::NikonCrop {
            left: crop.left - 2,
            top: crop.top + top - 2,
            width: width + 4,
            height: height + 4,
        };
        let core = if let Some(warp) = ca_warp {
            let mut halo = raw::reconstruct_planar_green_halo_roi(
                &frame, area, stage1, black as i32, wb, offset, noise[0],
            )
            .map_err(|error| anyhow!("reconstruction at strip {top}: {error}"))?;
            apply_linear_chromatic_aberration(
                &mut halo,
                area.width,
                area.height,
                -2,
                top as isize - 2,
                crop.width,
                crop.height,
                warp,
            );
            raw::highlight_and_aa(&halo, area.width, area.height, recovery, aa_params)
                .map_err(|error| anyhow!("OpenCL highlight+AA at strip {top}: {error}"))?
        } else {
            let stage2_f4 = raw::reconstruct_stage2_scaled_f4(
                &frame, area, stage1, black as i32, wb,
            )
            .map_err(|error| anyhow!("stage 2 at strip {top}: {error}"))?;
            raw::stage3_gg_highlight_aa(
                &stage2_f4,
                area.width + 12,
                area.height + 12,
                wb,
                offset,
                noise[0],
                recovery,
                aa_params,
            )
            .map_err(|error| anyhow!("OpenCL fused stage3+HL+AA at strip {top}: {error}"))?
        };
        ensure!(core.len() == width * height, "unexpected anti-alias strip size");
        on_tile(0, top, width, height, &core, iso, noise)?;
    }
    Ok((crop.width, crop.height))
}

#[cfg(windows)]
pub(in crate::aurora_raw) fn linear_rgba16(
    nef: &[u8],
    bayer: Option<(usize, usize, Vec<u16>)>,
    ca_warp: Option<&crate::mock::lens_profile::LensWarp>,
) -> Result<(usize, usize, Vec<u16>)> {
    let (w, h, out, _oriented) = linear_rgba16_oriented(nef, bayer, ca_warp, 1)?;
    Ok((w, h, out))
}

/// Reconstructs Z6III linear RGBA16, preferring the single-upload GPU-resident
/// native CUDA pipeline (which fuses EXIF orientation into the final RGBA16
/// write) and transparently falling back to OpenCL when CUDA is disabled,
/// unavailable, or OOM.
///
/// Returns `(width, height, rgba16, already_oriented)`.
#[cfg(windows)]
pub(in crate::aurora_raw) fn linear_rgba16_oriented(
    nef: &[u8],
    bayer: Option<(usize, usize, Vec<u16>)>,
    ca_warp: Option<&crate::mock::lens_profile::LensWarp>,
    orientation: u16,
) -> Result<(usize, usize, Vec<u16>, bool)> {
    use rayon::prelude::*;

    // Fast path: native CUDA GPU-resident full pipeline (single H2D, 0 mid-readbacks,
    // fused orient + RGBA16 pack, single D2H).
    if raw::cuda_enabled() {
        let cuda_attempt = (|| -> Result<Option<(usize, usize, Vec<u16>, bool)>> {
            let iso = raw::nikon_exif_sensitivity(nef)
                .map_err(|error| anyhow!("ISO: {error}"))?
                .context("missing NEF ISO")?;
            let noise = color::z6iii_noise::z6iii_noise_for_iso(iso)
                .context("unsupported Z6III ISO")?;
            let offset = raw::Z6iiiStage3::offset_for_noise(noise, (50, 0.0));
            let wb = raw::nikon_as_shot_rgb(nef)
                .map_err(|error| anyhow!("white balance: {error}"))?
                .map(color::prepare_z6iii_global_white_balance)
                .context("missing as-shot white balance")?
                .map(|gain| ((f64::from(gain) * 1000.0).round_ties_even() / 1000.0) as f32);
            let black = raw::nikon_black_levels(nef)
                .map_err(|error| anyhow!("black levels: {error}"))?
                .context("missing black levels")?
                .capture_one_scalar();
            ensure!(
                black < 65_535 && wb.into_iter().all(|gain| gain.is_finite() && gain > 0.0),
                "invalid NEF black level or white balance"
            );
            let stage1 = raw::Stage1Params::prepare(wb, noise, black, false);
            let frame = match bayer.as_ref() {
                Some((full_w, full_h, samples)) => {
                    let crop = raw::nikon_crop(nef)
                        .map_err(|error| anyhow!("crop: {error}"))?
                        .context("missing Z6III crop")?;
                    raw::NikonHeCropFrame::new(crop, *full_w, *full_h, samples.clone())
                        .map_err(|error| anyhow!("Bayer frame: {error}"))?
                }
                None => raw::decode_frame_cropped(nef)
                    .map_err(|error| anyhow!("HE decode: {error}"))?
                    .context("unsupported Nikon HE RAW variant")?,
            };
            let crop = frame.crop();
            ensure!(
                (crop.width, crop.height) == (6048, 4032) && crop.cfa_phase() == (0, 0),
                "unsupported Z6III active crop"
            );
            let recovery = highlight_parameters(black, wb);
            let iso_u32 = iso as u32;
            let aca_gain = if iso_u32 >= 3200 {
                1.0
            } else if iso_u32 >= 1600 {
                0.75
            } else {
                0.5
            };
            let aa_params = [0.05_f32.sqrt(), aca_gain];
            let cuda_in = raw::CudaPipelineInput {
                full_w: frame.full_width(),
                full_h: frame.full_height(),
                crop,
                samples: frame.full_samples(),
                stage1,
                black,
                wb,
                offset,
                noise0: noise[0],
                recovery,
                aa_params,
                ca_warp,
                orientation,
            };
            match raw::cuda_reconstruct_rgba16(&cuda_in) {
                Ok((w, h, out)) => Ok(Some((w, h, out, true))),
                Err(err) => {
                    log::warn!("Z6III CUDA pipeline unavailable/failed ({err:?}); falling back to OpenCL");
                    Ok(None)
                }
            }
        })();
        match cuda_attempt {
            Ok(Some(frame)) => return Ok(frame),
            Ok(None) => {}
            Err(err) => {
                // If OpenCL is also unavailable, surface the metadata/HE decode error.
                if !raw::gpu_available() {
                    return Err(err);
                }
                log::warn!("Z6III CUDA prep failed ({err:#}); falling back to OpenCL");
            }
        }
    }

    // Fallback: OpenCL strip pipeline (returns unoriented 6048x4032).
    // Both the neutral export and the editable source start at the same
    // reconstructed pixels. Color noise is an adjustment, so the source cache
    // must be independent of it.
    let mut out = vec![65535u16; 6048 * 4032 * 4];
    let (crop_w, crop_h) = render_linear_tiles(
        nef,
        bayer,
        ca_warp,
        |left, top, width, height, core, _iso, _noise| {
            if left == 0 && width == 6048 {
                let dst_slice = &mut out[top * 6048 * 4..(top + height) * 6048 * 4];
                dst_slice
                    .par_chunks_exact_mut(4)
                    .zip(core.par_iter())
                    .for_each(|(dst, pixel)| {
                        dst[0] = pixel[0].clamp(0.0, 65535.0) as u16;
                        dst[1] = pixel[1].clamp(0.0, 65535.0) as u16;
                        dst[2] = pixel[2].clamp(0.0, 65535.0) as u16;
                    });
            } else {
                for (y, row) in core.chunks_exact(width).enumerate() {
                    let offset = ((top + y) * 6048 + left) * 4;
                    for (dst, pixel) in out[offset..offset + width * 4].chunks_exact_mut(4).zip(row) {
                        for (channel, &value) in dst[..3].iter_mut().zip(pixel) {
                            *channel = value.clamp(0.0, 65535.0) as u16;
                        }
                    }
                }
            }
            Ok(())
        },
    )?;
    Ok((crop_w, crop_h, out, false))
}

#[cfg(windows)]
pub(super) fn neutral(nef: &[u8], assets: &ExportAssets<'_>) -> Result<Vec<u16>> {
    check_assets(assets)?;
    let film = color::film_nodes(assets.curve).context("invalid film curve")?;
    let output = OutputClut::from_camera49(assets.camera_icc, assets.adobe_rgb_icc)
        .context("unsupported Adobe RGB output CLUT")?;
    let mut pixels = vec![[0; 3]; 6048 * 4032];
    let mut detected_iso = 100u32;
    let (crop_w, _crop_h) = render_linear_tiles(nef, None, None, |left, top, width, _height, core, iso, _noise| {
        detected_iso = iso;
        let encoded = raw::export_color(core, &film, output.nodes())
            .with_context(|| format!("OpenCL Auto and Adobe RGB at ({left},{top})"))?;
        for (row, values) in encoded.chunks_exact(width).enumerate() {
            let dst = (top + row) * 6048 + left;
            pixels[dst..dst + width].copy_from_slice(values);
        }
        Ok(())
    })?;
    let mut pixels = pixels.into_flattened();
    apply_banding(&mut pixels, crop_w, detected_iso, assets.banding_tables);
    Ok(pixels)
}

pub(super) fn check_assets(assets: &ExportAssets<'_>) -> Result<()> {
    let fingerprint = |bytes: &[u8]| bytes.iter().fold(0xcbf29ce484222325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    for (name, bytes, expected) in [
        ("Nikon Z6III ProStandard", assets.camera_icc, 0x247108353276f0f5),
        ("Adobe RGB (1998)", assets.adobe_rgb_icc, 0x90f5bf257afb92cd),
    ] {
        ensure!(fingerprint(bytes) == expected, "unsupported {name} asset version");
    }
    ensure!(assets.banding_tables.map(<[u8]>::len) == [256, 256, 1024]
        && assets.banding_tables.map(fingerprint) == [0x0528b3d3556eb0f5, 0x24060a098de1ec13, 0x065f8d932bb7fdef3],
        "unsupported banding tables");
    Ok(())
}

#[cfg(windows)]
fn highlight_parameters(black: u32, mut gains: [f32; 3]) -> [f32; 16] {
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
}

/// Runs Capture One's OpenCL highlight recovery (`recover_highlights`) and
/// `AntiColorAliasingV10` (`anti_alias_opencl`) on any generic Bayer linear
/// RGB frame in 512x512 tiles with a 2-pixel clamped halo. Falls back silently
/// when OpenCL is unavailable.
#[cfg(windows)]
pub(in crate::aurora_raw) fn recover_and_antialias_frame(
    rgb: &mut [[f32; 3]],
    width: usize,
    height: usize,
    black: u32,
    wb: [f32; 3],
    iso: u32,
) {
    if width < 4 || height < 4 || rgb.len() != width * height || !raw::gpu_available() {
        return;
    }
    if black >= 65_535 || !wb.into_iter().all(|g| g.is_finite() && g > 0.0) {
        return;
    }
    let recovery = highlight_parameters(black, wb);
    let aca_gain = if iso >= 3200 { 1.0 } else if iso >= 1600 { 0.75 } else { 0.5 };
    let max_x = (width - 1) as isize;
    let max_y = (height - 1) as isize;
    let mut out = vec![[0.0f32; 3]; width * height];

    for top in (0..height).step_by(TILE) {
        let th = TILE.min(height - top);
        for left in (0..width).step_by(TILE) {
            let tw = TILE.min(width - left);
            let hw = tw + 4;
            let hh = th + 4;
            let mut halo = vec![[0.0f32; 3]; hw * hh];
            for y in 0..hh {
                let sy = (top as isize + y as isize - 2).clamp(0, max_y) as usize;
                let src_row = &rgb[sy * width..(sy + 1) * width];
                let dst_row = &mut halo[y * hw..(y + 1) * hw];
                for x in 0..hw {
                    let sx = (left as isize + x as isize - 2).clamp(0, max_x) as usize;
                    dst_row[x] = src_row[sx];
                }
            }
            if let Err(err) = raw::recover_highlights(&mut halo, recovery) {
                log::warn!("generic Bayer OpenCL highlight recovery skipped: {err:#}");
                return;
            }
            let core = match raw::anti_alias_opencl(&halo, hw, hh, [0.05_f32.sqrt(), aca_gain]) {
                Ok(core) if core.len() == tw * th => core,
                Ok(_) => return,
                Err(err) => {
                    log::warn!("generic Bayer OpenCL anti-alias skipped: {err:#}");
                    return;
                }
            };
            for row in 0..th {
                let dst = (top + row) * width + left;
                let src = row * tw;
                out[dst..dst + tw].copy_from_slice(&core[src..src + tw]);
            }
        }
    }
    rgb.copy_from_slice(&out);
}

fn apply_banding(rgb: &mut [u16], width: usize, iso: u32, tables: [&[u8]; 3]) {
    let [left, right, packed] = tables;
    let words: Vec<_> = packed.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect();
    let factor = match iso {
        0..=100 => 0.5,
        101..=200 => iso as f32 / 200.0,
        201..=1600 => 1.0 + (iso - 200) as f32 / 1400.0,
        _ => 2.0,
    };
    let gain = (((200.0 * factor) * (1.0 / 128.0)) * 1.25 * 65_536.0) as i32;
    let limit = (gain * 255) >> 16;
    let strength = (65_536.0_f32 / limit as f32) as i32;
    for (row, pixels) in rgb.chunks_exact_mut(3 * width).enumerate() {
        for (column, pixel) in pixels.chunks_exact_mut(3).enumerate() {
            let cell = ((column & 14) << 4) | (row & 15);
            let block = ((column & 255) >> 4) | (row & 240);
            let pair = words[(block + usize::from(left[cell])) & 255]
                .wrapping_sub(words[(block + usize::from(right[cell])) & 255]);
            let half = (pair >> (16 * (column & 1))) & 65_535;
            let high = (((half >> 8) as i32) * gain) >> 16;
            let low = (((half & 255) as i32) * gain) >> 16;
            for code in pixel {
                let value = i32::from(*code);
                let headroom = limit.min(65_535 - value);
                *code = (value + (((high - low) * headroom * strength) >> 16)).clamp(0, 65_535) as u16;
            }
        }
    }
}

/// C1 queue stage 16 (`FixChromaticAberration`, `p_041444160.cl`):
/// radially resamples linear R and B channels along `(p - center) / r` with
/// shifts clamped to +/-3 px while leaving G untouched.
pub(in crate::aurora_raw) fn apply_linear_chromatic_aberration(
    tile: &mut [[f32; 3]],
    tile_w: usize,
    tile_h: usize,
    origin_x: isize,
    origin_y: isize,
    full_w: usize,
    full_h: usize,
    warp: &crate::mock::lens_profile::LensWarp,
) {
    if (warp.red.is_empty() && warp.blue.is_empty()) || tile_w == 0 || tile_h == 0 {
        return;
    }
    let src = tile.to_vec();
    let cx = full_w as f32 * 0.5;
    let cy = full_h as f32 * 0.5;
    let half_diag = (cx * cx + cy * cy).sqrt().max(1.0);
    let max_x = (tile_w - 1) as f32;
    let max_y = (tile_h - 1) as f32;

    let sample_ch = |fx: f32, fy: f32, ch: usize| -> f32 {
        let sx = fx.clamp(0.0, max_x);
        let sy = fy.clamp(0.0, max_y);
        let x0 = sx.floor() as usize;
        let y0 = sy.floor() as usize;
        let x1 = (x0 + 1).min(tile_w - 1);
        let y1 = (y0 + 1).min(tile_h - 1);
        let tx = sx - x0 as f32;
        let ty = sy - y0 as f32;
        let p00 = src[y0 * tile_w + x0][ch];
        let p10 = src[y0 * tile_w + x1][ch];
        let p01 = src[y1 * tile_w + x0][ch];
        let p11 = src[y1 * tile_w + x1][ch];
        let top = p00 + (p10 - p00) * tx;
        let bot = p01 + (p11 - p01) * tx;
        top + (bot - top) * ty
    };

    use rayon::prelude::*;
    tile.par_chunks_exact_mut(tile_w)
        .enumerate()
        .for_each(|(y, dst_row)| {
            let gy = (origin_y + y as isize) as f32;
            let dy = gy - cy;
            for (x, pixel) in dst_row.iter_mut().enumerate() {
                let gx = (origin_x + x as isize) as f32;
                let dx = gx - cx;
                let r = (dx * dx + dy * dy).sqrt();
                if r < 1e-4 {
                    continue;
                }
                let inv_r = 1.0 / r;
                let dir_x = dx * inv_r;
                let dir_y = dy * inv_r;
                let t = (r / half_diag).clamp(0.0, 1.0);
                if !warp.red.is_empty() {
                    let shift = (crate::mock::lens_profile::LensWarp::sample(&warp.red, t) * half_diag)
                        .clamp(-3.0, 3.0);
                    pixel[0] = sample_ch(x as f32 + dir_x * shift, y as f32 + dir_y * shift, 0);
                }
                if !warp.blue.is_empty() {
                    let shift = (crate::mock::lens_profile::LensWarp::sample(&warp.blue, t) * half_diag)
                        .clamp(-3.0, 3.0);
                    pixel[2] = sample_ch(x as f32 + dir_x * shift, y as f32 + dir_y * shift, 2);
                }
            }
        });
}

/// C1 queue stages 22-34 (linear-domain directional luminance NR + LumaTransfer):
/// smooths high-ISO luminance grain along the minimum-gradient orientation in a
/// 5x5 neighbourhood and scales RGB proportionally to preserve chromaticity.
#[allow(dead_code)]
pub(in crate::aurora_raw) fn apply_linear_luma_noise_2018(
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    iso: u32,
    noise: [f32; 2],
) {
    use rayon::prelude::*;

    if iso < 400 || w < 5 || h < 5 {
        return;
    }
    let iso_strength = ((iso as f32 - 300.0) / 1700.0).clamp(0.0, 1.0);
    let mix = 0.25 + 0.50 * iso_strength;
    let slope = noise[0].max(0.5);
    let floor = noise[1].max(4.0);

    let luma: Vec<f32> = rgb
        .par_iter()
        .map(|p| (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).max(0.0))
        .collect();

    rgb.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let max_x = (w - 1) as isize;
        let max_y = (h - 1) as isize;
        let at = |ix: isize, iy: isize| -> f32 {
            luma[iy.clamp(0, max_y) as usize * w + ix.clamp(0, max_x) as usize]
        };
        let yi = y as isize;
        for x in 0..w {
            let xi = x as isize;
            let yc = luma[y * w + x];
            if yc < 1.0 {
                continue;
            }
            let sigma = (slope * yc + floor).sqrt() * (1.2 + 1.4 * iso_strength);
            let inv_two_s2 = 1.0 / (2.0 * sigma * sigma + 1e-6);

            // 4-orientation gradient energies (H, V, D1, D2).
            let gh = (at(xi - 1, yi) - at(xi + 1, yi)).abs()
                + 0.5 * (at(xi - 2, yi) - at(xi + 2, yi)).abs();
            let gv = (at(xi, yi - 1) - at(xi, yi + 1)).abs()
                + 0.5 * (at(xi, yi - 2) - at(xi, yi + 2)).abs();
            let gd1 = (at(xi - 1, yi - 1) - at(xi + 1, yi + 1)).abs()
                + 0.5 * (at(xi - 2, yi - 2) - at(xi + 2, yi + 2)).abs();
            let gd2 = (at(xi - 1, yi + 1) - at(xi + 1, yi - 1)).abs()
                + 0.5 * (at(xi - 2, yi + 2) - at(xi + 2, yi - 2)).abs();
            let g_min = gh.min(gv).min(gd1).min(gd2) + sigma * 0.5 + 1.0;
            let dir_w = [
                g_min / (gh + g_min),
                g_min / (gv + g_min),
                g_min / (gd1 + g_min),
                g_min / (gd2 + g_min),
            ];
            const TAPS: [(isize, isize, usize, f32); 16] = [
                (-1, 0, 0, 1.0), (1, 0, 0, 1.0), (-2, 0, 0, 0.5), (2, 0, 0, 0.5),
                (0, -1, 1, 1.0), (0, 1, 1, 1.0), (0, -2, 1, 0.5), (0, 2, 1, 0.5),
                (-1, -1, 2, 0.85), (1, 1, 2, 0.85), (-2, -2, 2, 0.4), (2, 2, 2, 0.4),
                (-1, 1, 3, 0.85), (1, -1, 3, 0.85), (-2, 2, 3, 0.4), (2, -2, 3, 0.4),
            ];
            let mut sum_y = yc;
            let mut sum_w = 1.0f32;
            for &(dx, dy, dir, sp) in &TAPS {
                let yn = at(xi + dx, yi + dy);
                let d = yn - yc;
                let wt = sp * dir_w[dir] * (-(d * d) * inv_two_s2).exp();
                sum_y += yn * wt;
                sum_w += wt;
            }
            let y_clean = yc + (sum_y / sum_w - yc) * mix;
            let ratio = (y_clean / yc).clamp(0.5, 1.5);
            out_row[x] = [
                out_row[x][0] * ratio,
                out_row[x][1] * ratio,
                out_row[x][2] * ratio,
            ];
        }
    });
}

/// C1 queue stages 35-38 (`ColorNoise2012` / `LABRangeFilter2012`) directly in
/// linear 16-bit camera RGB space (0..65535), eliminating 8-bit quantization loss.
/// `t_root` has variance units (`[intensity^2]`) matching `d = PX * du^2`, so it
/// scales directly with the camera's Poisson + read-noise model `[slope, floor]`.
pub(in crate::aurora_raw) fn apply_linear_color_noise_2012(
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    iso: u32,
    noise: [f32; 2],
) {
    use rayon::prelude::*;

    if w == 0 || h == 0 {
        return;
    }
    const K: [f32; 4] = [0.299, 0.587, 0.114, 35.759_254];
    const PX: f32 = 0.178_877_62;
    const PY: f32 = 0.377_672_3;
    const PZ_BASE: f32 = 0.401_832_58;
    const PW_BASE: f32 = 2.968_700_4;

    let iso_var = (iso as f32 / 100.0).max(1.0);
    // Separate chroma variance tolerance from luma edge threshold:
    // dark shadow color blotches at ISO 800-1600 have linear chroma variance
    // much larger than the ISO 100 read-noise floor, while luma edges must
    // stay strict to prevent color bleeding across bright/dark boundaries.
    let pz_luma = (noise[1] * 12.0).max(PZ_BASE * iso_var);
    let pw_luma = (noise[0] * 4.5).max(PW_BASE * iso_var);
    let pz_chroma = (noise[1] * 60.0 + 1800.0 * iso_var).max(4000.0 * iso_var);
    let pw_chroma = (noise[0] * 16.0).max(18.0 * iso_var);
    const STEP: isize = 4;
    const RADIUS: isize = 28;

    let n = w * h;
    let mut encoded = vec![[0.0f32; 3]; n];
    encoded.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &rgb[y * w..(y + 1) * w];
        for x in 0..w {
            let [r, g, b] = src[x];
            let l = (K[0] * r + K[1] * g + K[2] * b).max(0.0) + K[3];
            let inv_l = 1.0 / l;
            row[x] = [l, (r - g) * inv_l, (b - g) * inv_l];
        }
    });

    let mut horiz = vec![[0.0f32; 3]; n];
    horiz.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let src_row = &encoded[y * w..(y + 1) * w];
        let max_x = (w - 1) as isize;
        for x in 0..w {
            let [lc, uc, vc] = src_row[x];
            let t_l = (pz_luma + pw_luma * lc).max(1.0);
            let t_l2 = t_l * t_l;
            let t_c = (pz_chroma + pw_chroma * lc).max(1.0);
            let u_lc = uc * lc;
            let v_lc = vc * lc;
            let mut sum_u = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut sum_w = 0.0f32;
            let mut off = -RADIUS;
            while off <= RADIUS {
                let nx = (x as isize + off).clamp(0, max_x) as usize;
                let [li, ui, vi] = src_row[nx];
                let dl = lc - li;
                let du = u_lc - ui * li;
                let dv = v_lc - vi * li;
                let dl2 = dl * dl;
                let dc = (PX * du * du).max(PY * dv * dv);
                let w_l = t_l2 / (dl2 * dl2 + t_l2);
                let w_c = t_c / (dc + t_c);
                let wt = (w_l * w_c - 0.01).max(0.0);
                sum_u += ui * wt;
                sum_v += vi * wt;
                sum_w += wt;
                off += STEP;
            }
            if sum_w > 1e-6 {
                let inv_w = 1.0 / sum_w;
                out_row[x] = [lc, sum_u * inv_w, sum_v * inv_w];
            } else {
                out_row[x] = [lc, uc, vc];
            }
        }
    });

    let max_y = (h - 1) as isize;
    let mut vert = vec![[0.0f32; 3]; n];
    vert.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        for x in 0..w {
            let [lc, uc, vc] = horiz[y * w + x];
            let t_l = (pz_luma + pw_luma * lc).max(1.0);
            let t_l2 = t_l * t_l;
            let t_c = (pz_chroma + pw_chroma * lc).max(1.0);
            let u_lc = uc * lc;
            let v_lc = vc * lc;
            let mut sum_u = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut sum_w = 0.0f32;
            let mut off = -RADIUS;
            while off <= RADIUS {
                let ny = (y as isize + off).clamp(0, max_y) as usize;
                let [li, ui, vi] = horiz[ny * w + x];
                let dl = lc - li;
                let du = u_lc - ui * li;
                let dv = v_lc - vi * li;
                let dl2 = dl * dl;
                let dc = (PX * du * du).max(PY * dv * dv);
                let w_l = t_l2 / (dl2 * dl2 + t_l2);
                let w_c = t_c / (dc + t_c);
                let wt = (w_l * w_c - 0.01).max(0.0);
                sum_u += ui * wt;
                sum_v += vi * wt;
                sum_w += wt;
                off += STEP;
            }
            if sum_w > 1e-6 {
                let inv_w = 1.0 / sum_w;
                out_row[x] = [lc, sum_u * inv_w, sum_v * inv_w];
            } else {
                out_row[x] = [lc, uc, vc];
            }
        }
    });

    // Wide-scale second pass for ISO >= 400 to suppress low-frequency
    // 8-20px chroma blotches in underexposed shadows (e.g. ISO 1000 NEF).
    if iso >= 400 {
        let hi_blend = (((iso as f32).log2() - 400.0_f32.log2()) / 3.0).clamp(0.35, 1.0);
        const HI_STEP: isize = 6;
        const HI_RADIUS: isize = 42;
        horiz.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
            let src_row = &vert[y * w..(y + 1) * w];
            let max_x = (w - 1) as isize;
            for x in 0..w {
                let [lc, uc, vc] = src_row[x];
                let t_l = (pz_luma + pw_luma * lc).max(1.0);
                let t_l2 = t_l * t_l;
                let t_c = (pz_chroma * 2.5 + pw_chroma * 1.5 * lc).max(1.0);
                let u_lc = uc * lc;
                let v_lc = vc * lc;
                let mut sum_u = 0.0f32;
                let mut sum_v = 0.0f32;
                let mut sum_w = 0.0f32;
                let mut off = -HI_RADIUS;
                while off <= HI_RADIUS {
                    let nx = (x as isize + off).clamp(0, max_x) as usize;
                    let [li, ui, vi] = src_row[nx];
                    let dl = lc - li;
                    let du = u_lc - ui * li;
                    let dv = v_lc - vi * li;
                    let dl2 = dl * dl;
                    let dc = (PX * du * du).max(PY * dv * dv);
                    let w_l = t_l2 / (dl2 * dl2 + t_l2);
                    let w_c = t_c / (dc + t_c);
                    let wt = (w_l * w_c - 0.01).max(0.0);
                    sum_u += ui * wt;
                    sum_v += vi * wt;
                    sum_w += wt;
                    off += HI_STEP;
                }
                if sum_w > 1e-6 {
                    let inv_w = 1.0 / sum_w;
                    out_row[x] = [
                        lc,
                        uc + (sum_u * inv_w - uc) * hi_blend,
                        vc + (sum_v * inv_w - vc) * hi_blend,
                    ];
                } else {
                    out_row[x] = [lc, uc, vc];
                }
            }
        });
        vert.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
            for x in 0..w {
                let [lc, uc, vc] = horiz[y * w + x];
                let t_l = (pz_luma + pw_luma * lc).max(1.0);
                let t_l2 = t_l * t_l;
                let t_c = (pz_chroma * 2.5 + pw_chroma * 1.5 * lc).max(1.0);
                let u_lc = uc * lc;
                let v_lc = vc * lc;
                let mut sum_u = 0.0f32;
                let mut sum_v = 0.0f32;
                let mut sum_w = 0.0f32;
                let mut off = -HI_RADIUS;
                while off <= HI_RADIUS {
                    let ny = (y as isize + off).clamp(0, max_y) as usize;
                    let [li, ui, vi] = horiz[ny * w + x];
                    let dl = lc - li;
                    let du = u_lc - ui * li;
                    let dv = v_lc - vi * li;
                    let dl2 = dl * dl;
                    let dc = (PX * du * du).max(PY * dv * dv);
                    let w_l = t_l2 / (dl2 * dl2 + t_l2);
                    let w_c = t_c / (dc + t_c);
                    let wt = (w_l * w_c - 0.01).max(0.0);
                    sum_u += ui * wt;
                    sum_v += vi * wt;
                    sum_w += wt;
                    off += HI_STEP;
                }
                if sum_w > 1e-6 {
                    let inv_w = 1.0 / sum_w;
                    out_row[x] = [
                        lc,
                        uc + (sum_u * inv_w - uc) * hi_blend,
                        vc + (sum_v * inv_w - vc) * hi_blend,
                    ];
                } else {
                    out_row[x] = [lc, uc, vc];
                }
            }
        });
    }

    rgb.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let src_row = &vert[y * w..(y + 1) * w];
        for x in 0..w {
            let [lc, u, v] = src_row[x];
            let ul = u * lc;
            let vl = v * lc;
            let g = lc - ul * K[0] - vl * K[2] - K[3];
            let r0 = (ul + g).max(1.0);
            let g0 = g.max(1.0);
            let b0 = (vl + g).max(1.0);
            let sc = ((lc - (K[3] - 1.0)) / ((K[0] * r0 + K[1] * g0 + K[2] * b0).max(0.0) + 1.0))
                .min(1.5);
            out_row[x] = [r0 * sc, g0 * sc, b0 * sc];
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banding_preserves_saturation_and_phase() {
        let mut words = [0_u8; 1024];
        words[4..8].copy_from_slice(&0x01ff_ff01_u32.to_le_bytes());
        let mut pixels = [0, 1_000, 65_535, 0, 1_000, 65_535];
        apply_banding(&mut pixels, 2, 200, [&[1; 256], &[0; 256], &words]);
        assert_eq!(pixels, [494, 1_494, 65_535, 0, 505, 65_535]);

    }

    #[test]
    fn linear_ca_shifts_red_and_blue_leaving_green_unchanged() {
        let (w, h) = (32, 32);
        let mut tile = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let v = (x * 1000) as f32;
                tile[y * w + x] = [v, v, v];
            }
        }
        let before = tile.clone();
        let warp = crate::mock::lens_profile::LensWarp {
            red: vec![(0.0, 0.0), (1.0, 0.05)],
            blue: vec![(0.0, 0.0), (1.0, -0.05)],
            ..Default::default()
        };
        apply_linear_chromatic_aberration(&mut tile, w, h, 0, 0, w, h, &warp);
        let probe = 16 * w + 28;
        assert!(tile[probe][0] > before[probe][0], "positive red shift moves outward");
        assert!(tile[probe][2] < before[probe][2], "negative blue shift moves inward");
        assert_eq!(tile[probe][1], before[probe][1], "green channel must stay untouched");
    }

    #[test]
    fn linear_denoise_suppresses_high_iso_blotches_without_shifting_luma() {
        let (w, h) = (64, 64);
        let mut rgb = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                // 8px-wide red/green chroma blotches + mild luma grain.
                let blotch = if ((x / 8) + (y / 8)) % 2 == 0 { 1.0 } else { -1.0 };
                let grain = if (x + y) % 2 == 0 { 250.0 } else { -250.0 };
                let base = 8000.0 + grain;
                rgb[y * w + x] = [base + blotch * 900.0, base - blotch * 450.0, base + blotch * 450.0];
            }
        }
        let chroma_var = |buf: &[[f32; 3]]| -> f32 {
            let diffs: Vec<f32> = buf.iter().map(|p| p[0] - p[1]).collect();
            let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
            diffs.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / diffs.len() as f32
        };
        let before_c = chroma_var(&rgb);
        apply_linear_luma_noise_2018(&mut rgb, w, h, 1000, [18.0, 200.0]);
        apply_linear_color_noise_2012(&mut rgb, w, h, 1000, [18.0, 200.0]);
        let after_c = chroma_var(&rgb);
        assert!(after_c < before_c * 0.25, "linear ColorNoise2012 must strongly suppress 8px chroma blotches: before={before_c} after={after_c}");
        let mean_l: f32 = rgb.iter().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).sum::<f32>() / rgb.len() as f32;
        assert!((mean_l - 8000.0).abs() < 150.0, "mean luma should stay stable: {mean_l}");
    }
}
