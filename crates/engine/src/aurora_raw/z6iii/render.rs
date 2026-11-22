use super::{ExportAssets, color, reconstruct as raw};
use anyhow::{Context, Result, anyhow, ensure};
use color::output::OutputClut;

const TILE: usize = 512;

/// Runs the Z6III OpenCL reconstruction + highlight recovery + anti-aliasing
/// pipeline over 512x512 tiles, invoking `on_tile(left, top, width, height, core)`
/// for each linear camera-RGB tile.
pub(super) fn render_linear_tiles(
    nef: &[u8],
    bayer: Option<(usize, usize, Vec<u16>)>,
    mut on_tile: impl FnMut(usize, usize, usize, usize, &[[f32; 3]], u32) -> Result<()>,
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
    for top in (0..crop.height).step_by(TILE) {
        let height = TILE.min(crop.height - top);
        for left in (0..crop.width).step_by(TILE) {
            let width = TILE.min(crop.width - left);
            let area = raw::NikonCrop { left: crop.left + left - 2, top: crop.top + top - 2,
                width: width + 4, height: height + 4 };
            let mut halo = raw::reconstruct_planar_green_halo_roi(&frame, area, stage1, black as i32, wb, offset, noise[0])
                .map_err(|error| anyhow!("reconstruction at ({left},{top}): {error}"))?;
            raw::recover_highlights(&mut halo, recovery)
                .with_context(|| format!("OpenCL highlight recovery at ({left},{top})"))?;
            let core = raw::anti_alias_opencl(&halo, area.width, area.height, [0.05_f32.sqrt(), aca_gain])
                .map_err(|error| anyhow!("OpenCL anti-alias at ({left},{top}): {error}"))?;
            ensure!(core.len() == width * height, "unexpected anti-alias tile size");
            on_tile(left, top, width, height, &core, iso)?;
        }
    }
    Ok((crop.width, crop.height))
}

pub(in crate::aurora_raw) fn linear_rgba16(
    nef: &[u8],
    bayer: Option<(usize, usize, Vec<u16>)>,
) -> Result<(usize, usize, Vec<u16>)> {
    let mut out = vec![0u16; 6048 * 4032 * 4];
    let (crop_w, crop_h) = render_linear_tiles(nef, bayer, |left, top, width, _height, core, _iso| {
        for (row, values) in core.chunks_exact(width).enumerate() {
            let dst_row = ((top + row) * 6048 + left) * 4;
            for (col, px) in values.iter().enumerate() {
                let dst = dst_row + col * 4;
                out[dst] = px[0].clamp(0.0, 65535.0) as u16;
                out[dst + 1] = px[1].clamp(0.0, 65535.0) as u16;
                out[dst + 2] = px[2].clamp(0.0, 65535.0) as u16;
                out[dst + 3] = 65535;
            }
        }
        Ok(())
    })?;
    Ok((crop_w, crop_h, out))
}

pub(super) fn neutral(nef: &[u8], assets: &ExportAssets<'_>) -> Result<Vec<u16>> {
    check_assets(assets)?;
    let film = color::film_nodes(assets.curve).context("invalid film curve")?;
    let output = OutputClut::from_camera49(assets.camera_icc, assets.adobe_rgb_icc)
        .context("unsupported Adobe RGB output CLUT")?;
    let mut pixels = vec![[0; 3]; 6048 * 4032];
    let mut detected_iso = 100u32;
    let (crop_w, _crop_h) = render_linear_tiles(nef, None, |left, top, width, _height, core, iso| {
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

#[cfg(test)]
#[test]
fn banding_preserves_saturation_and_phase() {
    let mut words = [0_u8; 1024];
    words[4..8].copy_from_slice(&0x01ff_ff01_u32.to_le_bytes());
    let mut pixels = [0, 1_000, 65_535, 0, 1_000, 65_535];
    apply_banding(&mut pixels, 2, 200, [&[1; 256], &[0; 256], &words]);
    assert_eq!(pixels, [494, 1_494, 65_535, 0, 505, 65_535]);
}
