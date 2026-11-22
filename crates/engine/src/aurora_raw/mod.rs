//! Aurora Raw: Nikon HE RAW decoding and reconstruction pipeline, ported from the parse lanes.
//!
//! Decoding is byte-identical to the bundled LibRaw on the validated Z6III
//! samples; the reason to run it in the product is the demosaic that follows,
//! which LibRaw replaces with its own. Everything here is opt-in through
//! `AURORA_C1_RAW=1` in `crate::raw` until the demosaic output has a visual
//! acceptance of its own.

pub mod c1_raw_he;
#[cfg(all(windows, feature = "z6iii-raw-export"))]
pub mod z6iii;

pub use c1_raw_he::decode_frame_cropped;

use std::path::Path;

/// Full-frame RGB16 through the C1 chain: HE decode, black level, stage 1/2/3
/// over 128-pixel tiles, then clamp to 16-bit codes.
///
/// The tile size is the one the lane proved invariant (256 px whole vs four
/// 128 px parts: zero differing pixels). Returns `None` for anything that is
/// not the verified HE variant, so the caller can fall back to LibRaw.
pub fn decode_he_rgba16(path: &Path) -> Option<(u32, u32, Vec<u16>)> {
    let bytes = std::fs::read(path).ok()?;
    let orientation = crate::raw::exif_orientation_bytes(&bytes).unwrap_or(1);
    // Prefer LibRaw's Bayer plane when the loaded build supports the format.
    // The custom HE decoder remains the fallback for builds without LibRaw 0.23.
    let libraw_bayer = crate::raw::decode_bayer16(path)
        .ok()
        .map(|lib| (lib.raw_width, lib.raw_height, lib.data));

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    if let Some((width, height, rgba)) = z6iii::render_linear_rgba16(&bytes, libraw_bayer.clone()) {
        return Some(orient_rgba16(rgba, width, height, orientation));
    }

    let fallback = || decode_frame_cropped(&bytes).ok().flatten();
    let frame = match libraw_bayer {
        Some((raw_width, raw_height, data)) => match c1_raw_he::nikon_crop(&bytes) {
            Ok(Some(crop)) => {
                match c1_raw_he::NikonHeCropFrame::new(crop, raw_width, raw_height, data) {
                    Ok(frame) => frame,
                    Err(_) => fallback()?,
                }
            }
            _ => fallback()?,
        },
        None => fallback()?,
    };
    let wb = c1_raw_he::nikon_as_shot_rgb(&bytes).ok()??;
    let black = c1_raw_he::nikon_black_levels(&bytes).ok()??.capture_one_scalar();
    let params = c1_raw_he::Stage1Params::prepare(wb, [12.8, 0.0], black, false);
    let crop = frame.crop();
    let (width, height) = (crop.width, crop.height);
    if width == 0 || height == 0 {
        return None;
    }

    let mut rgb = vec![0f32; width * height * 3];
    for top in (0..height).step_by(128) {
        for left in (0..width).step_by(128) {
            let tile_width = (width - left).min(128);
            let tile_height = (height - top).min(128);
            let area = c1_raw_he::NikonCrop {
                left: crop.left + left,
                top: crop.top + top,
                width: tile_width,
                height: tile_height,
            };
            let tile = c1_raw_he::reconstruct_planar_active_mirror_roi(
                &frame,
                area,
                params,
                Some(black as i32),
                wb,
                128.0,
            )
            .ok()?;
            for y in 0..tile_height {
                for x in 0..tile_width {
                    let dst = ((top + y) * width + left + x) * 3;
                    rgb[dst..dst + 3].copy_from_slice(&tile[y * tile_width + x]);
                }
            }
        }
    }

    let mut out = Vec::with_capacity(width * height * 4);
    for pixel in rgb.chunks_exact(3) {
        for value in pixel {
            out.push(value.clamp(0.0, 65535.0) as u16);
        }
        out.push(65535);
    }
    let orientation = crate::raw::exif_orientation_bytes(&bytes).unwrap_or(1);
    Some(orient_rgba16(out, width, height, orientation))
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
}
