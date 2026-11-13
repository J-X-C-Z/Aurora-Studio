//! Exposure fusion for a bracketed HDR merge.
//!
//! Frames arrive as linear-light [`Source`] values. Each frame is scaled by its
//! EXIF exposure relative to the first frame, then combined with Mertens'
//! well-exposedness, contrast and saturation weights. Alignment and deghosting
//! are not applied.

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::mock::process::linear_to_srgb;
use crate::{Exif, Source};

const SIGMA: f32 = 0.2;
const EPS: f32 = 0.01;

pub struct FusedFrame {
    pub width: u32,
    pub height: u32,
    /// Row-major linear RGB, 0 ..= 1.
    pub rgb: Vec<[f32; 3]>,
}

/// Relative exposure of `frame` against `reference`, in stops.
///
/// A component only counts when both sides carry a positive tag. Missing EXIF
/// contributes zero rather than a guessed exposure.
pub fn relative_stops(reference: &Exif, frame: &Exif) -> f32 {
    let ratio = |a: Option<f32>, b: Option<f32>| match (a, b) {
        (Some(a), Some(b)) if a > 0.0 && b > 0.0 => Some(b / a),
        _ => None,
    };
    let mut stops = 0.0;
    if let Some(r) = ratio(reference.shutter, frame.shutter) {
        stops += r.log2();
    }
    if let Some(r) = ratio(reference.iso.map(|i| i as f32), frame.iso.map(|i| i as f32)) {
        stops += r.log2();
    }
    if let Some(r) = ratio(reference.aperture, frame.aperture) {
        stops -= 2.0 * r.log2();
    }
    stops
}

pub fn fuse(frames: &[&Source], stops: &[f32]) -> Result<FusedFrame> {
    let Some(first) = frames.first() else {
        bail!("HDR merge needs at least one frame");
    };
    if frames.len() != stops.len() {
        bail!("HDR merge: {} frames but {} exposures", frames.len(), stops.len());
    }
    let (width, height) = first.size();
    if frames.iter().any(|f| f.size() != (width, height)) {
        let (w, h) = frames.iter().find(|f| f.size() != (width, height)).unwrap().size();
        bail!("HDR merge frames must share one size: {width}x{height} vs {w}x{h}");
    }

    let reference = stops[0];
    let gains: Vec<f32> = stops.iter().map(|s| 2f32.powf(reference - s)).collect();
    let planes: Vec<&crate::mock::process::Plane> = frames.iter().map(|f| &f.levels[0]).collect();
    let rgb: Vec<[f32; 3]> = (0..(width as usize) * (height as usize))
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i as u32 % width, i as u32 / width);
            let mut acc = [0.0f32; 3];
            let mut sum = 0.0f32;
            for (plane, &gain) in planes.iter().zip(&gains) {
                let weight = weight(plane, x, y);
                let pixel = plane.data[i];
                for k in 0..3 {
                    acc[k] += pixel[k].clamp(0.0, 1.0) * gain * weight;
                }
                sum += weight;
            }
            if sum > 0.0 {
                acc.map(|v| (v / sum).clamp(0.0, 1.0))
            } else {
                [0.0; 3]
            }
        })
        .collect();
    Ok(FusedFrame { width, height, rgb })
}

/// Encodes fused linear values as the 16-bit sRGB transport used by TIFF.
pub fn to_srgb16(frame: &FusedFrame) -> Vec<u16> {
    let mut out = vec![0u16; frame.rgb.len() * 3];
    out.par_chunks_mut(3).zip(frame.rgb.par_iter()).for_each(|(out, pixel)| {
        for k in 0..3 {
            out[k] = (linear_to_srgb(pixel[k].clamp(0.0, 1.0)) * 65535.0 + 0.5) as u16;
        }
    });
    out
}

fn weight(plane: &crate::mock::process::Plane, x: u32, y: u32) -> f32 {
    let pixel = plane.data[(y * plane.width + x) as usize];
    let luma = luma(pixel);
    let well_exposed = (-(luma - 0.5).powi(2) / (2.0 * SIGMA * SIGMA)).exp();
    let contrast = laplacian(plane, x, y).abs();
    let mean = (pixel[0] + pixel[1] + pixel[2]) / 3.0;
    let saturation = (((pixel[0] - mean).powi(2) + (pixel[1] - mean).powi(2)
        + (pixel[2] - mean).powi(2)) / 3.0)
        .sqrt();
    (well_exposed * (contrast + EPS) * (saturation + EPS)).max(1e-8)
}

fn luma(pixel: [f32; 3]) -> f32 {
    0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
}

fn laplacian(plane: &crate::mock::process::Plane, x: u32, y: u32) -> f32 {
    let at = |x: u32, y: u32| luma(plane.data[(y * plane.width + x) as usize]);
    let center = at(x, y);
    let left = if x > 0 { at(x - 1, y) } else { center };
    let right = if x + 1 < plane.width { at(x + 1, y) } else { center };
    let up = if y > 0 { at(x, y - 1) } else { center };
    let down = if y + 1 < plane.height { at(x, y + 1) } else { center };
    4.0 * center - left - right - up - down
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [f32; 3]) -> Source {
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let p = pixel(x, y).map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16);
                rgba.extend_from_slice(&[p[0], p[1], p[2], u16::MAX]);
            }
        }
        Source::from_rgba16(width, height, &rgba)
    }

    #[test]
    fn single_frame_is_returned_unchanged() {
        let src = frame(8, 8, |x, y| [x as f32 / 7.0, y as f32 / 7.0, 0.4]);
        let fused = fuse(&[&src], &[0.0]).unwrap();
        for (actual, expected) in fused.rgb.iter().zip(&src.levels[0].data) {
            for k in 0..3 {
                assert!((actual[k] - expected[k]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn highlights_and_shadows_come_from_the_opposite_exposures() {
        let dark = frame(16, 16, |x, _| if x < 8 { [0.8; 3] } else { [0.0; 3] });
        let bright = frame(16, 16, |x, _| if x < 8 { [1.0; 3] } else { [0.008; 3] });
        let fused = fuse(&[&dark, &bright], &[0.0, 2.0]).unwrap();
        let left = fused.rgb[8 * 16 + 4];
        let right = fused.rgb[8 * 16 + 12];
        assert!((0.6..0.95).contains(&left[0]));
        assert!(right[0] > 0.0005);
    }

    #[test]
    fn relative_stops_reads_all_exposure_components() {
        let base = Exif { shutter: Some(1.0 / 125.0), iso: Some(100), aperture: Some(4.0), ..Default::default() };
        let longer = Exif { shutter: Some(1.0 / 31.25), ..base.clone() };
        assert!((relative_stops(&base, &longer) - 2.0).abs() < 1e-5);
        let higher_iso = Exif { iso: Some(400), ..base.clone() };
        assert!((relative_stops(&base, &higher_iso) - 2.0).abs() < 1e-5);
        let wider = Exif { aperture: Some(2.8), ..base.clone() };
        assert!((relative_stops(&base, &wider) - 1.03).abs() < 0.02);
        assert_eq!(relative_stops(&base, &Exif::default()), 0.0);
    }
}
