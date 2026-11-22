//! Exposure fusion for a bracketed HDR merge: no alignment, with deghosting.
//!
//! Frames arrive as linear-light [`Source`]s. Each frame is first scaled by its
//! EXIF exposure relative to the middle-exposure frame, then every pixel gets
//! Mertens' exposure-fusion weight: well-exposedness × contrast × saturation.
//! Averaging with those weights takes shadows from the longer exposures and
//! highlights from the shorter ones; the result stays at the reference frame's
//! exposure. A pixel that differs from the reference beyond `DEGHOST_THRESHOLD`
//! drops that frame's weight, so a moving object keeps the reference pixels
//! instead of being blended into a ghost.

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::mock::process::linear_to_srgb;
use crate::{DeghostMode, Exif, HdrOptions, Source};

/// Width of the well-exposedness bell, in linear light.
const SIGMA: f32 = 0.2;
/// Keeps flat, colourless regions from dropping to a zero weight.
const EPS: f32 = 0.01;
/// Largest linear-light channel difference still treated as the same scene.
const DEGHOST_THRESHOLD: f32 = 0.08;

/// A fused frame in linear light.
pub struct FusedFrame {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB, 0 ..= 1.
    pub rgb: Vec<[f32; 3]>,
}

/// Relative exposure of `frame` against `reference`, in stops.
///
/// `EV = log2(t) + log2(ISO) - 2·log2(f)`, so the difference is
/// `log2(t_b/t_a) + log2(ISO_b/ISO_a) - 2·log2(f_b/f_a)`. A component only
/// counts when both sides carry it; nothing is guessed from a missing tag.
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

/// Fuses `frames` at their relative `stops`, one `stops` entry per frame.
pub fn fuse(frames: &[&Source], stops: &[f32]) -> Result<FusedFrame> {
    fuse_with(frames, stops, true)
}

/// Fuses with the interface's options.
///
/// `DeghostMode::Off` takes the legacy no-deghost path unchanged. `Auto` and
/// `Manual` detect a moving pixel by its median luminance across the bracket
/// and, where one frame deviates by more than [`DEGHOST_THRESHOLD`], keep only
/// the reference frame's pixel. `Manual` uses
/// [`HdrOptions::deghost_reference`] when it is in range; otherwise the median
/// exposure is the reference.
pub fn fuse_with_options(
    frames: &[&Source],
    stops: &[f32],
    options: &HdrOptions,
) -> Result<FusedFrame> {
    if frames.len() == 1 && stops.len() == 1 {
        let first = frames[0];
        let (width, height) = first.size();
        return Ok(FusedFrame { width, height, rgb: first.levels[0].data.clone() });
    }
    if options.deghost == DeghostMode::Off {
        return fuse_with(frames, stops, false);
    }
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
    let middle = middle_exposure(stops);
    let reference = match options.deghost {
        DeghostMode::Manual => {
            options.deghost_reference.filter(|&index| index < frames.len()).unwrap_or(middle)
        }
        _ => middle,
    };
    let reference_stops = stops[reference];
    let gains: Vec<f32> = stops.iter().map(|s| 2f32.powf(reference_stops - s)).collect();
    let planes: Vec<&crate::mock::process::Plane> = frames.iter().map(|f| &f.levels[0]).collect();
    let reference_pixel = &planes[reference].data;
    let reference_gain = gains[reference];
    let deghost = options.deghost != DeghostMode::Off;
    let rgb: Vec<[f32; 3]> = (0..(width as usize) * (height as usize))
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i as u32 % width, i as u32 / width);
            if deghost && pixel_moves(&planes, &gains, i) {
                return reference_pixel[i].map(|v| (v * reference_gain).clamp(0.0, 1.0));
            }
            let mut acc = [0.0f32; 3];
            let mut sum = 0.0f32;
            for (plane, &gain) in planes.iter().zip(&gains) {
                let pixel = plane.data[i];
                let weight = weight(plane, x, y);
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

/// True when one frame's exposure-normalised luminance is too far from the
/// bracket's median at this pixel: the signature of a moving object.
fn pixel_moves(planes: &[&crate::mock::process::Plane], gains: &[f32], index: usize) -> bool {
    const STACK: usize = 16;
    let n = planes.len();
    if n <= STACK {
        let mut lumas = [0.0f32; STACK];
        for (slot, (plane, &gain)) in lumas[..n].iter_mut().zip(planes.iter().zip(gains)) {
            *slot = luma(plane.data[index]) * gain;
        }
        sort_f32(&mut lumas[..n]);
        median_sorted(&lumas[..n])
    } else {
        let mut lumas: Vec<f32> = planes
            .iter()
            .zip(gains)
            .map(|(plane, &gain)| luma(plane.data[index]) * gain)
            .collect();
        sort_f32(&mut lumas);
        median_sorted(&lumas)
    }
}

fn median_sorted(values: &[f32]) -> bool {
    let median = if values.len() % 2 == 1 {
        values[values.len() / 2]
    } else {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    };
    values.iter().any(|&value| (value - median).abs() > DEGHOST_THRESHOLD)
}

fn sort_f32(values: &mut [f32]) {
    values.sort_by(f32::total_cmp);
}

fn fuse_with(frames: &[&Source], stops: &[f32], deghost: bool) -> Result<FusedFrame> {
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

    let reference = middle_exposure(stops);
    let reference_stops = stops[reference];
    let gains: Vec<f32> = stops.iter().map(|s| 2f32.powf(reference_stops - s)).collect();
    let planes: Vec<&crate::mock::process::Plane> = frames.iter().map(|f| &f.levels[0]).collect();
    let reference_pixel = &planes[reference].data;
    let reference_gain = gains[reference];
    let rgb: Vec<[f32; 3]> = (0..(width as usize) * (height as usize))
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i as u32 % width, i as u32 / width);
            let mut acc = [0.0f32; 3];
            let mut sum = 0.0f32;
            for (index, (plane, &gain)) in planes.iter().zip(&gains).enumerate() {
                let pixel = plane.data[i];
                if deghost
                    && index != reference
                    && differs(pixel, gain, reference_pixel[i], reference_gain)
                {
                    continue;
                }
                let weight = weight(plane, x, y);
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

/// Index of the frame at the median exposure; ties pick the shorter exposure.
fn middle_exposure(stops: &[f32]) -> usize {
    let mut order: Vec<usize> = (0..stops.len()).collect();
    order.sort_by(|&a, &b| stops[a].total_cmp(&stops[b]));
    order[(order.len() - 1) / 2]
}

fn differs(pixel: [f32; 3], gain: f32, reference: [f32; 3], reference_gain: f32) -> bool {
    (0..3).any(|k| {
        let reference = reference[k] * reference_gain;
        // A clipped reference has no detail left for a shorter frame to protect.
        reference < 1.0 - 1e-3
            && (pixel[k] * gain - reference).abs() > DEGHOST_THRESHOLD
    })
}

/// 16-bit sRGB samples for a TIFF: the fused linear values go through the same
/// transfer the file engine applies when it reads a non-RAW file back.
pub fn to_srgb16(frame: &FusedFrame) -> Vec<u16> {
    let mut out = vec![0u16; frame.rgb.len() * 3];
    out.par_chunks_mut(3)
        .zip(frame.rgb.par_iter())
        .for_each(|(out, pixel)| {
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
    let saturation = (((pixel[0] - mean).powi(2) + (pixel[1] - mean).powi(2) + (pixel[2] - mean).powi(2))
        / 3.0)
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
    fn a_single_frame_is_returned_unchanged() {
        let src = frame(8, 8, |x, y| [x as f32 / 7.0, y as f32 / 7.0, 0.4]);
        let fused = fuse(&[&src], &[0.0]).unwrap();
        for (a, b) in fused.rgb.iter().zip(&src.levels[0].data) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-6, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn highlights_come_from_the_dark_frame_and_shadows_from_the_bright_one() {
        // First frame: correct exposure, shadow crushed to black, highlight detail.
        let dark = frame(16, 16, |x, _| if x < 8 { [0.8; 3] } else { [0.0; 3] });
        // Second frame: two stops brighter, so the highlight clipped and the
        // shadow carries usable signal.
        let bright = frame(16, 16, |x, _| if x < 8 { [1.0; 3] } else { [0.008; 3] });
        let fused = fuse(&[&dark, &bright], &[0.0, 2.0]).unwrap();
        let left = fused.rgb[8 * 16 + 4];
        let right = fused.rgb[8 * 16 + 12];
        assert!(left[0] < 0.95, "the clipped highlight stayed at 1.0: {left:?}");
        assert!(left[0] > 0.6, "the dark frame's highlight was lost: {left:?}");
        assert!(right[0] > 0.0005, "the bright frame's shadow was not used: {right:?}");
    }

    #[test]
    fn a_pixel_that_moves_out_of_frame_takes_the_reference_frame() {
        // The block is in the first two frames only; the bright frame sees the
        // background there. Blending it would make the block semi-transparent.
        let dark = frame(16, 16, |x, _| if (6..10).contains(&x) { [0.2; 3] } else { [0.1; 3] });
        let middle = frame(16, 16, |x, _| if (6..10).contains(&x) { [0.4; 3] } else { [0.2; 3] });
        let bright = frame(16, 16, |_x, _y| [0.4; 3]);
        let fused = fuse(&[&dark, &middle, &bright], &[-1.0, 0.0, 1.0]).unwrap();
        let block = fused.rgb[8 * 16 + 8];
        let reference = middle.levels[0].data[8 * 16 + 8];
        let ghosted = fuse_with(&[&dark, &middle, &bright], &[-1.0, 0.0, 1.0], false).unwrap();
        assert!(
            (ghosted.rgb[8 * 16 + 8][0] - reference[0]).abs() > 0.02,
            "the fixture does not exercise blending without deghosting"
        );
        assert!(
            (block[0] - reference[0]).abs() < 1e-4,
            "the moving block was blended: {block:?} vs reference {reference:?}"
        );
        let background = fused.rgb[8 * 16 + 2];
        assert!((background[0] - 0.2).abs() < 1e-4, "static background changed: {background:?}");
    }

    #[test]
    fn a_static_scene_is_unchanged_by_deghosting() {
        let dark = frame(16, 16, |x, _| [0.05 + x as f32 * 0.01; 3]);
        let middle = frame(16, 16, |x, _| [0.1 + x as f32 * 0.02; 3]);
        let bright = frame(16, 16, |x, _| [0.2 + x as f32 * 0.04; 3]);
        let frames = [&dark, &middle, &bright];
        let with = fuse_with(&frames, &[-1.0, 0.0, 1.0], true).unwrap();
        let without = fuse_with(&frames, &[-1.0, 0.0, 1.0], false).unwrap();
        assert_eq!(with.rgb.len(), without.rgb.len());
        for (a, b) in with.rgb.iter().zip(&without.rgb) {
            assert_eq!(a, b, "deghosting changed a static pixel: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn relative_stops_reads_shutter_iso_and_aperture() {
        let base = Exif {
            shutter: Some(1.0 / 125.0),
            iso: Some(100),
            aperture: Some(4.0),
            ..Default::default()
        };
        let longer = Exif { shutter: Some(1.0 / 31.25), ..base.clone() };
        assert!((relative_stops(&base, &longer) - 2.0).abs() < 1e-5);
        let higher_iso = Exif { iso: Some(400), ..base.clone() };
        assert!((relative_stops(&base, &higher_iso) - 2.0).abs() < 1e-5);
        let wider = Exif { aperture: Some(2.8), ..base.clone() };
        assert!((relative_stops(&base, &wider) - 1.03).abs() < 0.02);
        assert_eq!(relative_stops(&base, &Exif::default()), 0.0);
    }

    fn moving_bracket() -> (Source, Source, Source) {
        let dark = frame(16, 16, |x, _| if (6..10).contains(&x) { [0.2; 3] } else { [0.1; 3] });
        let middle = frame(16, 16, |x, _| if (6..10).contains(&x) { [0.4; 3] } else { [0.2; 3] });
        let bright = frame(16, 16, |_x, _y| [0.4; 3]);
        (dark, middle, bright)
    }

    #[test]
    fn deghost_off_is_byte_identical_to_the_legacy_no_deghost_path() {
        let (dark, middle, bright) = moving_bracket();
        let frames = [&dark, &middle, &bright];
        let stops = [-1.0, 0.0, 1.0];
        let legacy = fuse_with(&frames, &stops, false).unwrap();
        let off = fuse_with_options(
            &frames,
            &stops,
            &HdrOptions { deghost: DeghostMode::Off, ..HdrOptions::default() },
        )
        .unwrap();
        assert_eq!(legacy.rgb, off.rgb);
    }

    #[test]
    fn auto_deghost_keeps_the_median_exposure_reference() {
        let (dark, middle, bright) = moving_bracket();
        let frames = [&dark, &middle, &bright];
        let fused = fuse_with_options(
            &frames,
            &[-1.0, 0.0, 1.0],
            &HdrOptions { deghost: DeghostMode::Auto, ..HdrOptions::default() },
        )
        .unwrap();
        let block = fused.rgb[8 * 16 + 8];
        let reference = middle.levels[0].data[8 * 16 + 8];
        assert!((block[0] - reference[0]).abs() < 1e-6, "{block:?} vs {reference:?}");
        assert!(fused.rgb.iter().all(|p| p.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn manual_deghost_uses_the_selected_reference_frame() {
        let (dark, middle, bright) = moving_bracket();
        let frames = [&dark, &middle, &bright];
        let options = HdrOptions {
            deghost: DeghostMode::Manual,
            deghost_reference: Some(0),
            ..HdrOptions::default()
        };
        let fused = fuse_with_options(&frames, &[-1.0, 0.0, 1.0], &options).unwrap();
        let block = fused.rgb[8 * 16 + 8];
        let reference = dark.levels[0].data[8 * 16 + 8];
        assert!((block[0] - reference[0]).abs() < 1e-6, "{block:?} vs {reference:?}");

        let missing = HdrOptions { deghost_reference: Some(99), ..options };
        let fallback = fuse_with_options(&frames, &[-1.0, 0.0, 1.0], &missing).unwrap();
        let auto = fuse_with_options(
            &frames,
            &[-1.0, 0.0, 1.0],
            &HdrOptions { deghost: DeghostMode::Auto, ..HdrOptions::default() },
        )
        .unwrap();
        assert_eq!(fallback.rgb, auto.rgb);
    }

    #[test]
    fn a_single_frame_with_deghost_options_is_unchanged() {
        let src = frame(8, 8, |x, y| [x as f32 / 7.0, y as f32 / 7.0, 0.4]);
        for mode in DeghostMode::ALL {
            let fused = fuse_with_options(
                &[&src],
                &[0.0],
                &HdrOptions { deghost: mode, ..HdrOptions::default() },
            )
            .unwrap();
            assert_eq!(fused.rgb, src.levels[0].data);
        }
    }

    #[test]
    fn auto_deghost_leaves_a_static_scene_byte_identical() {
        let dark = frame(16, 16, |x, _| [0.05 + x as f32 * 0.01; 3]);
        let middle = frame(16, 16, |x, _| [0.1 + x as f32 * 0.02; 3]);
        let bright = frame(16, 16, |x, _| [0.2 + x as f32 * 0.04; 3]);
        let frames = [&dark, &middle, &bright];
        let stops = [-1.0, 0.0, 1.0];
        let legacy = fuse_with(&frames, &stops, false).unwrap();
        let auto = fuse_with_options(&frames, &stops, &HdrOptions::default()).unwrap();
        assert_eq!(legacy.rgb, auto.rgb);
    }
}
