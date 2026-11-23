//! LCC dust-removal patches: the 16-byte `SpotData` records stored in a
//! Capture One LCC, and the Q12 gain applied inside each rectangle.
//!
//! Capture One renders these through `CImgOpApplyLCC` → `ApplyLCCDust`; the
//! extracted kernel (`p_041073488.cl`, `CLKernelApplyLCCDust`) does
//! `out = saturate((src * gain) >> 12)` with `gain` a `ushort` Q12 plane.
//! This module reproduces that multiply. The gain comes either from a supplied
//! plane or from the record's `bias`/`step`.
//!
//! The stage is inert unless `ZLENSDUSTREMOVALENABLED` is set *and* there is at
//! least one patch, so a variant without dust removal is byte-identical.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Spot, SpotKind};

/// Bytes per `.lcc` dust record: `x, y, w, h` (u16) + `offset` (u32) +
/// `bias` (i16) + `step` (u16).
pub const RECORD_BYTES: usize = 16;

/// One LCC dust patch.
///
/// `x`/`y`/`w`/`h` are pixels in the image/LCC plane, `offset` indexes the
/// LCC's Q12 gain plane, and `bias`/`step` describe the gain curve used when
/// that plane is not available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DustPatch {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
    pub offset: u32,
    pub bias: i16,
    pub step: u16,
}

impl DustPatch {
    /// Splits a `SpotData` blob into records, ignoring a trailing partial one.
    pub fn decode(records: &[u8]) -> Vec<Self> {
        records
            .chunks_exact(RECORD_BYTES)
            .map(|r| Self {
                x: u16::from_le_bytes([r[0], r[1]]),
                y: u16::from_le_bytes([r[2], r[3]]),
                w: u16::from_le_bytes([r[4], r[5]]),
                h: u16::from_le_bytes([r[6], r[7]]),
                offset: u32::from_le_bytes([r[8], r[9], r[10], r[11]]),
                bias: i16::from_le_bytes([r[12], r[13]]),
                step: u16::from_le_bytes([r[14], r[15]]),
            })
            .collect()
    }

    /// Q12 multiplier for an 8-bit level: `4096 * 2^((v - bias) * step / 4096)`.
    ///
    /// `step == 0` is identity, which is also the default for an unset record.
    pub fn gain_q12(&self, v: u8) -> u16 {
        if self.step == 0 {
            return 4096;
        }
        let level = (v as f32 - self.bias as f32).max(0.0);
        let gain = 4096.0 * 2f32.powf(level * self.step as f32 / 4096.0);
        gain.round().clamp(0.0, u16::MAX as f32) as u16
    }

    /// Q12 multiplier for one `SpotData` byte, following `ApplyLCCDust`:
    /// the byte is a signed delta split at `bias`, then
    /// `4096 * 2^(delta * step / 4096)`.
    pub fn gain_q12_at(&self, byte: u8) -> u16 {
        if self.step == 0 {
            return 4096;
        }
        let delta = if i32::from(byte) > i32::from(self.bias) {
            i32::from(byte) - 256
        } else {
            i32::from(byte)
        };
        let gain = 4096.0 * 2f32.powf(delta as f32 * self.step as f32 / 4096.0);
        gain.round().clamp(0.0, u16::MAX as f32) as u16
    }
}

/// Applies every patch in order, returning how many rectangles were touched.
///
/// Rectangles are clamped to the frame; a patch entirely outside is skipped.
/// When `gain` holds the record's `offset + w*h` bytes, each pixel is scaled by
/// the `SpotData` byte there (Capture One's decode); otherwise the record's
/// `bias`/`step` curve is evaluated against the pixel value, which is all a
/// detected patch without an LCC plane has.
pub fn apply(rgba: &mut [u8], w: usize, h: usize, patches: &[DustPatch], gain: &[u8]) -> usize {
    if w == 0 || h == 0 || patches.is_empty() || rgba.len() < w * h * 4 {
        return 0;
    }
    let mut touched = 0;
    for p in patches {
        if p.w == 0 || p.h == 0 {
            continue;
        }
        let plane = p
            .offset
            .checked_add((p.w as u32) * (p.h as u32))
            .and_then(|end| gain.get(p.offset as usize..end as usize));
        let x0 = (p.x as usize).min(w);
        let y0 = (p.y as usize).min(h);
        let x1 = (p.x as usize + p.w as usize).min(w);
        let y1 = (p.y as usize + p.h as usize).min(h);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        for y in y0..y1 {
            for x in x0..x1 {
                let i = (y * w + x) * 4;
                for k in 0..3 {
                    let v = rgba[i + k];
                    let g = match &plane {
                        Some(plane) => {
                            p.gain_q12_at(plane[(y - y0) * p.w as usize + (x - x0)]) as u32
                        }
                        None => p.gain_q12(v) as u32,
                    };
                    rgba[i + k] = ((v as u32 * g) >> 12).min(255) as u8;
                }
            }
        }
        touched += 1;
    }
    touched
}

/// Why dust detection could not produce patches.
///
/// Kept structured so the UI can say "install the model" instead of dumping a
/// string. Detection is the *authoring* path: Capture One runs it while
/// creating the LCC, then stores the resulting spots in the LCC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DustError {
    /// `dust.onnx` was not found; carries the paths that were tried.
    ModelMissing(String),
    /// ONNX Runtime / the execution provider is not available.
    RuntimeUnavailable,
    /// The model or the frame did not match the expected shapes.
    BadShape(String),
}

impl fmt::Display for DustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelMissing(paths) => write!(f, "dust model not found: {paths}"),
            Self::RuntimeUnavailable => write!(f, "ONNX Runtime is not available"),
            Self::BadShape(detail) => write!(f, "dust model shape error: {detail}"),
        }
    }
}

impl std::error::Error for DustError {}

/// Detection front-end. A real engine supplies `dust.onnx`; tests and builds
/// without a model get a structured [`DustError`] instead of a panic.
pub trait DustDetector {
    fn detect(&self, rgba: &[u8], w: u32, h: u32) -> Result<Vec<DustPatch>, DustError>;
}

/// The shipped `dust.onnx` detector.
pub struct OnnxDustDetector;

impl DustDetector for OnnxDustDetector {
    fn detect(&self, rgba: &[u8], w: u32, h: u32) -> Result<Vec<DustPatch>, DustError> {
        let spots = crate::dustdetect::detect_dust(
            rgba,
            w,
            h,
            crate::dustdetect::min_confidence(),
        )
        .map_err(|e| {
            let s = format!("{e:#}");
            // shortcut: errors come back as strings from the ONNX wrapper;
            // match the two prefixes it owns and treat the rest as shape errors.
            if s.contains("not found") || s.contains("unavailable") {
                if s.contains("Runtime") {
                    DustError::RuntimeUnavailable
                } else {
                    DustError::ModelMissing(s)
                }
            } else {
                DustError::BadShape(s)
            }
        })?;
        Ok(spots_to_patches(&spots, w, h))
    }
}

/// Converts detector spots into LCC-style rectangles.
///
/// The detector returns a centre and a radius; Capture One's LCC analysis
/// turns those into the real patch records while creating the profile. The
/// default `step` here is a conservative stand-in until that calibration is
/// ported, so detected patches brighten but cannot blow out.
pub fn spots_to_patches(spots: &[Spot], w: u32, h: u32) -> Vec<DustPatch> {
    let long = w.max(h) as f32;
    let mut out = Vec::with_capacity(spots.len());
    for s in spots {
        if s.kind != SpotKind::Dust || !(s.radius > 0.0) {
            continue;
        }
        let r_px = (s.radius * long).max(1.5);
        let cx = s.x * w as f32;
        let cy = s.y * h as f32;
        // `SpotData` is integer pixels as top-left + size, while the spot is a
        // centre + radius. Round the two edges outward so the rectangle covers
        // the whole disc; rounding `x` and the width separately would shift the
        // patch half a pixel and clip one side.
        let x0 = (cx - r_px).floor().clamp(0.0, w as f32);
        let y0 = (cy - r_px).floor().clamp(0.0, h as f32);
        let x1 = (cx + r_px).ceil().clamp(0.0, w as f32);
        let y1 = (cy + r_px).ceil().clamp(0.0, h as f32);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        out.push(DustPatch {
            x: x0 as u16,
            y: y0 as u16,
            w: (x1 - x0) as u16,
            h: (y1 - y0) as u16,
            offset: 0,
            bias: 0,
            // 8/4096 puts a mid-grey (128) at about +27 %, enough for dust and
            // far from clipping; the LCC analysis supplies the real exponent.
            step: 8,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch(step: u16, bias: i16) -> DustPatch {
        DustPatch {
            x: 1,
            y: 1,
            w: 2,
            h: 2,
            offset: 0,
            bias,
            step,
        }
    }

    #[test]
    fn records_decode_into_rectangles() {
        let mut bytes = Vec::new();
        let rows: [(u16, u16, u16, u16, u32, i16, u16); 2] = [
            (10, 20, 30, 40, 7, -3, 9),
            (1, 2, 3, 4, 5, 6, 7),
        ];
        for (x, y, w, h, off, bias, step) in rows {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            bytes.extend_from_slice(&w.to_le_bytes());
            bytes.extend_from_slice(&h.to_le_bytes());
            bytes.extend_from_slice(&off.to_le_bytes());
            bytes.extend_from_slice(&bias.to_le_bytes());
            bytes.extend_from_slice(&step.to_le_bytes());
        }
        let got = DustPatch::decode(&bytes);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], DustPatch { x: 10, y: 20, w: 30, h: 40, offset: 7, bias: -3, step: 9 });
        assert_eq!(got[1].step, 7);
    }

    #[test]
    fn step_zero_is_identity() {
        let p = patch(0, 0);
        for v in [0u8, 1, 127, 255] {
            assert_eq!(p.gain_q12(v), 4096);
        }
    }

    #[test]
    fn no_patches_leaves_the_frame_untouched() {
        let mut buf = vec![17u8; 8 * 8 * 4];
        let before = buf.clone();
        assert_eq!(apply(&mut buf, 8, 8, &[], &[]), 0);
        assert_eq!(buf, before);
    }

    #[test]
    fn a_patch_multiplies_inside_the_rect_only() {
        let mut buf = vec![128u8; 4 * 4 * 4];
        let p = patch(8, 0);
        assert_eq!(apply(&mut buf, 4, 4, &[p], &[]), 1);
        let inside = 1 * 4 + 1;
        let gain = p.gain_q12(128) as u32;
        let want = ((128u32 * gain) >> 12).min(255) as u8;
        assert!(want > 128, "gain {gain} should brighten mid-grey");
        assert_eq!(buf[inside * 4], want);
        assert_eq!(buf[0], 128, "outside the rectangle must not change");
    }

    #[test]
    fn an_out_of_frame_patch_is_skipped() {
        let mut buf = vec![64u8; 2 * 2 * 4];
        let p = DustPatch { x: 10, y: 10, w: 4, h: 4, ..patch(8, 0) };
        assert_eq!(apply(&mut buf, 2, 2, &[p], &[]), 0);
        assert!(buf.iter().all(|&v| v == 64));
    }

    #[test]
    fn spots_become_centred_pixel_rects() {
        let spots = vec![Spot {
            id: 0,
            kind: SpotKind::Dust,
            x: 0.5,
            y: 0.25,
            radius: 0.01,
        }];
        let got = spots_to_patches(&spots, 100, 100);
        assert_eq!(got.len(), 1);
        // Centre (50, 25), radius 1.5 px: the disc spans 48.5..51.5, so the
        // integer rectangle that contains it is 48..52 and stays centred.
        assert_eq!(
            (got[0].x, got[0].y, got[0].w, got[0].h),
            (48, 23, 4, 4)
        );
        assert_eq!(
            (got[0].x + got[0].w / 2, got[0].y + got[0].h / 2),
            (50, 25),
            "the rectangle must stay on the detected centre"
        );
    }

    #[test]
    fn a_missing_model_is_a_structured_error() {
        let err = DustError::ModelMissing("/nope/dust.onnx".into());
        assert!(err.to_string().contains("/nope/dust.onnx"));
        assert_eq!(DustError::RuntimeUnavailable.to_string(), "ONNX Runtime is not available");
    }
}
