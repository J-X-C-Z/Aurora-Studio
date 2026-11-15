//! PF100007 direct pixel displacement contract for the Apple Metal host.
//!
//! The CPU oracle stays `kernels::smile_warp::pf100007_apply`. This module owns
//! only the 32-byte kernel ABI, the shared input validation and the MSL source;
//! `metal::MetalContext::smile` performs the lazy compile and dispatch.

use crate::kernels::smile_warp::Rect;
use crate::Refusal;

pub const SMILE_MSL: &str = include_str!("../native/metal/smile_warp.metal");
pub const SMILE_PARAMS_BYTES: usize = 32;

/// Exact 32-byte mirror of `SmileParams` in `smile_warp.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmileParams {
    pub width: u32,
    pub height: u32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub degree: f32,
    pub flags: u32,
}

const _: () = assert!(std::mem::size_of::<SmileParams>() == SMILE_PARAMS_BYTES);

#[derive(Clone, Copy, Debug)]
pub struct SmileInput<'a> {
    pub source_bgra: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub degree: f32,
    pub lip_mask: &'a [u8],
    pub mouth_mask: &'a [u8],
    pub model_dx: &'a [f32],
    pub model_dy: &'a [f32],
    pub roi: Rect,
}

impl SmileInput<'_> {
    /// Mirrors the CPU entry checks, including zero-size/multiplication
    /// overflow (`ShapeMismatch("pf100007 bgra")`), ROI subtraction bounds,
    /// degree range and finite displacement planes.
    pub fn validate(&self) -> Result<usize, Refusal> {
        if self.width == 0 || self.height == 0 {
            return Err(Refusal::ShapeMismatch("pf100007 bgra"));
        }
        let pixels = (self.width as usize)
            .checked_mul(self.height as usize)
            .ok_or(Refusal::ShapeMismatch("pf100007 bgra"))?;
        let bytes = pixels
            .checked_mul(4)
            .ok_or(Refusal::ShapeMismatch("pf100007 bgra"))?;
        if self.source_bgra.len() != bytes {
            return Err(Refusal::ShapeMismatch("pf100007 bgra"));
        }
        for (plane, name) in [
            (self.lip_mask.len(), "pf100007 lip_mask"),
            (self.mouth_mask.len(), "pf100007 mouth_mask"),
            (self.model_dx.len(), "pf100007 model_dx"),
            (self.model_dy.len(), "pf100007 model_dy"),
        ] {
            if plane != pixels {
                return Err(Refusal::MissingInput(name));
            }
        }
        if !self.degree.is_finite() || !(0.0..=1.0).contains(&self.degree) {
            return Err(Refusal::ValueOutOfRange("pf100007 degree"));
        }
        if self.roi.x > self.width
            || self.roi.width > self.width - self.roi.x
            || self.roi.y > self.height
            || self.roi.height > self.height - self.roi.y
        {
            return Err(Refusal::ValueOutOfRange("pf100007 roi"));
        }
        if !self
            .model_dx
            .iter()
            .chain(self.model_dy)
            .all(|v| v.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("pf100007 displacement"));
        }
        Ok(pixels)
    }

    pub fn params(&self) -> Result<SmileParams, Refusal> {
        self.validate()?;
        Ok(SmileParams {
            width: self.width,
            height: self.height,
            roi_x: self.roi.x,
            roi_y: self.roi.y,
            roi_width: self.roi.width,
            roi_height: self.roi.height,
            degree: self.degree,
            flags: 0,
        })
    }
}
