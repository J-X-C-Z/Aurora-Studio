//! Pure crop/aspect geometry shared by the developed-frame map and spot removal.
//!
//! These helpers take only [`Geometry`], the source size and plain values; they
//! carry no renderer state or I/O, so they live in `processing`. The legacy
//! `mock` geometry module re-exports them, which keeps every existing caller
//! (its own frame map and the `mock` process module) unchanged.

use crate::{AspectRatio, Geometry, NormRect};

pub(crate) const EPS: f32 = 1e-6;

pub(crate) fn finite(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

pub(crate) fn clean_rect(g: &Geometry) -> NormRect {
    let r = g.crop;
    let (width, height) = (finite(r.width), finite(r.height));
    if g.crop_outside_image {
        // `ZCROPOUTSIDEIMAGE`: C1 allows the crop rectangle to reach past the
        // image, so only non-finite values are repaired here.
        return NormRect { x: finite(r.x), y: finite(r.y), width, height };
    }
    let (width, height) = (width.clamp(0.0, 1.0), height.clamp(0.0, 1.0));
    NormRect {
        x: finite(r.x).clamp(0.0, 1.0 - width),
        y: finite(r.y).clamp(0.0, 1.0 - height),
        width,
        height,
    }
}

/// Oriented image size: a quarter turn of `rotation` or an odd `orientation` swaps the axes.
pub(crate) fn oriented_size(g: &Geometry, source_size: (u32, u32)) -> (f32, f32) {
    let (w, h) = (source_size.0 as f32, source_size.1 as f32);
    let quarter_turn = (g.orientation % 2 == 1) ^ is_quarter_turn(g.rotation);
    if quarter_turn {
        (h, w)
    } else {
        (w, h)
    }
}

pub(crate) fn is_quarter_turn(rotation_deg: f32) -> bool {
    let turns = finite(rotation_deg) / 90.0;
    (turns - turns.round()).abs() < 1e-4 && turns.round() as i64 % 2 != 0
}

/// `ZCANVASSIZE` when the crop was stored against it, otherwise the oriented
/// image. A full-frame crop keeps the image size, so an uncropped variant
/// renders exactly as it did before the canvas was carried.
pub(crate) fn canvas_size(g: &Geometry, source_size: (u32, u32)) -> (f32, f32) {
    let (ow, oh) = oriented_size(g, source_size);
    let (cw, ch) = (finite(g.canvas[0]), finite(g.canvas[1]));
    if cw > EPS && ch > EPS && g.crop != NormRect::FULL {
        (cw, ch)
    } else {
        (ow, oh)
    }
}

/// The crop rect implied by an aspect ratio + portrait flag, centred in the oriented frame.
pub(crate) fn crop_for_aspect(g: &Geometry, source_size: (u32, u32)) -> NormRect {
    let c = clean_rect(g);
    let target = match g.aspect {
        AspectRatio::Unconstrained | AspectRatio::Original => return c,
        other => {
            let (ow, oh) = oriented_size(g, source_size);
            match other.ratio(ow / oh.max(EPS)) {
                Some(r) if r.is_finite() && r > EPS => {
                    if g.portrait {
                        1.0 / r
                    } else {
                        r
                    }
                }
                _ => return c,
            }
        }
    };
    // The rect is in `ZCROP`'s pixel frame, so the current ratio and the
    // resized rect are measured there.
    let (ow, oh) = canvas_size(g, source_size);
    let (pw, ph) = (ow * c.width, oh * c.height);
    if pw <= EPS || ph <= EPS {
        return c;
    }
    let current = pw / ph;
    if (current - target).abs() < 1e-4 {
        return c;
    }
    let (w, h) = if current > target {
        (c.width * target / current, c.height)
    } else {
        (c.width, c.height * current / target)
    };
    let (cx, cy) = (c.x + c.width * 0.5, c.y + c.height * 0.5);
    NormRect {
        x: (cx - w * 0.5).clamp(0.0, 1.0 - w),
        y: (cy - h * 0.5).clamp(0.0, 1.0 - h),
        width: w,
        height: h,
    }
}
