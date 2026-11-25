//! Output-region sizing helpers shared by the preview and export paths.
//!
//! These bodies moved byte-for-byte out of the CPU renderer; the renderer keeps
//! its local copies until the integration lane switches to these paths. The
//! module is intentionally free of renderer state and pixel buffers.


use crate::NormRect;

/// Keeps a region inside the frame: the four corners are baked into the pixel
/// walks, so an out-of-range request would otherwise size the output up and
/// paint the empty border as the neutral fill.
pub(crate) fn clean_region(r: NormRect) -> NormRect {
    let width = if r.width.is_finite() { r.width.clamp(0.0, 1.0) } else { 1.0 };
    let height = if r.height.is_finite() { r.height.clamp(0.0, 1.0) } else { 1.0 };
    NormRect {
        x: if r.x.is_finite() { r.x.clamp(0.0, 1.0 - width) } else { 0.0 },
        y: if r.y.is_finite() { r.y.clamp(0.0, 1.0 - height) } else { 0.0 },
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, width: f32, height: f32) -> NormRect {
        NormRect { x, y, width, height }
    }

    /// The identity region stays byte-identical.
    #[test]
    fn full_frame_is_unchanged() {
        assert_eq!(clean_region(NormRect::FULL), NormRect::FULL);
    }

    /// A negative origin clamps to zero; an origin past the frame edge clamps
    /// to the largest origin that still leaves the extent inside the frame.
    #[test]
    fn origin_clamps_to_the_unit_square() {
        assert_eq!(clean_region(rect(-0.5, 2.0, 0.25, 0.5)), rect(0.0, 0.5, 0.25, 0.5));
    }

    /// An extent larger than the frame clamps to one and pins the origin to zero.
    #[test]
    fn oversized_extent_clamps_to_the_unit_square() {
        assert_eq!(clean_region(rect(0.4, 0.4, 3.0, 2.0)), rect(0.0, 0.0, 1.0, 1.0));
    }

    /// Non-finite extents become the full frame; non-finite origins become zero.
    #[test]
    fn non_finite_fields_fall_back_to_full_frame() {
        let out = clean_region(rect(f32::NAN, f32::INFINITY, f32::NAN, f32::INFINITY));
        assert_eq!(out, rect(0.0, 0.0, 1.0, 1.0));
    }

    /// A zero-width or zero-height region is preserved rather than repaired.
    #[test]
    fn degenerate_extent_is_preserved() {
        assert_eq!(clean_region(rect(0.5, 0.5, 0.0, 0.0)), rect(0.5, 0.5, 0.0, 0.0));
    }
}
