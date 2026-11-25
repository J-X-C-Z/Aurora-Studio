//! Layer mask-weight normalisation.
//!
//! `luma_range_weight` moved here byte-for-byte from the CPU renderer. The
//! sibling `layer_mask_weight` still takes the mask-grid handle owned by the
//! planning layer, so it cannot move without changing that layer's boundary;
//! see the G2 handoff note.


use crate::LumaRange;
use crate::processing::color::sstep;

pub(crate) fn luma_range_weight(r: &LumaRange, l: f32) -> f32 {
    let w = if l < r.low_inner {
        sstep(r.low_outer, r.low_inner.max(r.low_outer + 1e-3), l)
    } else if l > r.high_inner {
        1.0 - sstep(r.high_inner, r.high_outer.max(r.high_inner + 1e-3), l)
    } else {
        1.0
    };
    if r.invert { 1.0 - w } else { w }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(low_outer: f32, low_inner: f32, high_inner: f32, high_outer: f32) -> LumaRange {
        LumaRange {
            low_outer,
            low_inner,
            high_inner,
            high_outer,
            ..Default::default()
        }
    }

    /// Pins the smoothstep ramps at each end and the flat interior exactly as
    /// the renderer's current implementation produces them.
    #[test]
    fn band_ramps_at_both_ends_and_is_flat_inside() {
        let r = window(0.0, 0.2, 0.8, 1.0);
        assert_eq!(luma_range_weight(&r, 0.0), 0.0);
        assert!((luma_range_weight(&r, 0.1) - 0.5).abs() < 1e-6);
        assert_eq!(luma_range_weight(&r, 0.2), 1.0);
        assert_eq!(luma_range_weight(&r, 0.5), 1.0);
        assert_eq!(luma_range_weight(&r, 0.8), 1.0);
        assert!((luma_range_weight(&r, 0.9) - 0.5).abs() < 1e-6);
        assert_eq!(luma_range_weight(&r, 1.0), 0.0);
    }

    /// Inverting subtracts the band from one.
    #[test]
    fn invert_mirrors_the_band() {
        let r = LumaRange { invert: true, ..window(0.0, 0.2, 0.8, 1.0) };
        assert_eq!(luma_range_weight(&r, 0.5), 0.0);
        assert!((luma_range_weight(&r, 0.1) - 0.5).abs() < 1e-6);
        assert_eq!(luma_range_weight(&r, 1.0), 1.0);
    }

    /// A degenerate outer edge is expanded by 1e-3 so the ramp has width; the
    /// inner boundary still cuts hard.
    #[test]
    fn degenerate_outer_edge_still_ramps() {
        let r = window(0.0, 0.0, 0.5, 0.5);
        assert_eq!(luma_range_weight(&r, 0.5), 1.0);
        assert_eq!(luma_range_weight(&r, 0.6), 0.0);
    }
}
