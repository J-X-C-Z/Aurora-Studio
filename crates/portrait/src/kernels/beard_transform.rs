//! PF21026 post-delta transform: `cv::MatOp_AddEx` evaluation.
//!
//! Chain, all static:
//! `0xC59C4E8` is an 8-byte BSS slot filled by the CRT initializer `0x25E33E0`
//! with `0x8FA6D88`; that vtable is `.?AVMatOp_AddEx@cv@@` and
//! `[vtable+0x10] = 0x25DD230` (`cv::MatOp_AddEx::assign`).
//!
//! The expression built at `0x25D8EA0 -> 0x25D3870` has an empty `b` and `c`
//! (`+0x70`/`+0xd0` copies of empty `cv::Mat`), so `assign` reduces to
//! `dst = alpha * a + gamma` (`a.convertTo(dst, type, alpha, gamma)`).
//! This is a numeric value scale/offset, not ROI, affine, LUT or layout.
//! The final store is `cv::Mat::convertTo`/`copyTo`; `_type` decides whether a
//! temporary is used.
//!
//! Candidate module for integration. It lives in the r3c lane because the
//! write set forbids touching `crates/`.

use crate::{Image, Refusal};

pub const SLOT_RVA: u32 = 0x00C5_9C4E8;
pub const VTABLE_RVA: u32 = 0x08FA_6D88;
pub const ASSIGN_RVA: u32 = 0x025D_D230;
pub const BUILDER_RVA: u32 = 0x025D_8EA0;
pub const CONSTRUCTOR_RVA: u32 = 0x025D_3870;
pub const VTABLE_NAME: &str = ".?AVMatOp_AddEx@cv@@";

/// `dst = alpha * src + gamma`, the empty-`b`/empty-`c` AddEx path.
pub fn evaluate_add_ex(src: &[f32], alpha: f32, gamma: f32) -> Result<Vec<f32>, Refusal> {
    if !alpha.is_finite() || !gamma.is_finite() {
        return Err(Refusal::ValueOutOfRange("alpha/gamma"));
    }
    if src.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("src"));
    }
    Ok(src.iter().map(|value| alpha * value + gamma).collect())
}

/// `cv::convertTo(CV_8U)` store: `saturate_cast<uchar>(base + alpha*delta +
/// gamma)`, rounding half to even like `cvRound`. Alpha is copied from `base`.
pub fn store_to_base(
    base: &Image,
    delta: &[f32],
    alpha: f32,
    gamma: f32,
) -> Result<Image, Refusal> {
    let pixels = (base.width as usize)
        .checked_mul(base.height as usize)
        .ok_or(Refusal::ShapeMismatch("image"))?;
    base.validate()?;
    if delta.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("delta"));
    }
    let scaled = evaluate_add_ex(delta, alpha, gamma)?;

    let mut out = base.clone();
    for (index, value) in scaled.iter().enumerate() {
        let p = index / 3;
        let c = index % 3;
        let result = base.pixels[p * 4 + c] as f32 + value;
        out.pixels[p * 4 + c] = if result <= 0.0 {
            0
        } else if result >= 255.0 {
            255
        } else {
            result.round_ties_even() as u8
        };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addex_scales_then_offsets_and_saturates_on_store() {
        assert_eq!(evaluate_add_ex(&[1.0, -2.0], 0.5, 0.25).unwrap(), [0.75, -0.75]);
        let base = Image::solid(1, 1, [10, 200, 128, 77]);
        let odd = Image::solid(1, 1, [10, 200, 127, 77]);
        // Half-even ties land on the even neighbour, as cvRound does.
        assert_eq!(store_to_base(&base, &[20.0, -400.0, 0.5], 1.0, 0.0).unwrap().pixel(0, 0),
            [30, 0, 128, 77]);
        assert_eq!(store_to_base(&odd, &[20.0, -400.0, 0.5], 1.0, 0.0).unwrap().pixel(0, 0),
            [30, 0, 128, 77]);
        assert_eq!(evaluate_add_ex(&[f32::NAN], 1.0, 0.0), Err(Refusal::ValueOutOfRange("src")));
    }

    #[test]
    fn malformed_base_and_delta_are_refused() {
        let base = Image { width: 1, height: 1, pixels: vec![0; 3] };
        assert_eq!(store_to_base(&base, &[0.0; 3], 1.0, 0.0), Err(Refusal::ShapeMismatch("image")));
        assert_eq!(store_to_base(&Image::solid(1, 1, [0; 4]), &[0.0; 2], 1.0, 0.0),
            Err(Refusal::ShapeMismatch("delta")));
    }
}
