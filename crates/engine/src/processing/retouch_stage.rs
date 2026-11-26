//! Capture One Retouch Faces decision logic.
//!
//! The face ordering, impact scaling and per-slider mask gating moved here
//! byte-for-byte from the CPU renderer. Only the mask resampling and the union
//! helper travel with it; the pixel operators stay in `crate::retouch_c1` and
//! the renderer keeps its local copies until the integration lane switches to
//! these paths.


use rayon::prelude::*;

use crate::processing::frame_map::FrameMap;
use crate::{AdjustmentSet, NormRect};

/// Resamples one retouch mask into an `out_w x out_h` coverage plane using the
/// same oriented sample points as the layer masks.
pub(crate) fn retouch_plane(
    mask: &crate::RasterMask,
    map: &FrameMap,
    region: NormRect,
    width: usize,
    height: usize,
    xs: &[f32],
) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let fy = region.y + (y as f32 + 0.5) / height as f32 * region.height;
        for (x, &fx) in xs.iter().enumerate() {
            let (p, _) = map.map([fx, fy]);
            row[x] = (mask.sample(p) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    });
    out
}

/// Element-wise union of two coverage planes, used for the `Both` eye sliders.
pub(crate) fn union_plane(a: &Option<Vec<u8>>, b: &Option<Vec<u8>>) -> Option<Vec<u8>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.iter().zip(b.iter()).map(|(a, b)| (*a).max(*b)).collect()),
        (Some(a), None) => Some(a.clone()),
        (None, Some(b)) => Some(b.clone()),
        (None, None) => None,
    }
}

/// Capture One Retouch Faces, in the order of `BuildOperationChain`:
/// `SmoothSkin -> FaceSculpting -> TeethWhitening -> EyeBright`.
///
/// Each slider's amount is scaled by the face's `Impact` before the pixel op
/// runs, matching `TranslateToICRetouchingSettings` (`m' = alpha * m` is the
/// numerically equivalent form). A face with `Impact == 0` is skipped whole.
pub(crate) fn apply_retouch(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &AdjustmentSet,
    map: &FrameMap,
    region: NormRect,
    scale: f32,
) {
    if adj.retouch.is_empty() {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let xs: Vec<f32> =
        (0..w).map(|x| region.x + (x as f32 + 0.5) / w as f32 * region.width).collect();
    let (ow, oh) = map.oriented();
    for face in &adj.retouch {
        let impact = (face.impact / 100.0).clamp(0.0, 1.0);
        if impact <= 0.0 {
            continue;
        }
        let plane = |key: u8| {
            face.masks
                .get(&key)
                .map(|mask| retouch_plane(mask, map, region, w, h, &xs))
        };
        let amount = |value: f32| value * impact;
        if face.skin_amount != 0.0 || face.skin_texture != 0.0 {
            if let Some(mask) = plane(1) {
                let mut mask = mask;
                // C1's `IC_DetectBeautyMarks` returns mole rectangles on a
                // face-scaled canvas; Skin Evening must not smear them, so
                // they are carved out of the smoothing mask. Default off:
                // without `AURORA_BEAUTYMARKS_ONNX=1` nothing is detected and
                // every byte stays as it was.
                if crate::beautymarks::enabled() {
                    match crate::beautymarks::detect_marks_in(
                        rgba,
                        width,
                        height,
                        face.face,
                        crate::beautymarks::min_confidence(),
                    ) {
                        Ok(marks) if !marks.is_empty() => {
                            crate::beautymarks::protect_mask(
                                &mut mask,
                                width,
                                height,
                                &marks,
                                1.0,
                            );
                        }
                        Ok(_) => {}
                        Err(err) => log::warn!("beauty marks skipped: {err:#}"),
                    }
                }
                // `skin_evening` uses the face's long side to scale its blur
                // radius; a rectangle of that side keeps the size while
                // staying in the frame's orientation.
                let mut long = (face.face[2] * ow).max(face.face[3] * oh);
                if long <= 0.0 {
                    if let Some(skin) = face.masks.get(&1) {
                        long = ((skin.rect[2] - skin.rect[0]) * ow)
                            .max((skin.rect[3] - skin.rect[1]) * oh);
                    }
                }
                let long = (long * scale).max(1.0);
                let rect = [0.0, 0.0, long / width as f32, long / height as f32];
                crate::retouch_c1::skin_evening(
                    rgba,
                    width,
                    height,
                    &mask,
                    rect,
                    amount(face.skin_amount),
                    amount(face.skin_texture),
                );
            }
        }
        if face.sculpt != 0.0 {
            if let Some(mask) = plane(2) {
                crate::retouch_c1::face_sculpting(rgba, width, height, &mask, amount(face.sculpt));
            }
        }
        if face.teeth_brightness != 0.0 || face.teeth_desaturation != 0.0 {
            if let Some(mask) = plane(8) {
                crate::retouch_c1::teeth(
                    rgba,
                    width,
                    height,
                    &mask,
                    amount(face.teeth_brightness),
                    amount(face.teeth_desaturation),
                );
            }
        }
        let eyes = face.iris_left != 0.0 || face.iris_right != 0.0 || face.iris_both != 0.0;
        if eyes {
            let left = plane(4);
            let right = plane(6);
            if face.iris_both != 0.0 {
                if let Some(mask) = union_plane(&left, &right) {
                    crate::retouch_c1::iris(rgba, width, height, &mask, amount(face.iris_both));
                }
            }
            if face.iris_left != 0.0 {
                if let Some(mask) = &left {
                    crate::retouch_c1::iris(rgba, width, height, mask, amount(face.iris_left));
                }
            }
            if face.iris_right != 0.0 {
                if let Some(mask) = &right {
                    crate::retouch_c1::iris(rgba, width, height, mask, amount(face.iris_right));
                }
            }
        }
        let whites = face.sclera_left != 0.0 || face.sclera_right != 0.0 || face.sclera_both != 0.0;
        if whites {
            let left = plane(5);
            let right = plane(7);
            if face.sclera_both != 0.0 {
                if let Some(mask) = union_plane(&left, &right) {
                    crate::retouch_c1::sclera(rgba, width, height, &mask, amount(face.sclera_both));
                }
            }
            if face.sclera_left != 0.0 {
                if let Some(mask) = &left {
                    crate::retouch_c1::sclera(rgba, width, height, mask, amount(face.sclera_left));
                }
            }
            if face.sclera_right != 0.0 {
                if let Some(mask) = &right {
                    crate::retouch_c1::sclera(rgba, width, height, mask, amount(face.sclera_right));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Geometry, MaskOp, RasterMask, RetouchFace};

    fn full_mask(value: u8) -> RasterMask {
        RasterMask {
            op: MaskOp::Add,
            rect: [0.0, 0.0, 1.0, 1.0],
            width: 1,
            height: 1,
            data: vec![value],
        }
    }

    fn frame(w: u32, h: u32, v: u8) -> Vec<u8> {
        let mut out = vec![v; (w * h * 4) as usize];
        for px in out.chunks_exact_mut(4) {
            px[3] = 255;
        }
        out
    }

    fn xs(width: usize) -> Vec<f32> {
        (0..width).map(|x| (x as f32 + 0.5) / width as f32).collect()
    }

    /// A 1x1 full-value mask resamples to full coverage on the identity map.
    #[test]
    fn retouch_plane_resamples_full_coverage() {
        let map = FrameMap::new((4, 4), &Geometry::default());
        let plane = retouch_plane(&full_mask(255), &map, NormRect::FULL, 4, 4, &xs(4));
        assert_eq!(plane, vec![255u8; 16]);
    }

    /// A 50% mask rounds to the same byte the renderer produces today.
    #[test]
    fn retouch_plane_rounds_coverage_to_a_byte() {
        let map = FrameMap::new((2, 1), &Geometry::default());
        let plane = retouch_plane(&full_mask(128), &map, NormRect::FULL, 2, 1, &xs(2));
        assert_eq!(plane, vec![128u8; 2]);
    }

    /// The `Both` union is the element-wise maximum, with clones when only one
    /// side exists and `None` only when both sides are absent.
    #[test]
    fn union_plane_takes_the_elementwise_maximum() {
        let a = Some(vec![0u8, 128, 255]);
        let b = Some(vec![10u8, 50, 200]);
        assert_eq!(union_plane(&a, &b), Some(vec![10u8, 128, 255]));
        assert_eq!(union_plane(&a, &None), a);
        assert_eq!(union_plane(&None, &b), b);
        assert_eq!(union_plane(&None, &None), None);
    }

    /// No stored faces: the frame is returned untouched.
    #[test]
    fn empty_retouch_leaves_the_frame_untouched() {
        let (w, h) = (8u32, 8u32);
        let mut rgba = frame(w, h, 120);
        let before = rgba.clone();
        let map = FrameMap::new((w, h), &Geometry::default());
        apply_retouch(&mut rgba, w, h, &AdjustmentSet::default(), &map, NormRect::FULL, 1.0);
        assert_eq!(rgba, before);
    }

    /// Zero impact skips the whole face, and an active face without its mask
    /// entry skips the individual slider. Both leave every byte in place.
    #[test]
    fn zero_impact_and_missing_masks_skip_sliders() {
        let (w, h) = (8u32, 8u32);
        let map = FrameMap::new((w, h), &Geometry::default());

        let mut zero_impact = frame(w, h, 120);
        let before = zero_impact.clone();
        let mut adj = AdjustmentSet::default();
        adj.retouch.push(RetouchFace {
            impact: 0.0,
            skin_amount: 50.0,
            skin_texture: 10.0,
            sculpt: 50.0,
            teeth_brightness: 50.0,
            iris_both: 50.0,
            sclera_both: 50.0,
            ..Default::default()
        });
        apply_retouch(&mut zero_impact, w, h, &adj, &map, NormRect::FULL, 1.0);
        assert_eq!(zero_impact, before);

        let mut missing_mask = frame(w, h, 120);
        let before = missing_mask.clone();
        let mut adj = AdjustmentSet::default();
        adj.retouch.push(RetouchFace {
            impact: 100.0,
            skin_amount: 50.0,
            sculpt: 50.0,
            teeth_brightness: 50.0,
            iris_both: 50.0,
            sclera_both: 50.0,
            ..Default::default()
        });
        apply_retouch(&mut missing_mask, w, h, &adj, &map, NormRect::FULL, 1.0);
        assert_eq!(missing_mask, before);
    }
}
