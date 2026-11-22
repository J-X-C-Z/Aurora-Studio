//! Geometry-only keystone suggestion.
//!
//! Sobel edges are split into near-vertical and near-horizontal sets, then
//! each set is split into the two halves of the frame. Parallel lines cancel
//! out; a converging pair leaves half the convergence in each half. The result
//! is only a suggestion: nothing here writes adjustments.

/// Suggested keystone correction in degrees, matching [`crate::Keystone`]'s
/// vertical/horizontal controls.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Suggestion {
    pub vertical: f32,
    pub horizontal: f32,
}

/// Working resolution of the detector; the full frame is sampled at this side.
const MAX_SIDE: usize = 720;
/// Edge pixels that are not clearly vertical nor horizontal are ignored.
const AXIS_RATIO: f32 = 1.5;
/// Edges this far off their axis are probably content, not perspective.
const MAX_TILT: f32 = 25.0;
/// Fewer supporting pixels than this in one half means "no line found".
const MIN_GROUP: u32 = 24;
/// Absolute clamp, wider than any camera keystone we correct.
const MAX_OUT: f32 = 45.0;
/// Hough angle resolution and offset resolution, in pixels.
const ANGLE_STEP: f32 = 0.5;
const OFFSET_BIN: f32 = 2.0;
const ANGLE_BINS: usize = (2.0 * MAX_TILT / ANGLE_STEP) as usize + 1;

/// Votes for `offset = x - tan(angle) * y` (vertical) or
/// `offset = y - tan(angle) * x` (horizontal). A long line fills one cell;
/// the short staircase steps of the other line set spread over many.
struct Hough {
    votes: Vec<u32>,
    base: f32,
    bins: usize,
}

impl Hough {
    fn new(side: usize) -> Self {
        let bins = (3 * side) / 2 + 2;
        Self {
            votes: vec![0; ANGLE_BINS * bins],
            base: side as f32,
            bins,
        }
    }

    fn vote(&mut self, angle_bin: usize, offset: f32) {
        let i = (((offset + self.base) / OFFSET_BIN) as isize).clamp(0, self.bins as isize - 1)
            as usize;
        self.votes[angle_bin * self.bins + i] += 1;
    }

    /// The dominant line, only when it is long enough to be a real edge
    /// rather than a handful of similarly tilted texture pixels.
    fn best(&self, pixels: u32) -> Option<f32> {
        let (angle, votes) = self.peak();
        (votes >= 12 && votes * 10 >= pixels).then_some(angle)
    }

    fn peak(&self) -> (f32, u32) {
        let (mut best, mut bin) = (0u32, 0usize);
        for (i, &v) in self.votes.iter().enumerate() {
            if v > best {
                best = v;
                bin = i / self.bins;
            }
        }
        (-MAX_TILT + bin as f32 * ANGLE_STEP, best)
    }
}

/// Detects half the vertical and horizontal convergence, in degrees.
/// A flat image, a frame without axis-aligned edges, or a bad buffer returns
/// zeros instead of guessing.
pub fn detect(rgba: &[u8], w: u32, h: u32) -> Suggestion {
    let mut zero = Suggestion::default();
    let (wu, hu) = (w as usize, h as usize);
    let Some(len) = wu.checked_mul(hu).and_then(|n| n.checked_mul(4)) else {
        return zero;
    };
    if wu < 8 || hu < 8 || rgba.len() < len {
        return zero;
    }
    let step = (wu.max(hu) / MAX_SIDE).max(1);
    let (sw, sh) = (wu / step, hu / step);
    if sw < 8 || sh < 8 {
        return zero;
    }

    let luma: Vec<f32> = (0..sh)
        .flat_map(|y| {
            (0..sw).map(move |x| {
                let i = ((y * step) * wu + x * step) * 4;
                0.299 * rgba[i] as f32 + 0.587 * rgba[i + 1] as f32 + 0.114 * rgba[i + 2] as f32
            })
        })
        .collect();

    let mut gx = vec![0.0f32; sw * sh];
    let mut gy = vec![0.0f32; sw * sh];
    let mut mag = vec![0.0f32; sw * sh];
    let mut max = 0.0f32;
    for y in 1..sh - 1 {
        for x in 1..sw - 1 {
            let at = |dx: usize, dy: usize| luma[(y - 1 + dy) * sw + x - 1 + dx];
            let sx = at(2, 0) + 2.0 * at(2, 1) + at(2, 2) - at(0, 0) - 2.0 * at(0, 1) - at(0, 2);
            let sy = at(0, 2) + 2.0 * at(1, 2) + at(2, 2) - at(0, 0) - 2.0 * at(1, 0) - at(2, 0);
            let i = y * sw + x;
            gx[i] = sx;
            gy[i] = sy;
            mag[i] = (sx * sx + sy * sy).sqrt();
            max = max.max(mag[i]);
        }
    }
    let threshold = (max * 0.2).max(24.0);
    if max < 24.0 {
        return zero;
    }

    let mut vertical = [Hough::new(sw), Hough::new(sw)];
    let mut horizontal = [Hough::new(sh), Hough::new(sh)];
    let mut vertical_pixels = [0u32; 2];
    let mut horizontal_pixels = [0u32; 2];
    let slopes: Vec<f32> = (0..ANGLE_BINS)
        .map(|i| (-MAX_TILT + i as f32 * ANGLE_STEP).to_radians().tan())
        .collect();
    for y in 1..sh - 1 {
        for x in 1..sw - 1 {
            let i = y * sw + x;
            let m = mag[i];
            if m < threshold {
                continue;
            }
            let (sx, sy) = (gx[i], gy[i]);
            // A pixel's own tangent must already point along the axis; this
            // drops the perpendicular staircase corners of the other line set.
            let (mut tx, mut ty) = (-sy, sx);
            if sx.abs() >= AXIS_RATIO * sy.abs() {
                if ty < 0.0 {
                    tx = -tx;
                    ty = -ty;
                }
                if tx.atan2(ty).to_degrees().abs() <= MAX_TILT {
                    let half = (x * 2 / sw).min(1);
                    vertical_pixels[half] += 1;
                    for (bin, slope) in slopes.iter().enumerate() {
                        vertical[half].vote(bin, x as f32 - slope * y as f32);
                    }
                }
            } else if sy.abs() >= AXIS_RATIO * sx.abs() {
                if tx < 0.0 {
                    tx = -tx;
                    ty = -ty;
                }
                if ty.atan2(tx).to_degrees().abs() <= MAX_TILT {
                    let half = (y * 2 / sh).min(1);
                    horizontal_pixels[half] += 1;
                    for (bin, slope) in slopes.iter().enumerate() {
                        horizontal[half].vote(bin, y as f32 - slope * x as f32);
                    }
                }
            }
        }
    }

    if vertical_pixels[0] >= MIN_GROUP && vertical_pixels[1] >= MIN_GROUP {
        if let (Some(left), Some(right)) = (
            vertical[0].best(vertical_pixels[0]),
            vertical[1].best(vertical_pixels[1]),
        ) {
            zero.vertical = ((right - left) * 0.5).clamp(-MAX_OUT, MAX_OUT);
        }
    }
    if horizontal_pixels[0] >= MIN_GROUP && horizontal_pixels[1] >= MIN_GROUP {
        if let (Some(top), Some(bottom)) = (
            horizontal[0].best(horizontal_pixels[0]),
            horizontal[1].best(horizontal_pixels[1]),
        ) {
            zero.horizontal = ((bottom - top) * 0.5).clamp(-MAX_OUT, MAX_OUT);
        }
    }
    zero
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canvas(w: u32, h: u32) -> Vec<u8> {
        vec![255u8; (w * h * 4) as usize]
    }

    fn line(img: &mut [u8], w: u32, h: u32, from: (f32, f32), to: (f32, f32)) {
        let steps = (to.0 - from.0).abs().max((to.1 - from.1).abs()).max(1.0) as i32;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = from.0 + (to.0 - from.0) * t;
            let y = from.1 + (to.1 - from.1) * t;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (px, py) = (x as i32 + dx, y as i32 + dy);
                    if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                        let o = (py as usize * w as usize + px as usize) * 4;
                        img[o..o + 3].copy_from_slice(&[0, 0, 0]);
                    }
                }
            }
        }
    }

    /// Two sides of a trapezoid: the half-convergence is the expected angle.
    #[test]
    fn convergent_verticals_report_half_angle() {
        let (w, h) = (512, 384);
        let mut img = canvas(w, h);
        line(
            &mut img,
            w,
            h,
            (0.40 * w as f32, 0.0),
            (0.25 * w as f32, h as f32 - 1.0),
        );
        line(
            &mut img,
            w,
            h,
            (0.60 * w as f32, 0.0),
            (0.75 * w as f32, h as f32 - 1.0),
        );
        let expected = ((0.15 * w as f32) / h as f32).atan().to_degrees();
        let got = detect(&img, w, h);
        assert!(
            (got.vertical - expected).abs() < 1.5,
            "vertical {} want {expected}",
            got.vertical
        );
        assert!(got.horizontal.abs() < 1e-3, "horizontal {}", got.horizontal);
    }

    #[test]
    fn convergent_horizontals_report_half_angle() {
        let (w, h) = (512, 384);
        let mut img = canvas(w, h);
        line(
            &mut img,
            w,
            h,
            (0.0, 0.40 * h as f32),
            (w as f32 - 1.0, 0.25 * h as f32),
        );
        line(
            &mut img,
            w,
            h,
            (0.0, 0.60 * h as f32),
            (w as f32 - 1.0, 0.75 * h as f32),
        );
        let expected = ((0.15 * h as f32) / w as f32).atan().to_degrees();
        let got = detect(&img, w, h);
        assert!(
            (got.horizontal - expected).abs() < 1.5,
            "horizontal {} want {expected}",
            got.horizontal
        );
        assert!(got.vertical.abs() < 1e-3, "vertical {}", got.vertical);
    }

    /// A global 3° twist cancels out of both convergence measures.
    #[test]
    fn parallel_lines_are_level() {
        let (w, h) = (512, 384);
        let mut img = canvas(w, h);
        let slope = 3.0f32.to_radians().tan();
        for k in 1..5 {
            let x = w as f32 * k as f32 / 5.0;
            line(
                &mut img,
                w,
                h,
                (x, 0.0),
                (x + h as f32 * slope, h as f32 - 1.0),
            );
        }
        for k in 1..4 {
            let y = h as f32 * k as f32 / 4.0;
            line(
                &mut img,
                w,
                h,
                (0.0, y),
                (w as f32 - 1.0, y + w as f32 * slope),
            );
        }
        let got = detect(&img, w, h);
        assert!(got.vertical.abs() < 0.5, "vertical {}", got.vertical);
        assert!(got.horizontal.abs() < 0.5, "horizontal {}", got.horizontal);
    }

    #[test]
    fn flat_image_returns_defaults() {
        let img = vec![128u8; 320 * 240 * 4];
        assert_eq!(detect(&img, 320, 240), Suggestion::default());
    }

    #[test]
    fn bad_sizes_are_safe() {
        assert_eq!(detect(&[], 0, 0), Suggestion::default());
        assert_eq!(detect(&[0; 16], 100, 100), Suggestion::default());
        assert_eq!(detect(&[0; 64], 4, 4), Suggestion::default());
        assert_eq!(detect(&[0; 64], u32::MAX, u32::MAX), Suggestion::default());
    }
}
