//! Geometry-only keystone suggestion.
//!
//! Sobel edges are split into near-vertical and near-horizontal sets, then
//! each set is split into the two halves of the frame. Parallel lines cancel
//! out; a converging pair leaves half the convergence in each half. The result
//! is only a suggestion: nothing here writes adjustments.

/// Suggested keystone correction in degrees. [`auto_keystone`] inverts the
/// renderer's real keystone map to turn them into [`crate::Keystone`] slider
/// units (see [`mapped_half_angles`]); there is no fixed inset scale any more.
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

/// Half-edge inset angles, in degrees, the real renderer produces for one
/// [`crate::Keystone`] on a `w`×`h` frame — the same quantity [`detect`]
/// measures.
///
/// The suggestion is absolute: the caller renders the frame with the current
/// keystone removed, so the detector sees the source lines directly. A
/// correction is right when the keystone map moves the *frame's own* edges onto
/// those lines, i.e. when the mapped half-angle equals the detected one. We
/// therefore push the four frame corners through the renderer's
/// `CResampler::UpdateGeo` basis ([`crate::mock::geometry::FrameMap`]) and
/// measure the resulting edge tilts in frame pixels. This keeps the conversion
/// tied to the actual basis instead of the pre-`a4b9d27` `0.35·tilt` trapezoid,
/// so it follows the renderer whenever the basis changes.
fn mapped_half_angles(k: &crate::Keystone, w: f32, h: f32) -> Suggestion {
    let (wu, hu) = (w.max(1.0) as u32, h.max(1.0) as u32);
    let geometry = crate::Geometry { keystone: k.clone(), ..Default::default() };
    let map = crate::mock::geometry::FrameMap::new((wu, hu), &geometry);
    let pixel = |f: [f32; 2]| {
        let s = map.map(f).1;
        [s[0] * w, s[1] * h]
    };
    let (tl, tr, bl, br) = (
        pixel([0.0, 0.0]),
        pixel([1.0, 0.0]),
        pixel([0.0, 1.0]),
        pixel([1.0, 1.0]),
    );
    // Vertical edges are measured off the vertical axis, horizontal edges off
    // the horizontal one — exactly `atan(dx/dy)` / `atan(dy/dx)` as in the
    // Hough vote, but in pixels so the frame aspect is honoured.
    let off_vertical = |a: [f32; 2], b: [f32; 2]| (b[0] - a[0]).atan2(b[1] - a[1]).to_degrees();
    let off_horizontal = |a: [f32; 2], b: [f32; 2]| (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees();
    Suggestion {
        vertical: 0.5 * (off_vertical(tr, br) - off_vertical(tl, bl)),
        horizontal: 0.5 * (off_horizontal(bl, br) - off_horizontal(tl, tr)),
    }
}

/// Bisection budget. The map is allocation-free and Auto Keystone is a one-shot
/// user action, so the extra iterations cost nothing measurable; ~30 would
/// already reach f32 precision, 64 is a safe ceiling.
const SOLVE_ITERS: u32 = 64;
/// Stop once the mapped half-angle reproduces the detected one this closely.
const SOLVE_TOL_DEG: f32 = 0.01;

/// The axis a value is solved for; the other slider stays at zero.
#[derive(Clone, Copy)]
enum Axis {
    Vertical,
    Horizontal,
}

/// Inverts [`mapped_half_angles`] for one axis: the `-100..=100` slider unit
/// whose mapped half-angle equals `target_deg`.
///
/// The mapped angle rises monotonically with the slider, so a plain bisection
/// converges. Beyond the renderer's reach (the map tops out below the detector's
/// `MAX_OUT` 45°) the result saturates at the slider limit, and the exact zero
/// angle stays bit-for-bit zero (the identity path).
fn solve_axis(axis: Axis, target_deg: f32, aspect: f32, w: f32, h: f32) -> f32 {
    if !target_deg.is_finite() || target_deg == 0.0 {
        return 0.0;
    }
    let measure = |value: f32| {
        let k = match axis {
            Axis::Vertical => crate::Keystone { vertical: value, amount: 100.0, aspect, ..Default::default() },
            Axis::Horizontal => crate::Keystone { horizontal: value, amount: 100.0, aspect, ..Default::default() },
        };
        let angles = mapped_half_angles(&k, w, h);
        match axis {
            Axis::Vertical => angles.vertical,
            Axis::Horizontal => angles.horizontal,
        }
    };
    let (mut lo, mut hi) = (-100.0f32, 100.0f32);
    let (low, high) = (measure(lo), measure(hi));
    // A non-finite map or a non-monotone pair would make the bisection lie;
    // fall back to the neutral value rather than guessing.
    if !low.is_finite() || !high.is_finite() || low > high {
        return 0.0;
    }
    if target_deg <= low {
        return lo;
    }
    if target_deg >= high {
        return hi;
    }
    for _ in 0..SOLVE_ITERS {
        let mid = 0.5 * (lo + hi);
        let angle = measure(mid);
        if !angle.is_finite() {
            return 0.0;
        }
        if (angle - target_deg).abs() <= SOLVE_TOL_DEG {
            return mid;
        }
        if angle < target_deg {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Full [`crate::Keystone`] for the Auto button: absolute (the caller renders
/// the frame with the current keystone removed) and at full amount. `aspect`
/// only passes the user's stretch compensation through, and is included in the
/// inversion so the solved sliders level the frame *as rendered*.
pub fn auto_keystone(aspect: f32, rgba: &[u8], w: u32, h: u32) -> crate::Keystone {
    let half = detect(rgba, w, h);
    let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
    crate::Keystone {
        vertical: solve_axis(Axis::Vertical, half.vertical, aspect, wf, hf),
        horizontal: solve_axis(Axis::Horizontal, half.horizontal, aspect, wf, hf),
        amount: 100.0,
        aspect,
        ..Default::default()
    }
}

/// Lightroom Upright-style geometry correction modes, combined with C1's
/// `amount (0..100)` strength slider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UprightMode {
    Auto,
    Level,
    Vertical,
    Full,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UprightSolution {
    pub rotation_deg: f32,
    pub keystone: crate::Keystone,
}

/// Computes an Upright correction from image edges according to `mode`.
pub fn upright_keystone(
    mode: UprightMode,
    aspect: f32,
    rgba: &[u8],
    w: u32,
    h: u32,
) -> UprightSolution {
    let half = detect(rgba, w, h);
    let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
    match mode {
        UprightMode::Level => UprightSolution {
            rotation_deg: (-half.horizontal).clamp(-15.0, 15.0),
            keystone: crate::Keystone {
                amount: 100.0,
                aspect,
                ..Default::default()
            },
        },
        UprightMode::Vertical => UprightSolution {
            rotation_deg: 0.0,
            keystone: crate::Keystone {
                vertical: solve_axis(Axis::Vertical, half.vertical, aspect, wf, hf),
                horizontal: 0.0,
                amount: 100.0,
                aspect,
                ..Default::default()
            },
        },
        UprightMode::Auto | UprightMode::Full => UprightSolution {
            rotation_deg: 0.0,
            keystone: crate::Keystone {
                vertical: solve_axis(Axis::Vertical, half.vertical, aspect, wf, hf),
                horizontal: solve_axis(Axis::Horizontal, half.horizontal, aspect, wf, hf),
                amount: 100.0,
                aspect,
                ..Default::default()
            },
        },
    }
}

/// Solves a [`crate::Keystone`] from 2–4 user-drawn normalized guide segments
/// `([x0, y0], [x1, y1])` in `0.0..=1.0` (Guided Upright).
pub fn solve_guided_upright(
    guides: &[([f32; 2], [f32; 2])],
    aspect: f32,
    w: u32,
    h: u32,
) -> crate::Keystone {
    let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
    let mut v_left = Vec::new();
    let mut v_right = Vec::new();
    let mut h_top = Vec::new();
    let mut h_bot = Vec::new();

    for &(p0, p1) in guides {
        let dx = (p1[0] - p0[0]) * wf;
        let dy = (p1[1] - p0[1]) * hf;
        if dx.hypot(dy) < 4.0 {
            continue;
        }
        let mx = 0.5 * (p0[0] + p1[0]);
        let my = 0.5 * (p0[1] + p1[1]);
        if dy.abs() >= dx.abs() {
            // Vertical-ish guide: signed tilt angle in degrees (positive = tilts right as y grows).
            let signed_dy = if dy >= 0.0 { dy } else { -dy };
            let signed_dx = if dy >= 0.0 { dx } else { -dx };
            let angle = (signed_dx / signed_dy).atan().to_degrees();
            if mx < 0.5 {
                v_left.push(angle);
            } else {
                v_right.push(angle);
            }
        } else {
            let signed_dx = if dx >= 0.0 { dx } else { -dx };
            let signed_dy = if dx >= 0.0 { dy } else { -dy };
            let angle = (signed_dy / signed_dx).atan().to_degrees();
            if my < 0.5 {
                h_top.push(angle);
            } else {
                h_bot.push(angle);
            }
        }
    }

    let avg = |xs: &[f32]| -> Option<f32> {
        (!xs.is_empty()).then(|| xs.iter().sum::<f32>() / xs.len() as f32)
    };
    let half_v = match (avg(&v_left), avg(&v_right)) {
        (Some(l), Some(r)) => ((r - l) * 0.5).clamp(-MAX_OUT, MAX_OUT),
        _ => 0.0,
    };
    let half_h = match (avg(&h_top), avg(&h_bot)) {
        (Some(t), Some(b)) => ((b - t) * 0.5).clamp(-MAX_OUT, MAX_OUT),
        _ => 0.0,
    };

    crate::Keystone {
        vertical: solve_axis(Axis::Vertical, half_v, aspect, wf, hf),
        horizontal: solve_axis(Axis::Horizontal, half_h, aspect, wf, hf),
        amount: 100.0,
        aspect,
        ..Default::default()
    }
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

    /// Independent measurement of the half-edge inset the renderer applies,
    /// through the crate's `frame_to_source` entry point (the production helper
    /// uses `FrameMap`; the two must agree). Returns `(vertical, horizontal)` in
    /// degrees, matching what [`detect`] measures.
    fn mapped_angles(k: &crate::Keystone, w: u32, h: u32) -> (f32, f32) {
        let geometry = crate::Geometry { keystone: k.clone(), ..Default::default() };
        let pixel = |f: [f32; 2]| {
            let s = crate::mock::geometry::frame_to_source(f, &geometry, (w, h));
            [s[0] * w as f32, s[1] * h as f32]
        };
        let (tl, tr, bl, br) = (
            pixel([0.0, 0.0]),
            pixel([1.0, 0.0]),
            pixel([0.0, 1.0]),
            pixel([1.0, 1.0]),
        );
        let off_v = |a: [f32; 2], b: [f32; 2]| (b[0] - a[0]).atan2(b[1] - a[1]).to_degrees();
        let off_h = |a: [f32; 2], b: [f32; 2]| (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees();
        (
            0.5 * (off_v(tr, br) - off_v(tl, bl)),
            0.5 * (off_h(bl, br) - off_h(tl, tr)),
        )
    }

    /// Trapezoid whose two side edges tilt by `deg` from vertical. Positive
    /// narrows the top, which is what [`detect`] reports as positive.
    fn vertical_pair(w: u32, h: u32, deg: f32) -> Vec<u8> {
        let mut img = canvas(w, h);
        let cx = w as f32 * 0.5;
        let half_top = w as f32 * 0.18;
        let inset = deg.to_radians().tan() * (h as f32 - 1.0);
        let half_bottom = half_top + inset;
        line(&mut img, w, h, (cx - half_top, 0.0), (cx - half_bottom, h as f32 - 1.0));
        line(&mut img, w, h, (cx + half_top, 0.0), (cx + half_bottom, h as f32 - 1.0));
        img
    }

    /// Trapezoid whose two horizontal edges tilt by `deg`. Positive narrows the
    /// left, which is what [`detect`] reports as positive.
    fn horizontal_pair(w: u32, h: u32, deg: f32) -> Vec<u8> {
        let mut img = canvas(w, h);
        let cy = h as f32 * 0.5;
        let half_left = h as f32 * 0.18;
        let inset = deg.to_radians().tan() * (w as f32 - 1.0);
        let half_right = half_left + inset;
        line(&mut img, w, h, (0.0, cy - half_left), (w as f32 - 1.0, cy - half_right));
        line(&mut img, w, h, (0.0, cy + half_left), (w as f32 - 1.0, cy + half_right));
        img
    }

    /// The Auto button writes `Keystone` slider units, not degrees: the units it
    /// picks must, through the real renderer map, land back on the detected
    /// half-angle.
    #[test]
    fn auto_keystone_reproduces_the_detected_half_angle() {
        let (w, h) = (512, 384);
        let img = vertical_pair(w, h, 7.0);
        let half = detect(&img, w, h);
        assert!(half.vertical > 3.0, "fixture must converge, got {}", half.vertical);

        let k = auto_keystone(20.0, &img, w, h);
        let (mapped_v, mapped_h) = mapped_angles(&k, w, h);
        assert!(
            (mapped_v - half.vertical).abs() <= 0.02,
            "mapped {mapped_v} vs detected {}",
            half.vertical
        );
        assert!(mapped_h.abs() < 1e-3, "unexpected horizontal {mapped_h}");
        assert!((k.vertical - half.vertical).abs() > 1.0, "degrees leaked into the field: {}", k.vertical);
        assert_eq!(k.horizontal, 0.0);
        assert_eq!(k.amount, 100.0);
        assert_eq!(k.aspect, 20.0);
        assert_eq!((k.skew, k.focal_length, k.flags), (0.0, 0.0, 0));
    }

    #[test]
    fn auto_keystone_horizontal_reproduces_the_detected_half_angle() {
        let (w, h) = (400, 300);
        let img = horizontal_pair(w, h, 6.0);
        let half = detect(&img, w, h);
        assert!(half.horizontal > 3.0, "fixture must converge, got {}", half.horizontal);

        let k = auto_keystone(0.0, &img, w, h);
        let (mapped_v, mapped_h) = mapped_angles(&k, w, h);
        assert!(
            (mapped_h - half.horizontal).abs() <= 0.02,
            "mapped {mapped_h} vs detected {}",
            half.horizontal
        );
        assert!(mapped_v.abs() < 1e-3, "unexpected vertical {mapped_v}");
        assert_eq!(k.vertical, 0.0);
        assert_eq!(k.amount, 100.0);
    }

    /// Positive and negative convergence must give opposite, equal-magnitude
    /// sliders, so the suggestion is unbiased.
    #[test]
    fn auto_keystone_is_symmetric_about_zero() {
        let (w, h) = (512, 384);
        let pos = auto_keystone(0.0, &vertical_pair(w, h, 8.0), w, h);
        let neg = auto_keystone(0.0, &vertical_pair(w, h, -8.0), w, h);
        assert!(pos.vertical > 0.0 && neg.vertical < 0.0, "{} {}", pos.vertical, neg.vertical);
        assert!(
            (pos.vertical + neg.vertical).abs() < 1.0,
            "magnitudes differ: {} vs {}",
            pos.vertical,
            neg.vertical
        );

        let pos = auto_keystone(0.0, &horizontal_pair(w, h, 8.0), w, h);
        let neg = auto_keystone(0.0, &horizontal_pair(w, h, -8.0), w, h);
        assert!(pos.horizontal > 0.0 && neg.horizontal < 0.0, "{} {}", pos.horizontal, neg.horizontal);
        assert!(
            (pos.horizontal + neg.horizontal).abs() < 1.0,
            "magnitudes differ: {} vs {}",
            pos.horizontal,
            neg.horizontal
        );
    }

    /// Several angles on two frame sizes, all inverting through the real map to
    /// within the detector-independent solver tolerance.
    #[test]
    fn auto_keystone_conversion_is_consistent_across_angles_and_frames() {
        for &(w, h) in &[(512u32, 384u32), (400u32, 300u32)] {
            for &deg in &[1.0f32, 5.0, 10.0, 20.0] {
                let img = vertical_pair(w, h, deg);
                let half = detect(&img, w, h);
                assert!(half.vertical > 0.0, "{w}x{h} {deg}: fixture did not converge");
                let k = auto_keystone(0.0, &img, w, h);
                assert!(
                    k.vertical.is_finite() && (-100.0..=100.0).contains(&k.vertical),
                    "{w}x{h} {deg}: vertical out of range: {}",
                    k.vertical
                );
                let (mapped_v, _) = mapped_angles(&k, w, h);
                assert!(
                    (mapped_v - half.vertical).abs() <= 0.02,
                    "{w}x{h} {deg}: mapped {mapped_v} vs detected {}",
                    half.vertical
                );
            }
        }
    }

    /// The inverse itself (without detector noise): each requested half-angle,
    /// on several frames and both axes, maps back to the request. When the
    /// request exceeds what the slider range can produce the result saturates at
    /// `±100` and maps to the slider's own maximum, never past it.
    #[test]
    fn solve_axis_inverts_the_real_map() {
        let at_limit = |axis: Axis, w: u32, h: u32| {
            let k = match axis {
                Axis::Vertical => crate::Keystone { vertical: 100.0, amount: 100.0, ..Default::default() },
                Axis::Horizontal => crate::Keystone { horizontal: 100.0, amount: 100.0, ..Default::default() },
            };
            let (mv, mh) = mapped_angles(&k, w, h);
            match axis {
                Axis::Vertical => mv,
                Axis::Horizontal => mh,
            }
        };
        for &(w, h) in &[(512.0f32, 384.0f32), (400.0, 300.0), (640.0, 360.0)] {
            for axis in [Axis::Vertical, Axis::Horizontal] {
                let reach = at_limit(axis, w as u32, h as u32);
                assert!(reach > 1.0, "{w}x{h}: slider range is degenerate");
                for &deg in &[-20.0f32, -10.0, -5.0, -1.0, 1.0, 5.0, 10.0, 20.0] {
                    let value = solve_axis(axis, deg, 0.0, w, h);
                    let k = match axis {
                        Axis::Vertical => crate::Keystone { vertical: value, amount: 100.0, ..Default::default() },
                        Axis::Horizontal => crate::Keystone { horizontal: value, amount: 100.0, ..Default::default() },
                    };
                    let (mv, mh) = mapped_angles(&k, w as u32, h as u32);
                    let mapped = match axis {
                        Axis::Vertical => mv,
                        Axis::Horizontal => mh,
                    };
                    if deg.abs() <= reach {
                        assert!(
                            (mapped - deg).abs() <= 0.01 + 1e-4,
                            "{w}x{h} {deg}° axis={}: mapped {mapped}",
                            matches!(axis, Axis::Vertical),
                        );
                    } else {
                        assert_eq!(value, deg.signum() * 100.0, "{w}x{h} {deg}°: must saturate");
                        assert!(
                            (mapped - deg.signum() * reach).abs() < 1e-3,
                            "{w}x{h} {deg}°: saturated map {mapped}, reach {reach}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn auto_keystone_on_a_flat_frame_is_neutral() {
        let img = vec![128u8; 320 * 240 * 4];
        let k = auto_keystone(-3.0, &img, 320, 240);
        assert_eq!(
            k,
            crate::Keystone {
                vertical: 0.0,
                horizontal: 0.0,
                amount: 100.0,
                aspect: -3.0,
                ..Default::default()
            }
        );
    }

    #[test]
    fn bad_sizes_are_safe() {
        assert_eq!(detect(&[], 0, 0), Suggestion::default());
        assert_eq!(detect(&[0; 16], 100, 100), Suggestion::default());
        assert_eq!(detect(&[0; 64], 4, 4), Suggestion::default());
        assert_eq!(detect(&[0; 64], u32::MAX, u32::MAX), Suggestion::default());
    }

    #[test]
    fn guided_upright_solves_convergent_guides() {
        let ks = solve_guided_upright(
            &[
                ([0.2, 0.1], [0.25, 0.9]),
                ([0.8, 0.1], [0.75, 0.9]),
            ],
            0.0,
            1000,
            1000,
        );
        assert!(ks.vertical.abs() > 1.0 && ks.amount == 100.0, "{ks:?}");

        let flat = vec![128u8; 256 * 256 * 4];
        let level = upright_keystone(UprightMode::Level, 0.0, &flat, 256, 256);
        assert_eq!(level.rotation_deg, 0.0);
        assert_eq!(level.keystone.vertical, 0.0);
    }
}
