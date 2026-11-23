//! One inverse map from the developed frame back to oriented and source space,
//! shared by preview, export and the colour picker.
//!
//! Forward order (what the viewer shows): oriented+flip → keystone → crop →
//! aspect → rotation. So the inverse, applied here, is crop → keystone →
//! rotation → flip → orientation, matching `process::FrameMap`'s chain with the
//! keystone stage inserted between crop and rotation.
//!
//! `ZCROP` is axis-aligned in `ZCANVASSIZE`, which is not the oriented image
//! (C1 stores 4043.054×6056.673 for a 6048×4032 raw turned a quarter), so the
//! stored canvas scales the crop into the image before the flip and the turn.
//!
//! `frame_to_oriented` has no `source_size`, so it uses `g.crop` as is; an
//! aspect-ratio crop needs source pixels to size itself, so use
//! `frame_to_source` for that case.

use crate::{AspectRatio, Geometry, Keystone, NormRect};

const EPS: f32 = 1e-6;
const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn finite(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn finite_pair(p: [f32; 2]) -> [f32; 2] {
    [finite(p[0]), finite(p[1])]
}

fn clean_rect(g: &Geometry) -> NormRect {
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
fn oriented_size(g: &Geometry, source_size: (u32, u32)) -> (f32, f32) {
    let (w, h) = (source_size.0 as f32, source_size.1 as f32);
    let quarter_turn = (g.orientation % 2 == 1) ^ is_quarter_turn(g.rotation);
    if quarter_turn {
        (h, w)
    } else {
        (w, h)
    }
}

fn is_quarter_turn(rotation_deg: f32) -> bool {
    let turns = finite(rotation_deg) / 90.0;
    (turns - turns.round()).abs() < 1e-4 && turns.round() as i64 % 2 != 0
}

/// `ZCANVASSIZE` when the crop was stored against it, otherwise the oriented
/// image. A full-frame crop keeps the image size, so an uncropped variant
/// renders exactly as it did before the canvas was carried.
fn canvas_size(g: &Geometry, source_size: (u32, u32)) -> (f32, f32) {
    let (ow, oh) = oriented_size(g, source_size);
    let (cw, ch) = (finite(g.canvas[0]), finite(g.canvas[1]));
    if cw > EPS && ch > EPS && g.crop != NormRect::FULL {
        (cw, ch)
    } else {
        (ow, oh)
    }
}

/// Canvas size after a fine rotation: the AABB of the four rotated corners.
/// Quarter turns fall out of the same formula (`sin`/`cos` are 1/0 there).
pub(crate) fn rotated_size(w: f32, h: f32, rotation_deg: f32) -> (f32, f32) {
    let r = if rotation_deg.is_finite() { rotation_deg.to_radians() } else { 0.0 };
    let (s, c) = r.sin_cos();
    let (s, c) = (s.abs(), c.abs());
    (finite(w * c + h * s), finite(w * s + h * c))
}

/// Size of the developed frame in source pixels.
#[cfg(test)]
pub(crate) fn frame_size(g: &Geometry, source_size: (u32, u32)) -> (f32, f32) {
    FrameMap::new(source_size, g).frame_size()
}

/// `ZROTATION` includes the quarter turns; the H/V slider frame is rotated by
/// that total angle, not by the fine straighten component alone.
fn total_rotation(g: &Geometry) -> f32 {
    finite(g.rotation) + 90.0 * (g.orientation % 4) as f32
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

/// Keystone as a weak perspective plus C1's symmetric skew shear.
///
/// `vertical` insets the top edge horizontally (top narrower than bottom for a
/// positive value), `horizontal` insets the left edge vertically (left shorter
/// than right), each scaled by `amount`; `aspect` stretches the result about the
/// crop centre to undo the squeeze the correction introduces — Y for a positive
/// value, X for a negative one (`UpdateGeo` flips `.dca`). `skew` shears
/// both axes by `tan(atan(tan(skew°)·amount)/2)`. `focal_length` is C1's `c`:
/// stored tilt is written as `100·tan(pitch)/fLen`, so scaling it back cancels
/// the lens and leaves the physical angle.
pub(crate) struct KeystoneMap {
    /// Row-major 3x3 mapping a crop-space point to its source-crop point.
    m: [f32; 9],
    stretch: f32,
    /// C1 divides X for a negative `aspect` and Y for a positive one.
    stretch_x: bool,
    shear: f32,
}

impl KeystoneMap {
    pub(crate) fn new(k: &Keystone, rotation_deg: f32, flip_h: bool, flip_v: bool) -> Self {
        let mut stored = k.clone();
        stored.set_hv(k.horizontal, k.vertical, rotation_deg, flip_h, flip_v);
        let amount = (finite(k.amount) / 100.0).clamp(0.0, 1.2);
        // `aspectF = 1 + |aspect|/100`; a negative aspect swaps the axis via
        // `.dca = (0,1)` (UpdateGeo 0x141DB06/0x141DB0A).
        let aspect = finite(k.aspect);
        let stretch = (1.0 + aspect.abs() / 100.0).clamp(0.1, 10.0);
        let stretch_x = aspect < 0.0;
        let scale = stored.tilt_scale();
        let vertical = (finite(stored.vertical) / 100.0 * scale).clamp(-1.0, 1.0);
        let horizontal = (finite(stored.horizontal) / 100.0 * scale).clamp(-1.0, 1.0);
        let skew = finite(k.skew).clamp(-45.0, 45.0);
        let shear = if amount < EPS || skew.abs() < EPS {
            0.0
        } else {
            ((skew.to_radians().tan() * amount).atan() * 0.5).tan()
        };
        if amount < EPS || (vertical.abs() < EPS && horizontal.abs() < EPS && shear.abs() < EPS) {
            // `amount` only scales tilt and skew; `aspectF` is read straight
            // from the settings, so it survives a zero amount.
            return Self { m: IDENTITY, stretch, stretch_x, shear: 0.0 };
        }
        let (a, b) = (vertical * amount * 0.35, horizontal * amount * 0.35);
        // Frame corners are the unit square; they sample a trapezoid in the source.
        let from = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let to = [[a, b], [1.0 - a, 0.0], [1.0, 1.0 - b], [0.0, 1.0]];
        let m = solve_homography(from, to).unwrap_or(IDENTITY);
        Self { m, stretch, stretch_x, shear }
    }

    fn is_identity(&self) -> bool {
        self.m == IDENTITY && self.shear.abs() < EPS && (self.stretch - 1.0).abs() < EPS
    }

    fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let m = &self.m;
        let w = m[6] * p[0] + m[7] * p[1] + m[8];
        if !w.is_finite() || w.abs() < EPS {
            return finite_pair(p);
        }
        let x = (m[0] * p[0] + m[1] * p[1] + m[2]) / w;
        let y = (m[3] * p[0] + m[4] * p[1] + m[5]) / w;
        // Symmetric shear about the centre; C1 applies `basisX + t·basisY` on
        // both bases, so each axis is pushed by the other one.
        [finite(x + self.shear * (y - 0.5)), finite(y + self.shear * (x - 0.5))]
    }
}

/// C1's post-homography single-axis scale in the canvas→raw direction: the
/// forward tail (0x141EAF0) divides by `aspectF`, so the inverse multiplies.
fn aspect_scale(p: &mut [f32; 2], stretch: f32, stretch_x: bool) {
    if (stretch - 1.0).abs() <= EPS {
        return;
    }
    let i = if stretch_x { 0 } else { 1 };
    p[i] = finite((p[i] - 0.5) * stretch + 0.5);
}

/// Homography taking the four `from` points to the four `to` points.
/// Solves the 8 unknowns of a 3x3 matrix (h33 fixed to 1) by Gaussian elimination.
fn solve_homography(from: [[f32; 2]; 4], to: [[f32; 2]; 4]) -> Option<[f32; 9]> {
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = (from[i][0] as f64, from[i][1] as f64);
        let (u, v) = (to[i][0] as f64, to[i][1] as f64);
        a[i * 2] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[i * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for col in 0..8 {
        let pivot = (col..8).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            // Degenerate quad: callers fall back to the identity.
            return None;
        }
        a.swap(col, pivot);
        let inv = 1.0 / a[col][col];
        for k in col..9 {
            a[col][k] *= inv;
        }
        for row in 0..8 {
            if row == col {
                continue;
            }
            let f = a[row][col];
            if f == 0.0 {
                continue;
            }
            for k in col..9 {
                a[row][k] -= f * a[col][k];
            }
        }
    }
    let h = |i: usize| a[i][8] as f32;
    let m = [h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), 1.0];
    m.iter().all(|v| v.is_finite()).then_some(m)
}

/// The inverse chain, with an explicit crop rect so both public entry points share it.
#[cfg(test)]
fn map_frame(f: [f32; 2], g: &Geometry, crop: NormRect, source_size: (u32, u32)) -> ([f32; 2], [f32; 2]) {
    let f = finite_pair(f);
    let (ow, oh) = oriented_size(g, source_size);
    let (cw, ch) = canvas_size(g, source_size);
    let keystone = KeystoneMap::new(&g.keystone, total_rotation(g), g.flip_horizontal, g.flip_vertical);
    let mut p = [crop.x + f[0] * crop.width, crop.y + f[1] * crop.height];
    if !keystone.is_identity() {
        p = keystone.apply(p);
    }
    let rotation = finite(g.rotation).to_radians();
    if rotation.abs() > 1e-5 && ow > EPS && oh > EPS {
        let (cx, cy) = ((p[0] - 0.5) * cw, (p[1] - 0.5) * ch);
        let (s, c) = (-rotation).sin_cos();
        p = [(cx * c - cy * s) / ow + 0.5, (cx * s + cy * c) / oh + 0.5];
    } else if cw != ow || ch != oh {
        // Undo the canvas→image scale; a mirror commutes with it, so the flip
        // and the turn below stay in oriented coordinates.
        p = [((p[0] - 0.5) * cw / ow) + 0.5, ((p[1] - 0.5) * ch / oh) + 0.5];
    }
    // Aspect is the non-uniform scale C1 applies between the 3x3 and the flip.
    aspect_scale(&mut p, keystone.stretch, keystone.stretch_x);
    let oriented = finite_pair(p);
    let mut q = p;
    if g.flip_horizontal {
        q[0] = 1.0 - q[0];
    }
    if g.flip_vertical {
        q[1] = 1.0 - q[1];
    }
    let s = match g.orientation % 4 {
        1 => [q[1], 1.0 - q[0]],
        2 => [1.0 - q[0], 1.0 - q[1]],
        3 => [1.0 - q[1], q[0]],
        _ => q,
    };
    (oriented, finite_pair(s))
}

/// Maps a developed-frame point (0..1) back to an oriented-image point (0..1).
#[cfg(test)]
pub(crate) fn frame_to_oriented(f: [f32; 2], g: &crate::Geometry) -> [f32; 2] {
    map_frame(f, g, clean_rect(g), (1, 1)).0
}

/// Maps a developed-frame point back to the source image (after orientation/flip).
#[cfg(test)]
pub(crate) fn frame_to_source(f: [f32; 2], g: &crate::Geometry, source_size: (u32, u32)) -> [f32; 2] {
    map_frame(f, g, crop_for_aspect(g, source_size), source_size).1
}

/// The inverse map built once per frame: the keystone homography is solved in
/// `new`, so the per-pixel `map` is only interpolation and a rotation.
pub(crate) struct FrameMap {
    oriented: (f32, f32),
    canvas: (f32, f32),
    canvas_stored: bool,
    crop: NormRect,
    /// `None` when the keystone is a no-op, so the per-pixel path stays a plain
    /// crop + rotation with no homography division.
    keystone: Option<KeystoneMap>,
    stretch: f32,
    stretch_x: bool,
    quarter_turn: bool,
    rotation: f32,
    rotation_sin: f32,
    rotation_cos: f32,
    orientation: u8,
    flip_h: bool,
    flip_v: bool,
}

impl FrameMap {
    pub(crate) fn new(source_size: (u32, u32), g: &Geometry) -> Self {
        let rotation = finite(g.rotation).to_radians();
        let (rotation_sin, rotation_cos) = (-rotation).sin_cos();
        let keystone = KeystoneMap::new(&g.keystone, total_rotation(g), g.flip_horizontal, g.flip_vertical);
        let stretch = keystone.stretch;
        let stretch_x = keystone.stretch_x;
        let canvas = canvas_size(g, source_size);
        let canvas_stored = canvas != oriented_size(g, source_size);
        Self {
            oriented: oriented_size(g, source_size),
            canvas,
            canvas_stored,
            crop: crop_for_aspect(g, source_size),
            keystone: (!keystone.is_identity()).then_some(keystone),
            stretch,
            stretch_x,
            quarter_turn: is_quarter_turn(g.rotation),
            rotation,
            rotation_sin,
            rotation_cos,
            orientation: g.orientation % 4,
            flip_h: g.flip_horizontal,
            flip_v: g.flip_vertical,
        }
    }

    /// Size of the developed frame in source pixels.
    pub(crate) fn frame_size(&self) -> (f32, f32) {
        let (ow, oh) = self.oriented;
        if self.canvas_stored {
            // `ZCROP` is already axis-aligned in `ZCANVASSIZE`, so its pixel
            // size is the output size, including any part outside the image.
            return (
                finite(self.canvas.0 * self.crop.width),
                finite(self.canvas.1 * self.crop.height),
            );
        }
        let (w, h) = (finite(ow * self.crop.width), finite(oh * self.crop.height));
        if self.keystone.is_none() && (self.stretch - 1.0).abs() <= EPS {
            // `oriented_size` already folds exact quarter turns into its axes.
            return if self.quarter_turn { (w, h) } else { rotated_size(w, h, self.rotation.to_degrees()) };
        }
        // `CalculateGeo`: the canvas is the AABB of the four crop corners after
        // the 3x3 and the rotation; the aspect is C1's final forward step, so it
        // divides the chosen axis after the box.
        let (mut x0, mut y0) = (f32::INFINITY, f32::INFINITY);
        let (mut x1, mut y1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for f in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] {
            let p = self.map_pre_aspect(f);
            let (x, y) = (p[0] * ow, p[1] * oh);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        let (mut w, mut h) = (finite(x1 - x0), finite(y1 - y0));
        if (self.stretch - 1.0).abs() > EPS {
            if self.stretch_x { w /= self.stretch } else { h /= self.stretch };
        }
        (finite(w), finite(h))
    }

    /// Oriented (after quarter turns and rotation, before crop) size, for aspect-dependent masks.
    pub(crate) fn oriented(&self) -> (f32, f32) {
        self.oriented
    }

    /// Frame point → oriented point before C1's aspect scale.
    fn map_pre_aspect(&self, f: [f32; 2]) -> [f32; 2] {
        let f = finite_pair(f);
        let (ow, oh) = self.oriented;
        let (cw, ch) = self.canvas;
        let mut p = [self.crop.x + f[0] * self.crop.width, self.crop.y + f[1] * self.crop.height];
        if let Some(k) = &self.keystone {
            p = k.apply(p);
        }
        if self.rotation.abs() > 1e-5 && ow > EPS && oh > EPS {
            let (cx, cy) = ((p[0] - 0.5) * cw, (p[1] - 0.5) * ch);
            let (s, c) = (self.rotation_sin, self.rotation_cos);
            p = [(cx * c - cy * s) / ow + 0.5, (cx * s + cy * c) / oh + 0.5];
        } else if cw != ow || ch != oh {
            p = [((p[0] - 0.5) * cw / ow) + 0.5, ((p[1] - 0.5) * ch / oh) + 0.5];
        }
        finite_pair(p)
    }

    /// Frame point → (oriented-image point, source point).
    pub(crate) fn map(&self, f: [f32; 2]) -> ([f32; 2], [f32; 2]) {
        let mut p = self.map_pre_aspect(f);
        aspect_scale(&mut p, self.stretch, self.stretch_x);
        let oriented = finite_pair(p);
        let mut q = p;
        if self.flip_h {
            q[0] = 1.0 - q[0];
        }
        if self.flip_v {
            q[1] = 1.0 - q[1];
        }
        let s = match self.orientation {
            1 => [q[1], 1.0 - q[0]],
            2 => [1.0 - q[0], 1.0 - q[1]],
            3 => [1.0 - q[1], q[0]],
            _ => q,
        };
        (oriented, finite_pair(s))
    }

    pub(crate) fn frame_from_source(&self, source: [f32; 2]) -> Option<[f32; 2]> {
        let source = finite_pair(source);
        let mut frame = source;
        for _ in 0..16 {
            let (_, mapped) = self.map(frame);
            let rx = mapped[0] - source[0];
            let ry = mapped[1] - source[1];
            if rx.abs().max(ry.abs()) <= 1e-6 {
                return Some(frame);
            }
            let h = 1e-4;
            let (_, x_shift) = self.map([frame[0] + h, frame[1]]);
            let (_, y_shift) = self.map([frame[0], frame[1] + h]);
            let j00 = (x_shift[0] - mapped[0]) / h;
            let j10 = (x_shift[1] - mapped[1]) / h;
            let j01 = (y_shift[0] - mapped[0]) / h;
            let j11 = (y_shift[1] - mapped[1]) / h;
            let determinant = j00 * j11 - j01 * j10;
            if !determinant.is_finite() || determinant.abs() < 1e-8 {
                return None;
            }
            let dx = (j11 * rx - j01 * ry) / determinant;
            let dy = (-j10 * rx + j00 * ry) / determinant;
            let next = [frame[0] - dx, frame[1] - dy];
            if next
                .iter()
                .any(|value| !value.is_finite() || !(-4.0..=5.0).contains(value))
            {
                return None;
            }
            frame = next;
        }
        let (_, mapped) = self.map(frame);
        ((mapped[0] - source[0]).abs().max((mapped[1] - source[1]).abs()) <= 1e-5)
            .then_some(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: (u32, u32) = (400, 300);

    fn assert_near(a: [f32; 2], b: [f32; 2]) {
        assert!((a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4, "{a:?} != {b:?}");
    }

    #[test]
    fn default_centre_is_identity() {
        assert_near(frame_to_source([0.5, 0.5], &Geometry::default(), SRC), [0.5, 0.5]);
        assert_near(frame_to_oriented([0.5, 0.5], &Geometry::default()), [0.5, 0.5]);
    }

    /// All-zero keystone takes the old code path bit for bit: no homography,
    /// no stretch, no shear.
    #[test]
    fn zero_keystone_is_an_exact_identity_path() {
        let g = Geometry::default();
        let map = FrameMap::new(SRC, &g);
        assert!(map.keystone.is_none());
        assert_eq!(map.stretch, 1.0);
        for f in [[0.0, 0.0], [0.3, 0.7], [0.5, 0.5], [1.0, 1.0]] {
            assert_eq!(map.map(f), (f, f));
        }
        assert_eq!(map.frame_size(), (400.0, 300.0));
        assert!(KeystoneMap::new(&g.keystone, 0.0, false, false).is_identity());
    }

    #[test]
    fn orientation_quarter_turns_map_corners() {
        let expected: [([f32; 2], [f32; 2]); 4] = [
            // 0: identity; 1: (x, y) -> (y, 1-x); 2: mirror both; 3: (x, y) -> (1-y, x).
            ([0.0, 0.0], [0.0, 0.0]),
            ([0.0, 0.0], [0.0, 1.0]),
            ([0.0, 0.0], [1.0, 1.0]),
            ([0.0, 0.0], [1.0, 0.0]),
        ];
        for (o, (frame, source)) in expected.into_iter().enumerate() {
            let g = Geometry { orientation: o as u8, ..Default::default() };
            assert_near(frame_to_source(frame, &g, SRC), source);
            let far = [1.0 - frame[0], 1.0 - frame[1]];
            assert_near(frame_to_source(far, &g, SRC), [1.0 - source[0], 1.0 - source[1]]);
        }
    }

    #[test]
    fn flip_horizontal_mirrors_x() {
        let g = Geometry { flip_horizontal: true, ..Default::default() };
        assert_near(frame_to_source([0.25, 0.5], &g, SRC), [0.75, 0.5]);
        assert_near(frame_to_source([0.25, 0.5], &g, SRC), [1.0 - 0.25, 0.5]);
    }

    #[test]
    fn quarter_turn_swaps_frame_size() {
        let by_orientation = Geometry { orientation: 1, ..Default::default() };
        let (w, h) = frame_size(&by_orientation, SRC);
        assert!((w - 300.0).abs() < 1e-3 && (h - 400.0).abs() < 1e-3, "{w}x{h}");
        let by_rotation = Geometry { rotation: 90.0, ..Default::default() };
        let (w, h) = frame_size(&by_rotation, SRC);
        assert!((w - 300.0).abs() < 1e-3 && (h - 400.0).abs() < 1e-3, "{w}x{h}");
        assert_near(frame_to_source([0.5, 0.5], &by_rotation, SRC), [0.5, 0.5]);
    }

    #[test]
    fn crop_keeps_its_centre() {
        let g = Geometry { crop: NormRect { x: 0.25, y: 0.25, width: 0.5, height: 0.5 }, ..Default::default() };
        assert_near(frame_to_source([0.5, 0.5], &g, SRC), [0.5, 0.5]);
        assert_near(frame_to_source([0.0, 0.0], &g, SRC), [0.25, 0.25]);
    }

    #[test]
    fn keystone_amount_zero_is_identity() {
        let g = Geometry {
            keystone: Keystone { vertical: 80.0, horizontal: -60.0, amount: 0.0, aspect: 0.0, ..Default::default() },
            ..Default::default()
        };
        for f in [[0.0, 0.0], [1.0, 0.0], [0.3, 0.7], [0.5, 0.5]] {
            assert_near(frame_to_source(f, &g, SRC), f);
        }
        // `aspectF` does not depend on `amount` (0x141D980 reads cfg[0x28]
        // before the tilt branch), so aspect alone still scales the axis.
        let aspect_only = Geometry {
            keystone: Keystone { amount: 0.0, aspect: 20.0, ..Default::default() },
            ..Default::default()
        };
        assert_near(frame_to_source([0.5, 0.6], &aspect_only, SRC), [0.5, 0.5 + 0.1 * 1.2]);
    }

    /// Camera data writes `tilt = 100·tan(pitch)/fLen`; the `c` scale in the
    /// matrix must cancel it so the correction depends only on the pitch.
    #[test]
    fn focal_length_cancels_the_camera_data_tilt() {
        let tan_pitch = 20.0f32.to_radians().tan();
        let wanted = Geometry {
            keystone: Keystone { vertical: 100.0 * tan_pitch, amount: 80.0, ..Default::default() },
            ..Default::default()
        };
        for f in [0.2, 0.5, 1.0, 2.0] {
            let g = Geometry {
                keystone: Keystone {
                    vertical: 100.0 * tan_pitch / f,
                    amount: 80.0,
                    focal_length: f,
                    ..Default::default()
                },
                ..Default::default()
            };
            for p in [[0.0, 0.0], [0.3, 0.8], [0.5, 0.5], [1.0, 1.0]] {
                assert_near(frame_to_source(p, &g, SRC), frame_to_source(p, &wanted, SRC));
            }
        }
    }

    /// Skew is C1's symmetric shear: `t = tan(atan(tan(20°)·1)/2) = tan(10°)`.
    #[test]
    fn skew_twenty_degrees_is_tan_ten() {
        let k = Keystone { skew: 20.0, amount: 100.0, ..Default::default() };
        let map = KeystoneMap::new(&k, 0.0, false, false);
        assert_near(map.apply([0.5, 0.5]), [0.5, 0.5]);
        let t = 10.0f32.to_radians().tan();
        let got = map.apply([0.6, 0.5]);
        assert!((got[0] - 0.6).abs() < 1e-5 && (got[1] - 0.5 - 0.1 * t).abs() < 1e-5, "{got:?}");
        let got = map.apply([0.5, 0.6]);
        assert!((got[1] - 0.6).abs() < 1e-5 && (got[0] - 0.5 - 0.1 * t).abs() < 1e-5, "{got:?}");
    }

    #[test]
    fn canvas_aabb_follows_the_rotation_formula() {
        let (w, h) = (400.0, 300.0);
        for (rot, want) in [(0.0, (400.0, 300.0)), (90.0, (300.0, 400.0)), (180.0, (400.0, 300.0)), (270.0, (300.0, 400.0))] {
            let got = rotated_size(w, h, rot);
            assert!((got.0 - want.0).abs() < 1e-3 && (got.1 - want.1).abs() < 1e-3, "{rot}: {got:?}");
        }
        let diag = (w + h) / 2.0f32.sqrt();
        let got = rotated_size(w, h, 45.0);
        assert!((got.0 - diag).abs() < 1e-3 && (got.1 - diag).abs() < 1e-3, "{got:?}");
        let g = Geometry { rotation: 45.0, ..Default::default() };
        let (fw, fh) = frame_size(&g, SRC);
        assert!((fw - diag).abs() < 1e-3 && (fh - diag).abs() < 1e-3, "{fw}x{fh}");
    }

    /// `0x141DB06/0x141DB0A`: a negative aspect flips `.dca` to (0,1), so
    /// `aspectF = 1 - aspect/100` scales X instead of Y.
    #[test]
    fn negative_aspect_scales_the_horizontal_axis() {
        let up = Geometry {
            keystone: Keystone { aspect: 30.0, amount: 100.0, ..Default::default() },
            ..Default::default()
        };
        assert_near(frame_to_source([0.5, 0.65], &up, SRC), [0.5, 0.5 + 0.15 * 1.3]);
        assert_near(frame_to_source([0.65, 0.5], &up, SRC), [0.65, 0.5]);
        let down = Geometry {
            keystone: Keystone { aspect: -30.0, amount: 100.0, ..Default::default() },
            ..Default::default()
        };
        assert_near(frame_to_source([0.65, 0.5], &down, SRC), [0.5 + 0.15 * 1.3, 0.5]);
        assert_near(frame_to_source([0.5, 0.65], &down, SRC), [0.5, 0.65]);
        // Both signs shrink one canvas axis by the same factor (raw→canvas divides).
        let (w, h) = frame_size(&up, SRC);
        let (w2, h2) = frame_size(&down, SRC);
        assert!((h - 300.0 / 1.3).abs() < 1e-3 && (w - 400.0).abs() < 1e-3, "{w}x{h}");
        assert!((w2 - 400.0 / 1.3).abs() < 1e-3 && (h2 - 300.0).abs() < 1e-3, "{w2}x{h2}");
    }

    /// The canvas is the AABB of the four corners *after* the homography, so a
    /// vertical keystone widens the frame by more than the rotation formula.
    #[test]
    fn canvas_aabb_includes_the_keystone_perspective() {
        let g = Geometry {
            rotation: 10.0,
            keystone: Keystone {
                vertical: 60.0,
                horizontal: 40.0,
                amount: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let (w, h) = frame_size(&g, SRC);
        let map = FrameMap::new(SRC, &g);
        let (mut x0, mut y0) = (f32::INFINITY, f32::INFINITY);
        let (mut x1, mut y1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for f in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] {
            let p = map.map(f).0;
            x0 = x0.min(p[0] * 400.0);
            x1 = x1.max(p[0] * 400.0);
            y0 = y0.min(p[1] * 300.0);
            y1 = y1.max(p[1] * 300.0);
        }
        assert!((w - (x1 - x0)).abs() < 1e-3 && (h - (y1 - y0)).abs() < 1e-3, "{w}x{h}");
        let (rw, rh) = rotated_size(400.0, 300.0, 10.0);
        assert!((w - rw).abs() > 0.5 || (h - rh).abs() > 0.5, "perspective ignored: {w}x{h} vs {rw}x{rh}");
    }

    #[test]
    fn keystone_makes_top_and_bottom_edges_differ() {
        let g = Geometry {
            keystone: Keystone { vertical: 50.0, amount: 100.0, ..Default::default() },
            ..Default::default()
        };
        let top = frame_to_source([0.0, 0.0], &g, SRC);
        let top_right = frame_to_source([1.0, 0.0], &g, SRC);
        let bottom = frame_to_source([0.0, 1.0], &g, SRC);
        let bottom_right = frame_to_source([1.0, 1.0], &g, SRC);
        let top_width = top_right[0] - top[0];
        let bottom_width = bottom_right[0] - bottom[0];
        assert!((top_width - bottom_width).abs() > 1e-2, "top {top_width} vs bottom {bottom_width}");
        // A perspective quad leaves frame corners on the image border.
        assert!(top[1].abs() < 1e-4 && (bottom[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn crop_for_aspect_16x9_is_correct() {
        let g = Geometry { aspect: AspectRatio::R16x9, ..Default::default() };
        let c = crop_for_aspect(&g, SRC);
        let ratio = (c.width * SRC.0 as f32) / (c.height * SRC.1 as f32);
        assert!((ratio - 16.0 / 9.0).abs() < 1e-3, "{ratio}");
        assert!((c.x + c.width * 0.5 - 0.5).abs() < 1e-4 && (c.y + c.height * 0.5 - 0.5).abs() < 1e-4);
    }

    #[test]
    fn crop_for_aspect_free_is_unchanged() {
        let crop = NormRect { x: 0.1, y: 0.2, width: 0.3, height: 0.4 };
        let g = Geometry { crop, ..Default::default() };
        assert_eq!(crop_for_aspect(&g, SRC), crop);
        let g = Geometry { crop, aspect: AspectRatio::Original, ..Default::default() };
        assert_eq!(crop_for_aspect(&g, SRC), crop);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let bad = Geometry {
            rotation: f32::NAN,
            orientation: 9,
            crop: NormRect { x: f32::NAN, y: 0.0, width: f32::INFINITY, height: 1.0 },
            aspect: AspectRatio::R16x9,
            keystone: Keystone { vertical: f32::NAN, horizontal: 1e9, amount: 100.0, aspect: f32::NAN, ..Default::default() },
            ..Default::default()
        };
        for p in [[f32::NAN, 0.5], [0.5, f32::INFINITY], [0.0, 0.0]] {
            let s = frame_to_source(p, &bad, (0, 0));
            assert!(s[0].is_finite() && s[1].is_finite(), "{s:?}");
            let o = frame_to_oriented(p, &bad);
            assert!(o[0].is_finite() && o[1].is_finite(), "{o:?}");
        }
        let (w, h) = frame_size(&bad, (0, 0));
        assert!(w.is_finite() && h.is_finite());
        let _ = crop_for_aspect(&bad, (0, 0));
    }

    /// The map that lived in `process.rs` before this module existed, kept here
    /// only to prove the new chain is a superset: same result when keystone is
    /// untouched and the aspect is free.
    struct LegacyMap {
        oriented: (f32, f32),
        crop: NormRect,
        rotation: f32,
        orientation: u8,
        flip_h: bool,
        flip_v: bool,
    }

    impl LegacyMap {
        fn new(src: (u32, u32), g: &Geometry) -> Self {
            Self {
                oriented: if g.orientation % 2 == 1 { (src.1 as f32, src.0 as f32) } else { (src.0 as f32, src.1 as f32) },
                crop: g.crop,
                rotation: g.rotation.to_radians(),
                orientation: g.orientation % 4,
                flip_h: g.flip_horizontal,
                flip_v: g.flip_vertical,
            }
        }

        fn map(&self, f: [f32; 2]) -> [f32; 2] {
            let mut p = [self.crop.x + f[0] * self.crop.width, self.crop.y + f[1] * self.crop.height];
            if self.rotation.abs() > 1e-5 {
                let (w, h) = self.oriented;
                let (cx, cy) = ((p[0] - 0.5) * w, (p[1] - 0.5) * h);
                let (s, c) = (-self.rotation).sin_cos();
                p = [(cx * c - cy * s) / w + 0.5, (cx * s + cy * c) / h + 0.5];
            }
            let mut q = p;
            if self.flip_h {
                q[0] = 1.0 - q[0];
            }
            if self.flip_v {
                q[1] = 1.0 - q[1];
            }
            match self.orientation {
                1 => [q[1], 1.0 - q[0]],
                2 => [1.0 - q[0], 1.0 - q[1]],
                3 => [1.0 - q[1], q[0]],
                _ => q,
            }
        }
    }

    #[test]
    fn untouched_keystone_matches_the_legacy_map() {
        let cases = [
            Geometry::default(),
            Geometry { crop: NormRect { x: 0.1, y: 0.2, width: 0.6, height: 0.5 }, ..Default::default() },
            Geometry { orientation: 3, flip_horizontal: true, ..Default::default() },
            Geometry { rotation: 12.0, flip_vertical: true, ..Default::default() },
            Geometry { rotation: -30.0, crop: NormRect { x: 0.05, y: 0.0, width: 0.9, height: 0.8 }, ..Default::default() },
        ];
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75], [1.0, 1.0]];
        for g in cases {
            let new = FrameMap::new(SRC, &g);
            let old = LegacyMap::new(SRC, &g);
            for f in points {
                assert_near(new.map(f).1, old.map(f));
            }
        }
    }

    #[test]
    fn frame_map_agrees_with_frame_to_source() {
        let g = Geometry {
            rotation: 7.0,
            orientation: 1,
            crop: NormRect { x: 0.2, y: 0.1, width: 0.5, height: 0.6 },
            aspect: AspectRatio::R4x5,
            keystone: Keystone { vertical: 30.0, amount: 80.0, ..Default::default() },
            ..Default::default()
        };
        let map = FrameMap::new(SRC, &g);
        for f in [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.3, 0.9]] {
            assert_near(map.map(f).1, frame_to_source(f, &g, SRC));
        }
    }

    #[test]
    fn cached_rotation_matches_per_pixel_sincos() {
        let threshold_degrees = 1e-5f32.to_degrees();
        let cases = [
            Geometry::default(),
            Geometry { rotation: 0.0005, ..Default::default() },
            Geometry { rotation: 0.0006, ..Default::default() },
            Geometry { rotation: threshold_degrees, ..Default::default() },
            Geometry { rotation: 12.5, ..Default::default() },
            Geometry { rotation: -30.0, ..Default::default() },
            Geometry {
                rotation: 7.0,
                orientation: 1,
                crop: NormRect { x: 0.2, y: 0.1, width: 0.5, height: 0.6 },
                aspect: AspectRatio::R4x5,
                keystone: Keystone { vertical: 30.0, amount: 80.0, ..Default::default() },
                flip_horizontal: true,
                flip_vertical: true,
                ..Default::default()
            },
            Geometry { rotation: f32::NAN, ..Default::default() },
            Geometry { rotation: f32::INFINITY, ..Default::default() },
            Geometry { rotation: f32::NEG_INFINITY, ..Default::default() },
        ];
        let points = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75], [1.0, 1.0], [f32::NAN, 0.5], [0.5, f32::INFINITY]];
        for (i, g) in cases.into_iter().enumerate() {
            let map = FrameMap::new(SRC, &g);
            for f in points {
                let reference = map_frame(f, &g, crop_for_aspect(&g, SRC), SRC);
                assert_eq!(map.map(f), reference, "case {i}, point {f:?}");
            }
        }
    }

    #[test]
    fn square_aspect_crops_a_centred_square() {
        let g = Geometry { aspect: AspectRatio::Square, ..Default::default() };
        let c = crop_for_aspect(&g, (300, 200));
        assert!((c.width * 300.0 - c.height * 200.0).abs() < 1e-2);
        assert!((c.x + c.width * 0.5 - 0.5).abs() < 1e-4);
    }

    #[test]
    fn keystone_maps_the_frame_corners_onto_its_trapezoid() {
        let g = Geometry {
            keystone: Keystone { vertical: 60.0, horizontal: -40.0, amount: 100.0, aspect: 0.0, ..Default::default() },
            ..Default::default()
        };
        // 0.35 * control amount, the trapezoid `KeystoneMap` builds.
        let (a, b) = (0.6 * 0.35, -0.4 * 0.35);
        assert_near(frame_to_source([0.0, 0.0], &g, SRC), [a, b]);
        assert_near(frame_to_source([1.0, 0.0], &g, SRC), [1.0 - a, 0.0]);
        assert_near(frame_to_source([1.0, 1.0], &g, SRC), [1.0, 1.0 - b]);
        assert_near(frame_to_source([0.0, 1.0], &g, SRC), [0.0, 1.0]);
    }

    #[test]
    fn both_flips_compose_to_a_half_turn() {
        let g = Geometry { flip_horizontal: true, flip_vertical: true, ..Default::default() };
        assert_near(frame_to_source([0.2, 0.3], &g, SRC), [0.8, 0.7]);
    }

    /// `_DSC5673.NEF` as the catalog stores it: 6048×4032, canvas
    /// 4043.054×6056.673, `ZCROP` 2525.8076;2348.0876;3328.8486;4993.2734,
    /// `ZROTATION` 270, no flip.
    const DSC5673: (u32, u32) = (6048, 4032);
    const DSC_CANVAS: [f32; 2] = [4043.054, 6056.673];

    fn dsc5673() -> Geometry {
        let (x1, y1, x2, y2) = (2525.8076171875f32, 2348.0876464844, 3328.8486328125, 4993.2734375);
        Geometry {
            orientation: 3,
            crop: NormRect {
                x: x1 / DSC_CANVAS[0],
                y: y1 / DSC_CANVAS[1],
                width: (x2 - x1) / DSC_CANVAS[0],
                height: (y2 - y1) / DSC_CANVAS[1],
            },
            canvas: DSC_CANVAS,
            aspect: AspectRatio::Original,
            keystone: Keystone { amount: 100.0, focal_length: 2.7792820930480957, ..Default::default() },
            ..Default::default()
        }
    }

    fn assert_dsc_corners(g: &Geometry, want: [[f32; 2]; 4]) {
        let map = FrameMap::new(DSC5673, g);
        let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for (f, want) in corners.iter().zip(want) {
            let (wx, wy) = (want[0], want[1]);
            let s = map.map(*f).1;
            let (px, py) = (s[0] * DSC5673.0 as f32, s[1] * DSC5673.1 as f32);
            assert!((px - wx).abs() < 0.05 && (py - wy).abs() < 0.05, "{f:?}: ({px:.4},{py:.4}) != ({wx},{wy})");
        }
    }

    /// The output is the crop's size in `ZCANVASSIZE` pixels, not its size in
    /// the 4032×6048 oriented image (which is 2 px narrower and 4 px shorter).
    #[test]
    fn real_dsc5673_canvas_size() {
        let (w, h) = FrameMap::new(DSC5673, &dsc5673()).frame_size();
        assert!((w - 803.041015625).abs() < 1e-2 && (h - 2645.1857910156).abs() < 1e-2, "{w}x{h}");
        let plain = FrameMap::new(DSC5673, &Geometry { orientation: 3, crop: dsc5673().crop, ..Default::default() });
        let (pw, ph) = plain.frame_size();
        assert!((pw - w).abs() > 1.0 && (ph - h).abs() > 1.0, "the canvas must move the size: {pw}x{ph} vs {w}x{h}");
    }

    /// Canvas corner → source pixel, matching `evidence\dsc5673-corners.txt`.
    #[test]
    fn real_dsc5673_crop_corners() {
        assert_dsc_corners(
            &dsc5673(),
            [[3704.2489, 2520.2806], [3704.2489, 3323.3216], [1059.0631, 3323.3216], [1059.0631, 2520.2806]],
        );
    }

    /// Flip acts on the turned canvas: with a 270° turn an H flip mirrors the
    /// source's y axis, a V flip mirrors x, and both mirror both.
    #[test]
    fn real_dsc5673_flip_corners() {
        let flip = |h: bool, v: bool| Geometry { flip_horizontal: h, flip_vertical: v, ..dsc5673() };
        assert_dsc_corners(
            &flip(true, false),
            [[3704.2489, 1511.7194], [3704.2489, 708.6784], [1059.0631, 708.6784], [1059.0631, 1511.7194]],
        );
        assert_dsc_corners(
            &flip(false, true),
            [[2343.7511, 2520.2806], [2343.7511, 3323.3216], [4988.9369, 3323.3216], [4988.9369, 2520.2806]],
        );
        assert_dsc_corners(
            &flip(true, true),
            [[2343.7511, 1511.7194], [2343.7511, 708.6784], [4988.9369, 708.6784], [4988.9369, 1511.7194]],
        );
    }

    /// `ZROTATION=270` arrives through the catalog as orientation 3 plus the
    /// canvas, and the four corners land on the same source pixels.
    #[test]
    fn real_dsc5673_rotation_270_through_the_catalog() {
        let mut row = std::collections::HashMap::new();
        for (k, v) in [
            ("ZROTATION", "270"),
            ("ZFLIP", "0"),
            ("ZCROPOUTSIDEIMAGE", "0"),
            ("ZCROP", "2525.8076171875;2348.0876464844;3328.8486328125;4993.2734375000"),
            ("ZCROPASPECTRATIONAME", "Original"),
            ("ZCANVASSIZE", "4043.054;6056.673"),
            ("ZGEOMETRYKEYSTONEAMOUNT", "100"),
            ("ZGEOMETRYKEYSTONEFOCALLENGTH", "2.7792820930480957"),
        ] {
            row.insert(k, Some(v.to_string()));
        }
        let settings = crate::c1_model::from_layer_row(&row);
        let mut g = crate::c1_model::to_adjustment_set(&settings).geometry;
        g.crop = crate::c1_model::crop_on_canvas(settings.crop_px, &g, Some(DSC5673));
        assert_eq!(g.orientation, 3, "270 is a quarter turn");
        assert!(g.rotation.abs() < 1e-6, "270 must not stay a fine rotation");
        assert_eq!(g.canvas, DSC_CANVAS);
        let [cx, cy] = [2525.8076, 2348.0876];
        let [half_width, half_height] = [3328.8486 / 2.0, 4993.2734 / 2.0];
        assert_dsc_corners(&g, [
            [cx + half_height, cy - half_width],
            [cx + half_height, cy + half_width],
            [cx - half_height, cy + half_width],
            [cx - half_height, cy - half_width],
        ]);
    }

    /// A stored canvas with no crop must not change an uncropped frame.
    #[test]
    fn stored_canvas_without_a_crop_is_ignored() {
        let g = Geometry { canvas: DSC_CANVAS, ..Default::default() };
        let map = FrameMap::new(SRC, &g);
        assert_eq!(map.frame_size(), (400.0, 300.0));
        for f in [[0.0, 0.0], [0.3, 0.7], [0.5, 0.5], [1.0, 1.0]] {
            assert_eq!(map.map(f), (f, f));
        }
    }
}
