//! One inverse map from the developed frame back to oriented and source space,
//! shared by preview, export and the colour picker.
//!
//! Forward order (what the viewer shows): oriented+flip → keystone → crop →
//! aspect → rotation. So the inverse, applied here, is crop → keystone →
//! rotation → flip → orientation, matching `process::FrameMap`'s chain with the
//! keystone stage inserted between crop and rotation.
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

fn clean_rect(r: NormRect) -> NormRect {
    let width = finite(r.width).clamp(0.0, 1.0);
    let height = finite(r.height).clamp(0.0, 1.0);
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

/// Size of the developed frame in source pixels.
#[cfg(test)]
pub(crate) fn frame_size(g: &Geometry, source_size: (u32, u32)) -> (f32, f32) {
    let (ow, oh) = oriented_size(g, source_size);
    let c = crop_for_aspect(g, source_size);
    (finite(ow * c.width), finite(oh * c.height))
}

/// The crop rect implied by an aspect ratio + portrait flag, centred in the oriented frame.
pub(crate) fn crop_for_aspect(g: &Geometry, source_size: (u32, u32)) -> NormRect {
    let c = clean_rect(g.crop);
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
    let (ow, oh) = oriented_size(g, source_size);
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

/// Keystone as a weak perspective: one 4-point homography in crop space.
///
/// `vertical` insets the top edge horizontally (top narrower than bottom for a
/// positive value), `horizontal` insets the left edge vertically (left shorter
/// than right), each scaled by `amount`; `aspect` stretches the result about the
/// crop centre to undo the squeeze the correction introduces.
pub(crate) struct KeystoneMap {
    /// Row-major 3x3 mapping a crop-space point to its source-crop point.
    m: [f32; 9],
    stretch: f32,
}

impl KeystoneMap {
    pub(crate) fn new(k: &Keystone) -> Self {
        let amount = (finite(k.amount) / 100.0).clamp(0.0, 1.0);
        let stretch = (1.0 + finite(k.aspect) / 100.0).clamp(0.1, 10.0);
        let vertical = (finite(k.vertical) / 100.0).clamp(-1.0, 1.0);
        let horizontal = (finite(k.horizontal) / 100.0).clamp(-1.0, 1.0);
        if amount < EPS || (vertical.abs() < EPS && horizontal.abs() < EPS) {
            return Self { m: IDENTITY, stretch: 1.0 };
        }
        let (a, b) = (vertical * amount * 0.35, horizontal * amount * 0.35);
        // Frame corners are the unit square; they sample a trapezoid in the source.
        let from = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let to = [[a, b], [1.0 - a, 0.0], [1.0, 1.0 - b], [0.0, 1.0]];
        let m = solve_homography(from, to).unwrap_or(IDENTITY);
        Self { m, stretch }
    }

    fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let m = &self.m;
        let w = m[6] * p[0] + m[7] * p[1] + m[8];
        if !w.is_finite() || w.abs() < EPS {
            return finite_pair(p);
        }
        let x = (m[0] * p[0] + m[1] * p[1] + m[2]) / w;
        let y = (m[3] * p[0] + m[4] * p[1] + m[5]) / w;
        // Stretch about the centre, so `aspect` alone is also centred.
        [finite(x), finite((y - 0.5) / self.stretch + 0.5)]
    }
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
    let p = KeystoneMap::new(&g.keystone).apply([crop.x + f[0] * crop.width, crop.y + f[1] * crop.height]);
    let mut p = p;
    let rotation = finite(g.rotation).to_radians();
    if rotation.abs() > 1e-5 && ow > EPS && oh > EPS {
        let (cx, cy) = ((p[0] - 0.5) * ow, (p[1] - 0.5) * oh);
        let (s, c) = (-rotation).sin_cos();
        p = [(cx * c - cy * s) / ow + 0.5, (cx * s + cy * c) / oh + 0.5];
    }
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
    map_frame(f, g, clean_rect(g.crop), (1, 1)).0
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
    crop: NormRect,
    /// `None` when the keystone is a no-op, so the per-pixel path stays a plain
    /// crop + rotation with no homography division.
    keystone: Option<KeystoneMap>,
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
        Self {
            oriented: oriented_size(g, source_size),
            crop: crop_for_aspect(g, source_size),
            keystone: {
                let k = KeystoneMap::new(&g.keystone);
                (k.m != IDENTITY).then_some(k)
            },
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
        (finite(self.oriented.0 * self.crop.width), finite(self.oriented.1 * self.crop.height))
    }

    /// Oriented (after quarter turns and rotation, before crop) size, for aspect-dependent masks.
    pub(crate) fn oriented(&self) -> (f32, f32) {
        self.oriented
    }

    /// Frame point → (oriented-image point, source point).
    pub(crate) fn map(&self, f: [f32; 2]) -> ([f32; 2], [f32; 2]) {
        let f = finite_pair(f);
        let (ow, oh) = self.oriented;
        let mut p = [self.crop.x + f[0] * self.crop.width, self.crop.y + f[1] * self.crop.height];
        if let Some(k) = &self.keystone {
            p = k.apply(p);
        }
        if self.rotation.abs() > 1e-5 && ow > EPS && oh > EPS {
            let (cx, cy) = ((p[0] - 0.5) * ow, (p[1] - 0.5) * oh);
            let (s, c) = (self.rotation_sin, self.rotation_cos);
            p = [(cx * c - cy * s) / ow + 0.5, (cx * s + cy * c) / oh + 0.5];
        }
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
            keystone: Keystone { vertical: 80.0, horizontal: -60.0, amount: 0.0, aspect: 20.0 },
            ..Default::default()
        };
        for f in [[0.0, 0.0], [1.0, 0.0], [0.3, 0.7], [0.5, 0.5]] {
            assert_near(frame_to_source(f, &g, SRC), f);
        }
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
            keystone: Keystone { vertical: f32::NAN, horizontal: 1e9, amount: 100.0, aspect: f32::NAN },
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
            keystone: Keystone { vertical: 60.0, horizontal: -40.0, amount: 100.0, aspect: 0.0 },
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
}
