#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crop {
    pub center: [f64; 2],
    pub size: [f64; 2],
}

#[derive(Clone, Copy, Debug)]
pub enum Bounds {
    Mode1 { image: [u32; 2] },
    Mode8 { min: [f64; 2], max: [f64; 2] },
    Mode10 { size: [f64; 2] },
}

impl Bounds {
    fn mode(self) -> u8 {
        match self {
            Self::Mode1 { .. } => 2,
            Self::Mode8 { .. } => 9,
            Self::Mode10 { .. } => 10,
        }
    }

    fn edges(self) -> ([f64; 2], [f64; 2]) {
        match self {
            Self::Mode1 { image } => ([0.0; 2], image.map(f64::from)),
            Self::Mode8 { min, max } => (min, max),
            Self::Mode10 { size } => ([0.0; 2], size),
        }
    }

    fn distance(self, [x, y]: [f64; 2]) -> f64 {
        let ([left, top], [right, bottom]) = self.edges();
        let dx = (x - left).min(right - x);
        let dy = (y - top).min(bottom - y);
        if dx < 0.0 || dy < 0.0 {
            -dx.min(0.0).hypot(dy.min(0.0))
        } else {
            dx.min(dy)
        }
    }
}

/// Limit a canvas-space crop around its existing center. The caller supplies
/// the verified single-point and four-point transforms from the same job.
pub fn limit_center_keep_aspect(
    crop: Crop,
    bounds: Bounds,
    map_center: impl FnOnce(u8, [f64; 2]) -> [f64; 2],
    map_trial: impl Fn(u8, [[f64; 2]; 4]) -> [[f64; 2]; 4],
) -> Crop {
    let [cx, cy] = crop.center;
    let [w, h] = crop.size;
    if bounds.distance(map_center(bounds.mode(), crop.center)) <= 0.0 {
        return Crop { size: [0.0; 2], ..crop };
    }

    let left = cx - w / 2.0;
    let right = cx + w / 2.0;
    let top = cy - h / 2.0;
    let bottom = cy + h / 2.0;
    let sides = [
        [[left, top], [right, top]],
        [[left, bottom], [right, bottom]],
        [[left, top], [left, bottom]],
        [[right, top], [right, bottom]],
    ];
    let mut scale: f64 = 1.0;
    for [a, b] in sides {
        let points = [crop.center, a, b, crop.center];
        let span = (cx.max(a[0]).max(b[0]) - cx.min(a[0]).min(b[0]))
            .max(cy.max(a[1]).max(b[1]) - cy.min(a[1]).min(b[1]));
        scale = scale.min(search_side(span, |s| {
            let trial = points.map(|[x, y]| [cx + s * (x - cx), cy + s * (y - cy)]);
            map_trial(bounds.mode(), trial)
                .into_iter()
                .map(|point| bounds.distance(point))
                .fold(f64::INFINITY, f64::min)
        }));
    }
    if scale >= 1.0 {
        return crop;
    }
    Crop { size: [w * scale, h * scale], ..crop }
}

fn search_side(span: f64, mut distance: impl FnMut(f64) -> f64) -> f64 {
    if span < 0.01 {
        return 1.0;
    }
    let (mut lo, mut d_lo, mut hi, mut d_hi) = (0.0, -1.0, 1.0, -1.0);
    let mut s = 1.0;
    for _ in 0..51 {
        let d = distance(s);
        if d < 0.0 {
            (hi, d_hi) = (s, d);
        } else {
            (lo, d_lo) = (s, d);
        }
        let (a, b) = (d_hi.abs(), d_lo.abs());
        s = if b <= 5.0 * a && a <= 5.0 * b && d_lo >= 0.0 && d_hi != d_lo {
            (d_hi * lo - hi * d_lo) / (d_hi - d_lo)
        } else {
            (hi + lo) / 2.0
        };
        if (hi - lo) * span < 0.01 {
            break;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limit(crop: Crop, bounds: Bounds, mode: u8) -> Crop {
        limit_center_keep_aspect(crop, bounds, |m, p| {
            assert_eq!(m, mode);
            p
        }, |m, p| {
            assert_eq!(m, mode);
            p
        })
    }

    #[test]
    fn constraining_side_preserves_center_and_aspect() {
        let crop = Crop { center: [20.0, 35.0], size: [80.0, 40.0] };
        for (bounds, mode) in [
            (Bounds::Mode1 { image: [100, 100] }, 2),
            (Bounds::Mode10 { size: [100.0, 100.0] }, 10),
        ] {
            let got = limit(crop, bounds, mode);
            assert_eq!(got.center, crop.center);
            assert!((got.size[0] - 40.0).abs() < 0.01);
            assert_eq!(got.size[0], got.size[1] * 2.0);
        }
        let crop = Crop { center: [30.0, 55.0], ..crop };
        let bounds = Bounds::Mode8 { min: [10.0, 20.0], max: [110.0, 120.0] };
        assert!((limit(crop, bounds, 9).size[0] - 40.0).abs() < 0.01);
        let inside = Crop { center: [60.0, 70.0], ..crop };
        assert_eq!(limit(inside, bounds, 9), inside);
    }

    #[test]
    fn outside_and_boundary_centers_return_empty_without_trials() {
        for center in [[-1.0, 50.0], [0.0, 50.0], [100.0, 50.0], [50.0, 101.0]] {
            let crop = limit_center_keep_aspect(
                Crop { center, size: [30.0, 20.0] },
                Bounds::Mode10 { size: [100.0, 100.0] },
                |_, p| p,
                |_, _| panic!("outside center must not enter side search"),
            );
            assert_eq!(crop, Crop { center, size: [0.0; 2] });
        }
    }

    #[test]
    fn termination_uses_span_and_evaluation_cap() {
        assert_eq!(search_side(0.009, |_| panic!("degenerate span")), 1.0);
        let mut count = 0;
        assert_eq!(search_side(1.0, |_| { count += 1; 1.0 }), 1.0);
        assert_eq!(count, 1);
        let mut count = 0;
        search_side(f64::MAX, |s| { count += 1; if s > 0.5 { -1.0 } else { 1.0 } });
        assert_eq!(count, 51);
    }
}
