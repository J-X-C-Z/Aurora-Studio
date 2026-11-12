//! A small CPU development pipeline. It is not color-accurate; it exists so every
//! control in the interface has a visible, plausible effect during UI work.

use rayon::prelude::*;

use crate::*;

/// Source pixels with a mip pyramid for fast zoomed-out sampling.
#[doc(hidden)]
pub struct Source {
    pub(crate) levels: Vec<Plane>,
    /// Lens EXIF of the capture, for the optional C1 lens profile.
    pub(crate) lens_shot: Option<super::lens_profile::LensShot>,
    /// EXIF-matched lens profile, resolved once when the source is decoded.
    lens_auto: Option<super::lens_profile::LensWarp>,
    /// True when the pixels are the camera's own JPEG preview, not sensor data.
    camera_preview: bool,
}

pub(crate) struct Plane {
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Linear-light RGB.
    pub(crate) data: Vec<[f32; 3]>,
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Linear → sRGB over `0 ..= 4` linear, so highlight recovery has headroom to pull from.
struct GammaLut(Vec<f32>);

impl GammaLut {
    const SIZE: usize = 8192;
    const RANGE: f32 = 4.0;

    fn new() -> Self {
        Self((0..Self::SIZE).map(|i| linear_to_srgb(i as f32 / (Self::SIZE - 1) as f32 * Self::RANGE)).collect())
    }

    fn get(&self, v: f32) -> f32 {
        let i = (v.max(0.0) / Self::RANGE * (Self::SIZE - 1) as f32) as usize;
        self.0[i.min(Self::SIZE - 1)]
    }
}

impl Source {
    pub(crate) fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Self {
        let lut: Vec<f32> = (0..256).map(|i| srgb_to_linear(i as f32 / 255.0)).collect();
        let data: Vec<[f32; 3]> = rgba
            .par_chunks_exact(4)
            .map(|p| [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]])
            .collect();
        Self::from_linear(width, height, data)
    }

    /// 16-bit RGBA samples that are already linear light, which is what libraw
    /// hands over. Eight-bit sRGB transport quantises the shadows in 256 steps
    /// of the encoded range; this keeps 65535.
    #[doc(hidden)]
    pub fn from_rgba16(width: u32, height: u32, rgba: &[u16]) -> Self {
        let data: Vec<[f32; 3]> = rgba
            .par_chunks_exact(4)
            .map(|p| [p[0] as f32 / 65535.0, p[1] as f32 / 65535.0, p[2] as f32 / 65535.0])
            .collect();
        Self::from_linear(width, height, data)
    }

    /// Wraps level 0 and builds the mip chain shared by both bit depths.
    /// A zero-size or short frame becomes one black pixel so `sample` stays in bounds.
    fn from_linear(width: u32, height: u32, mut data: Vec<[f32; 3]>) -> Self {
        if width == 0 || height == 0 || data.len() < (width as usize) * (height as usize) {
            return Self {
                levels: vec![Plane { width: 1, height: 1, data: vec![[0.0; 3]] }],
                lens_shot: None,
                lens_auto: None,
                camera_preview: false,
            };
        }
        data.truncate((width as usize) * (height as usize));
        let mut levels = vec![Plane { width, height, data }];
        while levels.last().is_some_and(|l| l.width > 256 && l.height > 256) {
            let prev = levels.last().unwrap();
            let (w, h) = (prev.width / 2, prev.height / 2);
            let data: Vec<[f32; 3]> = (0..(w * h) as usize)
                .into_par_iter()
                .map(|i| {
                    let (x, y) = ((i as u32 % w) * 2, (i as u32 / w) * 2);
                    let mut acc = [0.0; 3];
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let p = prev.data[((y + dy) * prev.width + x + dx) as usize];
                        for k in 0..3 {
                            acc[k] += p[k] * 0.25;
                        }
                    }
                    acc
                })
                .collect();
            levels.push(Plane { width: w, height: h, data });
        }
        Self { levels, lens_shot: None, lens_auto: None, camera_preview: false }
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.levels[0].width, self.levels[0].height)
    }

    /// Marks pixels that came from the camera's embedded preview: they are already
    /// developed, so a camera profile or film curve would double-process them.
    pub(crate) fn with_camera_preview(mut self, preview: bool) -> Self {
        self.camera_preview = preview;
        self
    }

    pub(crate) fn with_lens_shot(mut self, shot: Option<super::lens_profile::LensShot>) -> Self {
        self.lens_auto = shot.as_ref().and_then(super::lens_profile::auto_warp);
        self.lens_shot = shot;
        self
    }

    fn sample(&self, level: usize, sx: f32, sy: f32) -> Option<[f32; 3]> {
        if !(0.0..=1.0).contains(&sx) || !(0.0..=1.0).contains(&sy) {
            return None;
        }
        let plane = &self.levels[level.min(self.levels.len() - 1)];
        // `.max(0.0)` keeps the one-pixel planes of an empty source in range.
        let x = (sx * plane.width as f32 - 0.5).clamp(0.0, (plane.width as f32 - 1.001).max(0.0));
        let y = (sy * plane.height as f32 - 0.5).clamp(0.0, (plane.height as f32 - 1.001).max(0.0));
        let (x0, y0) = (x as u32, y as u32);
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let x1 = (x0 + 1).min(plane.width - 1);
        let y1 = (y0 + 1).min(plane.height - 1);
        let at = |xx: u32, yy: u32| plane.data[(yy * plane.width + xx) as usize];
        let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        let mut out = [0.0; 3];
        for k in 0..3 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bottom = c[k] + (d[k] - c[k]) * tx;
            out[k] = top + (bottom - top) * ty;
        }
        Some(out)
    }

    /// Bicubic tap for a region magnified past 1:1. Bilinear turns a sharp edge
    /// into a ramp; the wider kernel is what keeps zoomed-in detail readable.
    fn sample_cubic(&self, level: usize, sx: f32, sy: f32) -> Option<[f32; 3]> {
        if !(0.0..=1.0).contains(&sx) || !(0.0..=1.0).contains(&sy) {
            return None;
        }
        let plane = &self.levels[level.min(self.levels.len() - 1)];
        let (w, h) = (plane.width as i64, plane.height as i64);
        let x = (sx * w as f32 - 0.5).clamp(0.0, (w as f32 - 1.001).max(0.0));
        let y = (sy * h as f32 - 0.5).clamp(0.0, (h as f32 - 1.001).max(0.0));
        let (x0, y0) = (x as i64, y as i64);
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let at = |xx: i64, yy: i64| {
            plane.data[(yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)) as usize]
        };
        let mut out = [0.0; 3];
        for k in 0..3 {
            let cols: [f32; 4] = std::array::from_fn(|i| {
                let xx = x0 - 1 + i as i64;
                catmull_rom(at(xx, y0 - 1)[k], at(xx, y0)[k], at(xx, y0 + 1)[k], at(xx, y0 + 2)[k], ty)
            });
            out[k] = catmull_rom(cols[0], cols[1], cols[2], cols[3], tx);
        }
        Some(out)
    }
}

/// Catmull-Rom blend of `b` and `c` at `t`, steering with the outer taps.
fn catmull_rom(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let (t2, t3) = (t * t, t * t * t);
    0.5 * (2.0 * b + (c - a) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 + (3.0 * b - 3.0 * c + d - a) * t3)
}

/// Approximate white point of a black body at `kelvin`, normalized to green.
fn kelvin_rgb(kelvin: f32) -> [f32; 3] {
    let t = kelvin.clamp(1000.0, 40000.0) / 100.0;
    let r = if t <= 66.0 { 255.0 } else { 329.698_73 * (t - 60.0).powf(-0.133_204_76) };
    let g = if t <= 66.0 { 99.470_8 * t.ln() - 161.119_57 } else { 288.122_16 * (t - 60.0).powf(-0.075_514_85) };
    let b = if t >= 66.0 {
        255.0
    } else if t <= 19.0 {
        0.0
    } else {
        138.517_73 * (t - 10.0).ln() - 305.044_8
    };
    let (r, g, b) = (r.clamp(1.0, 255.0), g.clamp(1.0, 255.0), b.clamp(1.0, 255.0));
    [srgb_to_linear(r / 255.0) / srgb_to_linear(g / 255.0), 1.0, srgb_to_linear(b / 255.0) / srgb_to_linear(g / 255.0)]
}

pub(crate) fn white_balance_gains(kelvin: f32, tint: f32) -> [f32; 3] {
    let shot = kelvin_rgb(5600.0);
    let set = kelvin_rgb(kelvin);
    let tint_gain = 1.0 - tint / 100.0 * 0.6;
    [shot[0] / set[0], tint_gain, shot[2] / set[2]]
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn sstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn hue_rgb(hue: f32) -> [f32; 3] {
    let h = (hue.rem_euclid(360.0)) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    }
}

fn rgb_to_hsv(c: [f32; 3]) -> (f32, f32, f32) {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let d = max - min;
    let h = if d < 1e-6 {
        0.0
    } else if max == c[0] {
        60.0 * ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if max == c[1] {
        60.0 * ((c[2] - c[0]) / d + 2.0)
    } else {
        60.0 * ((c[0] - c[1]) / d + 4.0)
    };
    let s = if max < 1e-6 { 0.0 } else { d / max };
    (h, s, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let base = hue_rgb(h);
    [v * (1.0 - s + s * base[0]), v * (1.0 - s + s * base[1]), v * (1.0 - s + s * base[2])]
}

fn hue_distance(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

/// Offset a color wheel adds, centered on neutral.
fn wheel_offset(w: &ColorWheel, strength: f32) -> [f32; 3] {
    let c = hue_rgb(w.hue);
    let mean = (c[0] + c[1] + c[2]) / 3.0;
    let s = w.saturation * strength;
    let l = w.lightness * 0.25;
    [(c[0] - mean) * s + l, (c[1] - mean) * s + l, (c[2] - mean) * s + l]
}

fn contrast(g: f32, amount: f32) -> f32 {
    if amount.abs() < 1e-4 {
        return g;
    }
    let k = 1.0 + amount;
    if k >= 1.0 {
        // Sigmoid-ish S curve that does not clip hard.
        let x = (g - 0.5) * k;
        0.5 + x / (1.0 + (x.abs() * (k - 1.0) * 0.8))
    } else {
        0.5 + (g - 0.5) * k
    }
}

/// The highlight shoulder: rolls anything above 0.9 back so it cannot clip.
fn shoulder(mut g: [f32; 3]) -> [f32; 3] {
    for v in &mut g {
        if *v > 0.9 {
            let over = *v - 0.9;
            *v = 0.9 + over / (1.0 + over * 10.0);
        }
    }
    g
}

fn contrast_all(g: [f32; 3], amount: f32) -> [f32; 3] {
    [contrast(g[0], amount), contrast(g[1], amount), contrast(g[2], amount)]
}

/// A coarse raster of one layer's shape mask in oriented-image space.
#[derive(Clone)]
struct MaskGrid {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

impl MaskGrid {
    /// `cell_px` is the size of one grid cell in source pixels, so the mask's
    /// pixel radii (`feather`, `refine`) convert to grid pixels.
    /// `guide` is an optional luminance grid (same layout) used by `refine`.
    ///
    /// `built` holds the masks of the layers already planned, in draw order;
    /// a [`MaskShape::Copy`] reads its source from there.
    fn build(
        mask: &Mask,
        aspect: f32,
        cell_px: f32,
        guide: Option<&[f32]>,
        built: &[(u64, MaskGrid)],
    ) -> Self {
        let (w, h) = grid_dims(aspect);
        let mut data = vec![0.0f32; w * h];
        // Resolve each AI shape's cached plane once, not per grid cell.
        let planes: Vec<Option<std::sync::Arc<crate::aimask::MaskPlane>>> = mask
            .shapes
            .iter()
            .map(|shape| match shape {
                MaskShape::Ai(ai) => ai.cache_key.as_deref().and_then(crate::aimask::lookup),
                _ => None,
            })
            .collect();
        for (i, v) in data.iter_mut().enumerate() {
            let p = [((i % w) as f32 + 0.5) / w as f32, ((i / w) as f32 + 0.5) / h as f32];
            let mut m = 0.0f32;
            for (shape, plane) in mask.shapes.iter().zip(&planes) {
                let (op, value) = match shape {
                    MaskShape::Copy { from_layer, op } => (
                        *op,
                        built
                            .iter()
                            .find(|(id, _)| id == from_layer)
                            .map_or(0.0, |(_, g)| g.sample(p)),
                    ),
                    other => shape_value(other, p, aspect, plane.as_deref()),
                };
                m = match op {
                    MaskOp::Add => m + value,
                    MaskOp::Subtract => m - value,
                }
                .clamp(0.0, 1.0);
            }
            *v = m;
        }
        let mut grid = Self { w, h, data };
        if mask.feather > 0.0 {
            let r = (mask.feather / cell_px.max(1e-3)).round() as i32;
            if r > 0 {
                grid.data = box_blur_grid(&grid.data, w, h, r);
            }
        }
        if mask.refine > 0.0 {
            let r = (mask.refine / cell_px.max(1e-3)).round() as i32;
            if r > 0 {
                if let Some(guide) = guide {
                    grid.data = guided_filter_grid(&grid.data, guide, w, h, r);
                }
            }
        }
        grid
    }

    fn sample(&self, p: [f32; 2]) -> f32 {
        let x = (p[0] * self.w as f32 - 0.5).clamp(0.0, self.w as f32 - 1.001);
        let y = (p[1] * self.h as f32 - 0.5).clamp(0.0, self.h as f32 - 1.001);
        let (x0, y0) = (x as usize, y as usize);
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let at = |xx: usize, yy: usize| self.data[yy * self.w + xx];
        let top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * tx;
        let bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * tx;
        top + (bottom - top) * ty
    }
}

/// Grid resolution of a mask raster: 256 cells across, the height follows the
/// oriented aspect.
fn grid_dims(aspect: f32) -> (usize, usize) {
    let w = 256usize;
    let h = ((256.0 / aspect).round() as usize).clamp(32, 512);
    (w, h)
}

/// The ellipse fallback looks like a real mask in the viewer, so say once why
/// it is being used.
fn warn_missing_ai_mask(ai: &AiMask) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        log::warn!(
            "no segmentation plane for the {:?} AI mask (cache key {:?}); using the placeholder shape",
            ai.kind,
            ai.cache_key
        );
    }
}

fn shape_value(
    shape: &MaskShape,
    p: [f32; 2],
    aspect: f32,
    plane: Option<&crate::aimask::MaskPlane>,
) -> (MaskOp, f32) {
    match shape {
        MaskShape::Brush(stroke) => {
            let mut best = 0.0f32;
            let r = stroke.size.max(1e-4);
            let hard = (stroke.hardness / 100.0).clamp(0.0, 0.99);
            for pt in &stroke.points {
                let dx = (p[0] - pt[0]) * aspect.max(1.0);
                let dy = (p[1] - pt[1]) * (1.0 / aspect).max(1.0);
                let d = (dx * dx + dy * dy).sqrt() / r;
                let v = 1.0 - sstep(hard, 1.0, d);
                best = best.max(v * pt[2].max(0.05));
            }
            (stroke.op, best * stroke.opacity / 100.0)
        }
        MaskShape::Linear(g) => {
            let d = [g.end[0] - g.start[0], g.end[1] - g.start[1]];
            let len2 = (d[0] * d[0] + d[1] * d[1]).max(1e-6);
            let t = ((p[0] - g.start[0]) * d[0] + (p[1] - g.start[1]) * d[1]) / len2;
            (g.op, 1.0 - sstep(0.0, 1.0, t))
        }
        MaskShape::Radial(g) => {
            let (s, c) = g.rotation.to_radians().sin_cos();
            let dx = (p[0] - g.center[0]) * aspect;
            let dy = p[1] - g.center[1];
            let rx = (dx * c + dy * s) / (g.radius[0] * aspect).max(1e-4);
            let ry = (-dx * s + dy * c) / g.radius[1].max(1e-4);
            let d = (rx * rx + ry * ry).sqrt();
            let inner = 1.0 - (g.feather / 100.0).clamp(0.0, 1.0);
            let v = 1.0 - sstep(inner, 1.0, d);
            (g.op, if g.inverted { 1.0 - v } else { v })
        }
        MaskShape::Ai(ai) => {
            if let Some(plane) = plane {
                return (ai.op, plane.sample(p));
            }
            warn_missing_ai_mask(ai);
            let subject = {
                let dx = (p[0] - 0.5) / 0.28;
                let dy = (p[1] - 0.58) / 0.42;
                1.0 - sstep(0.85, 1.0, (dx * dx + dy * dy).sqrt())
            };
            let v = match ai.kind {
                AiMaskKind::Subject | AiMaskKind::Person | AiMaskKind::Clothing => subject,
                AiMaskKind::Background => 1.0 - subject,
                AiMaskKind::Sky => 1.0 - sstep(0.35, 0.5, p[1]),
                AiMaskKind::Skin | AiMaskKind::Eyes | AiMaskKind::Lips => {
                    let dx = (p[0] - 0.5) / 0.15;
                    let dy = (p[1] - 0.42) / 0.2;
                    1.0 - sstep(0.8, 1.0, (dx * dx + dy * dy).sqrt())
                }
                AiMaskKind::Hair => {
                    let dx = (p[0] - 0.5) / 0.2;
                    let dy = (p[1] - 0.3) / 0.15;
                    1.0 - sstep(0.8, 1.0, (dx * dx + dy * dy).sqrt())
                }
            };
            (ai.op, v)
        }
        // Copies are resolved by `MaskGrid::build`, which has the other layers.
        MaskShape::Copy { op, .. } => (*op, 0.0),
    }
}

/// Separable box blur over a grid, radius `r` cells, edges clamped.
fn box_blur_grid(src: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
    if r <= 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[y * w + (x as i32 + k).clamp(0, w as i32 - 1) as usize];
            }
            tmp[y * w + x] = sum / (2 * r + 1) as f32;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += tmp[(y as i32 + k).clamp(0, h as i32 - 1) as usize * w + x];
            }
            out[y * w + x] = sum / (2 * r + 1) as f32;
        }
    }
    out
}

/// One guided filter pass: the mask is smoothed toward the guide's luminance
/// edges, so a refine radius snaps the mask to the object boundary instead of
/// smearing across it. `guide` holds luminance in 0..1.
///
/// This is the two-window He et al. filter: the local means come from box
/// blurs, the linear coefficients are averaged over the same window, and the
/// result is the filtered mask.
fn guided_filter_grid(mask: &[f32], guide: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
    let mean_mask = box_blur_grid(mask, w, h, r);
    let mean_guide = box_blur_grid(guide, w, h, r);
    let prod: Vec<f32> = mask.iter().zip(guide).map(|(a, b)| a * b).collect();
    let mean_prod = box_blur_grid(&prod, w, h, r);
    let sq: Vec<f32> = guide.iter().map(|v| v * v).collect();
    let mean_sq = box_blur_grid(&sq, w, h, r);
    let eps = 1e-3;
    let mut av = vec![0.0f32; w * h];
    let mut bv = vec![0.0f32; w * h];
    for i in 0..w * h {
        let cov = mean_prod[i] - mean_mask[i] * mean_guide[i];
        let var = (mean_sq[i] - mean_guide[i] * mean_guide[i]).max(0.0);
        let a = cov / (var + eps);
        av[i] = a;
        bv[i] = mean_mask[i] - a * mean_guide[i];
    }
    let mean_a = box_blur_grid(&av, w, h, r);
    let mean_b = box_blur_grid(&bv, w, h, r);
    (0..w * h).map(|i| (mean_a[i] * mask[i] + mean_b[i]).clamp(0.0, 1.0)).collect()
}

/// Oriented-image point to source-image point: flips first, then the
/// quarter turn, which is the inverse of what [`super::geometry::FrameMap`]
/// does to reach the source.
fn oriented_to_source(o: [f32; 2], g: &Geometry) -> [f32; 2] {
    let mut q = o;
    if g.flip_horizontal {
        q[0] = 1.0 - q[0];
    }
    if g.flip_vertical {
        q[1] = 1.0 - q[1];
    }
    match g.orientation % 4 {
        1 => [q[1], 1.0 - q[0]],
        2 => [1.0 - q[0], 1.0 - q[1]],
        3 => [1.0 - q[1], q[0]],
        _ => q,
    }
}

/// The oriented image as RGBA bytes, `long` pixels on the long side, which is
/// what the ONNX segmenters see. Their masks share the coordinate space of every
/// other mask, so geometry (orientation, flips) has to be baked in here.
pub(crate) fn ai_input_rgba(src: &Source, g: &Geometry, long: u32) -> (Vec<u8>, u32, u32) {
    let (ow, oh) = super::geometry::FrameMap::new(src.size(), g).oriented();
    let scale = long as f32 / ow.max(oh).max(1.0);
    let w = ((ow * scale).round() as u32).max(1);
    let h = ((oh * scale).round() as u32).max(1);
    let level = src
        .levels
        .iter()
        .rposition(|p| p.width >= w && p.height >= h)
        .unwrap_or(0);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let o = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32];
            let s = oriented_to_source(o, g);
            let c = src.sample(level, s[0], s[1]).unwrap_or([0.0; 3]);
            for value in c {
                rgba.push((linear_to_srgb(value.max(0.0)).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            rgba.push(255);
        }
    }
    (rgba, w, h)
}

/// A coarse luminance grid of the source in oriented-image space, used as the
/// guide for the mask's refine edge.
fn luma_guide(src: &Source, g: &Geometry, w: usize, h: usize) -> Vec<f32> {
    let level = src
        .levels
        .iter()
        .position(|p| p.width.max(p.height) <= 256)
        .unwrap_or(src.levels.len() - 1);
    (0..w * h)
        .map(|i| {
            let o = [
                (i % w) as f32 / w as f32 + 0.5 / w as f32,
                (i / w) as f32 / h as f32 + 0.5 / h as f32,
            ];
            let s = oriented_to_source(o, g);
            match src.sample(level, s[0], s[1]) {
                Some(c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).clamp(0.0, 1.0),
                None => 0.0,
            }
        })
        .collect()
}

fn luma_range_weight(r: &LumaRange, l: f32) -> f32 {
    let w = if l < r.low_inner {
        sstep(r.low_outer, r.low_inner.max(r.low_outer + 1e-3), l)
    } else if l > r.high_inner {
        1.0 - sstep(r.high_inner, r.high_outer.max(r.high_inner + 1e-3), l)
    } else {
        1.0
    };
    if r.invert { 1.0 - w } else { w }
}

fn color_range_weight(r: &ColorRange, c: [f32; 3]) -> f32 {
    let (h, s, _) = rgb_to_hsv(c);
    let half = r.width / 2.0;
    let d = hue_distance(h, r.hue);
    let hue_w = 1.0 - sstep(half, half + r.smoothness.max(0.1), d);
    hue_w * sstep(r.min_saturation * 0.5, r.min_saturation + 0.05, s)
}

struct LayerPlan<'a> {
    layer: &'a Layer,
    grid: Option<MaskGrid>,
    overlay: bool,
    wb_gains: Option<[f32; 3]>,
    /// The layer's curves, precomputed as 256-entry LUTs.
    curve_rgb: Option<[f32; 256]>,
    curve_r: Option<[f32; 256]>,
    curve_g: Option<[f32; 256]>,
    curve_b: Option<[f32; 256]>,
    curve_luma: Option<[f32; 256]>,
    levels: bool,
    /// Constants a per-pixel `local()` would otherwise recompute every time.
    exposure_gain: f32,
    saturation: f32,
    opacity: f32,
    /// Spatial operators; they need neighbours, so they run on the whole
    /// frame after the layers have been mixed.
    moire: Moire,
    sharpening: Option<Sharpening>,
    noise_reduction: Option<NoiseReduction>,
}

/// Builds the per-layer plan in draw order. Each mask is rasterized before the
/// next layer starts, so a copy mask can read the layers below it.
fn layer_plan<'a>(
    l: &'a Layer,
    adj: &AdjustmentSet,
    grid: Option<MaskGrid>,
    mask_overlay: Option<u64>,
) -> LayerPlan<'a> {
    let a = &l.adjustments;
    LayerPlan {
        layer: l,
        grid,
        overlay: mask_overlay == Some(l.id),
        wb_gains: a.white_balance.as_ref().map(|w| {
            let g = white_balance_gains(w.kelvin, w.tint);
            let base = white_balance_gains(adj.white_balance.kelvin, adj.white_balance.tint);
            [g[0] / base[0], g[1] / base[1], g[2] / base[2]]
        }),
        curve_rgb: lut_of(&a.curves.rgb),
        curve_r: lut_of(&a.curves.red),
        curve_g: lut_of(&a.curves.green),
        curve_b: lut_of(&a.curves.blue),
        curve_luma: lut_of(&a.curves.luma),
        levels: a.levels != Levels::default(),
        exposure_gain: 2f32.powf(a.exposure.exposure * 0.6),
        saturation: 1.0 + a.exposure.saturation / 100.0,
        opacity: (l.opacity / 100.0) * (l.mask.density.clamp(0.0, 100.0) / 100.0),
        moire: a.moire.clone(),
        sharpening: a.sharpening.clone(),
        noise_reduction: a.noise_reduction.clone(),
    }
}

fn layer_plans<'a>(
    adj: &'a AdjustmentSet,
    aspect: f32,
    cell_px: f32,
    guide: Option<&[f32]>,
    mask_overlay: Option<u64>,
) -> Vec<LayerPlan<'a>> {
    let mut built: Vec<(u64, MaskGrid)> = Vec::new();
    let mut plans = Vec::new();
    for l in adj.layers.iter().filter(|l| l.enabled && l.opacity > 0.0) {
        let grid = (!l.mask.shapes.is_empty())
            .then(|| MaskGrid::build(&l.mask, aspect, cell_px, guide, &built));
        if let Some(g) = &grid {
            built.push((l.id, g.clone()));
        }
        plans.push(layer_plan(l, adj, grid, mask_overlay));
    }
    plans
}

/// Everything precomputed once per render.
struct Plan<'a> {
    adj: &'a AdjustmentSet,
    /// Camera input profile; `None` keeps the source's own sRGB encoding.
    input_profile: Option<std::sync::Arc<crate::icc::Profile>>,
    /// Capture One film curve, applied to linear values before gamma.
    film_lut: Option<[f32; 256]>,
    gamma: GammaLut,
    /// Exact sRGB instead of the preview LUT, so 16-bit export is not capped
    /// at the LUT's 8192 steps. The film curve still takes precedence in `encode`.
    fine: bool,
    wb: [f32; 3],
    exposure: f32,
    base_contrast: f32,
    curve_rgb: Option<[f32; 256]>,
    curve_r: Option<[f32; 256]>,
    curve_g: Option<[f32; 256]>,
    curve_b: Option<[f32; 256]>,
    curve_luma: Option<[f32; 256]>,
    levels: bool,
    color_editor: bool,
    color_balance: bool,
    layers: Vec<LayerPlan<'a>>,
}

fn lut_of(c: &Curve) -> Option<[f32; 256]> {
    (!c.is_identity()).then(|| c.lut())
}

fn lut_get(lut: &[f32; 256], v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0) * 255.0;
    let i = (x as usize).min(254);
    let t = x - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * t
}

pub struct Rendered {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub(crate) histogram: Histogram,
}

/// A 16-bit frame for export. Previews keep using the 8-bit [`Rendered`].
pub struct Rendered16 {
    pub rgba: Vec<u16>,
    pub width: u32,
    pub height: u32,
}

#[doc(hidden)]
pub fn render(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
    mask_overlay: Option<u64>,
) -> Rendered {
    let (out_w, out_h, scale, level) = geometry_for(src, adj, region, max_size);
    render_sized(src, adj, region, out_w, out_h, scale, level, bypass, mask_overlay, false).0
}

/// Export render that keeps the 16-bit quantisation of the developed floats.
///
/// It allocates a second buffer, so previews must keep calling [`render`].
#[doc(hidden)]
pub fn render16(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
) -> Rendered16 {
    let (out_w, out_h, scale, level) = geometry_for(src, adj, region, max_size);
    let (narrow, wide) =
        render_sized(src, adj, region, out_w, out_h, scale, level, bypass, None, true);
    Rendered16 {
        rgba: wide.expect("a wide render always returns its 16-bit buffer"),
        width: narrow.width,
        height: narrow.height,
    }
}

/// Output size and source mip level a request at `max_size` resolves to.
fn geometry_for(src: &Source, adj: &AdjustmentSet, region: NormRect, max_size: u32) -> (u32, u32, f32, usize) {
    let region = clean_region(region);
    let map = super::geometry::FrameMap::new(src.size(), &adj.geometry);
    let (fw, fh) = map.frame_size();
    let (rw, rh) = (fw * region.width, fh * region.height);
    let scale = max_size as f32 / rw.max(rh).max(1.0);
    let out_w = ((rw * scale).round() as u32).max(1);
    let out_h = ((rh * scale).round() as u32).max(1);
    // One output pixel covers `1 / scale` region pixels, so the mip is picked
    // from the region, not the whole frame: a 1/8 region of a 6000 px frame
    // zoomed to 1200 px still reads level 0.
    let footprint = 1.0 / scale;
    let level = if footprint > 1.0 { footprint.log2().floor() as usize } else { 0 };
    (out_w, out_h, scale, level)
}

/// Keeps a region inside the frame: the four corners are baked into the pixel
/// walks, so an out-of-range request would otherwise size the output up and
/// paint the empty border as the neutral fill.
fn clean_region(r: NormRect) -> NormRect {
    let width = if r.width.is_finite() { r.width.clamp(0.0, 1.0) } else { 1.0 };
    let height = if r.height.is_finite() { r.height.clamp(0.0, 1.0) } else { 1.0 };
    NormRect {
        x: if r.x.is_finite() { r.x.clamp(0.0, 1.0 - width) } else { 0.0 },
        y: if r.y.is_finite() { r.y.clamp(0.0, 1.0 - height) } else { 0.0 },
        width,
        height,
    }
}

/// Output size a full render at `max_size` would have.
pub(crate) fn preview_size(src: &Source, adj: &AdjustmentSet, region: NormRect, max_size: u32) -> (u32, u32) {
    let (w, h, _, _) = geometry_for(src, adj, region, max_size);
    (w, h)
}

/// `fast` renders at half the output size and upsamples to the size a full
/// render would have, so callers get the same dimensions with less per-pixel work.
pub(crate) fn render_at(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
    mask_overlay: Option<u64>,
    fast: bool,
) -> Rendered {
    if !fast {
        return render(src, adj, region, max_size, bypass, mask_overlay);
    }
    let (tw, th) = preview_size(src, adj, region, max_size);
    let small = render(src, adj, region, max_size.div_ceil(2).max(1), bypass, mask_overlay);
    if (small.width, small.height) == (tw, th) {
        return small;
    }
    let rgba = upsample(&small.rgba, small.width, small.height, tw, th);
    let histogram = histogram(&rgba);
    Rendered { rgba, width: tw, height: th, histogram }
}

/// Bilinear upsample. `image`'s resize costs more than the half-size render it
/// would be scaling up, so the fast path does it here in parallel.
fn upsample(rgba: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    // The horizontal tap positions are the same for every row.
    let xs: Vec<(u32, u32, f32)> = (0..tw)
        .map(|x| {
            let sx = ((x as f32 + 0.5) * w as f32 / tw as f32 - 0.5).clamp(0.0, w as f32 - 1.0);
            let x0 = sx.floor() as u32;
            (x0, (x0 + 1).min(w - 1), sx - x0 as f32)
        })
        .collect();
    let mut out = vec![0u8; (tw * th * 4) as usize];
    out.par_chunks_mut((tw * 4) as usize).enumerate().for_each(|(y, line)| {
        let sy = ((y as f32 + 0.5) * h as f32 / th as f32 - 0.5).clamp(0.0, h as f32 - 1.0);
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(h - 1);
        let ty = sy - y0 as f32;
        for (x, &(x0, x1, tx)) in xs.iter().enumerate() {
            let at = |xx: u32, yy: u32| ((yy * w + xx) * 4) as usize;
            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            for k in 0..4 {
                let top = rgba[a + k] as f32 + (rgba[b + k] as f32 - rgba[a + k] as f32) * tx;
                let bottom = rgba[c + k] as f32 + (rgba[d + k] as f32 - rgba[c + k] as f32) * tx;
                let v = top + (bottom - top) * ty + 0.5;
                line[x * 4 + k] = v as u8;
            }
        }
    });
    out
}

fn render_sized(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    out_w: u32,
    out_h: u32,
    scale: f32,
    level: usize,
    bypass: bool,
    mask_overlay: Option<u64>,
    wide: bool,
) -> (Rendered, Option<Vec<u16>>) {
    let region = clean_region(region);
    let map = super::geometry::FrameMap::new(src.size(), &adj.geometry);
    let oriented = map.oriented();
    let aspect = oriented.0 / oriented.1;
    // One mask-grid cell covers `cell_px` source pixels, so the mask's pixel
    // radii (feather, refine) convert into grid cells.
    let cell_px = oriented.0 / 256.0;
    let (gw, gh) = grid_dims(aspect);
    // The refine edge only needs the luminance guide when a layer asks for it.
    let needs_guide = adj.layers.iter().any(|l| {
        l.enabled && l.opacity > 0.0 && l.mask.refine > 0.0 && !l.mask.shapes.is_empty()
    });
    let guide = needs_guide.then(|| luma_guide(src, &adj.geometry, gw, gh));

    let plan = Plan {
        adj,
        input_profile: if src.camera_preview {
            None
        } else {
            adj.base.icc_profile.as_deref().and_then(crate::icc::profile)
        },
        film_lut: if src.camera_preview {
            None
        } else {
            adj.base.film_curve.as_deref().and_then(crate::filmcurve::curve_lut)
        },
        gamma: GammaLut::new(),
        fine: wide,
        wb: white_balance_gains(adj.white_balance.kelvin, adj.white_balance.tint),
        exposure: 2f32.powf(adj.exposure.exposure),
        base_contrast: match adj.base.curve {
            ToneCurvePreset::Auto | ToneCurvePreset::FilmStandard => 0.12,
            ToneCurvePreset::FilmExtraShadow => 0.06,
            ToneCurvePreset::FilmHighContrast => 0.3,
            ToneCurvePreset::Linear => 0.0,
        },
        curve_rgb: lut_of(&adj.curves.rgb),
        curve_r: lut_of(&adj.curves.red),
        curve_g: lut_of(&adj.curves.green),
        curve_b: lut_of(&adj.curves.blue),
        curve_luma: lut_of(&adj.curves.luma),
        levels: adj.levels != Levels::default(),
        color_editor: adj.color_editor != ColorEditor::default(),
        color_balance: adj.color_balance != ColorBalance::default(),
        layers: layer_plans(adj, aspect, cell_px, guide.as_deref(), mask_overlay),
    };

    // The frame-point x of each column does not depend on the row.
    let xs: Vec<f32> =
        (0..out_w).map(|x| region.x + (x as f32 + 0.5) / out_w as f32 * region.width).collect();
    let len = (out_w * out_h * 4) as usize;
    let row = (out_w * 4) as usize;
    let mut rgba = vec![0u8; len];
    let mut rgba16 = wide.then(|| vec![0u16; len]);
    // Above 1:1 the region is magnified, so a bicubic tap keeps the detail the
    // zoom is for; at or below 1:1 the bilinear mip sample is the correct filter.
    let sharp = scale > 1.0;
    match rgba16.as_mut() {
        Some(wide_line) => {
            rgba.par_chunks_mut(row)
                .zip(wide_line.par_chunks_mut(row))
                .enumerate()
                .for_each(|(y, (line, line16))| {
                    develop_line(&plan, src, &map, region, level, out_h, &xs, y, bypass, sharp, line, Some(line16));
                });
        }
        None => {
            rgba.par_chunks_mut(row).enumerate().for_each(|(y, line)| {
                develop_line(&plan, src, &map, region, level, out_h, &xs, y, bypass, sharp, line, None);
            });
        }
    }
    // Detail, spots and layers are 8-bit stages; keeping their delta lets the
    // wide buffer carry the develop precision through them.
    let develop8 = rgba16.is_some().then(|| rgba.clone());
    if !bypass {
        let warp = if adj.lens.use_profile {
            src.lens_shot
                .as_ref()
                .and_then(|shot| super::lens_profile::warp_for(&adj.lens, shot, src.lens_auto.as_ref()))
        } else {
            None
        };
        match warp.as_ref() {
            Some(warp) => super::detail::post_process_with_lens(&mut rgba, out_w, out_h, adj, scale, Some(warp)),
            None => super::detail::post_process(&mut rgba, out_w, out_h, adj, scale),
        }
    }
    if !adj.spots.is_empty() && !bypass {
        super::spots::apply_spots(&mut rgba, out_w, out_h, adj, src.size());
    }
    // Layers come last: the spot removal sees the developed frame, and the
    // layer mix sees the healed one.
    if !bypass {
        apply_layers(&mut rgba, out_w, out_h, &plan, &map, region, scale);
    }
    if let (Some(wide), Some(before)) = (rgba16.as_mut(), develop8.as_deref()) {
        for ((px, base), after) in
            wide.chunks_exact_mut(4).zip(before.chunks_exact(4)).zip(rgba.chunks_exact(4))
        {
            for k in 0..3 {
                px[k] = (px[k] as i32 + (after[k] as i32 - base[k] as i32) * 257)
                    .clamp(0, 65535) as u16;
            }
            px[3] = if after[3] == base[3] { px[3] } else { after[3] as u16 * 257 };
        }
    }
    let histogram = histogram(&rgba);
    (Rendered { rgba, width: out_w, height: out_h, histogram }, rgba16)
}

#[allow(clippy::too_many_arguments)]
fn develop_line(
    plan: &Plan,
    src: &Source,
    map: &super::geometry::FrameMap,
    region: NormRect,
    level: usize,
    out_h: u32,
    xs: &[f32],
    y: usize,
    bypass: bool,
    sharp: bool,
    line: &mut [u8],
    line16: Option<&mut [u16]>,
) {
    let fy = region.y + (y as f32 + 0.5) / out_h as f32 * region.height;
    let mut line16 = line16;
    for (x, &fx) in xs.iter().enumerate() {
        let f = [fx, fy];
        let (_, s) = map.map(f);
        let sample = if sharp { src.sample_cubic(level, s[0], s[1]) } else { src.sample(level, s[0], s[1]) };
        let c = match sample {
            Some(lin) => develop(plan, lin, f, bypass),
            None => [0.11, 0.11, 0.11],
        };
        let px = &mut line[x * 4..x * 4 + 4];
        let mut px16 = line16.as_deref_mut().map(|l| &mut l[x * 4..x * 4 + 4]);
        for k in 0..3 {
            let v = c[k].clamp(0.0, 1.0);
            px[k] = (v * 255.0 + 0.5) as u8;
            if let Some(px16) = px16.as_deref_mut() {
                px16[k] = (v * 65535.0 + 0.5) as u16;
            }
        }
        px[3] = 255;
        if let Some(px16) = px16.as_deref_mut() {
            px16[3] = 65535;
        }
    }
}

/// Linear light to display codes: the capture's film curve when there is one,
/// otherwise the sRGB transfer curve.
fn encode(plan: &Plan, lin: [f32; 3]) -> [f32; 3] {
    match &plan.film_lut {
        Some(lut) => [lut_get(lut, lin[0]), lut_get(lut, lin[1]), lut_get(lut, lin[2])],
        None if plan.fine => [
            linear_to_srgb(lin[0].max(0.0)),
            linear_to_srgb(lin[1].max(0.0)),
            linear_to_srgb(lin[2].max(0.0)),
        ],
        None => [plan.gamma.get(lin[0]), plan.gamma.get(lin[1]), plan.gamma.get(lin[2])],
    }
}

fn develop(plan: &Plan, lin: [f32; 3], f: [f32; 2], bypass: bool) -> [f32; 3] {
    // The camera profile is an input transform: device RGB -> linear sRGB.
    let lin = match &plan.input_profile {
        Some(profile) => {
            let encoded = [
                plan.gamma.get(lin[0]),
                plan.gamma.get(lin[1]),
                plan.gamma.get(lin[2]),
            ];
            profile.to_linear_srgb(encoded)
        }
        None => lin,
    };
    let adj = plan.adj;
    if bypass {
        // The "before" image keeps the camera base look (gamma, shoulder, base
        // curve) but none of the user's edits, so an unedited photo looks the
        // same before and after.
        let g = encode(plan, lin);
        return contrast_all(shoulder(g), plan.base_contrast);
    }
    // Linear light: white balance and exposure.
    let e = plan.exposure;
    let mut c = [lin[0] * plan.wb[0] * e, lin[1] * plan.wb[1] * e, lin[2] * plan.wb[2] * e];

    // Lens light falloff is corrected toward flat; vignetting adds its own.
    let dx = f[0] - 0.5;
    let dy = f[1] - 0.5;
    let radius = match adj.vignetting.method {
        VignetteMethod::EllipticOnCrop => (dx * dx + dy * dy).sqrt() * std::f32::consts::SQRT_2,
        VignetteMethod::Circular | VignetteMethod::CircularCrop => (dx * dx + dy * dy).sqrt() * 1.6,
    };
    if adj.vignetting.amount.abs() > 1e-3 {
        let v = 2f32.powf(adj.vignetting.amount * sstep(0.35, 1.1, radius));
        c = [c[0] * v, c[1] * v, c[2] * v];
    }

    // Display-referred tone work. Capture One's .fcrv already lands on display
    // codes (0.18 maps to 0.49), so it replaces the sRGB encode rather than
    // stacking on top of it.
    let mut g = encode(plan, c);
    g = tone(g, &adj.hdr, &adj.exposure, plan.base_contrast, &adj.clarity, &adj.dehaze);

    if plan.levels {
        g = adj.levels.apply(g);
    }
    if let Some(lut) = &plan.curve_rgb {
        g = [lut_get(lut, g[0]), lut_get(lut, g[1]), lut_get(lut, g[2])];
    }
    if let Some(lut) = &plan.curve_r {
        g[0] = lut_get(lut, g[0]);
    }
    if let Some(lut) = &plan.curve_g {
        g[1] = lut_get(lut, g[1]);
    }
    if let Some(lut) = &plan.curve_b {
        g[2] = lut_get(lut, g[2]);
    }
    if let Some(lut) = &plan.curve_luma {
        let l = luma(g).max(1e-4);
        let k = lut_get(lut, l) / l;
        g = [g[0] * k, g[1] * k, g[2] * k];
    }

    // Color.
    let sat = 1.0 + adj.exposure.saturation / 100.0;
    if (sat - 1.0).abs() > 1e-4 {
        let l = luma(g);
        g = [l + (g[0] - l) * sat, l + (g[1] - l) * sat, l + (g[2] - l) * sat];
    }
    if plan.color_balance {
        g = color_balance(g, &adj.color_balance, 1.0);
    }
    if plan.color_editor {
        g = color_editor(g, &adj.color_editor, 1.0);
    }
    if adj.black_white.enabled {
        g = black_white(g, &adj.black_white);
    }

    g
}

fn tone(
    mut g: [f32; 3],
    hdr: &HighDynamicRange,
    exp: &Exposure,
    base_contrast: f32,
    clarity: &Clarity,
    dehaze: &Dehaze,
) -> [f32; 3] {
    let l = luma(g);
    let wh = sstep(0.45, 1.2, l);
    let ws = 1.0 - sstep(0.0, 0.55, l);
    let ww = sstep(0.65, 1.0, l);
    let wb = 1.0 - sstep(0.0, 0.25, l);
    let shift = hdr.highlights / 100.0 * 0.35 * wh + hdr.shadows / 100.0 * 0.3 * ws + hdr.whites / 100.0 * 0.2 * ww
        + hdr.blacks / 100.0 * 0.12 * wb;
    if shift.abs() > 1e-5 {
        let target = (l + shift).max(0.0);
        let k = target / l.max(1e-4);
        g = [g[0] * k, g[1] * k, g[2] * k];
    }
    // Soft shoulder for whatever still exceeds white.
    for v in &mut g {
        if *v > 0.9 {
            let over = *v - 0.9;
            *v = 0.9 + over / (1.0 + over * 10.0);
        }
    }
    if exp.brightness.abs() > 1e-3 {
        let gamma = 2f32.powf(-exp.brightness / 50.0 * 0.6);
        g = [g[0].max(0.0).powf(gamma), g[1].max(0.0).powf(gamma), g[2].max(0.0).powf(gamma)];
    }
    let c = base_contrast + exp.contrast / 50.0 * 0.8;
    g = [contrast(g[0], c), contrast(g[1], c), contrast(g[2], c)];
    let local = (clarity.clarity + clarity.structure * 0.5) / 100.0;
    if local.abs() > 1e-4 {
        let l = luma(g);
        let mid = 4.0 * l * (1.0 - l);
        let k = 1.0 + local * 0.6 * mid;
        g = [0.5 + (g[0] - 0.5) * k, 0.5 + (g[1] - 0.5) * k, 0.5 + (g[2] - 0.5) * k];
        if matches!(clarity.method, ClarityMethod::Punch) {
            let l = luma(g);
            let s = 1.0 + local * 0.25;
            g = [l + (g[0] - l) * s, l + (g[1] - l) * s, l + (g[2] - l) * s];
        }
    }
    if dehaze.amount > 0.0 {
        let a = dehaze.amount / 100.0 * 0.22;
        g = [(g[0] - a) / (1.0 - a), (g[1] - a) / (1.0 - a), (g[2] - a) / (1.0 - a)];
        let l = luma(g);
        let s = 1.0 + a;
        g = [l + (g[0] - l) * s, l + (g[1] - l) * s, l + (g[2] - l) * s];
    }
    g
}

fn color_balance(g: [f32; 3], cb: &ColorBalance, k: f32) -> [f32; 3] {
    let l = luma(g).clamp(0.0, 1.0);
    let ws = (1.0 - l).powi(2);
    let wh = l.powi(2);
    let wm = (1.0 - ws - wh).max(0.0);
    let mut out = g;
    for (wheel, w) in [(&cb.master, 1.0), (&cb.shadow, ws), (&cb.midtone, wm), (&cb.highlight, wh)] {
        if wheel.saturation == 0.0 && wheel.lightness == 0.0 {
            continue;
        }
        let o = wheel_offset(wheel, 0.35);
        for i in 0..3 {
            out[i] += o[i] * w * k;
        }
    }
    out
}

fn signed_hue_delta(h: f32, center: f32) -> f32 {
    let d = (h - center).rem_euclid(360.0);
    if d > 180.0 {
        d - 360.0
    } else {
        d
    }
}

/// Coverage of one color-editor band for a pixel at hue `h`, saturation `s`.
/// `extra_feather` widens every band edge (the global group's smoothness).
fn band_weight(b: &ColorBand, extra_feather: f32, h: f32, s: f32) -> f32 {
    let d = signed_hue_delta(h, b.center_hue);
    let feather = b.smoothness.max(0.1) + extra_feather;
    let hue_w = if d < b.hue_low {
        sstep(b.hue_low - feather, b.hue_low, d)
    } else if d > b.hue_high {
        1.0 - sstep(b.hue_high, b.hue_high + feather, d)
    } else {
        1.0
    };
    let ds = (s - b.center_saturation) * 100.0;
    let sat_feather = 10.0f32.max(b.smoothness);
    let sat_w = if ds < b.sat_low {
        sstep(b.sat_low - sat_feather, b.sat_low, ds)
    } else if ds > b.sat_high {
        1.0 - sstep(b.sat_high, b.sat_high + sat_feather, ds)
    } else {
        1.0
    };
    (hue_w * sat_w).clamp(0.0, 1.0)
}

fn apply_homogeneity(b: &ColorBand, w: f32, h: &mut f32, s: &mut f32, v: &mut f32) {
    let hue = b.homogeneity[0] / 100.0 * w;
    if hue != 0.0 {
        *h += signed_hue_delta(b.center_hue, *h) * hue;
    }
    let sat = b.homogeneity[1] / 100.0 * w;
    if sat != 0.0 {
        *s += (b.center_saturation - *s) * sat;
    }
    let light = b.homogeneity[2] / 100.0 * w;
    if light != 0.0 {
        *v += (b.center_value - *v) * light;
    }
}

fn color_editor(g: [f32; 3], ce: &ColorEditor, k: f32) -> [f32; 3] {
    let (h0, s0, v0) = rgb_to_hsv([g[0].max(0.0), g[1].max(0.0), g[2].max(0.0)]);
    if s0 < 0.02 {
        return g;
    }
    let extra = ce.global.smoothness.max(0.0) * 0.5;
    let mut h = h0;
    let mut s = s0;
    let mut v = v0;
    let mut dh = 0.0;
    let mut ds = 0.0;
    let mut dl = 0.0;

    if ce.global.enabled {
        let mut w = band_weight(&ce.global, 0.0, h0, s0);
        if ce.global.inverted {
            w = 1.0 - w;
        }
        apply_homogeneity(&ce.global, w, &mut h, &mut s, &mut v);
        dh += ce.global_shift.hue * w;
        ds += ce.global_shift.saturation * w;
        dl += ce.global_shift.lightness * w;
    }

    for band in ce.bands.iter().filter(|b| b.enabled) {
        let mut w = band_weight(band, extra, h0, s0);
        if band.inverted {
            w = 1.0 - w;
        }
        apply_homogeneity(band, w, &mut h, &mut s, &mut v);
    }

    for r in ce.advanced.iter().chain(ce.skin.iter()) {
        if let Some((_, band)) = ce.advanced_bands.iter().find(|(id, _)| *id == r.id) {
            if !band.enabled {
                continue;
            }
            let mut w = band_weight(band, 0.0, h0, s0);
            if band.inverted {
                w = 1.0 - w;
            }
            apply_homogeneity(band, w, &mut h, &mut s, &mut v);
        } else if let Some(u) = &r.uniformity {
            let w = color_range_weight(r, g);
            h += signed_hue_delta(r.hue, h) * (u.hue / 100.0 * w);
        }
    }

    for (i, shift) in ce.basic.iter().enumerate() {
        if *shift == HslShift::default() {
            continue;
        }
        let band = &ce.bands[i];
        if !band.enabled {
            continue;
        }
        let mut w = band_weight(band, extra, h, s);
        if band.inverted {
            w = 1.0 - w;
        }
        dh += shift.hue * w;
        ds += shift.saturation * w;
        dl += shift.lightness * w;
    }

    for r in ce.advanced.iter().chain(ce.skin.iter()) {
        let w = match ce.advanced_bands.iter().find(|(id, _)| *id == r.id) {
            Some((_, band)) if band.enabled => {
                let w = band_weight(band, 0.0, h0, s0);
                if band.inverted { 1.0 - w } else { w }
            }
            Some(_) => 0.0,
            None => color_range_weight(r, g),
        };
        dh += r.shift.hue * w;
        ds += r.shift.saturation * w;
        dl += r.shift.lightness * w;
    }
    let dh = dh.clamp(-180.0, 180.0);
    let ds = ds.clamp(-100.0, 100.0);
    let dl = dl.clamp(-100.0, 100.0);
    let s2 = (s * (1.0 + ds / 100.0 * k)).clamp(0.0, 1.0);
    let v2 = (v * (1.0 + dl / 100.0 * 0.5 * s * k)).max(0.0);
    hsv_to_rgb(h + dh * k, s2, v2)
}

fn black_white(g: [f32; 3], bw: &BlackWhite) -> [f32; 3] {
    let (h, s, _) = rgb_to_hsv([g[0].max(0.0), g[1].max(0.0), g[2].max(0.0)]);
    let centers = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
    let mut boost = 0.0;
    for (i, c) in centers.iter().enumerate() {
        let w = (1.0 - hue_distance(h, *c) / 60.0).max(0.0);
        boost += bw.sensitivity[i] / 100.0 * w;
    }
    let l = (luma(g) * (1.0 + boost * s * 0.8)).max(0.0);
    let mut out = [l, l, l];
    for (tone, w) in [(&bw.highlight_tone, sstep(0.4, 1.0, l)), (&bw.shadow_tone, 1.0 - sstep(0.0, 0.6, l))] {
        if tone.saturation > 0.0 {
            let c = hue_rgb(tone.hue);
            let a = tone.saturation / 100.0 * 0.35 * w;
            let mean = (c[0] + c[1] + c[2]) / 3.0;
            for i in 0..3 {
                out[i] += (c[i] - mean) * a;
            }
        }
    }
    out
}

fn local(g: [f32; 3], lp: &LayerPlan, m: f32) -> [f32; 3] {
    let a = &lp.layer.adjustments;
    if let Some(fill) = a.fill_color {
        return [g[0] + (fill[0] - g[0]) * m, g[1] + (fill[1] - g[1]) * m, g[2] + (fill[2] - g[2]) * m];
    }
    let mut c = g;
    if let Some(wb) = lp.wb_gains {
        c = [c[0] * wb[0], c[1] * wb[1], c[2] * wb[2]];
    }
    if lp.exposure_gain != 1.0 {
        let e = lp.exposure_gain;
        c = [c[0] * e, c[1] * e, c[2] * e];
    }
    c = tone(c, &a.hdr, &a.exposure, 0.0, &a.clarity, &a.dehaze);
    if lp.levels {
        c = a.levels.apply(c);
    }
    if let Some(lut) = &lp.curve_rgb {
        c = [lut_get(lut, c[0]), lut_get(lut, c[1]), lut_get(lut, c[2])];
    }
    if let Some(lut) = &lp.curve_r {
        c[0] = lut_get(lut, c[0]);
    }
    if let Some(lut) = &lp.curve_g {
        c[1] = lut_get(lut, c[1]);
    }
    if let Some(lut) = &lp.curve_b {
        c[2] = lut_get(lut, c[2]);
    }
    if let Some(lut) = &lp.curve_luma {
        let l = luma(c).max(1e-4);
        let k = lut_get(lut, l) / l;
        c = [c[0] * k, c[1] * k, c[2] * k];
    }
    let sat = lp.saturation;
    if (sat - 1.0).abs() > 1e-4 {
        let l = luma(c);
        c = [l + (c[0] - l) * sat, l + (c[1] - l) * sat, l + (c[2] - l) * sat];
    }
    if a.color_balance != ColorBalance::default() {
        c = color_balance(c, &a.color_balance, 1.0);
    }
    if a.color_editor != ColorEditor::default() {
        c = color_editor(c, &a.color_editor, 1.0);
    }
    [g[0] + (c[0] - g[0]) * m, g[1] + (c[1] - g[1]) * m, g[2] + (c[2] - g[2]) * m]
}

/// The mask weight of one layer at a point, including invert, the luma and
/// colour windows, the layer opacity and the mask density.
fn layer_weight(lp: &LayerPlan, p: [f32; 2], g: [f32; 3]) -> f32 {
    let mut m = lp.grid.as_ref().map_or(1.0, |grid| grid.sample(p));
    if lp.layer.mask.invert {
        m = 1.0 - m;
    }
    if let Some(r) = &lp.layer.mask.luma_range {
        m *= luma_range_weight(r, luma(g));
    }
    if let Some(r) = &lp.layer.mask.color_range {
        m *= color_range_weight(r, g);
    }
    m * lp.opacity
}

/// One layer's contribution to a pixel: the colour adjustments, blended in by
/// the mask weight `m`, plus the mask overlay tint.
fn layer_pixel(lp: &LayerPlan, g: [f32; 3], m: f32) -> [f32; 3] {
    let mut c = local(g, lp, m);
    if lp.overlay {
        c = [c[0] + (0.95 - c[0]) * m * 0.55, c[1] * (1.0 - m * 0.55), c[2] * (1.0 - m * 0.55)];
    }
    c
}

/// Applies every layer to the finished frame.
///
/// The colour adjustments are per pixel, but sharpening, noise reduction and
/// moire need neighbours, so they run over the whole frame and are weighted by
/// the layer's mask at each pixel.
fn apply_layers(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    plan: &Plan,
    map: &super::geometry::FrameMap,
    region: NormRect,
    scale: f32,
) {
    if plan.layers.is_empty() {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    let n = plan.layers.len();
    // Per-pixel mask weights, one slot per layer, pixel-major. The spatial
    // pass below only reads them for layers that actually run a filter.
    let mut weights = vec![0f32; w * h * n];
    let xs: Vec<f32> =
        (0..w).map(|x| region.x + (x as f32 + 0.5) / w as f32 * region.width).collect();
    // Colour pass: the layers stack, so each one sees the previous result.
    rgba.par_chunks_mut(w * 4).zip(weights.par_chunks_mut(w * n)).enumerate().for_each(|(y, (line, wline))| {
        let fy = region.y + (y as f32 + 0.5) / h as f32 * region.height;
        for x in 0..w {
            let px = x * 4;
            let mut g = [line[px] as f32 / 255.0, line[px + 1] as f32 / 255.0, line[px + 2] as f32 / 255.0];
            let f = [xs[x], fy];
            let (p, _) = map.map(f);
            for (li, lp) in plan.layers.iter().enumerate() {
                let m = layer_weight(lp, p, g);
                wline[x * n + li] = m;
                if m <= 1e-4 {
                    continue;
                }
                g = layer_pixel(lp, g, m);
            }
            for k in 0..3 {
                line[px + k] = (g[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    });
    // Spatial pass: the operators need neighbours, so they work on the frame.
    for (li, lp) in plan.layers.iter().enumerate() {
        let spatial = lp.sharpening.is_some() || lp.noise_reduction.is_some() || lp.moire.amount > 0.0;
        if !spatial {
            continue;
        }
        let wgt: Vec<f32> = (0..w * h).map(|i| weights[i * n + li]).collect();
        if wgt.iter().all(|v| *v <= 1e-4) {
            continue;
        }
        let luma = luma_plane(rgba, w, h);
        let mut target = luma.clone();
        if let Some(nr) = &lp.noise_reduction {
            let strength = (nr.luminance / 100.0).clamp(0.0, 1.0);
            if strength > 0.0 {
                let blur = mean3x3(&luma, w, h);
                for i in 0..w * h {
                    target[i] = luma[i] + (blur[i] - luma[i]) * strength;
                }
            }
        }
        if lp.moire.amount > 0.0 {
            let amount = (lp.moire.amount / 100.0).clamp(0.0, 1.0);
            let pattern = (lp.moire.pattern / 100.0).clamp(0.0, 1.0);
            let weight = amount * (0.25 + 0.75 * pattern);
            let blur = mean3x3(&target, w, h);
            for i in 0..w * h {
                let hf = (target[i] - blur[i]).abs();
                let blend = weight * (hf / 10.0).clamp(0.0, 1.0);
                target[i] += (blur[i] - target[i]) * blend;
            }
        }
        if let Some(sh) = &lp.sharpening {
            let amount = sh.amount / 100.0;
            if amount > 0.0 {
                let radius = ((sh.radius.max(0.1)) * scale).round().max(1.0) as i32;
                let blur = box_blur3(&target, w, h, radius);
                let threshold = sh.threshold.max(0.0);
                for i in 0..w * h {
                    let mut detail = target[i] - blur[i];
                    if detail.abs() < threshold {
                        detail = 0.0;
                    }
                    target[i] += detail * amount * 0.75;
                }
            }
        }
        // Write the new luminance back, keeping the hue ratios, and blend it
        // in by the mask weight.
        for i in 0..w * h {
            let px = i * 4;
            let old = luma[i];
            let new = target[i];
            let weight = wgt[i];
            if weight <= 1e-4 {
                continue;
            }
            let mixed = old + (new - old) * weight;
            let f = if old > 1.0 { mixed / old } else { 1.0 };
            for k in 0..3 {
                rgba[px + k] = (rgba[px + k] as f32 * f).clamp(0.0, 255.0).round() as u8;
            }
        }
    }
}

/// Luminance of the frame, 0..255.
fn luma_plane(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut plane = vec![0f32; w * h];
    for (i, p) in plane.iter_mut().enumerate() {
        let j = i * 4;
        *p = 0.2126 * rgba[j] as f32 + 0.7152 * rgba[j + 1] as f32 + 0.0722 * rgba[j + 2] as f32;
    }
    plane
}

/// 3x3 mean of a plane.
fn mean3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    sum += src[yy * w + xx];
                }
            }
            out[y * w + x] = sum / 9.0;
        }
    }
    out
}

/// Three box passes approximate a Gaussian of a comparable radius.
fn box_blur3(src: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
    let mut a = src.to_vec();
    let mut b = vec![0f32; w * h];
    for _ in 0..3 {
        box_blur_h(&a, &mut b, w, h, r);
        std::mem::swap(&mut a, &mut b);
        box_blur_v(&a, &mut b, w, h, r);
        std::mem::swap(&mut a, &mut b);
    }
    a
}

fn box_blur_h(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    for y in 0..h {
        let base = y * w;
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[base + (x as i32 + k).clamp(0, w as i32 - 1) as usize];
            }
            dst[base + x] = sum * inv;
        }
    }
}

fn box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[(y as i32 + k).clamp(0, h as i32 - 1) as usize * w + x];
            }
            dst[y * w + x] = sum * inv;
        }
    }
}

pub(crate) fn histogram(rgba: &[u8]) -> Histogram {
    // Fixed arrays keep the per-chunk fold off the heap; a ray working on a
    // 1400 px preview otherwise allocates thousands of small vectors.
    type Bins = ([u32; 256], [u32; 256], [u32; 256], [u32; 256], u64, u64);
    let empty = || -> Bins { ([0; 256], [0; 256], [0; 256], [0; 256], 0, 0) };
    let (r, g, b, l, lo, hi) = rgba
        .par_chunks(4 * 1024)
        .fold(
            empty,
            |mut acc, chunk| {
                for p in chunk.chunks_exact(4) {
                    acc.0[p[0] as usize] += 1;
                    acc.1[p[1] as usize] += 1;
                    acc.2[p[2] as usize] += 1;
                    let y = (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) as usize;
                    acc.3[y.min(255)] += 1;
                    if p[0] <= 1 && p[1] <= 1 && p[2] <= 1 {
                        acc.4 += 1;
                    }
                    if p[0] >= 254 || p[1] >= 254 || p[2] >= 254 {
                        acc.5 += 1;
                    }
                }
                acc
            },
        )
        .reduce(empty, |mut a, b| {
                for i in 0..256 {
                    a.0[i] += b.0[i];
                    a.1[i] += b.1[i];
                    a.2[i] += b.2[i];
                    a.3[i] += b.3[i];
                }
                a.4 += b.4;
                a.5 += b.5;
                a
            });
    let total = (rgba.len() / 4).max(1) as f32;
    Histogram {
        red: r.to_vec(),
        green: g.to_vec(),
        blue: b.to_vec(),
        luma: l.to_vec(),
        clipped_shadows: lo as f32 / total,
        clipped_highlights: hi as f32 / total,
    }
}

/// Processed color at a frame point, for the color picker readout.
pub(crate) fn sample_point(src: &Source, adj: &AdjustmentSet, f: [f32; 2]) -> [f32; 3] {
    let map = super::geometry::FrameMap::new(src.size(), &adj.geometry);
    let (p, s) = map.map(f);
    let plan_out = render_single(src, adj, p, s, f);
    [plan_out[0].clamp(0.0, 1.0), plan_out[1].clamp(0.0, 1.0), plan_out[2].clamp(0.0, 1.0)]
}

fn render_single(src: &Source, adj: &AdjustmentSet, p: [f32; 2], s: [f32; 2], f: [f32; 2]) -> [f32; 3] {
    let map = super::geometry::FrameMap::new(src.size(), &adj.geometry);
    let oriented = map.oriented();
    let aspect = oriented.0 / oriented.1;
    let cell_px = oriented.0 / 256.0;
    let plan = Plan {
        adj,
        input_profile: if src.camera_preview {
            None
        } else {
            adj.base.icc_profile.as_deref().and_then(crate::icc::profile)
        },
        film_lut: if src.camera_preview {
            None
        } else {
            adj.base.film_curve.as_deref().and_then(crate::filmcurve::curve_lut)
        },
        gamma: GammaLut::new(),
        fine: false,
        wb: white_balance_gains(adj.white_balance.kelvin, adj.white_balance.tint),
        exposure: 2f32.powf(adj.exposure.exposure),
        base_contrast: 0.12,
        curve_rgb: lut_of(&adj.curves.rgb),
        curve_r: lut_of(&adj.curves.red),
        curve_g: lut_of(&adj.curves.green),
        curve_b: lut_of(&adj.curves.blue),
        curve_luma: lut_of(&adj.curves.luma),
        levels: adj.levels != Levels::default(),
        color_editor: adj.color_editor != ColorEditor::default(),
        color_balance: adj.color_balance != ColorBalance::default(),
        layers: layer_plans(adj, aspect, cell_px, None, None),
    };
    let mut g = match src.sample(0, s[0], s[1]) {
        Some(lin) => develop(&plan, lin, f, false),
        None => [0.0; 3],
    };
    // The colour stack of the layers applies to the picked point too; the
    // spatial operators need neighbours, so they only run on a full frame.
    for lp in &plan.layers {
        let m = layer_weight(lp, p, g);
        if m <= 1e-4 {
            continue;
        }
        g = layer_pixel(lp, g, m);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Draft vs Final render cost on a 2000x1333 source.
    /// `cargo test -p lumen-engine bench_render_quality -- --ignored --nocapture`
    #[test]
    #[ignore = "timing benchmark, run explicitly with --ignored"]
    fn bench_render_quality() {
        use std::time::Instant;
        let (w, h) = (2000u32, 1333u32);
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        for (label, max) in [
            ("draft(800)", 800u32),
            ("draft(1000)", 1000u32),
            ("draft(1400)", 1400u32),
            ("final(8192)", 8192u32),
        ] {
            let adj = AdjustmentSet::default();
            let t = Instant::now();
            let out = render(&src, &adj, NormRect::FULL, max, false, None);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!(
                "{label:12} {ms:8.2} ms  {}x{}  {:.1} Mpx/s",
                out.width,
                out.height,
                (out.width * out.height) as f64 / ms / 1000.0
            );
        }

        let mut off = AdjustmentSet::default();
        off.sharpening.amount = 0.0;
        off.noise_reduction = NoiseReduction { luminance: 0.0, details: 0.0, color: 0.0, single_pixel: 0.0 };
        off.film_grain.impact = 0.0;
        off.moire.amount = 0.0;
        off.lens.distortion = 0.0;
        off.lens.light_falloff = 0.0;
        off.lens.chromatic_aberration = false;
        let t = Instant::now();
        render(&src, &off, NormRect::FULL, 1400, false, None);
        println!("draft, detail off {:8.2} ms", t.elapsed().as_secs_f64() * 1000.0);
    }

    /// A synthetic gradient with enough structure for the detail stages to act on.
    fn synthetic(w: u32, h: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let v = (x * 255 / w.max(1)) as u8;
                buf[i] = v;
                buf[i + 1] = (y * 255 / h.max(1)) as u8;
                buf[i + 2] = ((x + y) % 256) as u8;
                buf[i + 3] = 255;
            }
        }
        buf
    }

    /// Capture One's default layer is not neutral: ZUSMAMOUNT=160 and the noise
    /// reduction amounts are 50. Those are camera-matched baseline values, so an
    /// unedited photo still differs from the raw "before" image.
    #[test]
    fn default_adjustments_are_capture_ones_baseline() {
        let w = 64;
        let h = 48;
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let adj = AdjustmentSet::default();
        let edited = render(&src, &adj, NormRect::FULL, w, false, None);
        let before = render(&src, &adj, NormRect::FULL, w, true, None);
        assert_eq!(edited.rgba.len(), before.rgba.len());
        assert_eq!(adj.sharpening.amount, 160.0, "C1 default layer carries ZUSMAMOUNT=160");
        assert_eq!(adj.sharpening.radius, 0.8, "ZUSMRADIUS");
        assert_eq!(adj.sharpening.threshold, 1.0, "ZUSMTHRESHOLD");
        assert_eq!(adj.noise_reduction.luminance, 50.0, "ZNOISEREDUCTIONAMOUNT");
        assert_ne!(edited.rgba, before.rgba, "the baseline must be applied to the edited frame");
    }

    /// The detail stages must still be reachable: a non-default adjustment
    /// changes the frame, and the bypass frame stays raw.
    #[test]
    fn detail_stages_change_the_frame_only_when_edited() {
        let w = 64;
        let h = 48;
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let mut adj = AdjustmentSet::default();
        adj.sharpening.amount = 200.0;
        let edited = render(&src, &adj, NormRect::FULL, w, false, None);
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None);
        assert_ne!(edited.rgba, plain.rgba, "sharpening must change the frame");
        let before = render(&src, &adj, NormRect::FULL, w, true, None);
        let plain_before = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, true, None);
        assert_eq!(before.rgba, plain_before.rgba, "bypass must ignore the detail stages");
    }

    /// The aspect and keystone stages must reach the rendered frame, not just
    /// the mapping helpers.
    #[test]
    fn aspect_and_keystone_shape_the_rendered_frame() {
        let w = 120;
        let h = 80;
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let adj = AdjustmentSet { geometry: Geometry { aspect: AspectRatio::Square, ..Default::default() }, ..Default::default() };
        let square = render(&src, &adj, NormRect::FULL, w, false, None);
        let ratio = square.width as f32 / square.height as f32;
        assert!((ratio - 1.0).abs() < 0.08, "aspect 1:1 gave {ratio}");

        let flat = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None);
        let tilted = AdjustmentSet {
            geometry: Geometry {
                keystone: Keystone { vertical: 50.0, amount: 100.0, ..Default::default() },
                ..Default::default()
            },
            ..Default::default()
        };
        let keyed = render(&src, &tilted, NormRect::FULL, w, false, None);
        assert_ne!(keyed.rgba, flat.rgba, "keystone must move pixels");
    }

    /// A camera profile is opt-in: no name renders exactly as before, and a real
    /// C1 profile changes the frame.
    #[test]
    fn camera_profile_is_opt_in_and_changes_the_frame() {
        let w = 32;
        let h = 24;
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None);

        let mut adj = AdjustmentSet::default();
        adj.base.icc_profile = Some("does-not-exist".into());
        let missing = render(&src, &adj, NormRect::FULL, w, false, None);
        assert_eq!(missing.rgba, plain.rgba, "unknown profile must be ignored");

        if crate::icc::profile("NikonZ6III-Generic").is_none() {
            return; // no Capture One install on this machine
        }
        adj.base.icc_profile = Some("NikonZ6III-Generic".into());
        let profiled = render(&src, &adj, NormRect::FULL, w, false, None);
        assert_ne!(profiled.rgba, plain.rgba, "camera profile must change the frame");
    }

    /// Flat frames so the detail stages cannot change the result: sharpening and
    /// noise reduction act on local differences, and a uniform frame has none.
    fn flat(w: u32, h: u32, v: u8) -> Vec<u8> {
        [v, v, v, 255].repeat((w * h) as usize)
    }


    // ───────────────────────────── mask feather and refine ─────────────────────────────

    /// A brush stroke with a hard edge, so the mask is a clean 0/1 disc.
    fn hard_disc(size: f32) -> Mask {
        Mask {
            shapes: vec![MaskShape::Brush(BrushStroke {
                op: MaskOp::Add,
                points: vec![[0.5, 0.5, 1.0]],
                size,
                hardness: 100.0,
                opacity: 100.0,
                flow: 100.0,
                auto_mask: false,
            })],
            feather: 0.0,
            refine: 0.0,
            density: 100.0,
            ..Default::default()
        }
    }

    /// Without a feather the raster must be exactly what the old path
    /// produced: one shape evaluation per cell, no filtering.
    #[test]
    fn feather_zero_raster_is_the_unfiltered_shape() {
        let aspect = 4.0 / 3.0;
        let mask = hard_disc(0.3);
        let grid = MaskGrid::build(&mask, aspect, 4.0, None, &[]);
        assert_eq!(grid.w, 256);
        for i in 0..grid.w * grid.h {
            let p = [
                ((i % grid.w) as f32 + 0.5) / grid.w as f32,
                ((i / grid.w) as f32 + 0.5) / grid.h as f32,
            ];
            let mut want = 0.0f32;
            for shape in &mask.shapes {
                let (op, value) = shape_value(shape, p, aspect, None);
                want = match op {
                    MaskOp::Add => want + value,
                    MaskOp::Subtract => want - value,
                }
                .clamp(0.0, 1.0);
            }
            assert_eq!(grid.data[i], want, "cell {i} must be the raw shape value");
        }
    }

    /// A feather blurs the mask, so a cell that was empty next to the disc
    /// edge now holds a value strictly between zero and one.
    #[test]
    fn feather_blurs_the_mask_edge() {
        let aspect = 4.0 / 3.0;
        let mut mask = hard_disc(0.3);
        let sharp = MaskGrid::build(&mask, aspect, 4.0, None, &[]);
        mask.feather = 40.0;
        let soft = MaskGrid::build(&mask, aspect, 4.0, None, &[]);
        assert_ne!(sharp.data, soft.data, "the feather must change the raster");
        let softened = soft
            .data
            .iter()
            .zip(&sharp.data)
            .filter(|(b, a)| **b > 0.0 && **b < 1.0 && **a <= 0.0)
            .count();
        assert!(softened > 0, "the feather must leave intermediate values on the edge");
    }

    /// The refine edge follows the image's luminance, so the guide is the
    /// source's brightness at each cell.
    #[test]
    fn refine_guide_carries_the_image_tones() {
        let (w, h) = (64u32, 48u32);
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let geometry = Geometry::default();
        let (gw, gh) = grid_dims(w as f32 / h as f32);
        let guide = luma_guide(&src, &geometry, gw, gh);
        assert_eq!(guide.len(), gw * gh);
        assert!(guide.iter().any(|v| *v > 0.02 && *v < 0.98), "the guide must carry the image's tones");
    }

    // ───────────────────────────── brush + eraser ─────────────────────────────

    fn stroke(op: MaskOp, at: [f32; 2], size: f32) -> MaskShape {
        MaskShape::Brush(BrushStroke {
            op,
            points: vec![[at[0], at[1], 1.0]],
            size,
            hardness: 100.0,
            opacity: 100.0,
            flow: 100.0,
            auto_mask: false,
        })
    }

    /// The eraser is a subtractive brush: it cuts a hole out of the painted
    /// disc and cannot drive the mask below zero.
    #[test]
    fn eraser_subtracts_from_the_brush() {
        let aspect = 1.0;
        let mask = Mask {
            shapes: vec![
                stroke(MaskOp::Add, [0.5, 0.5], 0.3),
                stroke(MaskOp::Subtract, [0.5, 0.5], 0.1),
            ],
            ..Default::default()
        };
        let grid = MaskGrid::build(&mask, aspect, 4.0, None, &[]);
        assert_eq!(grid.sample([0.5, 0.5]), 0.0, "the erased centre must be empty");
        assert_eq!(grid.sample([0.72, 0.5]), 1.0, "the painted ring must survive");
        assert_eq!(grid.sample([0.05, 0.05]), 0.0, "outside stays empty");

        let only_erase = Mask {
            shapes: vec![stroke(MaskOp::Subtract, [0.5, 0.5], 0.3)],
            ..Default::default()
        };
        let empty = MaskGrid::build(&only_erase, aspect, 4.0, None, &[]);
        assert!(empty.data.iter().all(|v| *v == 0.0), "subtract clamps at zero");
    }

    // ───────────────────────────── copy masks ─────────────────────────────

    /// A copy mask takes the finished raster of the named layer below it.
    #[test]
    fn copy_mask_reuses_the_source_layer() {
        let aspect = 1.0;
        let source = Mask { shapes: vec![stroke(MaskOp::Add, [0.5, 0.5], 0.3)], ..Default::default() };
        let grid = MaskGrid::build(&source, aspect, 4.0, None, &[]);

        let copy = Mask {
            shapes: vec![MaskShape::Copy { from_layer: 7, op: MaskOp::Add }],
            ..Default::default()
        };
        let built = [(7u64, grid.clone())];
        let added = MaskGrid::build(&copy, aspect, 4.0, None, &built);
        assert_eq!(added.data, grid.data, "add-copy must equal the source mask");

        let subtract = Mask {
            shapes: vec![
                stroke(MaskOp::Add, [0.5, 0.5], 0.3),
                MaskShape::Copy { from_layer: 7, op: MaskOp::Subtract },
            ],
            ..Default::default()
        };
        let cut = MaskGrid::build(&subtract, aspect, 4.0, None, &built);
        assert!(cut.data.iter().all(|v| *v == 0.0), "subtract-copy must cancel the same mask");

        let missing = Mask { shapes: vec![MaskShape::Copy { from_layer: 99, op: MaskOp::Add }], ..Default::default() };
        let blank = MaskGrid::build(&missing, aspect, 4.0, None, &built);
        assert!(blank.data.iter().all(|v| *v == 0.0), "an unknown source adds nothing");
    }

    // ───────────────────────────── layers ─────────────────────────────

    fn tone_layer(id: u64, exposure: f32, brightness: f32) -> Layer {
        Layer {
            id,
            name: format!("Layer {id}"),
            kind: LayerKind::Adjustment,
            enabled: true,
            opacity: 100.0,
            mask: Mask::default(),
            adjustments: LocalAdjustments {
                exposure: Exposure { exposure, brightness, ..Default::default() },
                ..Default::default()
            },
        }
    }

    /// The renderer paints the list in order, so swapping two layers changes
    /// the frame; `ZINDEX` ordering feeds that list.
    #[test]
    fn layer_list_order_is_the_draw_order() {
        let (w, h) = (16u32, 16u32);
        let src = Source::from_rgba(w, h, &flat(w, h, 110));
        let mut ab = AdjustmentSet::default();
        ab.layers = vec![tone_layer(1, 0.8, 0.0), tone_layer(2, 0.0, 30.0)];
        let mut ba = AdjustmentSet::default();
        ba.layers = vec![tone_layer(2, 0.0, 30.0), tone_layer(1, 0.8, 0.0)];
        let a = render(&src, &ab, NormRect::FULL, w, false, None);
        let b = render(&src, &ba, NormRect::FULL, w, false, None);
        assert_ne!(a.rgba, b.rgba, "layer order must change the result");
    }

    /// A layer with a curve and a hard-edged mask on the left half.
    fn curve_layer(mask: Mask) -> Layer {
        Layer {
            id: 7,
            name: "Curve".into(),
            kind: LayerKind::Adjustment,
            enabled: true,
            opacity: 100.0,
            mask,
            adjustments: LocalAdjustments {
                curves: Curves {
                    rgb: Curve { points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(0.5, 0.95), CurvePoint::new(1.0, 1.0)] },
                    ..Default::default()
                },
                ..Default::default()
            },
        }
    }

    /// The layer's curve may only touch the pixels the mask covers.
    #[test]
    fn layer_curve_only_changes_the_masked_pixels() {
        let (w, h) = (64u32, 48u32);
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let mask = Mask {
            shapes: vec![MaskShape::Brush(BrushStroke {
                op: MaskOp::Add,
                points: vec![[0.25, 0.5, 1.0]],
                size: 0.2,
                hardness: 100.0,
                opacity: 100.0,
                flow: 100.0,
                auto_mask: false,
            })],
            ..Default::default()
        };
        let mut adj = AdjustmentSet::default();
        adj.layers.push(curve_layer(mask));
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None);
        let layered = render(&src, &adj, NormRect::FULL, w, false, None);
        let mut changed = 0;
        let mut same = 0;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let i = (y * w as usize + x) * 4;
                if plain.rgba[i..i + 3] == layered.rgba[i..i + 3] {
                    same += 1;
                } else {
                    changed += 1;
                }
            }
        }
        assert!(changed > 0, "the layer curve must change the frame");
        assert!(same > 0, "the layer curve must leave the rest of the frame alone");
        // Everything right of the mask's disc is outside the mask, so it is
        // untouched.
        for y in 0..h as usize {
            for x in (w as usize / 2)..w as usize {
                let i = (y * w as usize + x) * 4;
                assert_eq!(plain.rgba[i..i + 3], layered.rgba[i..i + 3], "the unmasked half must not change");
            }
        }
    }

    // ───────────────────────────── spots ─────────────────────────────

    /// An empty spot list leaves the frame exactly as it was, and a real spot
    /// does change it: the spots reach the rendered frame.
    #[test]
    fn empty_spot_list_leaves_the_frame_untouched() {
        let (w, h) = (64u32, 48u32);
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let mut adj = AdjustmentSet::default();
        adj.spots.clear();
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None);
        let spotted = render(&src, &adj, NormRect::FULL, w, false, None);
        assert_eq!(spotted.rgba, plain.rgba, "an empty spot list must not touch the frame");

        adj.spots.push(Spot { id: 1, kind: SpotKind::Spot, x: 0.5, y: 0.5, radius: 0.1 });
        let healed = render(&src, &adj, NormRect::FULL, w, false, None);
        assert_ne!(healed.rgba, plain.rgba, "a spot must change the frame");
    }

    // ─────────────────────────── color editor ───────────────────────────

    fn hue_of(c: [f32; 3]) -> f32 {
        rgb_to_hsv(c).0
    }

    fn chroma_of(c: [f32; 3]) -> f32 {
        let (_, s, v) = rgb_to_hsv(c);
        s * v
    }

    /// A shift in one sector must not leak into a different hue.
    #[test]
    fn color_editor_sectors_are_independent() {
        let mut ce = ColorEditor::default();
        ce.basic[HueSector::Magenta as usize] =
            HslShift { hue: 0.0, saturation: -100.0, lightness: 0.0 };

        let magenta = hue_rgb(315.0);
        let red = hue_rgb(0.0);
        let desaturated = color_editor(magenta, &ce, 1.0);
        assert!(
            chroma_of(desaturated) < chroma_of(magenta) * 0.1,
            "magenta sector should desaturate magenta: {desaturated:?}"
        );
        let untouched = color_editor(red, &ce, 1.0);
        for i in 0..3 {
            assert!(
                (untouched[i] - red[i]).abs() < 1e-3,
                "red must stay: {untouched:?} vs {red:?}"
            );
        }
    }

    /// The global group's smoothness must widen every sector transition.
    #[test]
    fn global_smoothness_widens_the_transition() {
        let mut ce = ColorEditor::default();
        ce.basic[HueSector::Red as usize] =
            HslShift { hue: 20.0, saturation: 0.0, lightness: 0.0 };
        let probe = hue_rgb(40.0);

        let tight = color_editor(probe, &ce, 1.0);
        assert!(
            signed_hue_delta(hue_of(tight), 40.0).abs() < 0.5,
            "40 deg is outside the default band: {tight:?}"
        );

        ce.global.smoothness = 30.0;
        let wide = color_editor(probe, &ce, 1.0);
        let tight_shift = signed_hue_delta(hue_of(tight), 40.0);
        let wide_shift = signed_hue_delta(hue_of(wide), 40.0);
        assert!(
            wide_shift > tight_shift + 2.0,
            "global smoothness must widen the band: tight {tight_shift}, wide {wide_shift}"
        );
    }

    #[test]
    fn color_editor_leaves_neutral_gray_alone() {
        let ce = ColorEditor::default();
        assert_eq!(color_editor([0.5, 0.5, 0.5], &ce, 1.0), [0.5, 0.5, 0.5]);
    }

    fn narrow_bands(ce: &mut ColorEditor) {
        for band in &mut ce.bands {
            band.hue_low = -6.0;
            band.hue_high = 6.0;
            band.smoothness = 2.0;
        }
    }

    /// Every basic sector must only move its own hue, not its neighbours.
    #[test]
    fn all_eight_basic_sectors_are_isolated() {
        for (index, sector) in HueSector::ALL.iter().enumerate() {
            let mut ce = ColorEditor::default();
            narrow_bands(&mut ce);
            ce.basic[index] = HslShift { hue: 0.0, saturation: -100.0, lightness: 0.0 };
            let own = hue_rgb(sector.hue());
            let out = color_editor(own, &ce, 1.0);
            assert!(
                chroma_of(out) < chroma_of(own) * 0.05,
                "{sector:?} must desaturate its own center"
            );
            for (other_index, other) in HueSector::ALL.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                let c = hue_rgb(other.hue());
                let out = color_editor(c, &ce, 1.0);
                for i in 0..3 {
                    assert!(
                        (out[i] - c[i]).abs() < 1e-3,
                        "{sector:?} leaked into {other:?}: {out:?} vs {c:?}"
                    );
                }
            }
        }
    }

    /// A correction whose shift and homogeneity fields are all zero must not
    /// alter the pixel, even though its band is enabled.
    #[test]
    fn zero_field_correction_leaves_pixel_unchanged() {
        let mut ce = ColorEditor::default();
        let band = ColorBand {
            center_hue: 0.0,
            center_saturation: 0.0,
            center_value: 0.0,
            hue_low: 0.0,
            hue_high: 0.0,
            sat_low: 0.0,
            sat_high: 0.0,
            smoothness: 0.0,
            homogeneity: [0.0; 3],
            inverted: false,
            enabled: true,
        };
        ce.bands[0] = band;
        ce.global = band;
        let pixel = [0.6, 0.2, 0.2];
        let out = color_editor(pixel, &ce, 1.0);
        for i in 0..3 {
            assert!((out[i] - pixel[i]).abs() < 1e-6, "zero fields moved the pixel: {out:?}");
        }
    }

    #[test]
    fn saturation_and_lightness_shifts_are_monotonic() {
        let mut ce = ColorEditor::default();
        narrow_bands(&mut ce);
        let pixel = hsv_to_rgb(0.0, 0.5, 0.6);
        ce.bands[HueSector::Red as usize].center_saturation = 0.5;
        let mut last = f32::MIN;
        for saturation in [-60.0, -30.0, 0.0, 30.0, 60.0] {
            ce.basic[HueSector::Red as usize] =
                HslShift { hue: 0.0, saturation, lightness: 0.0 };
            let chroma = chroma_of(color_editor(pixel, &ce, 1.0));
            assert!(chroma > last, "saturation must be monotonic at {saturation}");
            last = chroma;
        }
        ce.basic[HueSector::Red as usize] = HslShift::default();
        let mut last = f32::MIN;
        for lightness in [-50.0, -25.0, 0.0, 25.0, 50.0] {
            ce.basic[HueSector::Red as usize] =
                HslShift { hue: 0.0, saturation: 0.0, lightness };
            let v = rgb_to_hsv(color_editor(pixel, &ce, 1.0)).2;
            assert!(v > last, "lightness must be monotonic at {lightness}");
            last = v;
        }
    }

    /// A 0.1° sweep over both band edges must not step.
    #[test]
    fn boundary_scan_has_no_jump() {
        let mut ce = ColorEditor::default();
        narrow_bands(&mut ce);
        ce.bands[HueSector::Red as usize].hue_low = -8.0;
        ce.bands[HueSector::Red as usize].hue_high = 8.0;
        ce.bands[HueSector::Red as usize].smoothness = 4.0;
        ce.basic[HueSector::Red as usize] =
            HslShift { hue: 20.0, saturation: 0.0, lightness: 0.0 };
        let mut last: Option<f32> = None;
        let mut h = 340.0;
        while h <= 380.0 {
            let out = color_editor(hue_rgb(h), &ce, 1.0);
            let shift = signed_hue_delta(hue_of(out), h);
            if let Some(prev) = last {
                assert!(
                    (shift - prev).abs() < 1.0,
                    "jump at {h:.1}°: {prev:.3} -> {shift:.3}"
                );
            }
            last = Some(shift);
            h += 0.1;
        }
    }

    #[test]
    fn extreme_values_do_not_panic() {
        let mut ce = ColorEditor::default();
        let extreme = ColorBand {
            center_hue: 360.0,
            center_saturation: 2.0,
            center_value: 2.0,
            hue_low: -1.0e9,
            hue_high: 1.0e9,
            sat_low: -1.0e9,
            sat_high: 1.0e9,
            smoothness: 1.0e9,
            homogeneity: [1.0e9; 3],
            inverted: true,
            enabled: true,
        };
        ce.bands[0] = extreme;
        ce.global = extreme;
        ce.basic[0] = HslShift { hue: f32::MAX, saturation: f32::MAX, lightness: f32::MAX };
        ce.global_shift = HslShift { hue: -f32::MAX, saturation: -f32::MAX, lightness: -f32::MAX };
        for pixel in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.9, 0.1]] {
            let out = color_editor(pixel, &ce, 1.0);
            assert!(out.iter().all(|v| v.is_finite()), "extreme values produced {out:?}");
        }
    }

    /// Homogeneity lightness must pull V toward the selected color's value.
    #[test]
    fn lightness_homogeneity_converges() {
        let mut ce = ColorEditor::default();
        let band = &mut ce.bands[HueSector::Red as usize];
        band.hue_low = -10.0;
        band.hue_high = 10.0;
        band.smoothness = 2.0;
        band.center_saturation = 0.75;
        band.center_value = 0.8;
        band.homogeneity = [0.0, 0.0, 100.0];
        let dark = hsv_to_rgb(0.0, 0.75, 0.3);
        let light = hsv_to_rgb(0.0, 0.75, 0.95);
        let before = (rgb_to_hsv(dark).2 - rgb_to_hsv(light).2).abs();
        let after = (rgb_to_hsv(color_editor(dark, &ce, 1.0)).2
            - rgb_to_hsv(color_editor(light, &ce, 1.0)).2)
            .abs();
        assert!(after < before * 0.01, "V did not converge: {before} -> {after}");
    }

    /// The local=0 global group must act through its own shift and must not
    /// leak into the eight basic slots.
    #[test]
    fn local_global_group_does_not_leak_into_basic() {
        let mut ce = ColorEditor::default();
        ce.global = ColorBand {
            center_hue: 0.0,
            center_saturation: 1.0,
            center_value: 1.0,
            hue_low: -5.0,
            hue_high: 5.0,
            sat_low: -100.0,
            sat_high: 100.0,
            smoothness: 2.0,
            homogeneity: [0.0; 3],
            inverted: false,
            enabled: true,
        };
        ce.global_shift = HslShift { hue: 0.0, saturation: -100.0, lightness: 0.0 };
        let red = hue_rgb(0.0);
        let blue = hue_rgb(225.0);
        assert!(chroma_of(color_editor(red, &ce, 1.0)) < chroma_of(red) * 0.05);
        let untouched = color_editor(blue, &ce, 1.0);
        for i in 0..3 {
            assert!((untouched[i] - blue[i]).abs() < 1e-3, "global leaked into blue");
        }
        assert!(ce.basic.iter().all(|s| *s == HslShift::default()));
    }

    /// sRGB-encode a linear picture into 8-bit bytes, as `raw::decode` does.
    fn to_rgba8(linear: &[f32]) -> Vec<u8> {
        linear
            .iter()
            .flat_map(|&v| {
                let e = (linear_to_srgb(v) * 255.0 + 0.5) as u8;
                [e, e, e, 255]
            })
            .collect()
    }

    /// The same linear picture as 16-bit samples, as `raw::decode16` hands over.
    fn to_rgba16(linear: &[f32]) -> Vec<u16> {
        linear
            .iter()
            .flat_map(|&v| {
                let q = (v * 65535.0).round() as u16;
                [q, q, q, 65535]
            })
            .collect()
    }

    /// Same linear picture through both transports must render the same picture,
    /// within the one output code the 8-bit sRGB rounding can cost.
    #[test]
    fn sixteen_bit_source_matches_eight_bit_within_one_code() {
        let (w, h) = (256u32, 64u32);
        let linear: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                0.02 + 0.6 * x * (0.4 + 0.6 * y)
            })
            .collect();
        let adj = AdjustmentSet::default();
        let a = render(&Source::from_rgba(w, h, &to_rgba8(&linear)), &adj, NormRect::FULL, w, true, None);
        let b = render(&Source::from_rgba16(w, h, &to_rgba16(&linear)), &adj, NormRect::FULL, w, true, None);
        assert_eq!((a.width, a.height), (b.width, b.height));
        let worst = a
            .rgba
            .iter()
            .zip(&b.rgba)
            .enumerate()
            .filter(|(i, _)| i % 4 != 3)
            .map(|(_, (x, y))| x.abs_diff(*y))
            .max()
            .unwrap();
        assert!(worst <= 1, "8-bit and 16-bit renders differ by {worst} codes");
    }

    /// A dark ramp, linear 0.0015..0.0060: 8-bit sRGB transport keeps ~13 levels
    /// there, 16-bit linear keeps the whole ramp. That gap is the banding.
    #[test]
    fn sixteen_bit_keeps_shadow_gradation() {
        let (w, h) = (256u32, 8u32);
        let dark: Vec<f32> = (0..w).map(|x| 0.0015 + 0.0045 * x as f32 / (w - 1) as f32).collect();
        let rows: Vec<f32> = (0..w * h).map(|i| dark[(i % w) as usize]).collect();
        let level = |src: &Source| -> Vec<f32> {
            (0..w).map(|x| src.sample(0, (x as f32 + 0.5) / w as f32, 0.5).unwrap()[1]).collect()
        };
        let eight = level(&Source::from_rgba(w, h, &to_rgba8(&rows)));
        let sixteen = level(&Source::from_rgba16(w, h, &to_rgba16(&rows)));
        let count = |v: &[f32]| -> usize {
            let mut keys: Vec<u32> = v.iter().map(|f| (f / 1e-6).round() as u32).collect();
            keys.sort_unstable();
            keys.dedup();
            keys.len()
        };
        assert!(count(&eight) <= 16, "8-bit sRGB should collapse the ramp: {}", count(&eight));
        assert!(count(&sixteen) >= 200, "16-bit should keep the ramp: {}", count(&sixteen));
        assert!((sixteen[0] - dark[0]).abs() < 2.0 / 65535.0);
    }

    /// A level past the end of the chain must fall back to level 0, not panic.
    #[test]
    fn sample_beyond_the_mip_chain_falls_back_to_level_zero() {
        let rgba16: Vec<u16> = (0..64 * 64)
            .flat_map(|i| [i as u16 * 7, i as u16 * 5, 4000, 65535])
            .collect();
        let src = Source::from_rgba16(64, 64, &rgba16);
        assert_eq!(src.levels.len(), 1, "a 64x64 frame builds no mip chain");
        assert_eq!(src.sample(9, 0.5, 0.5), src.sample(0, 0.5, 0.5));
        let big = Source::from_rgba(1024, 1024, &vec![128u8; 1024 * 1024 * 4]);
        assert!(big.levels.len() > 1);
        assert!(big.sample(99, 0.25, 0.75).is_some());
    }

    /// The 8-bit path still decodes sRGB codes through its LUT, unchanged.
    #[test]
    fn eight_bit_path_decodes_srgb_through_the_lut() {
        let src = Source::from_rgba(1, 1, &[10, 128, 250, 255]);
        let c = src.sample(0, 0.5, 0.5).unwrap();
        for (got, code) in c.iter().zip([10.0f32, 128.0, 250.0]) {
            assert!((got - srgb_to_linear(code / 255.0)).abs() < 1e-6, "{got} vs code {code}");
        }
    }

    /// Empty and short buffers must not index past the plane.
    #[test]
    fn empty_and_short_sources_are_safe() {
        for src in [
            Source::from_rgba16(0, 0, &[]),
            Source::from_rgba16(4, 4, &[1, 2, 3, 4]),
            Source::from_rgba(0, 0, &[]),
            Source::from_rgba(4, 4, &[9, 9, 9, 255]),
        ] {
            assert_eq!(src.size(), (1, 1));
            assert_eq!(src.sample(0, 0.5, 0.5), Some([0.0; 3]));
            let out = render(&src, &AdjustmentSet::default(), NormRect::FULL, 16, true, None);
            assert_eq!(out.rgba.len(), (out.width * out.height * 4) as usize);
        }
    }

    /// Shadow-level and render-cost numbers for the report.
    /// `cargo test -p lumen-engine --offline shadow_levels_16_vs_8 -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement, run explicitly with --ignored"]
    fn shadow_levels_16_vs_8() {
        use std::collections::BTreeSet;
        use std::time::Instant;
        let (w, h) = (2000u32, 1333u32);
        let linear: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                0.0015 + 0.35 * x * (0.3 + 0.7 * y)
            })
            .collect();
        let t = Instant::now();
        let eight = Source::from_rgba(w, h, &to_rgba8(&linear));
        let build8 = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let sixteen = Source::from_rgba16(w, h, &to_rgba16(&linear));
        let build16 = t.elapsed().as_secs_f64() * 1000.0;
        println!("source build {w}x{h}: 8-bit {build8:.1} ms, 16-bit {build16:.1} ms");
        let adj = AdjustmentSet::default();
        for (label, src) in [("8-bit", &eight), ("16-bit", &sixteen)] {
            let t = Instant::now();
            let out = render(src, &adj, NormRect::FULL, 1400, false, None);
            println!("render draft(1400) {label}: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
            let codes: BTreeSet<u8> = out.rgba.chunks_exact(4).map(|p| p[1]).filter(|&g| g < 40).collect();
            println!("  distinct output green codes below 40: {}", codes.len());
        }
        // 512 px across linear 0.0015..0.0060, the range a shadow lift exposes.
        let dark: Vec<f32> = (0..512).map(|x| 0.0015 + 0.0045 * x as f32 / 511.0).collect();
        for (label, src) in [
            ("8-bit", Source::from_rgba(512, 1, &to_rgba8(&dark))),
            ("16-bit", Source::from_rgba16(512, 1, &to_rgba16(&dark))),
        ] {
            let v: Vec<f32> = (0..512)
                .map(|x| src.sample(0, (x as f32 + 0.5) / 512.0, 0.5).unwrap()[1] * 1e6)
                .collect();
            let mut keys: Vec<u32> = v.iter().map(|f| f.round() as u32).collect();
            keys.sort_unstable();
            keys.dedup();
            let jump = v.windows(2).map(|p| (p[1] - p[0]).round() as u32).max().unwrap();
            println!(
                "dark ramp 0.0015..0.0060 {label}: {} levels/512 px, max step {:.1}e-6, span {:.1}e-6",
                keys.len(),
                jump as f32,
                v[511] - v[0]
            );
        }
    }

    /// The export render must keep shadow codes the 8-bit preview quantises away.
    #[test]
    fn render16_keeps_more_shadow_codes_than_render() {
        use std::collections::BTreeSet;
        let (w, h) = (1024u32, 2u32);
        let rows: Vec<f32> =
            (0..w * h).map(|i| 0.001 + 0.004 * (i % w) as f32 / (w - 1) as f32).collect();
        let src = Source::from_rgba16(w, h, &to_rgba16(&rows));
        let adj = AdjustmentSet::default();
        let eight = render(&src, &adj, NormRect::FULL, w, false, None);
        let sixteen = render16(&src, &adj, NormRect::FULL, w, false);
        assert_eq!(sixteen.rgba.len(), (w * h * 4) as usize);
        assert_eq!((sixteen.width, sixteen.height), (eight.width, eight.height));
        let d8: BTreeSet<u8> = eight.rgba.chunks_exact(4).map(|p| p[1]).collect();
        let d16: BTreeSet<u16> = sixteen.rgba.chunks_exact(4).map(|p| p[1]).collect();
        assert!(d16.len() > d8.len(), "16-bit {} vs 8-bit {}", d16.len(), d8.len());
        assert!(sixteen.rgba.chunks_exact(4).all(|p| p[3] == 65535), "alpha stays opaque");
    }

    // ───────────────────────────── zoom regions ─────────────────────────────

    fn flat_source(w: u32, h: u32) -> Source {
        Source::from_rgba(w, h, &vec![128u8; (w * h * 4) as usize])
    }

    /// 100% zoom of a region shows source pixels one to one, so it must read the
    /// full-resolution plane even though the whole frame is much larger.
    #[test]
    fn hundred_percent_region_samples_level_zero() {
        let src = flat_source(2048, 1365);
        assert!(src.levels.len() > 2, "the source needs a mip chain for this test");
        let adj = AdjustmentSet::default();
        let region = NormRect { x: 0.5, y: 0.5, width: 1.0 / 8.0, height: 1.0 / 8.0 };
        let (ow, oh, scale, level) = geometry_for(&src, &adj, region, 256);
        assert!((scale - 1.0).abs() < 1e-6, "2048/8 = 256 region pixels at 256 out is 1:1");
        assert_eq!((ow, oh), (256, 171));
        assert_eq!(level, 0, "1:1 zoom must not fall into a mip");
    }

    /// A region shown well below its own pixel count may use a lower mip: four
    /// region pixels per output pixel is level 2.
    #[test]
    fn zoomed_out_region_reads_a_lower_mip() {
        let src = flat_source(2048, 1365);
        let adj = AdjustmentSet::default();
        let region = NormRect { x: 0.0, y: 0.0, width: 0.25, height: 0.25 };
        let (_, _, scale, level) = geometry_for(&src, &adj, region, 128);
        assert!((scale - 0.25).abs() < 1e-6, "512 region pixels into 128 px");
        assert_eq!(level, 2);
    }

    /// The output keeps the region's aspect and puts its long side at `max_size`.
    #[test]
    fn output_size_follows_the_region() {
        let src = flat_source(2048, 1365);
        let adj = AdjustmentSet::default();
        let region = NormRect { x: 0.25, y: 0.5, width: 0.25, height: 0.1 };
        let (ow, oh, _, _) = geometry_for(&src, &adj, region, 600);
        assert_eq!((ow, oh), (600, 160));
        let out = render(&src, &adj, region, 600, true, None);
        assert_eq!((out.width, out.height), (600, 160));
    }

    /// A region hanging over the frame edge is clamped, so the output neither
    /// grows past `max_size` nor paints the out-of-frame fill.
    #[test]
    fn out_of_range_region_is_clamped() {
        let src = flat_source(512, 512);
        let adj = AdjustmentSet::default();
        let wild = NormRect { x: 0.9, y: 0.8, width: 0.5, height: 0.5 };
        let clean = NormRect { x: 0.5, y: 0.5, width: 0.5, height: 0.5 };
        assert_eq!(geometry_for(&src, &adj, wild, 400), geometry_for(&src, &adj, clean, 400));
        let out = render(&src, &adj, wild, 400, true, None);
        assert_eq!((out.width, out.height), (400, 400));
        assert!(
            out.rgba.chunks_exact(4).all(|p| p[0] == 128 && p[1] == 128 && p[2] == 128),
            "a clamped region must stay inside the frame"
        );
    }

    /// Magnified regions use the bicubic tap; 1:1 positions stay byte-identical.
    #[test]
    fn magnified_regions_keep_edges_sharper_than_bilinear() {
        let (w, h) = (64u32, 4u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { 0 } else { 255 };
                let i = ((y * w + x) * 4) as usize;
                rgba[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let src = Source::from_rgba(w, h, &rgba);
        // A quarter pixel either side of the edge: the bicubic tap traverses the
        // edge faster, so the same step keeps more contrast.
        let at = |x: f32| (x + 0.5) / w as f32;
        let (bilinear, cubic) = (src.sample(0, at(31.25), 0.5).unwrap()[0], src.sample_cubic(0, at(31.25), 0.5).unwrap()[0]);
        let (bilinear_hi, cubic_hi) = (src.sample(0, at(31.75), 0.5).unwrap()[0], src.sample_cubic(0, at(31.75), 0.5).unwrap()[0]);
        assert!(
            cubic_hi - cubic > bilinear_hi - bilinear + 0.05,
            "cubic step {cubic}..{cubic_hi} must beat bilinear {bilinear}..{bilinear_hi}"
        );
        let centre = 0.5 / w as f32;
        assert_eq!(src.sample(0, centre, 0.5), src.sample_cubic(0, centre, 0.5));
    }

    /// The report's case: 6000x4000, an eighth of the frame at 1200 px.
    /// `cargo test -p lumen-engine --offline zoom_region_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "timing benchmark, run explicitly with --ignored"]
    fn zoom_region_bench() {
        use std::time::Instant;
        let (w, h) = (6000u32, 4000u32);
        let rgba: Vec<u8> = (0..w as usize * h as usize)
            .flat_map(|i| {
                let x = (i % w as usize) as u32;
                let y = (i / w as usize) as u32;
                let v = if (x / 2 + y / 2) % 2 == 0 { 30 } else { 225 };
                [v, v, v, 255]
            })
            .collect();
        let src = Source::from_rgba(w, h, &rgba);
        let adj = AdjustmentSet::default();
        let eighth = 1.0 / 8.0;
        let region = NormRect { x: 0.5 - eighth / 2.0, y: 0.5 - eighth / 2.0, width: eighth, height: eighth };
        // 750 px is 1:1 (bilinear), 1200 px is the reported 1.6x magnification.
        for max in [750u32, 1200] {
            let (ow, oh, scale, level) = geometry_for(&src, &adj, region, max);
            let t = Instant::now();
            let out = render(&src, &adj, region, max, false, None);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let (lo, hi) = out
                .rgba
                .chunks_exact(4)
                .map(|p| p[0])
                .fold((255u8, 0u8), |(lo, hi), v| (lo.min(v), hi.max(v)));
            println!(
                "6000x4000 1/8 region -> {ow}x{oh} (scale {scale:.2}, level {level}) in {ms:.1} ms \
                 ({:.0} Mpx/s), detail span {}",
                (ow * oh) as f64 / ms / 1000.0,
                hi.saturating_sub(lo)
            );
            assert_eq!(level, 0, "a 1/8 region at {max} px must read level 0");
            assert!(hi - lo > 150, "the 2 px checker must survive {max} px: {lo}..{hi}");
        }
    }

    #[cfg(feature = "external-fixtures")]
    include!("../../local-fixtures/mock_process-tests.rs");
}
