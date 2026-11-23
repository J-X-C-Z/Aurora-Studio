//! A small CPU development pipeline. It is not color-accurate; it exists so every
//! control in the interface has a visible, plausible effect during UI work.

use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock, atomic::{AtomicU64, Ordering}};

use rayon::prelude::*;

use crate::*;

/// Source pixels with a mip pyramid for fast zoomed-out sampling.
#[doc(hidden)]
pub struct Source {
    snapshot_id: u64,
    pub(crate) levels: Vec<Plane>,
    /// Lens EXIF of the capture, for the optional C1 lens profile.
    pub(crate) lens_shot: Option<super::lens_profile::LensShot>,
    /// EXIF-matched lens profile, resolved once when the source is decoded.
    lens_auto: Option<super::lens_profile::LensWarp>,
    /// True when the pixels are the camera's own JPEG preview, not sensor data.
    camera_preview: bool,
    /// True when linear camera-space RGB came from the Z6III C1 RAW pipeline.
    c1_z6iii_linear: bool,
    /// Skin/hair/face masks, computed lazily for portrait effects.
    portrait_masks: OnceLock<crate::portrait_stage::PortraitMasks>,
    /// Stable fingerprint of the decoded source snapshot.
    fingerprint: OnceLock<u64>,
    /// Packed sRGB RGBA for the full decoded source frame.
    face_rgba8: OnceLock<crate::PortraitImage>,
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

pub(crate) fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Linear → sRGB over `0 ..= 4` linear, so highlight recovery has headroom to pull from.
const GAMMA_LUT_SIZE: usize = 8192;
struct GammaLut(&'static [f32; GAMMA_LUT_SIZE]);

impl GammaLut {
    const SIZE: usize = GAMMA_LUT_SIZE;
    const RANGE: f32 = 4.0;

    fn new() -> Self {
        static TABLE: OnceLock<[f32; GammaLut::SIZE]> = OnceLock::new();
        Self(TABLE.get_or_init(|| {
            std::array::from_fn(|i| {
                linear_to_srgb(i as f32 / (Self::SIZE - 1) as f32 * Self::RANGE)
            })
        }))
    }

    fn get(&self, v: f32) -> f32 {
        let i = (v.max(0.0) / Self::RANGE * (Self::SIZE - 1) as f32) as usize;
        self.0[i.min(Self::SIZE - 1)]
    }
}

#[derive(Clone, Copy)]
struct SourceSample {
    rgb: [f32; 3],
    valid: bool,
}

fn source_sample_at(
    src: &Source,
    map: &super::geometry::FrameMap,
    level: usize,
    cubic: bool,
    point: [f32; 2],
) -> SourceSample {
    let (_, sampled_point) = map.map(point);
    let sampled = if cubic {
        src.sample_cubic(level, sampled_point[0], sampled_point[1])
    } else {
        src.sample(level, sampled_point[0], sampled_point[1])
    };
    sampled.map_or(
        SourceSample { rgb: [0.0; 3], valid: false },
        |rgb| SourceSample { rgb, valid: true },
    )
}

#[derive(Clone, Debug, PartialEq)]
struct SourceSampleKey {
    source: u64,
    geometry: Geometry,
    region: NormRect,
    out_w: u32,
    out_h: u32,
    level: usize,
    cubic: bool,
}

type SourceSampleCache = Mutex<Option<(SourceSampleKey, Arc<Vec<SourceSample>>)>>;
/// Frames above this would allocate a full sample vector without caching it.
const SOURCE_SAMPLE_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

fn streams_source_samples(out_w: u32, out_h: u32) -> bool {
    (out_w as usize)
        .saturating_mul(out_h as usize)
        .saturating_mul(std::mem::size_of::<SourceSample>())
        > SOURCE_SAMPLE_CACHE_MAX_BYTES
}

/// One exact sampled source frame. A slider that only changes colour reuses it;
/// geometry, crop, output size, mip and interpolation all participate in the key.
fn source_samples(
    src: &Source,
    map: &super::geometry::FrameMap,
    geometry: &Geometry,
    region: NormRect,
    out_w: u32,
    out_h: u32,
    level: usize,
    cubic: bool,
    xs: &[f32],
) -> Arc<Vec<SourceSample>> {
    static CACHE: OnceLock<SourceSampleCache> = OnceLock::new();
    let key = SourceSampleKey {
        source: src.snapshot_id(),
        geometry: geometry.clone(),
        region,
        out_w,
        out_h,
        level,
        cubic,
    };
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    if let Some((cached_key, samples)) = cache.lock().unwrap_or_else(|error| error.into_inner()).as_ref() {
        if cached_key == &key {
            return Arc::clone(samples);
        }
    }

    let mut samples = vec![SourceSample { rgb: [0.0; 3], valid: false }; (out_w * out_h) as usize];
    samples.par_chunks_mut(out_w as usize).enumerate().for_each(|(y, line)| {
        let fy = region.y + (y as f32 + 0.5) / out_h as f32 * region.height;
        for (x, &fx) in xs.iter().enumerate() {
            line[x] = source_sample_at(src, map, level, cubic, [fx, fy]);
        }
    });
    let samples = Arc::new(samples);
    if samples.len().saturating_mul(std::mem::size_of::<SourceSample>())
        <= SOURCE_SAMPLE_CACHE_MAX_BYTES
    {
        *cache.lock().unwrap_or_else(|error| error.into_inner()) = Some((key, Arc::clone(&samples)));
    }
    samples
}

impl Source {
    fn next_snapshot_id() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn snapshot_id(&self) -> u64 {
        self.snapshot_id
    }

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

    /// Packed RGB (no alpha) 16-bit samples, used by the raw decode cache.
    pub(crate) fn from_rgb16(width: u32, height: u32, rgb: &[u16]) -> Self {
        let data: Vec<[f32; 3]> = rgb
            .par_chunks_exact(3)
            .map(|p| [p[0] as f32 / 65535.0, p[1] as f32 / 65535.0, p[2] as f32 / 65535.0])
            .collect();
        Self::from_linear(width, height, data)
    }

    /// Wraps level 0 and builds the mip chain shared by both bit depths.
    /// A zero-size or short frame becomes one black pixel so `sample` stays in bounds.
    fn from_linear(width: u32, height: u32, mut data: Vec<[f32; 3]>) -> Self {
        if width == 0 || height == 0 || data.len() < (width as usize) * (height as usize) {
            return Self {
                snapshot_id: Self::next_snapshot_id(),
                levels: vec![Plane { width: 1, height: 1, data: vec![[0.0; 3]] }],
                lens_shot: None,
                lens_auto: None,
                camera_preview: false,
                c1_z6iii_linear: false,
                portrait_masks: OnceLock::new(),
                fingerprint: OnceLock::new(),
                face_rgba8: OnceLock::new(),
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
        Self {
            snapshot_id: Self::next_snapshot_id(),
            levels,
            lens_shot: None,
            lens_auto: None,
            camera_preview: false,
            c1_z6iii_linear: false,
            portrait_masks: OnceLock::new(),
            fingerprint: OnceLock::new(),
            face_rgba8: OnceLock::new(),
        }
    }

    pub(crate) fn size(&self) -> (u32, u32) {
        (self.levels[0].width, self.levels[0].height)
    }

    pub(crate) fn face_input_image(&self) -> crate::PortraitImage {
        self.face_rgba8
            .get_or_init(|| {
                let plane = &self.levels[0];
                let mut rgba = Vec::with_capacity((plane.width * plane.height * 4) as usize);
                for pixel in &plane.data {
                    for channel in pixel {
                        rgba.push(
                            (linear_to_srgb((*channel).clamp(0.0, 1.0)) * 255.0 + 0.5) as u8,
                        );
                    }
                    rgba.push(255);
                }
                crate::PortraitImage::new(plane.width, plane.height, rgba)
                    .expect("the decoded source level has exactly one RGBA plane")
            })
            .clone()
    }

    /// Marks pixels that came from the camera's embedded preview: they are already
    /// developed, so a camera profile or film curve would double-process them.
    pub(crate) fn with_camera_preview(mut self, preview: bool) -> Self {
        self.camera_preview = preview;
        self
    }

    pub(crate) fn with_c1_z6iii_linear(mut self, enabled: bool) -> Self {
        self.c1_z6iii_linear = enabled;
        self
    }

    pub(crate) fn with_lens_shot(mut self, shot: Option<super::lens_profile::LensShot>) -> Self {
        self.lens_auto = shot.as_ref().and_then(super::lens_profile::auto_warp);
        self.lens_shot = shot;
        self
    }

    /// Portrait masks at 512x512, computed once per source image.
    pub(crate) fn portrait_masks(&self) -> &crate::portrait_stage::PortraitMasks {
        self.portrait_masks.get_or_init(|| {
            let edge = 512usize;
            let rgba = self.portrait_mask_rgba(edge);
            crate::portrait_stage::PortraitMasks::from_rgba(&rgba, edge as u32, edge as u32)
        })
    }

    fn portrait_mask_rgba(&self, edge: usize) -> Vec<u8> {
        let mut rgba = vec![0u8; edge * edge * 4];
        rgba.par_chunks_mut(edge * 4).enumerate().for_each(|(y, row)| {
            for x in 0..edge {
                let px = self
                    .sample(
                        0,
                        (x as f32 + 0.5) / edge as f32,
                        (y as f32 + 0.5) / edge as f32,
                    )
                    .unwrap_or([0.0; 3]);
                for k in 0..3 {
                    row[x * 4 + k] =
                        (linear_to_srgb(px[k]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
                row[x * 4 + 3] = 255;
            }
        });
        rgba
    }

    fn fingerprint(&self) -> u64 {
        *self.fingerprint.get_or_init(|| {
            let hashes: Vec<u64> = self.levels[0]
                .data
                .par_chunks(4096)
                .map(|chunk| {
                    let mut hash = 0xcbf29ce484222325u64;
                    for pixel in chunk {
                        for value in pixel {
                            hash ^= u64::from(value.to_bits());
                            hash = hash.wrapping_mul(0x100000001b3);
                        }
                    }
                    hash
                })
                .collect();
            let mut hash = 0xcbf29ce484222325u64;
            for chunk_hash in hashes {
                hash ^= chunk_hash;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash | 1
        })
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

/// Pixel gains: exact stored C1 gains when the import had them, otherwise the
/// Kelvin/Tint approximation. K/T is display-only once exact gains exist.
fn render_wb_gains(w: &crate::WhiteBalance) -> [f32; 3] {
    w.gains
        .filter(|g| g.iter().all(|v| v.is_finite() && *v > 0.0))
        .unwrap_or_else(|| white_balance_gains(w.kelvin, w.tint))
}

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Rec.601 luma, the weight Capture One's `ZGRADATIONCURVEY` stage uses.
fn luma601(c: [f32; 3]) -> f32 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

fn sstep(a: f32, b: f32, x: f32) -> f32 {
    if b <= a {
        return if x < a { 0.0 } else { 1.0 };
    }
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `CImgOpLensLight`'s manual branch: `gain = exp2(-amount * f(radius))`.
/// `amount` is the vignetting slider in EV, so positive values darken the
/// corners while the centre stays at gain 1. `f` is C1's spline over the
/// normalised radius; a smoothstep is the B-level stand-in for its control
/// points.
fn vignette_gain(x: f32, y: f32, method: VignetteMethod, amount: f32) -> f32 {
    let (dx, dy) = (x - 0.5, y - 0.5);
    let radius = match method {
        VignetteMethod::EllipticOnCrop => (dx * dx + dy * dy).sqrt() * std::f32::consts::SQRT_2,
        VignetteMethod::Circular | VignetteMethod::CircularCrop => (dx * dx + dy * dy).sqrt() * 1.6,
    };
    2f32.powf(-amount * sstep(0.35, 1.1, radius))
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
        built: &[(u64, std::sync::Arc<MaskGrid>)],
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
                grid.data = match guide {
                    Some(guide) => guided_filter_grid(&grid.data, guide, w, h, r),
                    None => box_blur_grid(&grid.data, w, h, r),
                };
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

#[derive(Clone, Debug, PartialEq)]
struct MaskGridCacheKey {
    source: u64,
    geometry: Geometry,
    layers: Vec<(u64, Mask)>,
    has_guide: bool,
}

fn cached_mask_grids(
    src: &Source,
    adj: &AdjustmentSet,
    aspect: f32,
    cell_px: f32,
    guide: Option<&[f32]>,
) -> Arc<Vec<Option<Arc<MaskGrid>>>> {
    const MAX_CACHED_BYTES: usize = 8 * 1024 * 1024;
    static CACHE: OnceLock<
        Mutex<Vec<(MaskGridCacheKey, Arc<Vec<Option<Arc<MaskGrid>>>>)>>,
    > =
        OnceLock::new();
    let active: Vec<&Layer> =
        adj.layers.iter().filter(|layer| layer.enabled && layer.opacity > 0.0).collect();
    let key = MaskGridCacheKey {
        source: src.snapshot_id(),
        geometry: adj.geometry.clone(),
        layers: active.iter().map(|layer| (layer.id, layer.mask.clone())).collect(),
        has_guide: guide.is_some(),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut entries = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(index) = entries.iter().position(|(cached_key, _)| cached_key == &key) {
        let entry = entries.remove(index);
        let grids = Arc::clone(&entry.1);
        entries.push(entry);
        return grids;
    }

    let mut built: Vec<(u64, Arc<MaskGrid>)> = Vec::new();
    let mut grids = Vec::with_capacity(active.len());
    for layer in active {
        let grid = (!layer.mask.shapes.is_empty())
            .then(|| Arc::new(MaskGrid::build(&layer.mask, aspect, cell_px, guide, &built)));
        if let Some(grid) = &grid {
            built.push((layer.id, Arc::clone(grid)));
        }
        grids.push(grid);
    }
    let bytes = grids
        .iter()
        .flatten()
        .map(|grid| grid.data.len() * std::mem::size_of::<f32>())
        .sum::<usize>();
    let grids = Arc::new(grids);
    if bytes <= MAX_CACHED_BYTES {
        while entries
            .iter()
            .map(|(_, cached)| {
                cached
                    .iter()
                    .flatten()
                    .map(|grid| grid.data.len() * std::mem::size_of::<f32>())
                    .sum::<usize>()
            })
            .sum::<usize>()
            .saturating_add(bytes)
            > 32 * 1024 * 1024
        {
            entries.remove(0);
        }
        entries.push((key, Arc::clone(&grids)));
    }
    grids
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
            // `size` is a fraction of the long side; distance is measured in
            // short-side units so aspect and size share one scale.
            let long = aspect.max(1.0);
            let r = (stroke.size * long).max(1e-4);
            let hard = (stroke.hardness / 100.0).clamp(0.0, 0.99);
            for pt in &stroke.points {
                let dx = (p[0] - pt[0]) * aspect;
                let dy = p[1] - pt[1];
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
        MaskShape::Raster(r) => (r.op, r.sample(p)),
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
pub(crate) fn oriented_to_source(o: [f32; 2], g: &Geometry) -> [f32; 2] {
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
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    rgba.par_chunks_mut((w * 4) as usize).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let o = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32];
            let s = oriented_to_source(o, g);
            let c = src.sample(level, s[0], s[1]).unwrap_or([0.0; 3]);
            for k in 0..3 {
                row[(x * 4) as usize + k] =
                    (linear_to_srgb(c[k].max(0.0)).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            row[(x * 4 + 3) as usize] = 255;
        }
    });
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
        .into_par_iter()
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
    grid: Option<Arc<MaskGrid>>,
    overlay: bool,
    wb_gains: Option<[f32; 3]>,
    /// The layer's curves, precomputed as 256-entry LUTs.
    curve_rgb: Option<[f32; 256]>,
    curve_r: Option<[f32; 256]>,
    curve_g: Option<[f32; 256]>,
    curve_b: Option<[f32; 256]>,
    curve_luma: Option<[f32; 256]>,
    levels: Option<Levels>,
    /// Constants a per-pixel `local()` would otherwise recompute every time.
    exposure_gain: f32,
    saturation: f32,
    simple_exposure: bool,
    opacity: f32,
    /// Spatial operators; they need neighbours, so they run on the whole
    /// frame after the layers have been mixed.
    moire: Moire,
    sharpening: Option<Sharpening>,
    noise_reduction: Option<NoiseReduction>,
}

fn layer_has_spatial(a: &LocalAdjustments) -> bool {
    a.moire.amount > 0.0
        || a.sharpening.as_ref().is_some_and(|sharp| sharp.amount > 0.0)
        || a.noise_reduction.as_ref().is_some_and(|nr| {
            nr.luminance > 0.0 || nr.color > 0.0 || nr.single_pixel > 0.0
        })
}

/// Builds the per-layer plan in draw order. Each mask is rasterized before the
/// next layer starts, so a copy mask can read the layers below it.
fn layer_plan<'a>(
    l: &'a Layer,
    adj: &AdjustmentSet,
    grid: Option<Arc<MaskGrid>>,
    mask_overlay: Option<u64>,
) -> LayerPlan<'a> {
    let a = &l.adjustments;
    let opacity = (l.opacity / 100.0) * (l.mask.density.clamp(0.0, 100.0) / 100.0);
    let simple_exposure = a.fill_color.is_none()
        && a.hdr.highlights == 0.0
        && a.hdr.shadows == 0.0
        && a.hdr.whites == 0.0
        && a.hdr.blacks == 0.0
        && a.exposure.brightness == 0.0
        && a.exposure.contrast == 0.0
        && a.clarity.clarity == 0.0
        && a.clarity.structure == 0.0
        && a.dehaze.amount <= 0.0
        && a.levels.is_none()
        && a.curves.is_none()
        && a.exposure.saturation == 0.0
        && a.color_balance == ColorBalance::default()
        && a.color_editor == ColorEditor::default()
        && !a.black_white.as_ref().is_some_and(|bw| bw.enabled);
    LayerPlan {
        layer: l,
        grid,
        overlay: mask_overlay == Some(l.id),
        wb_gains: a.white_balance.as_ref().map(|w| {
            let g = render_wb_gains(w);
            let base = render_wb_gains(&adj.white_balance);
            [g[0] / base[0], g[1] / base[1], g[2] / base[2]]
        }),
        curve_rgb: a.curves.as_ref().and_then(|c| lut_of(&c.rgb)),
        curve_r: a.curves.as_ref().and_then(|c| lut_of(&c.red)),
        curve_g: a.curves.as_ref().and_then(|c| lut_of(&c.green)),
        curve_b: a.curves.as_ref().and_then(|c| lut_of(&c.blue)),
        curve_luma: a.curves.as_ref().and_then(|c| lut_of(&c.luma)),
        levels: a.levels.clone(),
        exposure_gain: 2f32.powf(a.exposure.exposure * 0.6),
        saturation: 1.0 + a.exposure.saturation / 100.0,
        simple_exposure,
        opacity,
        moire: a.moire.clone(),
        sharpening: a.sharpening.clone(),
        noise_reduction: a.noise_reduction.clone(),
    }
}

fn layer_plans<'a>(
    src: &Source,
    adj: &'a AdjustmentSet,
    aspect: f32,
    cell_px: f32,
    guide: Option<&[f32]>,
    mask_overlay: Option<u64>,
) -> Vec<LayerPlan<'a>> {
    let grids = cached_mask_grids(src, adj, aspect, cell_px, guide);
    adj.layers
        .iter()
        .filter(|layer| layer.enabled && layer.opacity > 0.0)
        .zip(grids.iter().cloned())
        .map(|(layer, grid)| layer_plan(layer, adj, grid, mask_overlay))
        .collect()
}

/// Everything precomputed once per render.
struct Plan<'a> {
    uniform_light: Option<(crate::dust_lcc::UniformLight<'a>, &'a super::geometry::FrameMap, [f32; 2])>,
    adj: &'a AdjustmentSet,
    source: u64,
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    c1_color: Option<crate::z6iii::Z6iiiColorStage>,
    gamma: GammaLut,
    /// Exact sRGB instead of the preview LUT, so 16-bit export is not capped
    /// at the LUT's 8192 steps.
    fine: bool,
    wb: [f32; 3],
    exposure: f32,
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

#[derive(Clone, Debug, PartialEq)]
struct LayerOutputWeightKey {
    source: u64,
    geometry: Geometry,
    layers: Vec<(u64, Mask)>,
    region: NormRect,
    width: usize,
    height: usize,
}

fn cached_layer_weights(
    plan: &Plan<'_>,
    map: &super::geometry::FrameMap,
    region: NormRect,
    width: usize,
    height: usize,
) -> Option<Arc<Vec<f32>>> {
    if plan.layers.is_empty()
        || plan
            .layers
            .iter()
            .any(|layer| layer.layer.mask.luma_range.is_some() || layer.layer.mask.color_range.is_some())
    {
        return None;
    }
    const MAX_ENTRY_BYTES: usize = 32 * 1024 * 1024;
    const MAX_TOTAL_BYTES: usize = 128 * 1024 * 1024;
    static CACHE: OnceLock<
        Mutex<Vec<(LayerOutputWeightKey, Arc<Vec<f32>>)>>,
    > = OnceLock::new();
    let key = LayerOutputWeightKey {
        source: plan.source,
        geometry: plan.adj.geometry.clone(),
        layers: plan
            .layers
            .iter()
            .map(|layer| (layer.layer.id, layer.layer.mask.clone()))
            .collect(),
        region,
        width,
        height,
    };
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut entries = cache.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(index) = entries.iter().position(|(cached_key, _)| cached_key == &key) {
        let entry = entries.remove(index);
        let weights = Arc::clone(&entry.1);
        entries.push(entry);
        return Some(weights);
    }

    let n = plan.layers.len();
    let xs: Vec<f32> =
        (0..width).map(|x| region.x + (x as f32 + 0.5) / width as f32 * region.width).collect();
    let mut weights = vec![0.0f32; width * height * n];
    weights.par_chunks_mut(width * n).enumerate().for_each(|(y, row)| {
        let fy = region.y + (y as f32 + 0.5) / height as f32 * region.height;
        for (x, &fx) in xs.iter().enumerate() {
            let (p, _) = map.map([fx, fy]);
            for (layer_index, layer) in plan.layers.iter().enumerate() {
                row[x * n + layer_index] = layer_mask_weight(
                    &layer.layer.mask,
                    layer.grid.as_deref(),
                    layer.opacity,
                    p,
                    [0.0; 3],
                );
            }
        }
    });
    let weights = Arc::new(weights);
    let bytes = weights.len() * std::mem::size_of::<f32>();
    if bytes <= MAX_ENTRY_BYTES {
        while entries.len() >= 4
            || entries
                .iter()
                .map(|(_, cached)| cached.len() * std::mem::size_of::<f32>())
                .sum::<usize>()
                .saturating_add(bytes)
                > MAX_TOTAL_BYTES
        {
            entries.remove(0);
        }
        entries.push((key, Arc::clone(&weights)));
    }
    Some(weights)
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

fn portrait_job_id(
    src: &Source,
    adj: &AdjustmentSet,
    region: NormRect,
    out_w: u32,
    out_h: u32,
    level: usize,
    wide: bool,
) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    src.fingerprint().hash(&mut hash);
    src.size().hash(&mut hash);
    level.hash(&mut hash);
    wide.hash(&mut hash);
    out_w.hash(&mut hash);
    out_h.hash(&mut hash);
    region.x.to_bits().hash(&mut hash);
    region.y.to_bits().hash(&mut hash);
    region.width.to_bits().hash(&mut hash);
    region.height.to_bits().hash(&mut hash);
    format!("{adj:?}").hash(&mut hash);
    hash.finish() | 1
}

fn needs_stage_face_input(extra: &std::collections::BTreeMap<String, f32>) -> bool {
    extra.iter().any(|(key, value)| {
        if value.abs() <= 1.0e-6 {
            return false;
        }
        let Some(pf) = key.strip_prefix("pc:").and_then(|s| s.parse::<u32>().ok()) else {
            return false;
        };
        matches!(
            pf,
            242 | 243
                | 247
                | 248
                | 8001..=8033
                | 30073
                | 30079..=30093
                | 41922
                | 44124
                | 48019
                | 49221
        )
    })
}

fn needs_stage_face_parsing(extra: &std::collections::BTreeMap<String, f32>) -> bool {
    extra.iter().any(|(key, value)| {
        if value.abs() <= 1.0e-6 {
            return false;
        }
        let Some(pf) = key.strip_prefix("pc:").and_then(|s| s.parse::<u32>().ok()) else {
            return false;
        };
        matches!(pf, 108 | 109 | 110)
    })
}

fn teeth_ui_percent(extra: &std::collections::BTreeMap<String, f32>, key: &str) -> f32 {
    extra.get(key).copied().unwrap_or(0.7) * 100.0
}

fn stage_face_bundle(
    job_id: u64,
    frame: &crate::PortraitImage,
    include_parsing: bool,
) -> Result<crate::FaceInputBundle, aurora_portrait::Refusal> {
    let session = match crate::FaceInputSession::open_from_env(include_parsing) {
        Ok(session) => session,
        Err(refusal) if include_parsing => {
            log::warn!(
                "canonical FaceParsingV2 session unavailable; retrying slim-only face input: {refusal:?}"
            );
            crate::FaceInputSession::open_from_env(false)?
        }
        Err(refusal) => return Err(refusal),
    };
    match session.produce(job_id, frame) {
        Ok(bundle) => Ok(bundle),
        Err(refusal) if include_parsing => {
            log::warn!(
                "canonical FaceParsingV2 unavailable; retrying slim-only face input: {refusal:?}"
            );
            crate::FaceInputSession::open_from_env(false)?.produce(job_id, frame)
        }
        Err(refusal) => Err(refusal),
    }
}

pub struct Rendered {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub(crate) histogram: Histogram,
    pub portrait_report: Option<crate::portrait_stage::PortraitRunReport>,
}

/// A 16-bit frame for export. Previews keep using the 8-bit [`Rendered`].
pub struct Rendered16 {
    pub rgba: Vec<u16>,
    pub width: u32,
    pub height: u32,
    pub portrait_report: Option<crate::portrait_stage::PortraitRunReport>,
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
    render_sized(
        src,
        adj,
        region,
        out_w,
        out_h,
        scale,
        level,
        bypass,
        mask_overlay,
        false,
        true,
    )
    .0
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
        render_sized(
            src,
            adj,
            region,
            out_w,
            out_h,
            scale,
            level,
            bypass,
            None,
            true,
            false,
        );
    Rendered16 {
        rgba: wide.expect("a wide render always returns its 16-bit buffer"),
        width: narrow.width,
        height: narrow.height,
        portrait_report: narrow.portrait_report,
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
    Rendered {
        rgba,
        width: tw,
        height: th,
        histogram,
        portrait_report: small.portrait_report,
    }
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

#[allow(clippy::too_many_arguments)]
fn build_plan<'a>(
    src: &Source,
    adj: &'a AdjustmentSet,
    uniform_light: Option<(crate::dust_lcc::UniformLight<'a>, &'a super::geometry::FrameMap, [f32; 2])>,
    fine: bool,
    aspect: f32,
    cell_px: f32,
    guide: Option<&[f32]>,
    mask_overlay: Option<u64>,
) -> Plan<'a> {
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    let c1_color = if !src.camera_preview {
        crate::z6iii::Z6iiiColorStage::from_names_for_camera(
            adj.base.icc_profile.as_deref(),
            adj.base.film_curve.as_deref(),
            src.c1_z6iii_linear,
        )
        .or_else(|| {
            src.c1_z6iii_linear.then(|| {
                crate::z6iii::Z6iiiColorStage::from_names_for_camera(
                    Some("NikonZ6III-ProStandard"),
                    adj.base.film_curve.as_deref(),
                    true,
                )
            })?
        })
    } else {
        None
    };

    Plan {
        uniform_light,
        adj,
        source: src.snapshot_id(),
        #[cfg(all(windows, feature = "z6iii-raw-export"))]
        c1_color,
        gamma: GammaLut::new(),
        fine,
        wb: render_wb_gains(&adj.white_balance),
        exposure: 2f32.powf(adj.exposure.exposure),
        curve_rgb: lut_of(&adj.curves.rgb),
        curve_r: lut_of(&adj.curves.red),
        curve_g: lut_of(&adj.curves.green),
        curve_b: lut_of(&adj.curves.blue),
        curve_luma: lut_of(&adj.curves.luma),
        levels: adj.levels != Levels::default(),
        color_editor: adj.color_editor != ColorEditor::default(),
        color_balance: adj.color_balance != ColorBalance::default(),
        layers: layer_plans(src, adj, aspect, cell_px, guide, mask_overlay),
    }
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
    with_histogram: bool,
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

    let uniform_light = crate::dust_lcc::UniformLight::prepare(
        adj.lens.uniform_light_grid.as_deref(),
        !bypass && adj.lens.uniform_light_enabled,
        // shortcut: current UI percent and soft cap, replace with verified C1 runtime policy
        (adj.lens.uniform_light / 100.0).clamp(0.0, 1.2),
        crate::dust_lcc::LightFalloffLimit::Soft,
    ).map(|prepared| (prepared, &map, [src.size().0 as f32, src.size().1 as f32]));
    let plan = build_plan(
        src,
        adj,
        uniform_light,
        wide,
        aspect,
        cell_px,
        guide.as_deref(),
        mask_overlay,
    );

    let len = (out_w * out_h * 4) as usize;
    let row = (out_w * 4) as usize;
    let mut rgba = vec![0u8; len];
    let mut rgba16 = wide.then(|| vec![0u16; len]);
    let xs: Vec<f32> =
        (0..out_w).map(|x| region.x + (x as f32 + 0.5) / out_w as f32 * region.width).collect();
    // Above 1:1 the region is magnified, so a bicubic tap keeps the detail the
    // zoom is for; at or below 1:1 the bilinear mip sample is the correct filter.
    let sharp = scale > 1.0;
    if streams_source_samples(out_w, out_h) {
        match rgba16.as_mut() {
            Some(wide_line) => {
                rgba.par_chunks_mut(row)
                    .zip(wide_line.par_chunks_mut(row))
                    .enumerate()
                    .for_each(|(y, (line, line16))| {
                        develop_line_streamed(
                            &plan,
                            src,
                            &map,
                            region,
                            out_h,
                            level,
                            sharp,
                            &xs,
                            y,
                            bypass,
                            line,
                            Some(line16),
                        );
                    });
            }
            None => {
                rgba.par_chunks_mut(row).enumerate().for_each(|(y, line)| {
                    develop_line_streamed(
                        &plan, src, &map, region, out_h, level, sharp, &xs, y, bypass, line, None,
                    );
                });
            }
        }
    } else {
        let samples =
            source_samples(src, &map, &adj.geometry, region, out_w, out_h, level, sharp, &xs);
        match rgba16.as_mut() {
            Some(wide_line) => {
                rgba.par_chunks_mut(row)
                    .zip(wide_line.par_chunks_mut(row))
                    .enumerate()
                    .for_each(|(y, (line, line16))| {
                        develop_line(&plan, region, out_h, &xs, &samples, y, bypass, line, Some(line16));
                    });
            }
            None => {
                rgba.par_chunks_mut(row).enumerate().for_each(|(y, line)| {
                    develop_line(&plan, region, out_h, &xs, &samples, y, bypass, line, None);
                });
            }
        }
    }
    // Detail, spots and layers are 8-bit stages; keeping their delta lets the
    // wide buffer carry the develop precision through them.
    let mut develop8 = rgba16.is_some().then(|| rgba.clone());
    if !bypass {
        let warp = if adj.lens.use_profile {
            super::lens_profile::warp_for(&adj.lens, src.lens_shot.as_ref(), src.lens_auto.as_ref())
        } else {
            None
        };
        match warp.as_ref() {
            // A warp is not a small delta: apply it to both buffers. The 8-bit
            // buffer feeds previews and the histogram; converting its warp to
            // a delta would replace the export's 16-bit detail with 8-bit
            // quantisation noise wherever the warp moves pixels.
            Some(warp) if rgba16.is_some() => {
                let mut preview = adj.clone();
                preview.lens.distortion = 0.0;
                preview.lens.light_falloff = 0.0;
                preview.lens.uniform_light = 0.0;
                preview.lens.uniform_light_enabled = false;
                preview.lens.sharpness_falloff = 0.0;
                preview.lens.shift_x = 0.0;
                preview.lens.shift_y = 0.0;
                preview.lens.tilt = 0.0;
                preview.lens.chromatic_aberration = false;
                preview.lens.purple_fringing = 0.0;
                preview.lens.use_profile = false;
                super::detail::post_process(&mut rgba, out_w, out_h, &preview, scale);
                if let (Some(wide), Some(before)) = (rgba16.as_mut(), develop8.as_deref()) {
                    merge_wide_delta(wide, before, &rgba, out_w as usize);
                }
                super::detail::stage_lens(&mut rgba, out_w as usize, out_h as usize, &adj.lens, Some(warp));
                // The extras are 8-bit stages; the wide buffer keeps the profile
                // warp but not these (documented gap).
                super::detail::stage_lens_extras(&mut rgba, out_w as usize, out_h as usize, &adj.lens);
                if let Some(wide) = rgba16.as_mut() {
                    super::detail::stage_lens_profile16(
                        wide,
                        out_w as usize,
                        out_h as usize,
                        &adj.lens,
                        warp,
                        (adj.lens.distortion / 100.0).clamp(-1.0, 1.0),
                        (adj.lens.light_falloff / 100.0).clamp(0.0, 1.0),
                        adj.lens.chromatic_aberration,
                    );
                }
                // Everything after this point (spots, layers) is local and can
                // merge as before.
                develop8 = Some(rgba.clone());
            }
            Some(warp) => super::detail::post_process_with_lens(&mut rgba, out_w, out_h, adj, scale, Some(warp)),
            None => super::detail::post_process(&mut rgba, out_w, out_h, adj, scale),
        }
    }
    if !adj.spots.is_empty() && !bypass {
        super::spots::apply_spots(&mut rgba, out_w, out_h, adj, src.size());
    }
    // Capture One runs its retouch chain after clone/heal and before the tone
    // stages; here spots stand in for clone/heal, so retouch goes next.
    if !bypass && !adj.retouch.is_empty() {
        apply_retouch(&mut rgba, out_w, out_h, adj, &map, region, scale);
    }
    // Layers come last: the spot removal sees the developed frame, and the
    // layer mix sees the healed one.
    if !bypass {
        apply_layers(&mut rgba, out_w, out_h, &plan, &map, region, scale, src.size());
    }
    let portrait_report = if !adj.extra.is_empty() && !bypass {
        let masks = src.portrait_masks();
        let receipt = crate::portrait_stage::PortraitJobReceipt::from_renderer_receipt(
            &rgba,
            out_w,
            out_h,
            region,
            &adj.extra,
            masks,
        );
        // Face landmarks come from the full decoded source and are mapped
        // through the renderer's FrameMap into the current out_w/out_h stage.
        let mut producer_inputs = adj.portrait_producer_inputs.as_deref().cloned();
        if needs_stage_face_input(&adj.extra) {
            let geometric_lens = adj.lens.use_profile
                || adj.lens.distortion.abs() > 1.0e-6
                || adj.lens.shift_x.abs() > 1.0e-6
                || adj.lens.shift_y.abs() > 1.0e-6
                || adj.lens.tilt.abs() > 1.0e-6;
            if geometric_lens {
                log::warn!(
                    "{}: geometric lens transform not recovered",
                    crate::FACE_STAGE_TRANSFORM_REFUSAL
                );
            } else if let Some(stage_frame) = crate::PortraitImage::new(out_w, out_h, rgba.clone()) {
                let source_frame = src.face_input_image();
                match stage_face_bundle(receipt.job_id(), &source_frame, false) {
                    Ok(bundle) => match crate::FaceStageTransform::new(
                        [src.size().0, src.size().1],
                        [out_w, out_h],
                        [region.x, region.y, region.width, region.height],
                        &adj.geometry,
                    ) {
                        Ok(transform) => match bundle.slim_inputs_for_stage(&transform, &stage_frame)
                        {
                            Ok(slim) => {
                                let mut inputs = producer_inputs.take().unwrap_or_default();
                                inputs.slim = Some(slim);
                                producer_inputs = Some(inputs);
                            }
                            Err(refusal) => {
                                log::warn!("stage face input refused: {refusal:?}");
                            }
                        },
                        Err(refusal) => {
                            log::warn!("stage face transform refused: {refusal:?}");
                        }
                    },
                    Err(refusal) => {
                        log::warn!("stage face inference refused: {refusal:?}");
                    }
                }
            }
        }
        // Keep R17's PF108 path on the stage frame; its parsing semantics are
        // owned by the separate teeth lane.
        if needs_stage_face_parsing(&adj.extra) && region == NormRect::FULL {
            if let Some(stage_frame) = crate::PortraitImage::new(out_w, out_h, rgba.clone()) {
                match stage_face_bundle(receipt.job_id(), &stage_frame, true) {
                    Ok(bundle) => {
                        let mut inputs = producer_inputs.take().unwrap_or_default();
                        let mut attached = false;
                        if inputs.slim.is_none() {
                            match bundle.slim_inputs(&stage_frame) {
                                Ok(slim) => {
                                    inputs.slim = Some(slim);
                                    attached = true;
                                }
                                Err(refusal) => {
                                    log::warn!("stage face input refused: {refusal:?}");
                                }
                            }
                        }
                        let bright_ui = teeth_ui_percent(&adj.extra, "pc:109");
                        let deyellow_ui = teeth_ui_percent(&adj.extra, "pc:110");
                        match bundle.attach_pf108_inputs(
                            &stage_frame,
                            bright_ui,
                            deyellow_ui,
                            &mut inputs,
                        ) {
                            Ok(()) => attached = true,
                            Err(refusal) => {
                                log::warn!("canonical FaceParsingV2 input refused: {refusal:?}");
                            }
                        }
                        if attached {
                            producer_inputs = Some(inputs);
                        }
                    }
                    Err(refusal) => {
                        log::warn!("stage face inference refused: {refusal:?}");
                    }
                }
            }
        }
        let mut inputs =
            crate::portrait_stage::PortraitJobInputs::from_renderer_receipt(receipt, masks);
        if let Some(producer) = producer_inputs.as_ref() {
            inputs.fill_from_producer_inputs(producer);
        }
        Some(crate::portrait_stage::apply_portrait_effects_with_job(
            &mut rgba,
            out_w,
            out_h,
            region,
            &adj.extra,
            &inputs,
        ))
    } else {
        None
    };
    if let (Some(wide), Some(before)) = (rgba16.as_mut(), develop8.as_deref()) {
        merge_wide_delta(wide, before, &rgba, out_w as usize);
    }
    let histogram = if with_histogram { histogram(&rgba) } else { Histogram::default() };
    (
        Rendered {
            rgba,
            width: out_w,
            height: out_h,
            histogram,
            portrait_report,
        },
        rgba16,
    )
}

fn merge_wide_delta(wide: &mut [u16], before: &[u8], after: &[u8], width: usize) {
    if width == 0 {
        return;
    }
    let row = width * 4;
    wide.par_chunks_mut(row)
        .zip(before.par_chunks(row))
        .zip(after.par_chunks(row))
        .for_each(|((wide_line, base_line), after_line)| {
            for ((px, base), after) in wide_line
                .chunks_exact_mut(4)
                .zip(base_line.chunks_exact(4))
                .zip(after_line.chunks_exact(4))
            {
                for k in 0..3 {
                    px[k] = (px[k] as i32 + (after[k] as i32 - base[k] as i32) * 257)
                        .clamp(0, 65535) as u16;
                }
                px[3] = if after[3] == base[3] { px[3] } else { after[3] as u16 * 257 };
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn develop_line(
    plan: &Plan,
    region: NormRect,
    out_h: u32,
    xs: &[f32],
    samples: &[SourceSample],
    y: usize,
    bypass: bool,
    line: &mut [u8],
    line16: Option<&mut [u16]>,
) {
    let pixels = line.len() / 4;
    let row = &samples[y * pixels..(y + 1) * pixels];
    develop_line_with(plan, region, out_h, xs, y, bypass, line, line16, |x, _, _| row[x]);
}

#[allow(clippy::too_many_arguments)]
fn develop_line_streamed(
    plan: &Plan,
    src: &Source,
    map: &super::geometry::FrameMap,
    region: NormRect,
    out_h: u32,
    level: usize,
    cubic: bool,
    xs: &[f32],
    y: usize,
    bypass: bool,
    line: &mut [u8],
    line16: Option<&mut [u16]>,
) {
    develop_line_with(plan, region, out_h, xs, y, bypass, line, line16, |_, _, point| {
        source_sample_at(src, map, level, cubic, point)
    });
}

#[allow(clippy::too_many_arguments)]
fn develop_line_with<F>(
    plan: &Plan,
    region: NormRect,
    out_h: u32,
    xs: &[f32],
    y: usize,
    bypass: bool,
    line: &mut [u8],
    line16: Option<&mut [u16]>,
    mut sample_at: F,
) where
    F: FnMut(usize, f32, [f32; 2]) -> SourceSample,
{
    let fy = region.y + (y as f32 + 0.5) / out_h as f32 * region.height;
    let mut line16 = line16;
    for (x, &fx) in xs.iter().enumerate() {
        let f = [fx, fy];
        let sample = sample_at(x, fx, f);
        let c = if sample.valid {
            let mut rgb = sample.rgb;
            if let Some((uniform, map, size)) = &plan.uniform_light {
                let (_, source) = map.map(f);
                let source = [source[0] * size[0] - 0.5, source[1] * size[1] - 0.5];
                rgb = uniform.apply_source_rgb(rgb, source);
            }
            develop(plan, rgb, f, bypass)
        } else {
            [0.11, 0.11, 0.11]
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

/// Linear light to display sRGB codes (used only for non-RAW display-referred
/// images such as JPEG/PNG/TIFF or camera embedded previews).
fn encode_srgb(plan: &Plan, lin: [f32; 3]) -> [f32; 3] {
    if plan.fine {
        [
            linear_to_srgb(lin[0].max(0.0)),
            linear_to_srgb(lin[1].max(0.0)),
            linear_to_srgb(lin[2].max(0.0)),
        ]
    } else {
        [plan.gamma.get(lin[0]), plan.gamma.get(lin[1]), plan.gamma.get(lin[2])]
    }
}

fn linear_edits(plan: &Plan, lin: [f32; 3], f: [f32; 2]) -> [f32; 3] {
    let adj = plan.adj;
    // Linear light: white balance. Exposure is C1's color-stage 3, after the
    // RAW tone map, so it is multiplied in below.
    let mut c = [lin[0] * plan.wb[0], lin[1] * plan.wb[1], lin[2] * plan.wb[2]];

    // Lens light falloff is corrected toward flat; vignetting adds its own.
    // C1's manual vignette (`CImgOpLensLight`, no profile branch) is
    // `gain = exp2(-amount * f(radius))` with `amount` in EV, so positive
    // amounts darken the corners and the centre stays at gain 1. The profile
    // branch is `exp2(amount% * (maxProfile - profile(x,y)))` and is carried by
    // `lens.light_falloff` instead; the two are mutually exclusive in C1.
    if adj.vignetting.amount.abs() > 1e-3 {
        let v = vignette_gain(f[0], f[1], adj.vignetting.method, adj.vignetting.amount);
        c = [c[0] * v, c[1] * v, c[2] * v];
    }

    // Dehaze is a scene-linear operator: `CImgOpDehaze2020` consumes 16-bit
    // scene values, so it runs before the display encode instead of inside
    // `tone_user()`.
    c = apply_dehaze(c, &adj.dehaze);

    // `CImgOpLocalToneMappingV8` is raw stage 85: after Dehaze, before the
    // color list's `ApplyExposure`.
    c = apply_hdr(c, &adj.hdr);
    let e = plan.exposure;
    [c[0] * e, c[1] * e, c[2] * e]
}

fn develop(plan: &Plan, lin: [f32; 3], f: [f32; 2], bypass: bool) -> [f32; 3] {
    let adj = plan.adj;
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    let mut g = if let Some(c1) = &plan.c1_color {
        if bypass {
            return c1.develop_srgb(lin);
        }
        let c = linear_edits(plan, lin, f);
        tone_user(c1.develop_srgb(c), &adj.exposure, &adj.clarity)
    } else {
        if bypass {
            return encode_srgb(plan, lin);
        }
        let c = linear_edits(plan, lin, f);
        tone_user(encode_srgb(plan, c), &adj.exposure, &adj.clarity)
    };
    #[cfg(not(all(windows, feature = "z6iii-raw-export")))]
    let mut g = {
        if bypass {
            return encode_srgb(plan, lin);
        }
        let c = linear_edits(plan, lin, f);
        tone_user(encode_srgb(plan, c), &adj.exposure, &adj.clarity)
    };

    if plan.levels {
        g = adj.levels.apply_input(g);
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
        let l = luma601(g).max(1e-4);
        let k = lut_get(lut, l) / l;
        g = [g[0] * k, g[1] * k, g[2] * k];
    }
    // `ApplyTargetLevelsCommon` is C1's stage 18, after `GradationCurves`.
    if plan.levels {
        g = adj.levels.apply_output(g);
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
    if let Some(gains) = &adj.color_balance_gains {
        g = color_balance_gains(g, gains);
    }
    if plan.color_editor {
        g = color_editor(g, &adj.color_editor, 1.0);
    }
    if adj.black_white.enabled {
        g = black_white(g, &adj.black_white);
    }

    g
}

/// The hard toe of `CImgOpLocalToneMappingV8`: kernel `z2714f67010` in the
/// OpenCL blob at ImgCoreDll file offset 41,541,440 (`z3dbd1ff682 == 13`).
/// `k` is the op's curve strength; the kernel applies it to `0.9`.
fn hdr_toe(x: f32, k: f32) -> f32 {
    let u = x * (2.0 * 0.9);
    let v = u.abs() + 1.0;
    let r = (u * v) / (v + u * u) * 0.5 + 0.5;
    let r = r.max(u - r + 1.0) * (1.0 / (2.0 * k));
    let s = r * r;
    let num = (1.0 + r + s) * (r + s * s);
    num / (1.0 + num) * k
}

/// `z386cc3479b` — the mirrored toe the shadow and white branches use.
fn hdr_inv_toe(x: f32, k: f32) -> f32 {
    hdr_toe(-x, k)
}

/// Curve strengths `op+0x8c` / `op+0x90`. The kernel scales, it does not
/// change the shape's sign, so a placeholder of 1 keeps every branch monotone.
const HDR_K_SHADOW: f32 = 1.0;
const HDR_K_HIGHLIGHT: f32 = 1.0;

/// `CImgOpLocalToneMappingV8`: one gain per pixel from a single luma plane,
/// then `out = rgb * gain`, so the panel is channel-neutral.
///
/// Highlight and shadow are kernel-exact. Whites/blacks still use the old
/// zone-window approximation: their branch needs `op+0x94` (white curve) and
/// `op+0xb4` (white offset), which are not recovered yet.
///
/// shortcut: whites/blacks stay approximate until those two fields are read
/// from a C1 export; see `SESSION-c1-parse-hdr-highlights.md`.
fn apply_hdr(mut g: [f32; 3], hdr: &HighDynamicRange) -> [f32; 3] {
    let x = luma(g);
    let mut shift = 0.0;
    if hdr.whites != 0.0 {
        shift += hdr.whites / 100.0 * 0.2 * sstep(0.65, 1.0, x);
    }
    if hdr.blacks != 0.0 {
        shift += hdr.blacks / 100.0 * 0.12 * (1.0 - sstep(0.0, 0.25, x));
    }
    if shift.abs() > 1e-5 {
        let k = (x + shift).max(0.0) / x.max(1e-4);
        g = [g[0] * k, g[1] * k, g[2] * k];
    }
    if hdr.highlights == 0.0 && hdr.shadows == 0.0 {
        return g;
    }
    let x = luma(g);
    let m = (hdr.highlights / 100.0).clamp(-1.0, 1.0);
    let b = if m != 0.0 {
        let t = hdr_toe(x, HDR_K_HIGHLIGHT);
        2f32.powf(if m < 0.0 { -t } else { t }) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };
    let m = (hdr.shadows / 100.0).clamp(-1.0, 1.0);
    let a = if m != 0.0 {
        let t = hdr_inv_toe(x, HDR_K_SHADOW);
        2f32.powf(if m < 0.0 { -t } else { t }) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };
    let gain = a / b;
    [g[0] * gain, g[1] * gain, g[2] * gain]
}

fn tone_user(
    mut g: [f32; 3],
    exp: &Exposure,
    clarity: &Clarity,
) -> [f32; 3] {
    if exp.brightness.abs() > 1e-3 {
        let gamma = 2f32.powf(-exp.brightness / 50.0 * 0.6);
        g = [g[0].max(0.0).powf(gamma), g[1].max(0.0).powf(gamma), g[2].max(0.0).powf(gamma)];
    }
    let c = exp.contrast / 50.0 * 0.8;
    g = [contrast(g[0], c), contrast(g[1], c), contrast(g[2], c)];
    let local = clarity.c1_amount() + clarity.c1_structure();
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
    g
}

/// Capture One's `soft(x,a)` knee: `max(x,0) + max(a-|x|,0)³/(6a²)`.
fn dehaze_soft(x: f32, a: f32) -> f32 {
    if a <= 0.0 {
        return x.max(0.0);
    }
    let t = (a - x.abs()).max(0.0);
    x.max(0.0) + t * t * t / (6.0 * a * a)
}

/// The `CImgOpDehaze2020` kernel (`__kernel void zec60c625ee` in the OpenCL
/// source embedded at `ImgCoreDll!0x277a0d8`). `ZDEHAZECOLOR` is the haze
/// colour `C` (0..1); the host packs `A = 65535 * W * C`.
///
/// The knee radius is `Ah = C * min(1, 2*omega)` — the *colour*, not the
/// 16-bit `A`. With `C` around 0.6 the `soft()` knee is inert for normal
/// pixels; using `A` there shifts every pixel down by the whole atmospheric
/// light and crushes the frame.
///
/// `level_low/high` are the per-channel black/white points (auto levels,
/// 0/1 by default). `omega` is `DehazeAmount * 0.01`.
fn apply_dehaze(g: [f32; 3], d: &Dehaze) -> [f32; 3] {
    let omega = d.amount / 100.0;
    if omega <= 0.0 || !omega.is_finite() {
        return g;
    }
    let c = d.shadow_tone.map(|v| v.max(0.0));
    let a = c.map(|v| v * 65535.0);
    if a.iter().all(|v| *v <= f32::EPSILON) {
        return g;
    }
    // Auto levels normalise the plane before the atmospheric-light estimate.
    let level: [(bool, f32, f32); 3] = std::array::from_fn(|c| {
        let (lo, hi) = (d.level_low[c], d.level_high[c]);
        (hi - lo > 1e-3, lo, hi - lo)
    });
    let i: [f32; 3] = std::array::from_fn(|c| {
        if level[c].0 {
            (g[c] - level[c].1) / level[c].2 * 65535.0
        } else {
            g[c] * 65535.0
        }
    });
    let eps = 1e-6;
    let delta = (2.0 * omega).min(1.0);
    let ah = c.map(|v| v * delta);
    let mut i1 = [0.0f32; 3];
    let mut a1 = [0.0f32; 3];
    for c in 0..3 {
        i1[c] = dehaze_soft(i[c] - ah[c], ah[c]);
        a1[c] = dehaze_soft(a[c] - ah[c], ah[c]);
    }
    let mut recovered = [0.0f32; 3];
    let li = luma(i1);
    let la = luma(a1);
    let keep = la / (li + la).max(eps);
    for c in 0..3 {
        recovered[c] = (1.0 - keep) * i1[c];
        i1[c] *= keep;
    }
    let m_raw = (i1[0] / a1[0].max(eps))
        .min(i1[1] / a1[1].max(eps))
        .min(i1[2] / a1[2].max(eps));
    let m0 = m_raw.clamp(0.0, 1.0);
    let mag = (i1[0] * i1[0] + i1[1] * i1[1] + i1[2] * i1[2]).sqrt();
    let dvec = a1.map(|v| m0 * v);
    let mag_d =
        ((i1[0] - dvec[0]).powi(2) + (i1[1] - dvec[1]).powi(2) + (i1[2] - dvec[2]).powi(2))
            .sqrt();
    let ratio = (mag + eps).max(eps);
    let val = (ratio * 0.375 - (mag_d + eps) * (4.0 / 7.0)) / ratio;
    let t = 1.0 / (1.0 + ((0.5 - val) * 4.0).exp());
    let m = m0 * t;
    for c in 0..3 {
        recovered[c] += (1.0 - t) * i1[c];
        i1[c] *= t;
    }
    let t0 = (1.0 - omega * m).max(0.05);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        out[c] = (i1[c] - a1[c]) / t0 + a1[c] + recovered[c];
    }
    // Saturation guard: never add more chroma than `sqrt(1+(0.5+2m)²)` allows.
    let max_i = i1[0].max(i1[1]).max(i1[2]);
    if max_i > eps {
        let sat_i = (max_i - i1[0].min(i1[1]).min(i1[2])) / max_i;
        // The kernel anchors the guard on the *unclamped* ratio.
        let limit = sat_i * (1.0 + (0.5 + 2.0 * m_raw).powi(2)).sqrt();
        let max_o = out[0].max(out[1]).max(out[2]);
        if max_o > eps {
            let sat_o = (max_o - out[0].min(out[1]).min(out[2])) / max_o;
            if sat_o > limit {
                let k = limit / sat_o;
                let l = luma(out);
                out = [k * out[0] + (1.0 - k) * l, k * out[1] + (1.0 - k) * l, k * out[2] + (1.0 - k) * l];
            }
        }
    }
    std::array::from_fn(|c| {
        if level[c].0 {
            (level[c].1 + out[c] * level[c].2 / 65535.0).max(0.0)
        } else {
            (out[c] / 65535.0).max(0.0)
        }
    })
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

/// Capture One's `ColorBalance` / `ColorBalanceMaster` triples. Master is the
/// separate `CImgOpColorBalanceMaster` channel multiplier; the three zones add a
/// `(gain - min(gain)) / min(gain)` increment weighted by `wH = L²·protect`,
/// `wM = (4L(1-L))³/2`, `wS = (1-L)²/2`, where `L` is the maximum channel. A
/// final luma ratio keeps brightness. Neutral (all 1.0) returns the pixel
/// untouched.
///
/// Evidence: `lanes/c1-parse/c1_dehaze_cb_r266/evidence/colorbalance_kernel.cl.txt`.
fn color_balance_gains(g: [f32; 3], gains: &ColorBalanceGains) -> [f32; 3] {
    // Kernel order: master first; zone step = (gain - min) / min weighted by
    // L²·protect, (4L(1-L))³/2, (1-L)²/2 along rgb/L; one luma ratio last.
    if gains.is_neutral() {
        return g;
    }
    let l_in = luma(g);
    let mut out = [
        g[0] * gains.master[0],
        g[1] * gains.master[1],
        g[2] * gains.master[2],
    ];
    let maxc = out[0].max(out[1]).max(out[2]);
    if maxc > 1e-6 && maxc.is_finite() {
        let [wh, wm, ws] = color_balance_zone_weights(maxc);
        let norm = [out[0] / maxc, out[1] / maxc, out[2] / maxc];
        for (zone, w) in [(gains.highlight, wh), (gains.midtone, wm), (gains.shadow, ws)] {
            let gmin = zone[0].min(zone[1]).min(zone[2]);
            let gmax = zone[0].max(zone[1]).max(zone[2]);
            // The kernel's own gate: a uniform triple is skipped before dividing.
            if !w.is_finite() || w.abs() <= 1e-6 || !(gmin > 1e-6) || gmax - gmin <= 1e-3 {
                continue;
            }
            for c in 0..3 {
                let step = (zone[c] - gmin) / gmin;
                if step.is_finite() {
                    out[c] += w * step * norm[c];
                }
            }
        }
    }
    let l_out = luma(out);
    if l_out.abs() > 1e-6 && l_in > 0.0 && l_out.is_finite() {
        let k = (l_in / l_out).clamp(0.0, 8.0);
        out = [out[0] * k, out[1] * k, out[2] * k];
    }
    out.map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 })
}

/// `wH = L²·protect`, `wM = (4L(1-L))³/2`, `wS = (1-L)²/2` at maximum channel `l`.
fn color_balance_zone_weights(l: f32) -> [f32; 3] {
    // shortcut: the ramp end `.sf` is unresolved; SDR pixels (`l <= 1`) use protect = 1.
    let protect = if l <= 1.0 { 1.0 } else { (2.0 - l).clamp(0.0, 1.0) };
    let bell = 4.0 * l * (1.0 - l);
    [l * l * protect, bell * bell * bell * 0.5, (1.0 - l) * (1.0 - l) * 0.5]
}

fn signed_hue_delta(h: f32, center: f32) -> f32 {
    let d = (h - center).rem_euclid(360.0);
    if d > 180.0 {
        d - 360.0
    } else {
        d
    }
}

/// Coverage of one color-editor band for a pixel at hue `h`, saturation `s`,
/// value `v`. `extra_feather` widens every band edge (the global group's
/// smoothness).
fn band_weight(b: &ColorBand, extra_feather: f32, h: f32, s: f32, v: f32) -> f32 {
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
    // Fixed 0.05 V feather: C1 has no value band, so there is no stored unit to borrow.
    let v = v.clamp(0.0, 1.0);
    let value_w = if v < b.value_low {
        sstep(b.value_low - 0.05, b.value_low, v)
    } else if v > b.value_high {
        1.0 - sstep(b.value_high, b.value_high + 0.05, v)
    } else {
        1.0
    };
    (hue_w * sat_w * value_w).clamp(0.0, 1.0)
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
        let mut w = band_weight(&ce.global, 0.0, h0, s0, v0);
        if ce.global.inverted {
            w = 1.0 - w;
        }
        dh += ce.global_shift.hue * w;
        ds += ce.global_shift.saturation * w;
        dl += ce.global_shift.lightness * w;
    }

    // Homogeneity is a skin term only. C1's `pie_skin` branch pulls toward the
    // selected Lab chroma/L*; the basic, advanced and max-smooth branches never
    // read their homogeneity fields.
    for r in &ce.skin {
        if let Some((_, band)) = ce.advanced_bands.iter().find(|(id, _)| *id == r.id) {
            if !band.enabled {
                continue;
            }
            let mut w = band_weight(band, 0.0, h0, s0, v0);
            if band.inverted {
                w = 1.0 - w;
            }
            apply_homogeneity(band, w, &mut h, &mut s, &mut v);
        } else if let Some(u) = &r.uniformity {
            let w = color_range_weight(r, g);
            h += signed_hue_delta(r.hue, h) * (u.hue / 100.0 * w);
            if u.saturation > 0.0 {
                let target_s = (r.min_saturation + 0.35).clamp(0.15, 0.55);
                s += (target_s - s) * (u.saturation / 100.0 * w);
            }
            if u.lightness > 0.0 {
                let target_v = 0.65;
                v += (target_v - v) * (u.lightness / 100.0 * w * 0.5);
            }
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
        let mut w = band_weight(band, extra, h, s, v);
        if band.inverted {
            w = 1.0 - w;
        }
        dh += shift.hue * w;
        ds += shift.saturation * w;
        dl += shift.lightness * w;
    }

    // MaxSmooth (`kind=3`) runs C1's basic `pie_smooth` window, hence `extra`.
    for (r, extra_feather) in ce
        .advanced
        .iter()
        .map(|r| (r, 0.0))
        .chain(ce.skin.iter().map(|r| (r, 0.0)))
        .chain(ce.max_smooth.iter().map(|r| (r, extra)))
    {
        let w = match ce.advanced_bands.iter().find(|(id, _)| *id == r.id) {
            Some((_, band)) if band.enabled => {
                let w = band_weight(band, extra_feather, h0, s0, v0);
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
    // Zero sliders keep the exact display luma and therefore the old buckets.
    let y = if bw.sensitivity.iter().all(|&v| v == 0.0) {
        luma(g)
    } else {
        bw_sector_y(g, &bw.sensitivity)
    }
    .max(0.0);
    if bw.highlight_tone.saturation <= 0.0 && bw.shadow_tone.saturation <= 0.0 {
        return [y, y, y];
    }
    bw_split_tone(y, bw)
}

/// The tonemap kernel's own luma basis, Q10 `0.3328 / 0.6121 / 0.0551`.
const BW_C1_LUMA: [f32; 3] = [0.3328, 0.6121, 0.0551];
const BW_C1_LUMA_Q10: [i32; 3] = [340, 626, 56];

/// `CImgOpBlackWhiteV8_splittone`, host prep at `0x1816B0B40`: the tint is
/// `0.1 * pow(mix(1, hue_ramp, saturation), 2.2)` per channel. Hue is degrees,
/// saturation `0 ..= 100`; channel offsets are `1, 2/3, 1/3` (R, G, B).
fn bw_tint(hue: f32, saturation: f32) -> [f32; 3] {
    let s = (saturation / 100.0).clamp(0.0, 1.0);
    [1.0f32, 2.0 / 3.0, 1.0 / 3.0].map(|off| {
        let x = (hue / 360.0 + off).fract();
        let ramp = ((x * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0);
        0.1 * ((1.0 - s) + s * ramp).powf(2.2)
    })
}

/// The main kernel (`p_041248720.cl`): `dark_hi` is the light tint's luma,
/// `light_lo` the dark tint's max channel; the middle band blends the two
/// linear ramps. The 65536-entry LUT is identity in this build.
fn bw_split_tone(y: f32, bw: &BlackWhite) -> [f32; 3] {
    let light = bw_tint(bw.highlight_tone.hue, bw.highlight_tone.saturation);
    let dark = bw_tint(bw.shadow_tone.hue, bw.shadow_tone.saturation);
    let dark_hi = light[0] * BW_C1_LUMA[0] + light[1] * BW_C1_LUMA[1] + light[2] * BW_C1_LUMA[2];
    let light_lo = dark[0].max(dark[1]).max(dark[2]);
    if light_lo <= dark_hi {
        return [y, y, y];
    }
    let dark_ramp = dark.map(|c| c * y / light_lo);
    let light_ramp = light.map(|c| c * y / (light_lo - dark_hi));
    let out = if y < dark_hi {
        dark_ramp
    } else if y > light_lo {
        light_ramp
    } else {
        let t = (y - dark_hi) / (light_lo - dark_hi);
        [0, 1, 2].map(|i| dark_ramp[i] + (light_ramp[i] - dark_ramp[i]) * t)
    };
    out.map(|v| v.clamp(0.0, 1.0))
}

/// Q10 filter weights from the host builder at `0x1816B1FC0`. Slider order is
/// `[red, yellow, green, cyan, blue, magenta]`; each unit is `0.01` of full
/// scale, i.e. `10.24` Q10 counts. The two tables are complementary at zero:
/// `sum = 1024`, based on the tonemap's own luma basis (not Rec.601).
fn bw_weights(sensitivity: &[f32; 6]) -> ([i32; 3], [i32; 3]) {
    const BASE: [f64; 3] = [340.7872, 626.7904, 56.4224];
    let luma = [
        BASE[0] + 10.24 * sensitivity[0] as f64,
        BASE[1] + 10.24 * sensitivity[2] as f64,
        BASE[2] + 10.24 * sensitivity[4] as f64,
    ];
    let comp = [
        (1024.0 - BASE[0]) + 10.24 * sensitivity[3] as f64,
        (1024.0 - BASE[1]) + 10.24 * sensitivity[1] as f64,
        (1024.0 - BASE[2]) + 10.24 * sensitivity[5] as f64,
    ];
    let q = |w: [f64; 3]| w.map(|v| v as i32);
    (q(comp), q(luma))
}

/// `CImgOpBlackWhiteV8_tonemap` (`p_041248720.cl`), 16-bit path: sort RGB, pick
/// the sector by `(argmax, argmin)`, then
/// `t = min + ((mid-min)*wMin + (max-mid)*wMax)/1024` (C truncating division,
/// not a shift). A neutral floor follows when `t*8 < luma`.
fn bw_sector_y(g: [f32; 3], sensitivity: &[f32; 6]) -> f32 {
    let (w_min, w_max) = bw_weights(sensitivity);
    let q = g.map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as i32);
    // The host's argmax favours R unless B >= R, and argmin favours R unless
    // G < R; ties therefore pick R. Keep that order so sector boundaries match.
    let (i_max, max) = if q[0] > q[2] { (0, q[0]) } else { (2, q[2]) };
    let (i_max, max) = if q[1] > max { (1, q[1]) } else { (i_max, max) };
    let (i_min, min) = if q[1] < q[0] { (1, q[1]) } else { (0, q[0]) };
    let (i_min, min) = if q[2] < min { (2, q[2]) } else { (i_min, min) };
    let mid = q[3 - i_max - i_min];
    let mut t = min + ((mid - min) * w_min[i_min] + (max - mid) * w_max[i_max]) / 1024;
    let y = (q[0] * BW_C1_LUMA_Q10[0] + q[1] * BW_C1_LUMA_Q10[1] + q[2] * BW_C1_LUMA_Q10[2]) / 1024;
    if t * 8 < y {
        t = if t <= 0 {
            0
        } else {
            let yu = y as u32;
            let a = ((yu * yu) >> 5) * 25;
            let b = 1 + (yu.wrapping_mul(288).wrapping_sub((t as u32).wrapping_mul(1024)) >> 5);
            ((a / b) as i32 - (yu / 32) as i32).max(0)
        };
    }
    let shift = bw_shift(&w_min, &w_max);
    (((t >> shift) << shift).clamp(0, 65535) as f32) / 65535.0
}

/// The op pairs a right shift with the splittone stage's left shift; it keeps
/// the Q10 weights inside `0 ..= 1024` when a slider pushes past full scale.
fn bw_shift(w_min: &[i32; 3], w_max: &[i32; 3]) -> u32 {
    let mut m = w_min.iter().chain(w_max).copied().max().unwrap_or(0).max(0);
    let mut shift = 0;
    while m > 1024 {
        m >>= 1;
        shift += 1;
    }
    shift
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
    if lp.simple_exposure {
        return [g[0] + (c[0] - g[0]) * m, g[1] + (c[1] - g[1]) * m, g[2] + (c[2] - g[2]) * m];
    }
    // A layer sees display codes already, so its dehaze keeps that domain and
    // only loses the `tone_user()` stages that used to run in front of it.
    c = apply_dehaze(c, &a.dehaze);
    c = apply_hdr(c, &a.hdr);
    c = tone_user(c, &a.exposure, &a.clarity);
    if let Some(levels) = &lp.levels {
        c = levels.apply_input(c);
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
        let l = luma601(c).max(1e-4);
        let k = lut_get(lut, l) / l;
        c = [c[0] * k, c[1] * k, c[2] * k];
    }
    if let Some(levels) = &lp.levels {
        c = levels.apply_output(c);
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
    if let Some(bw) = a.black_white.as_ref().filter(|bw| bw.enabled) {
        c = black_white(c, bw);
    }
    [g[0] + (c[0] - g[0]) * m, g[1] + (c[1] - g[1]) * m, g[2] + (c[2] - g[2]) * m]
}

/// The mask weight of one layer at a point, including invert, the luma and
/// colour windows, the layer opacity and the mask density.
fn layer_weight(lp: &LayerPlan, p: [f32; 2], g: [f32; 3]) -> f32 {
    layer_mask_weight(&lp.layer.mask, lp.grid.as_deref(), lp.opacity, p, g)
}

fn layer_mask_weight(
    mask: &Mask,
    grid: Option<&MaskGrid>,
    opacity: f32,
    p: [f32; 2],
    g: [f32; 3],
) -> f32 {
    let mut m = grid.map_or(1.0, |grid| grid.sample(p));
    if mask.invert {
        m = 1.0 - m;
    }
    if let Some(r) = &mask.luma_range {
        m *= luma_range_weight(r, luma(g));
    }
    if let Some(r) = &mask.color_range {
        m *= color_range_weight(r, g);
    }
    m * opacity
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

/// Resamples one retouch mask into an `out_w x out_h` coverage plane using the
/// same oriented sample points as the layer masks.
fn retouch_plane(
    mask: &crate::RasterMask,
    map: &super::geometry::FrameMap,
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
fn union_plane(a: &Option<Vec<u8>>, b: &Option<Vec<u8>>) -> Option<Vec<u8>> {
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
fn apply_retouch(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &AdjustmentSet,
    map: &super::geometry::FrameMap,
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
    source_size: (u32, u32),
) {
    if plan.layers.is_empty() {
        return;
    }
    // Heal/clone layers repair the pixels before their own colour edits stack.
    for lp in &plan.layers {
        if !lp.layer.cover.is_empty() {
            super::spots::apply_cover_patches(
                rgba,
                width,
                height,
                &plan.adj.geometry,
                &lp.layer.cover,
                source_size,
            );
        } else if matches!(lp.layer.kind, LayerKind::Heal | LayerKind::Clone) {
            if let Some(src_px) = lp.layer.select_point {
                let sw = source_size.0.max(1) as f32;
                let sh = source_size.1.max(1) as f32;
                let src_norm = [(src_px[0] / sw).clamp(0.0, 1.0), (src_px[1] / sh).clamp(0.0, 1.0)];
                let cover = CoverPatch {
                    kind: if lp.layer.kind == LayerKind::Heal {
                        CoverKind::Heal
                    } else {
                        CoverKind::Clone
                    },
                    src: src_norm,
                    dst: [0.5, 0.5],
                    half: [0.05, 0.05],
                };
                super::spots::apply_cover_patches(
                    rgba,
                    width,
                    height,
                    &plan.adj.geometry,
                    &[cover],
                    source_size,
                );
            }
        }
    }
    let w = width as usize;
    let h = height as usize;
    let n = plan.layers.len();
    let xs: Vec<f32> =
        (0..w).map(|x| region.x + (x as f32 + 0.5) / w as f32 * region.width).collect();
    let static_weights = cached_layer_weights(plan, map, region, w, h);
    // Luma/colour windows depend on the stack at the moment this layer ran, so
    // only those layers need a full-frame weight recording.
    let record_dynamic = static_weights.is_none()
        && plan
            .layers
            .iter()
            .any(|layer| layer_has_spatial(&layer.layer.adjustments));
    let mut recorded = record_dynamic.then(|| vec![0f32; w * h * n]);
    // Colour pass: the layers stack, so each one sees the previous result.
    if let Some(weights) = static_weights.as_ref() {
        rgba.par_chunks_mut(w * 4)
            .zip(weights.par_chunks(w * n))
            .enumerate()
            .for_each(|(y, (line, weights))| {
                apply_layer_color_row(
                    line,
                    Some(weights),
                    None,
                    n,
                    plan,
                    map,
                    region,
                    &xs,
                    y,
                    h,
                );
            });
    } else if let Some(weights) = recorded.as_mut() {
        rgba.par_chunks_mut(w * 4)
            .zip(weights.par_chunks_mut(w * n))
            .enumerate()
            .for_each(|(y, (line, wline))| {
                apply_layer_color_row(
                    line,
                    None,
                    Some(wline),
                    n,
                    plan,
                    map,
                    region,
                    &xs,
                    y,
                    h,
                );
            });
    } else {
        rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
            apply_layer_color_row(line, None, None, n, plan, map, region, &xs, y, h);
        });
    }
    // Spatial pass: the operators need neighbours, so they work on the frame.
    for (li, lp) in plan.layers.iter().enumerate() {
        if !layer_has_spatial(&lp.layer.adjustments) {
            continue;
        }
        let wgt = static_weights.is_none().then(|| {
            let weights = recorded.as_ref().expect("dynamic layer weights");
            (0..w * h)
                .into_par_iter()
                .map(|i| weights[i * n + li])
                .collect::<Vec<_>>()
        });
        let all_zero = wgt.as_ref().map_or_else(
            || {
                static_weights.as_ref().is_some_and(|weights| {
                    (0..w * h)
                        .into_par_iter()
                        .all(|i| weights[i * n + li] <= 1e-4)
                })
            },
            |weights| weights.iter().all(|value| *value <= 1e-4),
        );
        if all_zero || !static_mask_can_apply(lp) {
            continue;
        }
        let luma = luma_plane(rgba, w, h);
        let mut target = luma.clone();
        if let Some(nr) = &lp.noise_reduction {
            let strength = (nr.luminance / 100.0).clamp(0.0, 1.0);
            if strength > 0.0 {
                let blur = mean3x3(&luma, w, h);
                target
                    .par_iter_mut()
                    .zip(&luma)
                    .zip(&blur)
                    .for_each(|((target, &luma), &blur)| {
                        *target = luma + (blur - luma) * strength;
                    });
            }
        }
        if lp.moire.amount > 0.0 {
            let amount = (lp.moire.amount / 100.0).clamp(0.0, 1.0);
            let pattern = (lp.moire.pattern / 100.0).clamp(0.0, 1.0);
            let weight = amount * (0.25 + 0.75 * pattern);
            let blur = mean3x3(&target, w, h);
            target.par_iter_mut().zip(&blur).for_each(|(target, &blur)| {
                let hf = (*target - blur).abs();
                let blend = weight * (hf / 10.0).clamp(0.0, 1.0);
                *target += (blur - *target) * blend;
            });
        }
        if let Some(sh) = &lp.sharpening {
            let amount = sh.amount / 100.0;
            if amount > 0.0 {
                let radius = ((sh.radius.max(0.1)) * scale).round().max(1.0) as i32;
                let blur = box_blur3(&target, w, h, radius);
                let threshold = sh.threshold.max(0.0);
                target.par_iter_mut().zip(&blur).for_each(|(target, &blur)| {
                    let mut detail = *target - blur;
                    if detail.abs() < threshold {
                        detail = 0.0;
                    }
                    *target += detail * amount * 0.75;
                });
            }
        }
        // Write the new luminance back, keeping the hue ratios, and blend it
        // in by the mask weight.
        rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
            let fy = region.y + (y as f32 + 0.5) / h as f32 * region.height;
            for x in 0..w {
                let i = y * w + x;
                let old = luma[i];
                let new = target[i];
                let weight = static_weights
                    .as_ref()
                    .map_or_else(|| wgt.as_ref().expect("spatial weights")[i], |weights| {
                        weights[i * n + li]
                    });
                if weight <= 1e-4 {
                    continue;
                }
                let mixed = old + (new - old) * weight;
                let f = if old > 1.0 { mixed / old } else { 1.0 };
                for k in 0..3 {
                    line[x * 4 + k] =
                        (line[x * 4 + k] as f32 * f).clamp(0.0, 255.0).round() as u8;
                }
            }
        });
    }
}

fn static_mask_can_apply(lp: &LayerPlan<'_>) -> bool {
    if lp.opacity <= 1e-4 {
        return false;
    }
    let Some(grid) = &lp.grid else {
        return true;
    };
    grid.data.iter().any(|value| {
        let value = if lp.layer.mask.invert { 1.0 - *value } else { *value };
        value * lp.opacity > 1e-4
    })
}

#[allow(clippy::too_many_arguments)]
fn apply_layer_color_row(
    line: &mut [u8],
    static_weights: Option<&[f32]>,
    recorded_weights: Option<&mut [f32]>,
    n: usize,
    plan: &Plan,
    map: &super::geometry::FrameMap,
    region: NormRect,
    xs: &[f32],
    y: usize,
    h: usize,
) {
    let fy = region.y + (y as f32 + 0.5) / h as f32 * region.height;
    let mut recorded_weights = recorded_weights;
    for x in 0..xs.len() {
        let px = x * 4;
        let mut g = [
            line[px] as f32 / 255.0,
            line[px + 1] as f32 / 255.0,
            line[px + 2] as f32 / 255.0,
        ];
        let p = static_weights
            .is_none()
            .then(|| map.map([xs[x], fy]).0);
        for (li, lp) in plan.layers.iter().enumerate() {
            let m = if let Some(weights) = static_weights {
                weights[x * n + li]
            } else {
                layer_weight(lp, p.expect("layer positions"), g)
            };
            if let Some(weights) = recorded_weights.as_deref_mut() {
                weights[x * n + li] = m;
            }
            if m <= 1e-4 {
                continue;
            }
            g = layer_pixel(lp, g, m);
        }
        for k in 0..3 {
            line[px + k] = (g[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
}

/// Luminance of the frame, 0..255.
fn luma_plane(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut plane = vec![0f32; w * h];
    plane.par_iter_mut().enumerate().for_each(|(i, p)| {
        let j = i * 4;
        *p = 0.2126 * rgba[j] as f32 + 0.7152 * rgba[j + 1] as f32 + 0.0722 * rgba[j + 2] as f32;
    });
    plane
}

/// 3x3 mean of a plane.
fn mean3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    sum += src[yy * w + xx];
                }
            }
            row[x] = sum / 9.0;
        }
    });
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
    if w == 0 || h == 0 {
        return;
    }
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w;
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[base + (x as i32 + k).clamp(0, w as i32 - 1) as usize];
            }
            row[x] = sum * inv;
        }
    });
}

fn box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    if w == 0 || h == 0 {
        return;
    }
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[(y as i32 + k).clamp(0, h as i32 - 1) as usize * w + x];
            }
            row[x] = sum * inv;
        }
    });
}

pub(crate) fn histogram(rgba: &[u8]) -> Histogram {
    // Fixed arrays keep the per-chunk fold off the heap; a ray working on a
    // 1400 px preview otherwise allocates thousands of small vectors.
    type Bins = ([u32; 256], [u32; 256], [u32; 256], [u32; 256], u64, u64);
    let empty = || -> Bins { ([0; 256], [0; 256], [0; 256], [0; 256], 0, 0) };
    let chunk_bytes = if rgba.is_empty() {
        4
    } else {
        rgba.len()
            .div_ceil(rayon::current_num_threads().max(1) * 4)
            .max(4 * 1024)
            .next_multiple_of(4)
    };
    let (r, g, b, l, lo, hi) = rgba
        .par_chunks(chunk_bytes)
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
    let plan = build_plan(src, adj, None, false, aspect, cell_px, None, None);
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

    #[test]
    fn lcc_uniform_light_lens_stage_uses_source_coordinates_and_identity_gates() {
        let fixture = [4, 88, 67, 111, 110, 0, 0, 6, 190, 186, 254, 202, 0, 2, 4, 84, 89, 80, 69, 0, 0, 14, 67, 97, 112, 116, 117, 114, 101, 79, 110, 101, 32, 76, 67, 67, 3, 86, 69, 82, 0, 0, 2, 6, 0, 12, 76, 105, 103, 104, 116, 70, 97, 108, 108, 111, 102, 102, 1, 0, 3, 72, 100, 114, 0, 0, 16, 16, 0, 2, 0, 2, 0, 10, 0, 20, 0, 4, 0, 8, 0, 1, 0, 5, 77, 111, 100, 101, 108, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 254, 65, 72];
        let src = Source::from_linear(32, 32, vec![[0.025; 3]; 32 * 32]);
        let mut adj = AdjustmentSet::default();
        let render_bytes = |adj: &AdjustmentSet| render(&src, adj, NormRect::FULL, 32, false, None).rgba;
        let before = render_bytes(&adj);
        adj.lens.uniform_light_enabled = true;
        adj.lens.uniform_light = 100.0;
        assert_eq!(render_bytes(&adj), before, "missing grid");
        adj.lens.load_dust_lcc(&fixture).unwrap();
        adj.lens.uniform_light_enabled = false;
        assert_eq!(render_bytes(&adj), before, "disabled gate");
        adj.lens.uniform_light_enabled = true;
        adj.lens.uniform_light = 0.0;
        assert_eq!(render_bytes(&adj), before, "zero strength");
        adj.lens.uniform_light = 100.0;
        let after = render_bytes(&adj);
        let at = |x: usize, y: usize| (y * 32 + x) * 4;
        assert!(after[at(10, 20)] > before[at(10, 20)]);
        assert_eq!(after[at(14, 28)], before[at(14, 28)]);
        assert_eq!(after[at(10, 20) + 3], before[at(10, 20) + 3]);

        let base = adj.lens.uniform_light_grid.as_deref().unwrap();
        let vectors = r#"case	tile_x	tile_y	local_x	local_y	inverse_scale	strength	gate	mode	override_dx_00	field_ev	raw_ev	limited_ev	gain
top_left	10	20	0	0	1	1	1	uncapped	0	0	2	2	4
top_right	10	20	4	0	1	1	1	uncapped	0	1	1	1	2
bottom_left	10	20	0	8	1	1	1	uncapped	0	1	1	1	2
bottom_right	10	20	4	8	1	1	1	uncapped	0	2	0	0	1
center	10	20	2	4	1	1	1	uncapped	0	1	1	1	2
source_scale	5	10	1	2	2	1	1	uncapped	0	1	1	1	2
outside_left	8	20	0	0	1	1	1	uncapped	0	-0.5	2.5	2.5	5.656854249
positive_x_slope	10	20	2	0	1	1	1	uncapped	8192	0.625	1.375	1.375	2.593679109
gate_off	10	20	0	0	1	1	0	uncapped	0	0	0	0	1
zero_strength	10	20	0	0	1	0	1	uncapped	0	0	0	0	1
hard_limit	10	20	0	0	1	2	1	hard	0	0	4	3	8
soft_knee	10	20	0	0	1	2	1	soft	0	0	4	3.75	13.45434264
soft_plateau	10	20	0	0	1	3	1	soft	0	0	6	4	16
uncapped	10	20	0	0	1	3	1	uncapped	0	0	6	6	64
"#;
        let mut count = 0;
        for row in vectors.lines().skip(1) {
            let c: Vec<_> = row.split('\t').collect();
            use crate::dust_lcc::{LightFalloffLimit, UniformLight};
            let limit = match c[8] {
                "hard" => LightFalloffLimit::Hard,
                "soft" => LightFalloffLimit::Soft,
                "uncapped" => LightFalloffLimit::Uncapped,
                _ => unreachable!(),
            };
            let mut grid = base.clone();
            grid.nodes[0].dx = c[9].parse().unwrap();
            let scale: f32 = c[5].parse().unwrap();
            let source = [0, 1].map(|axis| scale
                * (c[1+axis].parse::<f32>().unwrap() + c[3+axis].parse::<f32>().unwrap()));
            let rgb = [0.125, 0.25, 0.75];
            let out = UniformLight::prepare(Some(&grid), c[7]=="1", c[6].parse().unwrap(), limit)
                .map_or(rgb, |stage| stage.apply_source_rgb(rgb, source));
            let gain: f32 = c[13].parse().unwrap();
            for (before, after) in rgb.into_iter().zip(out) {
                assert!((before * gain - after).abs() <= before * gain * 1e-5, "{}", c[0]);
            }
            count += 1;
        }
        assert_eq!(count, 14);
    }

    use super::*;

    fn gray(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h * 4) as usize]
    }

    /// Rec.601 is the weight Capture One's `ZGRADATIONCURVEY` stage uses.
    #[test]
    fn gradation_luma_uses_rec601_weights() {
        assert!((luma601([1.0, 0.0, 0.0]) - 0.299).abs() < 1e-6);
        assert!((luma601([0.0, 1.0, 0.0]) - 0.587).abs() < 1e-6);
        assert!((luma601([0.0, 0.0, 1.0]) - 0.114).abs() < 1e-6);
        // The Rec.709 weights the rest of the pipeline uses differ by 21% on green.
        assert!((luma601([0.0, 1.0, 0.0]) - luma([0.0, 1.0, 0.0])).abs() > 0.1);
    }

    /// C1's manual vignette is `exp2(-amount * f(r))` in EV. The centre must be
    /// gain 1, +EV darkens, -EV brightens, and the two directions are mirror
    /// images. Profile correction is a different branch and lives in the lens
    /// light-falloff stage, not here.
    #[test]
    fn vignette_gain_is_flat_at_centre_and_signed_in_ev() {
        for m in VignetteMethod::ALL {
            assert!((vignette_gain(0.5, 0.5, m, 2.0) - 1.0).abs() < 1e-6, "{m:?} centre");
            assert!((vignette_gain(0.0, 0.0, m, 0.0) - 1.0).abs() < 1e-6, "{m:?} zero amount");
            let dark = vignette_gain(0.0, 0.0, m, 2.0);
            let light = vignette_gain(0.0, 0.0, m, -2.0);
            assert!(dark < 0.5, "{m:?} +2EV: {dark}");
            assert!(light > 2.0, "{m:?} -2EV: {light}");
            assert!((dark * light - 1.0).abs() < 1e-6, "{m:?} mirror");
            assert!(
                vignette_gain(0.0, 0.5, m, 2.0) >= vignette_gain(0.0, 0.0, m, 2.0) - 1e-6,
                "{m:?} edge is milder than corner"
            );
        }
    }

    /// The fifth gradation curve scales the frame's luma; an identity curve
    /// builds no LUT and leaves the frame alone.
    #[test]
    fn luma_curve_reaches_the_frame() {
        let (w, h) = (8u32, 8u32);
        let src = Source::from_rgba(w, h, &gray(w, h, 100));
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None).rgba;
        let mut adj = AdjustmentSet::default();
        adj.curves.luma = Curve {
            points: vec![
                CurvePoint::new(0.0, 0.0),
                CurvePoint::new(0.5, 0.8),
                CurvePoint::new(1.0, 1.0),
            ],
        };
        let curved = render(&src, &adj, NormRect::FULL, w, false, None).rgba;
        assert!(curved[0] > plain[0], "a lift curve must brighten the frame");
        assert!(lut_of(&Curve::default()).is_none(), "identity curves build no LUT");
    }

    /// C1 applies the target levels at stage 18, after `GradationCurves`.
    #[test]
    fn target_levels_apply_after_the_curves() {
        let (w, h) = (8u32, 8u32);
        let src = Source::from_rgba(w, h, &gray(w, h, 120));
        let mut adj = AdjustmentSet::default();
        adj.curves.rgb = Curve {
            points: vec![
                CurvePoint::new(0.0, 0.0),
                CurvePoint::new(0.5, 0.9),
                CurvePoint::new(1.0, 1.0),
            ],
        };
        let lifted = render(&src, &adj, NormRect::FULL, w, false, None).rgba;
        adj.levels.rgb.output_white = 0.5;
        let capped = render(&src, &adj, NormRect::FULL, w, false, None).rgba;
        assert!(lifted[0] > capped[0], "{} vs {}", lifted[0], capped[0]);
        assert!(capped[0] <= 128, "the target white caps the lifted value");
    }

    fn levels_with_targets(rgb: [f32; 2], channels: [[f32; 2]; 3]) -> Levels {
        let mut levels = Levels::default();
        levels.rgb.output_black = rgb[0];
        levels.rgb.output_white = rgb[1];
        levels.red.output_black = channels[0][0];
        levels.red.output_white = channels[0][1];
        levels.green.output_black = channels[1][0];
        levels.green.output_white = channels[1][1];
        levels.blue.output_black = channels[2][0];
        levels.blue.output_white = channels[2][1];
        levels
    }

    /// `ApplyTargetLevelsCommon`: one RGB-master affine over `.x`, no clamping,
    /// and the stored r/g/b components are never read.
    #[test]
    fn target_levels_affine_uses_only_the_master_pair() {
        let identity = Levels::default();
        for v in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_eq!(identity.apply_output([v; 3]), [v; 3]);
        }

        let levels = levels_with_targets([0.1, 0.9], [[0.0, 1.0]; 3]);
        let out = levels.apply_output([0.1, 0.5, 0.9]);
        assert!((out[0] - 0.18).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 0.5).abs() < 1e-6, "{out:?}");
        assert!((out[2] - 0.82).abs() < 1e-6, "{out:?}");

        // The stated 0.1 / 0.5 / 0.9 -> 0 / 0.5 / 1.0 triple belongs to the
        // *input* points (stage 16), not to the target pair.
        let mut levels = Levels::default();
        levels.rgb.input_black = 0.1;
        levels.rgb.input_white = 0.9;
        assert!(levels.apply_input([0.1; 3])[0].abs() < 1e-6);
        assert!((levels.apply_input([0.5; 3])[0] - 0.5).abs() < 1e-6);
        assert!((levels.apply_input([0.9; 3])[0] - 1.0).abs() < 1e-6);
    }

    /// Stage 18 wiring: with every other stage neutral, a flat gray or color
    /// frame maps through `out = in * gain + shadow` per channel, and shadow
    /// 0 / highlight 1 stays byte-identical.
    #[test]
    fn target_levels_stage_maps_gray_and_color() {
        let (w, h) = (16u32, 16u32);
        let gray_src = Source::from_rgba(w, h, &gray(w, h, 120));
        let flat: Vec<u8> = (0..w * h).flat_map(|_| [90u8, 110, 160, 255]).collect();
        let color_src = Source::from_rgba(w, h, &flat);
        let mut adj = AdjustmentSet::default();
        let neutral = levels_with_targets([0.0, 1.0], [[0.0, 1.0]; 3]);
        adj.levels = neutral.clone();
        for src in [&gray_src, &color_src] {
            let base = render(src, &AdjustmentSet::default(), NormRect::FULL, w, false, None).rgba;
            let same = render(src, &adj, NormRect::FULL, w, false, None).rgba;
            assert_eq!(base, same, "shadow 0 / highlight 1 must be identity");
        }

        adj.levels = levels_with_targets([0.1, 0.9], [[0.0, 1.0]; 3]);
        let (shadow, gain) = (0.1f32, 0.8f32);
        for src in [&gray_src, &color_src] {
            let base = render(src, &AdjustmentSet::default(), NormRect::FULL, w, false, None).rgba;
            let mapped = render(src, &adj, NormRect::FULL, w, false, None).rgba;
            for (a, b) in base.chunks_exact(4).zip(mapped.chunks_exact(4)) {
                for c in 0..3 {
                    let want = (a[c] as f32 * gain + shadow * 255.0).round() as i32;
                    let diff = (b[c] as i32 - want).abs();
                    assert!(diff <= 1, "got {} want {} (base {})", b[c], want, a[c]);
                }
            }
        }
    }

    /// A/B on a 256-step gradient: the replaced per-channel output chain vs the
    /// master affine. `cargo test -p aurora-engine target_levels_ab_table -- --ignored --nocapture`
    #[test]
    #[ignore = "A/B table, run explicitly"]
    fn target_levels_ab_table() {
        let (w, h) = (256u32, 1u32);
        let mut gray_line = vec![0u8; (w * 4) as usize];
        let mut color_line = vec![0u8; (w * 4) as usize];
        for x in 0..w as usize {
            let v = x as u8;
            gray_line[x * 4..x * 4 + 4].copy_from_slice(&[v, v, v, 255]);
            color_line[x * 4..x * 4 + 4].copy_from_slice(&[v, (x * 2 / 3) as u8, 255 - v, 255]);
        }
        let gray_src = Source::from_rgba(w, h, &gray_line);
        let color_src = Source::from_rgba(w, h, &color_line);
        let base = |src: &Source| render(src, &AdjustmentSet::default(), NormRect::FULL, w, false, None).rgba;
        let legacy = |l: &Levels, v: [f32; 3]| {
            [
                l.rgb.map_output(l.red.map_output(v[0])),
                l.rgb.map_output(l.green.map_output(v[1])),
                l.rgb.map_output(l.blue.map_output(v[2])),
            ]
        };
        let sets = [
            ("S1 identity      ", levels_with_targets([0.0, 1.0], [[0.0, 1.0]; 3])),
            ("S2 master 0.1/0.9", levels_with_targets([0.1, 0.9], [[0.0, 1.0]; 3])),
            (
                "S3 r/g/b 0.2-0.6 ",
                levels_with_targets([0.1, 0.9], [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]),
            ),
        ];
        println!("set               | image | max |A-B| | mean |A-B| | differing");
        for (label, levels) in &sets {
            for (kind, src) in [("gray ", &gray_src), ("color", &color_src)] {
                let pixels = base(src);
                let (mut max, mut sum, mut differing) = (0f32, 0f64, 0usize);
                for px in pixels.chunks_exact(4) {
                    let g = [px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0];
                    let a = legacy(levels, g);
                    let b = levels.apply_output(g);
                    for c in 0..3 {
                        let d = (a[c] - b[c]).abs() * 255.0;
                        max = max.max(d);
                        sum += d as f64;
                        if d > 0.5 {
                            differing += 1;
                        }
                    }
                }
                println!(
                    "{label} | {kind} | {max:5.1} | {:.3} | {differing:3}/768",
                    sum / 768.0
                );
            }
        }

        let identity = &sets[0].1;
        let master = &sets[1].1;
        assert_eq!(legacy(identity, [0.4, 0.6, 0.8]), identity.apply_output([0.4, 0.6, 0.8]));
        assert_eq!(legacy(master, [0.4, 0.6, 0.8]), master.apply_output([0.4, 0.6, 0.8]));
        assert_ne!(legacy(&sets[2].1, [0.0; 3]), sets[2].1.apply_output([0.0; 3]));
    }

    /// A C1 row whose curve, target and flip columns are all default must render
    /// exactly like the same frame with those stages forced neutral.
    #[test]
    fn default_curve_and_target_columns_are_byte_identical() {
        let (w, h) = (16u32, 16u32);
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let row: std::collections::HashMap<&str, Option<String>> = std::collections::HashMap::new();
        let set = crate::c1_model::to_adjustment_set(&crate::c1_model::from_layer_row(&row));
        assert_eq!(set.levels, Levels::default());
        assert_eq!(set.curves, Curves::default());
        let mut neutral = set.clone();
        neutral.levels = Levels::default();
        neutral.curves = Curves::default();
        let a = render(&src, &set, NormRect::FULL, w, false, None).rgba;
        let b = render(&src, &neutral, NormRect::FULL, w, false, None).rgba;
        assert_eq!(a, b);
    }

    /// Draft vs Final render cost on a 2000x1333 source.
    /// `cargo test -p aurora-engine bench_render_quality -- --ignored --nocapture`
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

    #[test]
    fn gamma_lut_is_shared_and_matches_the_formula_exactly() {
        let first = GammaLut::new();
        let second = GammaLut::new();
        assert!(std::ptr::eq(first.0, second.0));
        for i in [0usize, 1, 127, 4097, GAMMA_LUT_SIZE - 1] {
            let expected = linear_to_srgb(i as f32 / (GAMMA_LUT_SIZE - 1) as f32 * GammaLut::RANGE);
            assert_eq!(first.0[i].to_bits(), expected.to_bits(), "entry {i}");
        }
    }

    /// The pre-r263 hue tent, kept only to show where the sector kernel departs.
    fn tent_bw_y(g: [f32; 3], sensitivity: &[f32; 6]) -> f32 {
        let (h, s, _) = rgb_to_hsv([g[0].max(0.0), g[1].max(0.0), g[2].max(0.0)]);
        let centers = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
        let mut boost = 0.0;
        for (i, c) in centers.iter().enumerate() {
            let w = (1.0 - hue_distance(h, *c) / 60.0).max(0.0);
            boost += sensitivity[i] / 100.0 * w;
        }
        (luma(g) * (1.0 + boost * s * 0.8)).max(0.0)
    }

    /// Zero sliders must keep the exact luma the old path produced, so default
    /// renders stay byte-identical; an equal-channel pixel is identical in the
    /// sector kernel too.
    #[test]
    fn zero_bw_sliders_keep_the_exact_luma() {
        let bw = BlackWhite { enabled: true, ..BlackWhite::default() };
        for g in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],
            [0.9, 0.9, 0.9],
            [0.37, 0.37, 0.37],
        ] {
            let y = bw_sector_y(g, &bw.sensitivity);
            assert!((y - g[0]).abs() < 2e-3, "{g:?}: sector {y}");
            assert_eq!(black_white(g, &bw), [luma(g); 3], "{g:?} default path changed");
        }
    }

    /// Same six UI weights, old tent vs new sector: neutral keeps its luma,
    /// saturated colours take different paths.
    #[test]
    fn bw_sector_and_tent_differ_on_colour_but_not_on_neutral() {
        let sensitivity = [40.0, -20.0, 30.0, -10.0, 50.0, 20.0];
        let cases = [
            ("neutral", [0.5, 0.5, 0.5]),
            ("red", [1.0, 0.0, 0.0]),
            ("yellow", [1.0, 1.0, 0.0]),
            ("green", [0.0, 1.0, 0.0]),
            ("cyan", [0.0, 1.0, 1.0]),
            ("blue", [0.0, 0.0, 1.0]),
            ("magenta", [1.0, 0.0, 1.0]),
            ("skin", [0.8, 0.6, 0.5]),
        ];
        let mut differing = 0;
        for (name, g) in cases {
            let old = tent_bw_y(g, &sensitivity);
            let new = bw_sector_y(g, &sensitivity);
            println!("{name:8} tent={old:.6} sector={new:.6} delta={:.6}", new - old);
            if (new - old).abs() > 0.02 {
                differing += 1;
            }
        }
        assert!((tent_bw_y([0.5; 3], &sensitivity) - bw_sector_y([0.5; 3], &sensitivity)).abs() < 1e-4);
        assert!(differing >= 5, "only {differing} colours changed");
    }

    /// `0x1816B1FC0`: six Q10 weights, base `0.3328/0.6121/0.0551` and their
    /// complements, each slider worth `10.24` counts.
    #[test]
    fn bw_weights_match_the_host_builder() {
        let (comp, luma) = bw_weights(&[0.0; 6]);
        assert_eq!(luma, [340, 626, 56]);
        assert_eq!(comp, [683, 397, 967]);
        assert!(luma.iter().zip(comp).all(|(a, b)| a + b == 1023 || a + b == 1024));

        let (comp, luma) = bw_weights(&[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(luma[0], 1364, "red drives the max-side weight");
        assert_eq!(comp, [683, 397, 967]);

        let (comp, _) = bw_weights(&[0.0, 0.0, 0.0, 100.0, 0.0, 0.0]);
        assert_eq!(comp[0], 1707, "cyan drives the min-side weight");
    }

    /// Saturated primaries and secondaries at zero sliders resolve to the
    /// tonemap's own luma basis, not Rec.601.
    #[test]
    fn bw_sector_endpoints_use_the_c1_luma_basis() {
        let z = [0.0; 6];
        let c1 = [
            ([1.0, 0.0, 0.0], 340.0 / 1024.0),
            ([0.0, 1.0, 0.0], 626.0 / 1024.0),
            ([0.0, 0.0, 1.0], 56.0 / 1024.0),
            ([0.0, 1.0, 1.0], 683.0 / 1024.0),
            ([1.0, 0.0, 1.0], 397.0 / 1024.0),
            ([1.0, 1.0, 0.0], 967.0 / 1024.0),
        ];
        for (g, want) in c1 {
            let got = bw_sector_y(g, &z);
            assert!((got - want).abs() < 2e-3, "{g:?}: got {got} want {want}");
        }
        // Positive red lightens pure red; negative red crushes it to black.
        let bright = bw_sector_y([1.0, 0.0, 0.0], &[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let dark = bw_sector_y([1.0, 0.0, 0.0], &[-50.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(bright > 0.8, "{bright}");
        assert_eq!(dark, 0.0);
    }

    /// `BlackWhiteV8_splittone`: dark tone owns the low end, light tone the top,
    /// the middle is a two-tone blend; hue sets the direction.
    #[test]
    fn bw_split_tone_mixes_dark_and_light_tones() {
        let mut bw = BlackWhite { enabled: true, ..BlackWhite::default() };
        bw.shadow_tone = SplitTone { hue: 0.0, saturation: 100.0 };
        bw.highlight_tone = SplitTone { hue: 240.0, saturation: 100.0 };
        let shadow = black_white([0.002, 0.002, 0.002], &bw);
        assert!(shadow[0] > shadow[2] + 0.001, "shadow should lean red: {shadow:?}");
        let highlight = black_white([0.9, 0.9, 0.9], &bw);
        assert!(highlight[2] > highlight[0] + 0.5, "highlight should lean blue: {highlight:?}");
        let mid = black_white([0.05, 0.05, 0.05], &bw);
        assert!(mid.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(mid != [mid[0]; 3], "middle band must be tinted: {mid:?}");

        let neutral = BlackWhite { enabled: true, ..BlackWhite::default() };
        assert_eq!(black_white([0.5, 0.5, 0.5], &neutral), [luma([0.5; 3]); 3]);
    }

    /// Neutral `ColorBalance*` values (the `1;1;1` default) must not move a pixel.
    #[test]
    fn neutral_color_balance_gains_keep_the_pixels() {
        let rgba = [20u8, 20, 20, 255, 220, 220, 220, 255, 60, 120, 200, 255, 200, 80, 40, 255];
        let src = Source::from_rgba(2, 2, &rgba);
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, 2, false, None);
        let mut adj = AdjustmentSet::default();
        adj.color_balance_gains = Some(ColorBalanceGains::default());
        let neutral = render(&src, &adj, NormRect::FULL, 2, false, None);
        assert_eq!(plain.rgba, neutral.rgba);
    }

    /// Amount 0 is the neutral dehaze; colour and level columns must not leak in.
    #[test]
    fn zero_dehaze_keeps_the_pixels_byte_identical() {
        let rgba = [20u8, 30, 40, 255, 220, 210, 200, 255, 60, 120, 200, 255];
        let src = Source::from_rgba(3, 1, &rgba);
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, 3, false, None);
        let mut adj = AdjustmentSet::default();
        adj.dehaze = Dehaze {
            amount: 0.0,
            shadow_tone: [0.9, 0.8, 0.7],
            level_low: [0.1; 3],
            level_high: [0.9; 3],
        };
        let neutral = render(&src, &adj, NormRect::FULL, 3, false, None);
        assert_eq!(plain.rgba, neutral.rgba);
    }

    /// Atmospheric light ~0.8 on a brighter hazy pixel must darken it, and the
    /// red channel (highest `I/A`) the least.
    #[test]
    fn dehaze_pulls_a_hazy_pixel_toward_clear_air() {
        let d = Dehaze { amount: 100.0, shadow_tone: [0.8, 0.82, 0.85], ..Default::default() };
        let hazy = [0.9, 0.92, 0.95];
        let out = apply_dehaze(hazy, &d);
        assert!(out.iter().zip(hazy).all(|(o, i)| *o < i), "{out:?}");
        assert!(out.iter().all(|v| (0.0..=1.0).contains(v)), "{out:?}");
    }

    /// Auto levels narrow the working range; the default 0/1 pair leaves it alone.
    #[test]
    fn dehaze_levels_change_the_result_only_when_written() {
        let base = Dehaze { amount: 40.0, shadow_tone: [0.75; 3], ..Default::default() };
        let mut levelled = base.clone();
        levelled.level_low = [0.1; 3];
        levelled.level_high = [0.9; 3];
        let plain = apply_dehaze([0.5, 0.55, 0.6], &base);
        let with_levels = apply_dehaze([0.5, 0.55, 0.6], &levelled);
        assert_ne!(plain, with_levels);
    }

    /// The zone step is relative to `min(gain)`, so a uniform triple adds nothing.
    #[test]
    fn uniform_color_balance_triples_are_neutral() {
        let gains = ColorBalanceGains {
            master: [0.5; 3],
            shadow: [0.5; 3],
            ..Default::default()
        };
        for pixel in [[0.2, 0.4, 0.6], [0.5; 3]] {
            let out = color_balance_gains(pixel, &gains);
            for c in 0..3 {
                assert!((out[c] - pixel[c]).abs() < 1e-4, "{pixel:?} -> {out:?}");
            }
        }
    }

    /// Highlight gains act on bright pixels; shadows barely move.
    #[test]
    fn highlight_gain_moves_bright_pixels_more_than_shadows() {
        let gains = ColorBalanceGains { highlight: [1.3, 1.0, 0.7], ..Default::default() };
        let dark = color_balance_gains([0.1; 3], &gains);
        let bright = color_balance_gains([0.9; 3], &gains);
        let dark_delta = (dark[0] - 0.1).abs() + (dark[2] - 0.1).abs();
        let bright_delta = (bright[0] - 0.9).abs() + (bright[2] - 0.9).abs();
        assert!(bright_delta > dark_delta, "{bright_delta} vs {dark_delta}");
    }

    /// Near zero the step scales linearly with the gain delta.
    #[test]
    fn small_color_balance_steps_are_linear() {
        let shift = |d: f32| {
            let gains = ColorBalanceGains { midtone: [1.0 + d, 1.0, 1.0], ..Default::default() };
            color_balance_gains([0.5; 3], &gains)[0] - 0.5
        };
        let one = shift(2e-3);
        let two = shift(4e-3);
        assert!(one > 0.0, "small step must still move the pixel");
        assert!((two / one - 2.0).abs() < 0.01, "{one} vs {two}");
    }

    /// Every zone's increment is followed by one shared luma ratio.
    #[test]
    fn color_balance_keeps_the_luma() {
        let gains = ColorBalanceGains {
            master: [1.0; 3],
            shadow: [0.7, 1.0, 1.2],
            midtone: [1.1, 1.0, 0.9],
            highlight: [1.0; 3],
        };
        let pixel = [0.2, 0.4, 0.6];
        let out = color_balance_gains(pixel, &gains);
        assert!((luma(out) - luma(pixel)).abs() < 1e-4, "{out:?}");
    }

    #[test]
    fn source_sample_cache_is_exact_and_geometry_keyed() {
        let (w, h) = (31u32, 23u32);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let src = Source::from_rgba(w, h, &rgba);
        let geometry = Geometry::default();
        let region = NormRect { x: 0.13, y: 0.21, width: 0.67, height: 0.55 };
        let (out_w, out_h) = (17, 13);
        let map = super::super::geometry::FrameMap::new(src.size(), &geometry);
        let xs: Vec<f32> =
            (0..out_w).map(|x| region.x + (x as f32 + 0.5) / out_w as f32 * region.width).collect();
        let first = source_samples(&src, &map, &geometry, region, out_w, out_h, 0, false, &xs);
        let second = source_samples(&src, &map, &geometry, region, out_w, out_h, 0, false, &xs);
        assert!(Arc::ptr_eq(&first, &second));

        for y in 0..out_h as usize {
            let fy = region.y + (y as f32 + 0.5) / out_h as f32 * region.height;
            for (x, &fx) in xs.iter().enumerate() {
                let (_, s) = map.map([fx, fy]);
                let expected = src.sample(0, s[0], s[1]);
                let actual = first[y * out_w as usize + x];
                assert_eq!(actual.valid, expected.is_some());
                if let Some(rgb) = expected {
                    assert_eq!(
                        actual.rgb.map(f32::to_bits),
                        rgb.map(f32::to_bits),
                        "sample {x},{y}"
                    );
                }
            }
        }

        let changed = Geometry { rotation: 3.0, ..geometry };
        let changed_map = super::super::geometry::FrameMap::new(src.size(), &changed);
        let third =
            source_samples(&src, &changed_map, &changed, region, out_w, out_h, 0, false, &xs);
        assert!(!Arc::ptr_eq(&first, &third));
    }

    #[test]
    fn source_sample_cache_hit_matches_a_fresh_render_exactly() {
        let (w, h) = (64u32, 48u32);
        let rgba = synthetic(w, h);
        let first = Source::from_rgba(w, h, &rgba);
        let second = Source::from_rgba(w, h, &rgba);
        let mut adj = AdjustmentSet::default();
        let _ = render(&first, &adj, NormRect::FULL, 53, false, None);
        adj.exposure.exposure = 0.37;
        adj.white_balance.kelvin += 420.0;
        let cached = render(&first, &adj, NormRect::FULL, 53, false, None);
        let fresh = render(&second, &adj, NormRect::FULL, 53, false, None);
        assert_eq!(cached.rgba, fresh.rgba);
        assert_eq!(cached.histogram, fresh.histogram);
    }

    #[test]
    fn oversized_exports_stream_samples_instead_of_allocating_the_unused_cache_vector() {
        assert_eq!(std::mem::size_of::<SourceSample>(), 16);
        assert!(!streams_source_samples(1920, 1080));
        assert!(streams_source_samples(3840, 2160));
        assert!(streams_source_samples(6000, 4000));
        assert_eq!(3840usize * 2160 * std::mem::size_of::<SourceSample>(), 132_710_400);
        assert_eq!(6000usize * 4000 * std::mem::size_of::<SourceSample>(), 384_000_000);
    }

    /// A shadow-only red gain hits darks harder than highlights.
    #[test]
    fn shadow_gain_dims_dark_pixels_more_than_bright_ones() {
        let gains = ColorBalanceGains { shadow: [0.2, 1.0, 1.0], ..Default::default() };
        let dark = color_balance_gains([0.1; 3], &gains);
        let bright = color_balance_gains([0.9; 3], &gains);
        assert!(dark[0] < bright[0] * 0.95, "{dark:?} vs {bright:?}");
        assert!(bright.iter().all(|v| (0.8..1.0).contains(v)), "{bright:?}");
        let extreme = color_balance_gains([1.5, -0.2, 0.5], &gains);
        assert!(extreme.iter().all(|v| v.is_finite()), "{extreme:?}");
    }

    #[test]
    fn zero_hdr_does_not_change_the_tone_path() {
        let exposure = Exposure::default();
        let clarity = Clarity::default();
        for g in [[0.01, 0.2, 0.8], [0.18, 0.4, 0.65], [0.5, 0.55, 0.7]] {
            assert_eq!(apply_hdr(g, &HighDynamicRange::default()), g);
            assert_eq!(tone_user(g, &exposure, &clarity), g);
        }
        let active = HighDynamicRange { highlights: -40.0, ..Default::default() };
        assert_ne!(apply_hdr([0.65; 3], &active), [0.65; 3]);
    }

    /// `CImgOpLocalToneMappingV8`: highlight+ recovers brights, shadow+ lifts
    /// darks, and the single gain keeps channel ratios intact.
    #[test]
    fn hdr_sliders_act_on_their_own_zone() {
        let dark = [0.05; 3];
        let bright = [0.85; 3];
        let hi = HighDynamicRange { highlights: 100.0, ..Default::default() };
        let sh = HighDynamicRange { shadows: 100.0, ..Default::default() };
        let hi_dark = apply_hdr(dark, &hi)[0] / dark[0];
        let hi_bright = apply_hdr(bright, &hi)[0] / bright[0];
        let sh_dark = apply_hdr(dark, &sh)[0] / dark[0];
        let sh_bright = apply_hdr(bright, &sh)[0] / bright[0];
        assert!(hi_bright < hi_dark && hi_bright < 1.0, "{hi_bright} vs {hi_dark}");
        assert!(sh_dark > sh_bright && sh_dark > 1.0, "{sh_dark} vs {sh_bright}");
        let tinted = [0.6, 0.3, 0.15];
        let out = apply_hdr(tinted, &hi);
        assert!((out[0] / out[1] - tinted[0] / tinted[1]).abs() < 1e-5);
    }

    /// The master gain shifts the channel ratio; the luma stays put.
    #[test]
    fn master_gain_scales_only_its_channel() {
        let gains = ColorBalanceGains { master: [0.5, 1.0, 1.0], ..Default::default() };
        let out = color_balance_gains([0.5, 0.5, 0.5], &gains);
        assert!((out[1] / out[0] - 2.0).abs() < 1e-4, "{out:?}");
        assert!((luma(out) - 0.5).abs() < 1e-4, "{out:?}");
    }

    /// The three zone weights are the kernel formulas, with `protect = 1` on [0,1].
    #[test]
    fn zone_weights_match_the_kernel_formula() {
        for l in [0.0f32, 0.15, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let [wh, wm, ws] = color_balance_zone_weights(l);
            let bell = 4.0 * l * (1.0 - l);
            assert!((wh - l * l).abs() < 1e-6, "wH at {l}");
            assert!((wm - bell * bell * bell * 0.5).abs() < 1e-6, "wM at {l}");
            assert!((ws - (1.0 - l) * (1.0 - l) * 0.5).abs() < 1e-6, "wS at {l}");
        }
        assert!((color_balance_zone_weights(0.5)[1] - 0.5).abs() < 1e-6);
        assert!((color_balance_zone_weights(0.0)[2] - 0.5).abs() < 1e-6);
        assert!((color_balance_zone_weights(1.0)[0] - 1.0).abs() < 1e-6);
    }

    /// A full-scale `(g - min)/min` increment keeps the luma within 0.5 %.
    #[test]
    fn full_scale_zone_steps_conserve_luma() {
        let sets = [
            ColorBalanceGains { shadow: [2.0, 1.5, 1.0], ..Default::default() },
            ColorBalanceGains { midtone: [2.0, 1.5, 1.0], ..Default::default() },
            ColorBalanceGains { highlight: [2.0, 1.5, 1.0], ..Default::default() },
            ColorBalanceGains { shadow: [1.0, 0.5, 0.5], midtone: [1.0, 0.5, 0.5], ..Default::default() },
        ];
        for gains in sets {
            for l in [0.05f32, 0.2, 0.5, 0.8, 0.95] {
                let pixel = [l, l, l];
                let out = color_balance_gains(pixel, &gains);
                let drift = (luma(out) - luma(pixel)).abs() / luma(pixel).max(1e-6);
                assert!(drift < 0.005, "{gains:?} at {l}: {out:?} drift {drift}");
            }
        }
    }

    /// Each zone's response follows its own weight curve: mid peaks at 0.5,
    /// highlight rises with L, shadow falls.
    #[test]
    fn zone_response_follows_the_weight_curves() {
        let mid = ColorBalanceGains { midtone: [1.4, 1.0, 1.0], ..Default::default() };
        let hi = ColorBalanceGains { highlight: [1.4, 1.0, 1.0], ..Default::default() };
        let lo = ColorBalanceGains { shadow: [1.4, 1.0, 1.0], ..Default::default() };
        let delta = |g: &ColorBalanceGains, l: f32| {
            (color_balance_gains([l, l, l], g)[0] - l).abs() / l.max(1e-6)
        };
        assert!(delta(&mid, 0.5) > delta(&mid, 0.2), "mid peak");
        assert!(delta(&mid, 0.5) > delta(&mid, 0.8), "mid peak");
        assert!(delta(&hi, 0.9) > delta(&hi, 0.5), "highlight rises");
        assert!(delta(&lo, 0.1) > delta(&lo, 0.5), "shadow falls");
        // wH = L² is 4x from 0.45 to 0.9; the luma ratio compresses the visible shift.
        let ratio = delta(&hi, 0.9) / delta(&hi, 0.45);
        assert!(ratio > 1.5, "highlight wH ratio {ratio}");
    }

    /// Old vs new pixel path on a 256-step gradient, three slider sets.
    #[test]
    fn ab_color_balance_before_and_after() {
        fn before(g: [f32; 3], gains: &ColorBalanceGains) -> [f32; 3] {
            if gains.is_neutral() {
                return g;
            }
            let l_in = luma(g);
            let mut out = [
                g[0] * gains.master[0],
                g[1] * gains.master[1],
                g[2] * gains.master[2],
            ];
            let maxc = out[0].max(out[1]).max(out[2]);
            if maxc > 1e-6 && maxc.is_finite() {
                let l = maxc.clamp(0.0, 1.0);
                let norm = [out[0] / maxc, out[1] / maxc, out[2] / maxc];
                let protect = if l > 0.8 { ((1.0 - l) / 0.2).clamp(0.0, 1.0) } else { 1.0 };
                let wh = l * l * protect;
                let bell = 4.0 * l * (1.0 - l);
                let wm = bell * bell * bell * 0.5;
                let ws = (1.0 - l) * (1.0 - l) * 0.5;
                for (zone, w) in [(gains.highlight, wh), (gains.midtone, wm), (gains.shadow, ws)] {
                    if w <= 1e-6 {
                        continue;
                    }
                    let gmin = zone[0].min(zone[1]).min(zone[2]);
                    if !(gmin > 1e-6) || !gmin.is_finite() {
                        continue;
                    }
                    for c in 0..3 {
                        let step = (zone[c] - gmin) / gmin;
                        if step.is_finite() {
                            out[c] += w * step * norm[c];
                        }
                    }
                }
            }
            let l_out = luma(out);
            if l_out.abs() > 1e-6 && l_in > 0.0 && l_out.is_finite() {
                let k = (l_in / l_out).clamp(0.0, 8.0);
                out = [out[0] * k, out[1] * k, out[2] * k];
            }
            out.map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 })
        }
        let sets = [
            ("warm-highlight", ColorBalanceGains { highlight: [1.25, 1.0, 0.75], ..Default::default() }),
            ("cool-shadow", ColorBalanceGains { shadow: [0.8, 1.0, 1.2], ..Default::default() }),
            ("master+mid", ColorBalanceGains {
                master: [1.05, 1.0, 0.95],
                midtone: [1.1, 1.0, 0.9],
                ..Default::default()
            }),
        ];
        let mut worst = 0.0f32;
        let mut moved = 0usize;
        for (name, gains) in &sets {
            let mut set_worst = 0.0f32;
            let mut set_moved = 0usize;
            for i in 0..256 {
                let l = i as f32 / 255.0;
                let pixel = [l, l, l];
                let old = before(pixel, gains);
                let new = color_balance_gains(pixel, gains);
                let d = (0..3).map(|c| (old[c] - new[c]).abs()).fold(0.0f32, f32::max);
                if d > 1e-4 {
                    set_moved += 1;
                }
                set_worst = set_worst.max(d);
                println!(
                    "AB,{name},{i},{l:.6},{:.6},{:.6},{:.6},{:.6},{d:.6}",
                    old[0], old[2], new[0], new[2]
                );
            }
            println!("AB,{name},summary,moved={set_moved},worst={set_worst:.6}");
            worst = worst.max(set_worst);
            moved += set_moved;
        }
        println!("AB,total,moved={moved},worst={worst:.6}");
        // The highlight ramp is the intended change, so only guard against runaway output.
        assert!(worst < 0.5, "unexpectedly large old/new delta {worst}");
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

    fn film_dir() -> std::sync::MutexGuard<'static, ()> {
        let guard = crate::filmcurve::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::filmcurve::use_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/film-curves"));
        guard
    }

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    #[test]
    fn linear_film_curve_replaces_the_srgb_encode() {
        if crate::icc::profile("NikonZ6III-ProStandard").is_none() {
            return;
        }
        let _guard = film_dir();
        let (w, h) = (8, 6);
        let src = Source::from_rgba(w, h, &flat(w, h, 128));
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None);
        let mut adj = AdjustmentSet::default();
        adj.base.icc_profile = Some("NikonZ6III-ProStandard".into());
        adj.base.film_curve = Some("AppleiPhonePro-Linear Response".into());
        let curved = render(&src, &adj, NormRect::FULL, w, false, None);
        assert_ne!(curved.rgba, plain.rgba, "C1 color stage with linear response replaces default sRGB");
    }

    #[test]
    fn unknown_film_curve_falls_back_to_the_default_path() {
        let _guard = film_dir();
        let (w, h) = (8, 6);
        let src = Source::from_rgba(w, h, &flat(w, h, 128));
        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None);
        let mut adj = AdjustmentSet::default();
        adj.base.film_curve = Some("no such curve".into());
        let missing = render(&src, &adj, NormRect::FULL, w, false, None);
        assert_eq!(missing.rgba, plain.rgba);
    }

    /// Capture One's `.fcrv` is evaluated inside `Z6iiiColorStage` via cubic
    /// Hermite `film_at` + `couple_float` + `encode_camera` + 49^3 CLUT.
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    #[test]
    fn film_curve_replaces_gamma() {
        if crate::icc::profile("NikonZ6III-ProStandard").is_none() {
            return;
        }
        let _guard = film_dir();
        let (w, h) = (4, 4);
        let src = Source::from_rgba(w, h, &flat(w, h, 128));
        let mut adj = AdjustmentSet::default();
        adj.base.icc_profile = Some("NikonZ6III-ProStandard".into());
        adj.base.curve = ToneCurvePreset::Linear;
        adj.base.film_curve = Some("TestHalfDown".into());
        let out = render(&src, &adj, NormRect::FULL, w, false, None);

        let stage = crate::z6iii::Z6iiiColorStage::from_names_for_camera(
            Some("NikonZ6III-ProStandard"),
            Some("TestHalfDown"),
            false,
        )
        .expect("C1 color stage");
        let lin = srgb_to_linear(128.0 / 255.0);
        let expected = (stage.develop_srgb([lin, lin, lin])[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        assert!(
            out.rgba[0].abs_diff(expected) <= 1,
            "C1 color stage output {} must match expected {}",
            out.rgba[0],
            expected
        );
    }

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    #[test]
    fn film_curve_darkens_midtones() {
        if crate::icc::profile("NikonZ6III-ProStandard").is_none() {
            return;
        }
        let _guard = film_dir();
        let (w, h) = (4, 4);
        let src = Source::from_rgba(w, h, &flat(w, h, 96));
        let mut adj = AdjustmentSet::default();
        adj.base.icc_profile = Some("NikonZ6III-ProStandard".into());
        adj.base.curve = ToneCurvePreset::Linear;
        let plain = render(&src, &adj, NormRect::FULL, w, false, None);
        adj.base.film_curve = Some("TestHalfDown".into());
        let curved = render(&src, &adj, NormRect::FULL, w, false, None);
        assert!(curved.rgba[0] < plain.rgba[0], "0.5 -> 0.2 must darken");
    }

    #[test]
    fn film_curve_index_is_built_once() {
        let _guard = film_dir();
        let (w, h) = (4, 4);
        let src = Source::from_rgba(w, h, &flat(w, h, 128));
        let mut adj = AdjustmentSet::default();
        adj.base.film_curve = Some("TestHalfDown".into());
        render(&src, &adj, NormRect::FULL, w, false, None);
        let scans = crate::filmcurve::scan_count();
        render(&src, &adj, NormRect::FULL, w, false, None);
        assert_eq!(crate::filmcurve::scan_count(), scans, "renders must not rescan the directory");
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
                hair_brush: false,
                use_pen_pressure: false,
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

    // ──────────────────────────── C1 raster (`.comask`) masks ────────────────────────────

    /// Left half of the frame, full coverage, optionally feathered.
    fn left_half_raster(feather: f32) -> Mask {
        let (w, h) = (64, 32);
        Mask {
            shapes: vec![MaskShape::Raster(RasterMask {
                op: MaskOp::Add,
                rect: [0.0, 0.0, 0.5, 1.0],
                width: w,
                height: h,
                data: vec![255; w * h],
            })],
            feather,
            ..Default::default()
        }
    }

    fn wb_layer(mask: Mask) -> Layer {
        Layer {
            id: 1,
            name: "local wb".into(),
            kind: LayerKind::Adjustment,
            enabled: true,
            opacity: 100.0,
            mask,
            adjustments: LocalAdjustments {
                white_balance: Some(WhiteBalance {
                    kelvin: 7000.0,
                    tint: 20.0,
                    gains: None,
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// One WB layer on a raster mask must move masked pixels and leave every
    /// pixel outside the mask byte-identical to the layer-less render.
    #[test]
    fn raster_wb_changes_only_the_masked_pixels() {
        let (w, h) = (64u32, 32u32);
        let src = Source::from_rgba(w, h, &flat(w, h, 110));
        let base = render(&src, &AdjustmentSet::default(), NormRect::FULL, w, false, None);
        let mut adj = AdjustmentSet::default();
        adj.layers.push(wb_layer(left_half_raster(0.0)));
        let out = render(&src, &adj, NormRect::FULL, w, false, None);
        let at = |frame: &Rendered, x: u32, y: u32| {
            let i = ((y * frame.width + x) * 4) as usize;
            [frame.rgba[i], frame.rgba[i + 1], frame.rgba[i + 2]]
        };
        assert_ne!(at(&out, 8, 16), at(&base, 8, 16), "inside the mask the WB must apply");
        assert_eq!(at(&out, 56, 16), at(&base, 56, 16), "outside the mask nothing may change");
    }

    /// The mask edge gets a ramp instead of a step when `Mask::feather` is set.
    #[test]
    fn raster_feather_fades_across_the_mask_edge() {
        let sharp = MaskGrid::build(&left_half_raster(0.0), 2.0, 1.0, None, &[]);
        let soft = MaskGrid::build(&left_half_raster(16.0), 2.0, 1.0, None, &[]);
        assert!((sharp.sample([0.25, 0.5]) - 1.0).abs() < 1e-6);
        assert!((sharp.sample([0.75, 0.5])).abs() < 1e-6);
        assert!(soft.sample([0.25, 0.5]) > 0.9, "well inside stays at full effect");
        let edge = soft.sample([0.5, 0.5]);
        assert!(edge > 0.05 && edge < 0.95, "the feathered edge must be partial, got {edge}");
        assert!(soft.sample([0.75, 0.5]) < 0.1, "well outside stays at zero");
    }

    /// The layer-less pipeline must stay byte-identical on the clean sRGB path
    /// (without the removed legacy `base_contrast` and `shoulder` stages).
    #[test]
    fn default_render_bytes_are_the_pre_patch_bytes() {
        let (w, h) = (257usize, 193usize);
        let mut px = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 65535) / (w - 1)) as u16;
                let g = ((y * 65535) / (h - 1)) as u16;
                let b = (((x + y) * 32767) / (w + h - 2)) as u16;
                px.extend_from_slice(&[r, g, b, 65535]);
            }
        }
        let src = Source::from_rgba16(w as u32, h as u32, &px);
        let frame = render(&src, &AdjustmentSet::default(), NormRect::FULL, 256, false, None);
        let fnv = frame
            .rgba
            .iter()
            .fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3));
        assert_eq!(fnv, 3637764891915511363u64);
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

    // ───────────────────────────── mask parameter geometry ─────────────────────────────

    fn plan_with<'a>(layer: &'a Layer, adj: &'a AdjustmentSet) -> LayerPlan<'a> {
        let grid = MaskGrid::build(&layer.mask, 4.0 / 3.0, 4.0, None, &[]);
        layer_plan(layer, adj, Some(Arc::new(grid)), None)
    }

    #[test]
    fn cached_mask_grids_share_rasters_and_preserve_samples() {
        let src = Source::from_rgba(64, 48, &synthetic(64, 48));
        let mut adj = AdjustmentSet::default();
        for id in 1..=8 {
            adj.layers.push(Layer {
                id,
                opacity: 100.0,
                mask: hard_disc(0.1 + id as f32 * 0.03),
                ..Default::default()
            });
        }

        let first = cached_mask_grids(&src, &adj, 4.0 / 3.0, 4.0, None);
        let second = cached_mask_grids(&src, &adj, 4.0 / 3.0, 4.0, None);
        assert!(Arc::ptr_eq(&first, &second));

        for (a, b) in first.iter().zip(second.iter()) {
            let (Some(a), Some(b)) = (a.as_ref(), b.as_ref()) else {
                assert!(a.is_none() && b.is_none());
                continue;
            };
            assert!(Arc::ptr_eq(a, b));
            for p in [[0.0, 0.0], [0.25, 0.5], [0.5, 0.5], [1.0, 1.0]] {
                assert_eq!(a.sample(p).to_bits(), b.sample(p).to_bits());
            }
        }

        let expected =
            MaskGrid::build(&adj.layers[0].mask, 4.0 / 3.0, 4.0, None, &[]);
        assert_eq!(expected.data, first[0].as_ref().unwrap().data);
    }

    /// Layer opacity 100 must leave the mask's own weight untouched.
    #[test]
    fn full_opacity_is_the_identity_mask_weight() {
        let adj = AdjustmentSet::default();
        let layer = Layer { id: 1, opacity: 100.0, mask: hard_disc(0.3), ..Default::default() };
        let lp = plan_with(&layer, &adj);
        assert!((layer_weight(&lp, [0.5, 0.5], [0.0; 3]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn layer_opacity_scales_the_mask_weight() {
        let adj = AdjustmentSet::default();
        let mut layer = Layer { id: 1, opacity: 50.0, mask: hard_disc(0.3), ..Default::default() };
        let half = layer_weight(&plan_with(&layer, &adj), [0.5, 0.5], [0.0; 3]);
        assert!((half - 0.5).abs() < 1e-6, "50% opacity must halve the weight, got {half}");
        layer.opacity = 25.0;
        let quarter = layer_weight(&plan_with(&layer, &adj), [0.5, 0.5], [0.0; 3]);
        assert!((quarter - 0.25).abs() < 1e-6, "25% opacity must quarter the weight, got {quarter}");
    }

    #[test]
    fn spatial_mask_samples_the_output_frame_not_the_mask_grid() {
        let (w, h) = (640u32, 480u32);
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let mut layer = Layer {
            id: 1,
            opacity: 73.0,
            mask: Mask { invert: true, density: 81.0, ..hard_disc(0.31) },
            ..Default::default()
        };
        layer.adjustments.exposure.exposure = 0.4;
        layer.adjustments.sharpening =
            Some(Sharpening { amount: 120.0, ..Default::default() });
        let mut static_adj = AdjustmentSet::default();
        static_adj.layers.push(layer.clone());
        let mut recorded_adj = static_adj.clone();
        recorded_adj.layers[0].mask.luma_range = Some(LumaRange {
            low_outer: 0.0,
            low_inner: 0.0,
            high_inner: 1.0,
            high_outer: 1.0,
            ..Default::default()
        });
        let static_out = render(&src, &static_adj, NormRect::FULL, 420, false, None);
        let recorded_out = render(&src, &recorded_adj, NormRect::FULL, 420, false, None);
        assert_eq!(static_out.rgba, recorded_out.rgba);
        assert_eq!(static_out.histogram, recorded_out.histogram);
    }

    #[test]
    fn simple_local_exposure_matches_the_general_tone_path_exactly() {
        let (w, h) = (320u32, 240u32);
        let src = Source::from_rgba(w, h, &synthetic(w, h));
        let mut layer = Layer { id: 1, mask: hard_disc(0.35), ..Default::default() };
        layer.adjustments.exposure.exposure = 0.6;
        let mut simple = AdjustmentSet::default();
        simple.layers.push(layer.clone());
        let mut general = simple.clone();
        general.layers[0].adjustments.exposure.contrast = 0.001;
        let simple_out = render(&src, &simple, NormRect::FULL, 280, false, None);
        let general_out = render(&src, &general, NormRect::FULL, 280, false, None);
        assert_eq!(simple_out.rgba, general_out.rgba);
        assert_eq!(simple_out.histogram, general_out.histogram);
    }

    #[test]
    fn zero_opacity_or_disabled_layer_is_skipped() {
        let mut adj = AdjustmentSet::default();
        let src = Source::from_rgba(8, 6, &flat(8, 6, 128));
        adj.layers.push(Layer { id: 1, opacity: 0.0, mask: hard_disc(0.3), ..Default::default() });
        assert!(layer_plans(&src, &adj, 4.0 / 3.0, 4.0, None, None).is_empty());
        adj.layers[0].opacity = 100.0;
        adj.layers[0].enabled = false;
        assert!(layer_plans(&src, &adj, 4.0 / 3.0, 4.0, None, None).is_empty());
    }

    #[test]
    fn linear_gradient_ramps_once_from_inner_to_outer() {
        let shape = MaskShape::Linear(LinearGradient {
            op: MaskOp::Add,
            start: [0.2, 0.5],
            end: [0.8, 0.5],
        });
        let at = |x: f32| shape_value(&shape, [x, 0.5], 1.0, None).1;
        assert!((at(0.1) - 1.0).abs() < 1e-6, "before the inner line must be full effect");
        assert!((at(0.9)).abs() < 1e-6, "past the outer line must be zero");
        let ramp: Vec<f32> = (0..=10).map(|i| at(0.2 + 0.06 * i as f32)).collect();
        assert!((0.3..0.7).contains(&at(0.5)), "the middle must be a partial value");
        assert!(ramp.windows(2).all(|w| w[0] >= w[1]), "ramp must be monotonic: {ramp:?}");
        // Swapping the two lines is the exact linear inversion; the model has
        // no separate flag, matching how C1 stores the pair as geometry.
        let flipped = MaskShape::Linear(LinearGradient {
            op: MaskOp::Add,
            start: [0.8, 0.5],
            end: [0.2, 0.5],
        });
        let at_flipped = |x: f32| shape_value(&flipped, [x, 0.5], 1.0, None).1;
        assert!((at_flipped(0.1) - 0.0).abs() < 1e-6);
        assert!((at_flipped(0.9) - 1.0).abs() < 1e-6);
    }

    /// C1 names the zero-effect line first; `file_engine` feeds it in reversed
    /// because this model is full at `start`.
    #[test]
    fn a_c1_gradient_reaches_full_effect_at_the_line_it_names_second() {
        let shape = MaskShape::Linear(LinearGradient {
            op: MaskOp::Add,
            start: [0.8, 0.5],
            end: [0.2, 0.5],
        });
        let at = |x: f32| shape_value(&shape, [x, 0.5], 1.0, None).1;
        assert!(at(0.1) < 1e-6, "the C1 zero line must stay at zero effect");
        assert!((at(0.9) - 1.0).abs() < 1e-6, "the C1 full line must reach full effect");
        assert!((at(0.5) - 0.5).abs() < 1e-6);
    }

    /// The radial gradient peaks at its centre, reaches zero on the outer
    /// radius, and its `inverted` flag flips both.
    #[test]
    fn radial_gradient_centre_edge_and_invert() {
        let radial = |inverted| {
            MaskShape::Radial(RadialGradient {
                op: MaskOp::Add,
                center: [0.5, 0.5],
                radius: [0.2, 0.2],
                rotation: 0.0,
                feather: 50.0,
                inverted,
            })
        };
        let at = |shape: &MaskShape, p: [f32; 2]| shape_value(shape, p, 1.0, None).1;
        let normal = radial(false);
        assert!((at(&normal, [0.5, 0.5]) - 1.0).abs() < 1e-6);
        assert!((0.3..0.7).contains(&at(&normal, [0.65, 0.5])), "the feather must fade");
        assert!(at(&normal, [0.7, 0.5]).abs() < 1e-6, "the outer radius must be zero");
        let inverted = radial(true);
        assert!(at(&inverted, [0.5, 0.5]).abs() < 1e-6);
        assert!((at(&inverted, [0.7, 0.5]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn whole_mask_invert_flips_the_weight() {
        let adj = AdjustmentSet::default();
        let layer = Layer {
            id: 1,
            mask: Mask { invert: true, ..hard_disc(0.3) },
            ..Default::default()
        };
        let lp = plan_with(&layer, &adj);
        assert!(layer_weight(&lp, [0.5, 0.5], [0.0; 3]).abs() < 1e-6);
        assert!((layer_weight(&lp, [0.02, 0.02], [0.0; 3]) - 1.0).abs() < 1e-6);
    }

    /// A hard radial edge is a step, not a NaN, and a feathered edge crosses
    /// the boundary without a visible jump.
    #[test]
    fn feather_edges_are_finite_and_continuous() {
        let hard = MaskShape::Radial(RadialGradient {
            op: MaskOp::Add,
            center: [0.5, 0.5],
            radius: [0.2, 0.2],
            rotation: 0.0,
            feather: 0.0,
            inverted: false,
        });
        for i in 0..=100 {
            let d = i as f32 / 50.0;
            let v = shape_value(&hard, [0.5 + 0.2 * d, 0.5], 1.0, None).1;
            assert!(v.is_finite(), "radius {d} produced {v}");
            assert!((0.0..=1.0).contains(&v));
        }
        let mut mask = hard_disc(0.3);
        mask.feather = 40.0;
        let grid = MaskGrid::build(&mask, 4.0 / 3.0, 4.0, None, &[]);
        assert!(grid.data.iter().all(|v| v.is_finite()));
        let row = grid.h / 2;
        let jump = (1..grid.w)
            .map(|x| (grid.data[row * grid.w + x] - grid.data[row * grid.w + x - 1]).abs())
            .fold(0.0f32, f32::max);
        assert!(jump < 0.25, "feather must not step across the edge, jump {jump}");
    }

    /// `ZSURFACESMOOTHNESSAMOUNT` reaches `Mask::refine`; without a luminance
    /// guide the refine still smooths instead of silently doing nothing.
    #[test]
    fn surface_smoothness_smooths_the_mask_edge() {
        let mut mask = hard_disc(0.3);
        let sharp = MaskGrid::build(&mask, 4.0 / 3.0, 4.0, None, &[]);
        mask.refine = 40.0;
        let soft = MaskGrid::build(&mask, 4.0 / 3.0, 4.0, None, &[]);
        assert_ne!(sharp.data, soft.data, "refine must change the raster");
        assert!(soft.data.iter().all(|v| v.is_finite()));
        let softened = soft
            .data
            .iter()
            .zip(&sharp.data)
            .filter(|(b, a)| **b > 0.0 && **b < 1.0 && **a <= 0.0)
            .count();
        assert!(softened > 0, "refine must leave intermediate values on the edge");
    }

    /// A brush radius is physical: on a 16:9 image the same fractional radius
    /// must reach equally far in x and y, not be squashed by the aspect.
    #[test]
    fn brush_radius_is_round_on_a_wide_image() {
        let aspect = 16.0 / 9.0;
        let mut brush = BrushStroke {
            op: MaskOp::Add,
            points: vec![[0.5, 0.5, 1.0]],
            size: 0.1,
            hardness: 0.0,
            opacity: 100.0,
            flow: 100.0,
            auto_mask: false,
            hair_brush: false,
            use_pen_pressure: false,
        };
        let shape = MaskShape::Brush(brush.clone());
        let x = shape_value(&shape, [0.55, 0.5], aspect, None).1;
        let y = shape_value(&shape, [0.5, 0.5 + 0.05 * aspect], aspect, None).1;
        assert!((x - y).abs() < 1e-3, "radius must be round: x={x} y={y}");
        brush.hardness = 100.0;
        let shape = MaskShape::Brush(brush);
        assert!(shape_value(&shape, [0.5, 0.5], aspect, None).1 > 0.99);
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
            hair_brush: false,
            use_pen_pressure: false,
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
        let built = [(7u64, Arc::new(grid.clone()))];
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
            ..Default::default()
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
                curves: Some(Curves {
                    rgb: Curve { points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(0.5, 0.95), CurvePoint::new(1.0, 1.0)] },
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
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
                hair_brush: false,
                use_pen_pressure: false,
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

    /// The default 0 ..= 1 value band must not change one byte of the old
    /// two-factor (hue × saturation) weight.
    #[test]
    fn default_value_band_matches_the_old_weight_over_v() {
        let mut band = ColorBand {
            center_hue: 30.0,
            center_saturation: 0.4,
            hue_low: -20.0,
            hue_high: 15.0,
            sat_low: -40.0,
            sat_high: 25.0,
            smoothness: 8.0,
            ..ColorBand::default()
        };
        for extra in [0.0, 5.0] {
            for h in (0..=360).step_by(5) {
                for s in [0.0, 0.1, 0.4, 0.8, 1.0] {
                    let reference = band_weight(&band, extra, h as f32, s, 0.5);
                    for v in [0.0, 0.05, 0.2, 0.5, 0.8, 1.0, 1.4] {
                        assert_eq!(
                            band_weight(&band, extra, h as f32, s, v),
                            reference,
                            "default value band moved weight at h={h} s={s} v={v}"
                        );
                    }
                }
            }
        }
        band.inverted = true;
        assert_eq!(band_weight(&band, 0.0, 30.0, 0.4, 0.9), band_weight(&band, 0.0, 30.0, 0.4, 0.1));
    }

    /// Old `.portrait` documents have no value bounds and must load as 0 / 1.
    #[test]
    fn old_documents_load_the_full_value_band() {
        let band: ColorBand =
            serde_json::from_str(r#"{"center_hue":120.0,"smoothness":15.0}"#).unwrap();
        assert_eq!((band.value_low, band.value_high), (0.0, 1.0));
        assert_eq!(ColorBand::default().value_low, 0.0);
        assert_eq!(ColorBand::default().value_high, 1.0);
    }

    /// Tightening the value band must keep dark and bright pixels untouched.
    #[test]
    fn tightened_value_band_only_edits_its_window() {
        let mut ce = ColorEditor::default();
        let band = &mut ce.bands[HueSector::Red as usize];
        band.hue_low = -25.0;
        band.hue_high = 25.0;
        band.sat_low = -100.0;
        band.sat_high = 100.0;
        band.smoothness = 10.0;
        band.center_saturation = 1.0;
        band.value_low = 0.4;
        band.value_high = 0.6;
        ce.basic[HueSector::Red as usize] =
            HslShift { hue: 30.0, saturation: 0.0, lightness: 0.0 };

        let mid = hsv_to_rgb(0.0, 1.0, 0.5);
        let dark = hsv_to_rgb(0.0, 1.0, 0.1);
        let bright = hsv_to_rgb(0.0, 1.0, 0.95);
        let shift = |p: [f32; 3]| signed_hue_delta(hue_of(color_editor(p, &ce, 1.0)), 0.0);
        assert!(shift(mid) > 20.0, "midtone must still shift: {}", shift(mid));
        assert!(shift(dark).abs() < 1e-3, "dark pixel moved: {}", shift(dark));
        assert!(shift(bright).abs() < 1e-3, "bright pixel moved: {}", shift(bright));
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
            value_low: 0.0,
            value_high: 1.0,
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
            value_low: -1.0e9,
            value_high: 1.0e9,
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

    /// Homogeneity lightness pulls V toward the selected color's value on the
    /// skin list, and is ignored by every other list.
    #[test]
    fn lightness_homogeneity_converges_on_skin_only() {
        let band = ColorBand {
            center_hue: 0.0,
            center_saturation: 0.75,
            center_value: 0.8,
            hue_low: -10.0,
            hue_high: 10.0,
            smoothness: 2.0,
            homogeneity: [0.0, 0.0, 100.0],
            ..ColorBand::default()
        };
        let range = ColorRange { id: 1, ..ColorRange::default() };
        let mut ce = ColorEditor::default();
        ce.advanced_bands.push((1, band));
        ce.skin.push(range.clone());

        let dark = hsv_to_rgb(0.0, 0.75, 0.3);
        let light = hsv_to_rgb(0.0, 0.75, 0.95);
        let before = (rgb_to_hsv(dark).2 - rgb_to_hsv(light).2).abs();
        let after = (rgb_to_hsv(color_editor(dark, &ce, 1.0)).2
            - rgb_to_hsv(color_editor(light, &ce, 1.0)).2)
            .abs();
        assert!(after < before * 0.01, "skin V did not converge: {before} -> {after}");

        // The same band on another list must not move the pixel at all.
        for list in 0..2 {
            let mut ce = ColorEditor::default();
            ce.advanced_bands.push((1, band));
            if list == 0 {
                ce.advanced.push(range.clone());
            } else {
                ce.max_smooth.push(range.clone());
            }
            for pixel in [dark, light] {
                let out = color_editor(pixel, &ce, 1.0);
                for i in 0..3 {
                    assert!(
                        (out[i] - pixel[i]).abs() < 1e-6,
                        "homogeneity leaked out of the skin list: {out:?} vs {pixel:?}"
                    );
                }
            }
        }
    }

    /// C1 renders `pie_maxSmooth` with the basic `pie_smooth` window, so a
    /// max-smooth group must match the basic slot with the same band and shift
    /// (global feather included), not the narrower advanced one.
    #[test]
    fn max_smooth_renders_like_basic_not_advanced() {
        let band = ColorBand {
            center_hue: 30.0,
            center_saturation: 0.5,
            center_value: 0.6,
            hue_low: -10.0,
            hue_high: 10.0,
            sat_low: -60.0,
            sat_high: 60.0,
            smoothness: 5.0,
            ..ColorBand::default()
        };
        let shift = HslShift { hue: 15.0, saturation: 20.0, lightness: 10.0 };

        let mut basic = ColorEditor::default();
        basic.global.smoothness = 20.0;
        basic.bands[HueSector::Red as usize] = band;
        basic.basic[HueSector::Red as usize] = shift;

        let mut smooth = ColorEditor::default();
        smooth.global.smoothness = 20.0;
        smooth.advanced_bands.push((1, band));
        smooth.max_smooth.push(ColorRange { id: 1, shift, ..ColorRange::default() });

        let mut advanced = ColorEditor::default();
        advanced.global.smoothness = 20.0;
        advanced.advanced_bands.push((1, band));
        advanced.advanced.push(ColorRange { id: 1, shift, ..ColorRange::default() });

        for h in [25.0, 30.0, 38.0, 41.0, 44.0, 60.0] {
            let pixel = hsv_to_rgb(h, 0.5, 0.6);
            let b = color_editor(pixel, &basic, 1.0);
            let m = color_editor(pixel, &smooth, 1.0);
            let a = color_editor(pixel, &advanced, 1.0);
            assert_eq!(m, b, "max smooth must match basic at h={h}");
            if (41.0..=44.0).contains(&h) {
                assert!(
                    (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs() > 1e-4,
                    "basic and advanced feathers must differ at h={h}: {a:?} vs {b:?}"
                );
            }
        }
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
            value_low: 0.0,
            value_high: 1.0,
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
    /// `cargo test -p aurora-engine --offline shadow_levels_16_vs_8 -- --ignored --nocapture`
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

    #[test]
    fn export_render_skips_preview_histogram_without_changing_pixels() {
        let (w, h) = (67u32, 43u32);
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i.wrapping_mul(37) % 251) as u8).collect();
        let src = Source::from_rgba(w, h, &rgba);
        let adj = AdjustmentSet::default();
        let (preview, preview_wide) = render_sized(
            &src,
            &adj,
            NormRect::FULL,
            w,
            h,
            1.0,
            0,
            false,
            None,
            true,
            true,
        );
        let (export, export_wide) = render_sized(
            &src,
            &adj,
            NormRect::FULL,
            w,
            h,
            1.0,
            0,
            false,
            None,
            true,
            false,
        );
        assert_ne!(preview.histogram, Histogram::default());
        assert_eq!(export.histogram, Histogram::default());
        assert_eq!(export.rgba, preview.rgba);
        assert_eq!(export_wide, preview_wide);
    }

    #[test]
    fn parallel_histogram_matches_serial_exact_counts() {
        let pixels = 4099usize;
        let rgba: Vec<u8> = (0..pixels * 4).map(|i| (i.wrapping_mul(37) % 256) as u8).collect();
        let (mut red, mut green, mut blue, mut luma) = ([0u32; 256], [0u32; 256], [0u32; 256], [0u32; 256]);
        let (mut shadows, mut highlights) = (0u64, 0u64);
        for pixel in rgba.chunks_exact(4) {
            red[pixel[0] as usize] += 1;
            green[pixel[1] as usize] += 1;
            blue[pixel[2] as usize] += 1;
            let y = (0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32) as usize;
            luma[y.min(255)] += 1;
            shadows += u64::from(pixel[0] <= 1 && pixel[1] <= 1 && pixel[2] <= 1);
            highlights += u64::from(pixel[0] >= 254 || pixel[1] >= 254 || pixel[2] >= 254);
        }
        let expected = Histogram {
            red: red.to_vec(),
            green: green.to_vec(),
            blue: blue.to_vec(),
            luma: luma.to_vec(),
            clipped_shadows: shadows as f32 / pixels as f32,
            clipped_highlights: highlights as f32 / pixels as f32,
        };
        assert_eq!(histogram(&rgba), expected);
    }

    #[test]
    #[ignore = "profiles 24MP preview histogram"]
    fn bench_histogram_24mp() {
        let (w, h) = (6000usize, 4000usize);
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i.wrapping_mul(37) % 251) as u8).collect();
        let mut times = Vec::with_capacity(9);
        for _ in 0..9 {
            let started = std::time::Instant::now();
            std::hint::black_box(histogram(&rgba));
            times.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        println!(
            "24MP histogram P50={:.3} ms P95={:.3} ms",
            times[4],
            times[8]
        );
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
    /// `cargo test -p aurora-engine --offline zoom_region_bench -- --ignored --nocapture`
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

    #[cfg(windows)]
    #[test]
    #[ignore = "needs a real NEF, portrait model, and NVIDIA CUDA device"]
    fn portrait_mask_route_runs_through_preview_and_export_entries() {
        use std::path::Path;

        let path = std::env::var("AURORA_PORTRAIT_BENCH_NEF")
            .unwrap_or_else(|_| r"C:\Users\Jxcz2\Desktop\_DSC5673.NEF".to_owned());
        let (w, h, rgba16) = crate::raw::decode16(Path::new(&path)).expect("decode NEF");
        assert_eq!(crate::raw::from_sensor(Path::new(&path)), Some(true));
        let src = Source::from_rgba16(w, h, &rgba16);
        let mut adj = AdjustmentSet::default();
        adj.extra.insert("pc:21015".into(), 0.9);

        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, 512, false, None);
        assert!(plain.portrait_report.is_none());
        let preview = render(&src, &adj, NormRect::FULL, 512, false, None);
        let export = render16(&src, &adj, NormRect::FULL, 512, false);

        for (entry, report) in [
            ("preview", preview.portrait_report.as_ref()),
            ("export16", export.portrait_report.as_ref()),
        ] {
            let report = report.unwrap_or_else(|| panic!("{entry} report missing"));
            assert!(report.job_id.is_some(), "{entry} job id missing");
            assert_eq!(report.effects.len(), 1, "{entry} report");
            let effect = &report.effects[0];
            assert_eq!(effect.pf, 21015);
            assert!(effect.executed, "{entry} did not execute: {effect:?}");
            assert_eq!(
                effect.backend,
                crate::portrait_stage::PortraitBackend::Cuda,
                "{entry} must execute CUDA on this acceptance machine: {effect:?}"
            );
            assert!(!effect.dispatches.is_empty(), "{entry} has no dispatch");
            println!(
                "{entry} job={:?} backend={:?} dispatch={:?}",
                report.job_id, effect.backend, effect.dispatches
            );
        }
        assert_ne!(preview.rgba, plain.rgba, "PF21015 must change preview pixels");
        assert!(export.rgba.chunks_exact(4).all(|pixel| pixel[3] == 65535));
    }

    fn serial_mean3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
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

    fn serial_luma_guide(src: &Source, g: &Geometry, w: usize, h: usize) -> Vec<f32> {
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
                    Some(c) => {
                        (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).clamp(0.0, 1.0)
                    }
                    None => 0.0,
                }
            })
            .collect()
    }

    fn serial_ai_input(src: &Source, g: &Geometry, long: u32) -> (Vec<u8>, u32, u32) {
        let (ow, oh) = crate::mock::geometry::FrameMap::new(src.size(), g).oriented();
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
                    rgba.push(
                        (linear_to_srgb(value.max(0.0)).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    );
                }
                rgba.push(255);
            }
        }
        (rgba, w, h)
    }

    fn serial_portrait_mask_rgba(src: &Source, edge: usize) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(edge * edge * 4);
        for y in 0..edge {
            for x in 0..edge {
                let px = src
                    .sample(
                        0,
                        (x as f32 + 0.5) / edge as f32,
                        (y as f32 + 0.5) / edge as f32,
                    )
                    .unwrap_or([0.0; 3]);
                for value in px {
                    rgba.push(
                        (linear_to_srgb(value).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                    );
                }
                rgba.push(255);
            }
        }
        rgba
    }

    fn serial_box_blur_h(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
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

    fn serial_box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
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

    fn serial_box_blur3(src: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
        let mut a = src.to_vec();
        let mut b = vec![0.0f32; w * h];
        for _ in 0..3 {
            serial_box_blur_h(&a, &mut b, w, h, r);
            std::mem::swap(&mut a, &mut b);
            serial_box_blur_v(&a, &mut b, w, h, r);
            std::mem::swap(&mut a, &mut b);
        }
        a
    }

    #[test]
    fn parallel_spatial_filters_match_serial_exactly() {
        let (w, h) = (37usize, 23usize);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let plane: Vec<f32> = (0..w * h)
            .map(|i| ((i * 29 % 997) as f32) / 17.0 - 13.0)
            .collect();

        let luma = luma_plane(&rgba, w, h);
        let serial_luma: Vec<f32> = rgba
            .chunks_exact(4)
            .map(|p| {
                0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
            })
            .collect();
        assert_eq!(luma, serial_luma);
        assert_eq!(mean3x3(&plane, w, h), serial_mean3x3(&plane, w, h));

        let mut horizontal = vec![0.0f32; w * h];
        let mut serial_horizontal = vec![0.0f32; w * h];
        box_blur_h(&plane, &mut horizontal, w, h, 3);
        serial_box_blur_h(&plane, &mut serial_horizontal, w, h, 3);
        assert_eq!(horizontal, serial_horizontal);

        let mut vertical = vec![0.0f32; w * h];
        let mut serial_vertical = vec![0.0f32; w * h];
        box_blur_v(&plane, &mut vertical, w, h, 3);
        serial_box_blur_v(&plane, &mut serial_vertical, w, h, 3);
        assert_eq!(vertical, serial_vertical);
        assert_eq!(box_blur3(&plane, w, h, 3), serial_box_blur3(&plane, w, h, 3));
    }

    #[test]
    fn parallel_luma_guide_matches_serial_exactly() {
        let (sw, sh) = (640u32, 480u32);
        let rgba: Vec<u8> = (0..sw * sh)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let src = Source::from_rgba(sw, sh, &rgba);
        let g = Geometry::default();
        assert_eq!(
            luma_guide(&src, &g, 256, 192),
            serial_luma_guide(&src, &g, 256, 192)
        );
    }

    #[test]
    fn parallel_ai_input_matches_serial_exactly() {
        let (sw, sh) = (1024u32, 768u32);
        let rgba: Vec<u8> = (0..sw * sh)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let src = Source::from_rgba(sw, sh, &rgba);
        let g = Geometry::default();
        assert_eq!(ai_input_rgba(&src, &g, 1024), serial_ai_input(&src, &g, 1024));
    }

    #[test]
    fn parallel_portrait_mask_input_matches_serial_exactly() {
        let (sw, sh) = (640u32, 480u32);
        let rgba: Vec<u8> = (0..sw * sh)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let src = Source::from_rgba(sw, sh, &rgba);
        assert_eq!(
            src.portrait_mask_rgba(128),
            serial_portrait_mask_rgba(&src, 128)
        );
    }

    #[test]
    fn parallel_wide_delta_matches_serial_exactly() {
        let (w, h) = (37usize, 23usize);
        let mut wide: Vec<u16> = (0..w * h * 4)
            .map(|i| ((i * 4051) % 65536) as u16)
            .collect();
        let before: Vec<u8> =
            (0..w * h * 4).map(|i| ((i * 37 + i / 5) % 256) as u8).collect();
        let mut after = before.clone();
        for (i, value) in after.iter_mut().enumerate() {
            *value = value.wrapping_add(((i * 29) % 19) as u8);
        }
        for pixel in after.chunks_exact_mut(4).take(w) {
            pixel[3] = 127;
        }
        let mut expected = wide.clone();
        for ((px, base), result) in expected
            .chunks_exact_mut(4)
            .zip(before.chunks_exact(4))
            .zip(after.chunks_exact(4))
        {
            for k in 0..3 {
                px[k] = (px[k] as i32 + (result[k] as i32 - base[k] as i32) * 257)
                    .clamp(0, 65535) as u16;
            }
            px[3] = if result[3] == base[3] { px[3] } else { result[3] as u16 * 257 };
        }
        merge_wide_delta(&mut wide, &before, &after, w);
        assert_eq!(wide, expected);
    }

    #[test]
    #[ignore = "profiles 4K local spatial filters"]
    fn bench_parallel_spatial_filters() {
        let (w, h) = (3840usize, 2160usize);
        let plane: Vec<f32> = (0..w * h)
            .map(|i| ((i * 37 % 4093) as f32) / 91.0 - 17.0)
            .collect();
        let run = |serial: bool| {
            let start = std::time::Instant::now();
            if serial {
                std::hint::black_box(serial_mean3x3(&plane, w, h));
            } else {
                std::hint::black_box(mean3x3(&plane, w, h));
            }
            start.elapsed().as_secs_f64() * 1000.0
        };
        let mut serial_mean = Vec::new();
        let mut parallel_mean = Vec::new();
        for _ in 0..3 {
            serial_mean.push(run(true));
            parallel_mean.push(run(false));
        }
        serial_mean.sort_by(f64::total_cmp);
        parallel_mean.sort_by(f64::total_cmp);

        let run_blur = |serial: bool| {
            let start = std::time::Instant::now();
            if serial {
                std::hint::black_box(serial_box_blur3(&plane, w, h, 4));
            } else {
                std::hint::black_box(box_blur3(&plane, w, h, 4));
            }
            start.elapsed().as_secs_f64() * 1000.0
        };
        let mut serial_blur = Vec::new();
        let mut parallel_blur = Vec::new();
        for _ in 0..3 {
            serial_blur.push(run_blur(true));
            parallel_blur.push(run_blur(false));
        }
        serial_blur.sort_by(f64::total_cmp);
        parallel_blur.sort_by(f64::total_cmp);

        println!(
            "4K local spatial filters: mean3x3 {:.2}x ({:.3}->{:.3} ms), box3 {:.2}x ({:.3}->{:.3} ms)",
            serial_mean[1] / parallel_mean[1].max(f64::EPSILON),
            serial_mean[1],
            parallel_mean[1],
            serial_blur[1] / parallel_blur[1].max(f64::EPSILON),
            serial_blur[1],
            parallel_blur[1]
        );
    }

    #[test]
    #[ignore = "profiles the 256-cell refine guide"]
    fn bench_luma_guide() {
        let (sw, sh) = (2400u32, 1600u32);
        let rgba: Vec<u8> = (0..sw * sh)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let src = Source::from_rgba(sw, sh, &rgba);
        let g = Geometry::default();
        let mut serial = Vec::new();
        let mut parallel = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            std::hint::black_box(serial_luma_guide(&src, &g, 256, 171));
            serial.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = std::time::Instant::now();
            std::hint::black_box(luma_guide(&src, &g, 256, 171));
            parallel.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        serial.sort_by(f64::total_cmp);
        parallel.sort_by(f64::total_cmp);
        println!(
            "256-cell luma guide: serial={:.3} ms parallel={:.3} ms ({:.2}x)",
            serial[2],
            parallel[2],
            serial[2] / parallel[2].max(f64::EPSILON)
        );
    }

    #[test]
    #[ignore = "profiles 1024 px AI input preparation"]
    fn bench_ai_input() {
        let (sw, sh) = (2400u32, 1600u32);
        let rgba: Vec<u8> = (0..sw * sh)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let src = Source::from_rgba(sw, sh, &rgba);
        let g = Geometry::default();
        let mut serial = Vec::new();
        let mut parallel = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            std::hint::black_box(serial_ai_input(&src, &g, 1024));
            serial.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = std::time::Instant::now();
            std::hint::black_box(ai_input_rgba(&src, &g, 1024));
            parallel.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        serial.sort_by(f64::total_cmp);
        parallel.sort_by(f64::total_cmp);
        println!(
            "1024 px AI input: serial={:.3} ms parallel={:.3} ms ({:.2}x)",
            serial[2],
            parallel[2],
            serial[2] / parallel[2].max(f64::EPSILON),
        );
    }

    #[test]
    #[ignore = "profiles 512 px portrait mask input"]
    fn bench_portrait_mask_input() {
        let (sw, sh) = (2400u32, 1600u32);
        let rgba: Vec<u8> = (0..sw * sh)
            .flat_map(|i| {
                [
                    (i * 37 % 251) as u8,
                    (i * 73 % 241) as u8,
                    (i * 109 % 239) as u8,
                    255,
                ]
            })
            .collect();
        let src = Source::from_rgba(sw, sh, &rgba);
        let mut serial = Vec::new();
        let mut parallel = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            std::hint::black_box(serial_portrait_mask_rgba(&src, 512));
            serial.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = std::time::Instant::now();
            std::hint::black_box(src.portrait_mask_rgba(512));
            parallel.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        serial.sort_by(f64::total_cmp);
        parallel.sort_by(f64::total_cmp);
        println!(
            "512 px portrait mask input: serial={:.3} ms parallel={:.3} ms ({:.2}x)",
            serial[2],
            parallel[2],
            serial[2] / parallel[2].max(f64::EPSILON),
        );
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "needs a real portrait NEF and an NVIDIA CUDA device"]
    fn portrait_lut_route_runs_through_preview_and_export_entries() {
        use std::path::Path;

        let path = std::env::var("AURORA_PORTRAIT_BENCH_NEF")
            .unwrap_or_else(|_| r"C:\Users\Jxcz2\Desktop\_DSC5673.NEF".to_owned());
        let (w, h, rgba16) = crate::raw::decode16(Path::new(&path)).expect("decode NEF");
        assert_eq!(crate::raw::from_sensor(Path::new(&path)), Some(true));
        let src = Source::from_rgba16(w, h, &rgba16);
        let mut adj = AdjustmentSet::default();
        adj.extra.insert("pc:90007".into(), 0.9);

        let plain = render(&src, &AdjustmentSet::default(), NormRect::FULL, 512, false, None);
        let plain16 = render16(&src, &AdjustmentSet::default(), NormRect::FULL, 512, false);
        let preview = render(&src, &adj, NormRect::FULL, 512, false, None);
        let export = render16(&src, &adj, NormRect::FULL, 512, false);

        let skin = src
            .portrait_masks()
            .mask_u8(
                crate::portrait_stage::MaskKind::Skin,
                preview.width,
                preview.height,
                NormRect::FULL,
            )
            .expect("real skin mask");
        let skin_pixels = skin.iter().filter(|&&value| value > 0).count();
        assert!(skin_pixels > 0, "the real portrait job produced an empty skin mask");

        for (entry, report) in [
            ("preview", preview.portrait_report.as_ref()),
            ("export16", export.portrait_report.as_ref()),
        ] {
            let report = report.unwrap_or_else(|| panic!("{entry} report missing"));
            assert!(report.job_id.is_some(), "{entry} job id missing");
            assert_eq!(report.effects.len(), 1, "{entry} report");
            let effect = &report.effects[0];
            assert_eq!(effect.pf, 90007);
            assert!(effect.executed, "{entry} did not execute: {effect:?}");
            assert_eq!(
                effect.backend,
                crate::portrait_stage::PortraitBackend::Cuda,
                "{entry} must execute CUDA on this acceptance machine: {effect:?}"
            );
            assert!(
                effect
                    .dispatches
                    .iter()
                    .any(|dispatch| dispatch.kernel == "skin_adjust_lut_kernel"),
                "{entry} did not dispatch the ruddy LUT kernel: {effect:?}"
            );
            println!(
                "{entry} job={:?} backend={:?} dispatch={:?}",
                report.job_id, effect.backend, effect.dispatches
            );
        }

        let preview_diff: Vec<u8> = preview
            .rgba
            .iter()
            .zip(&plain.rgba)
            .map(|(&after, &before)| after.abs_diff(before))
            .collect();
        let preview_channels = preview_diff
            .chunks_exact(4)
            .flat_map(|pixel| &pixel[..3])
            .filter(|&&difference| difference != 0)
            .count();
        let preview_pixels = preview_diff
            .chunks_exact(4)
            .filter(|pixel| pixel[..3].iter().any(|&difference| difference != 0))
            .count();
        let preview_max = preview_diff.iter().copied().max().unwrap_or_default();
        let preview_sum: u64 = preview_diff.chunks_exact(4).flat_map(|pixel| &pixel[..3])
            .map(|&difference| u64::from(difference))
            .sum();

        let export_diff: Vec<u16> = export
            .rgba
            .iter()
            .zip(&plain16.rgba)
            .map(|(&after, &before)| after.abs_diff(before))
            .collect();
        let export_pixels = export_diff
            .chunks_exact(4)
            .filter(|pixel| pixel[..3].iter().any(|&difference| difference != 0))
            .count();
        let export_max = export_diff.iter().copied().max().unwrap_or_default();
        let export_sum: u64 = export_diff.chunks_exact(4).flat_map(|pixel| &pixel[..3])
            .map(|&difference| u64::from(difference))
            .sum();

        println!(
            "source={path} decoded={w}x{h} skin_pixels={skin_pixels} preview_changed_pixels={preview_pixels} preview_changed_channels={preview_channels} preview_max={preview_max} preview_mean_rgb={:.6} export_changed_pixels={export_pixels} export_max={export_max} export_mean_rgb={:.6}",
            preview_sum as f64 / (preview.width * preview.height * 3) as f64,
            export_sum as f64 / (export.width * export.height * 3) as f64,
        );
        assert_ne!(preview.rgba, plain.rgba, "PF90007 must change preview pixels");
        assert_ne!(export.rgba, plain16.rgba, "PF90007 must change exported pixels");
        assert!(export.rgba.chunks_exact(4).all(|pixel| pixel[3] == 65535));
    }

    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    #[test]
    fn z6iii_linear_source_uses_c1_color_stage_without_shadow_crush() {
        if crate::icc::profile("NikonZ6III-ProStandard").is_none()
            || !crate::filmcurve::curve_dir().join("NikonZ6III-Auto.fcrv").is_file()
        {
            return;
        }
        let samples = [
            [640u16, 900, 1200],
            [5129, 5129, 5129],
            [33626, 33626, 33626],
            [65535, 65535, 65535],
        ];
        let mut rgb = Vec::with_capacity(16 * 3);
        for sample in samples {
            for _ in 0..4 {
                rgb.extend_from_slice(&sample);
            }
        }
        let src = Source::from_rgb16(4, 4, &rgb).with_c1_z6iii_linear(true);
        for curve in ["NikonZ6III-Auto", "NikonZ6III-Linear Response"] {
            let mut adj = AdjustmentSet::default();
            adj.base.icc_profile = Some("NikonZ6III-ProStandard".into());
            adj.base.film_curve = Some(curve.into());
            adj.base.curve = ToneCurvePreset::Auto;
            adj.sharpening.amount = 0.0;
            adj.noise_reduction = crate::NoiseReduction {
                luminance: 0.0,
                details: 0.0,
                color: 0.0,
                single_pixel: 0.0,
            };

            let rendered = render(&src, &adj, NormRect::FULL, 4, false, None);
            let bypassed = render(&src, &adj, NormRect::FULL, 4, true, None);
            assert_eq!(rendered.rgba, bypassed.rgba, "{curve}: unedited render must match bypass");

            let stage = crate::z6iii::Z6iiiColorStage::from_names(
                Some("NikonZ6III-ProStandard"),
                Some(curve),
            )
            .expect("Z6iiiColorStage should build");
            for (row_idx, sample) in samples.into_iter().enumerate() {
                let lin = sample.map(|v| v as f32 / 65535.0);
                let expected = stage
                    .develop_srgb(lin)
                    .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                let off = row_idx * 16;
                let got = [rendered.rgba[off], rendered.rgba[off + 1], rendered.rgba[off + 2]];
                for ch in 0..3 {
                    assert!(
                        got[ch].abs_diff(expected[ch]) <= 1,
                        "{curve} row {row_idx} ch {ch}: got {got:?} expected {expected:?}"
                    );
                }
            }
            // Row 0 is dark shadow (~1% linear): must not collapse to 0.
            assert!(rendered.rgba[0] > 0 && rendered.rgba[1] > 0 && rendered.rgba[2] > 0);
            // Row 3 is full white: must reach >= 250 without artificial 0.95 shoulder cap.
            assert!(rendered.rgba[48] >= 250 && rendered.rgba[49] >= 250 && rendered.rgba[50] >= 250);
        }

        // Even if a stale sidecar stored an output space like "Adobe RGB (1998)" in
        // base.icc_profile, a Z6III linear sensor source must stay on the C1 pipeline.
        let mut stale = AdjustmentSet::default();
        stale.base.icc_profile = Some("Adobe RGB (1998)".into());
        stale.base.film_curve = Some("NikonZ6III-Auto".into());
        stale.sharpening.amount = 0.0;
        stale.noise_reduction = crate::NoiseReduction {
            luminance: 0.0,
            details: 0.0,
            color: 0.0,
            single_pixel: 0.0,
        };
        let mut clean = stale.clone();
        clean.base.icc_profile = Some("NikonZ6III-ProStandard".into());
        assert_eq!(
            render(&src, &stale, NormRect::FULL, 4, false, None).rgba,
            render(&src, &clean, NormRect::FULL, 4, false, None).rgba,
            "Z6III linear source must not fall back when icc_profile is an output space"
        );
    }
}
