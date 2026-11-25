//! Per-render planning types extracted from `mock::process`.
//!
//! These values hold adjustment data and precomputed tables for one frame
//! (`MaskGrid`, `LayerPlan`, `Plan`, ...). They used to be defined inside the
//! mock renderer; they live here so the plan layer is no longer part of the
//! mock module. The pixel stages themselves stay in the sibling `processing`
//! modules and the renderer entry points stay in `mock`.
//!
//! Fields are `pub(crate)` because `mock::process` builds and reads them in
//! place; the layout and semantics are byte-identical to the previous
//! definitions.

use std::sync::Arc;

use crate::{
    AdjustmentSet, AiMask, AiMaskKind, ColorBalance, ColorEditor, Geometry, Layer, Levels,
    LocalAdjustments, Mask, MaskOp, MaskShape, Moire, NoiseReduction, NormRect, Sharpening,
};
use crate::processing::color::{GammaLut, lut_of, render_wb_gains, sstep};
use crate::processing::frame_map::FrameMap;

/// A coarse raster of one layer's shape mask in oriented-image space.
#[derive(Clone)]
pub(crate) struct MaskGrid {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) data: Vec<f32>,
}

impl MaskGrid {
    /// `cell_px` is the size of one grid cell in source pixels, so the mask's
    /// pixel radii (`feather`, `refine`) convert to grid pixels.
    /// `guide` is an optional luminance grid (same layout) used by `refine`.
    ///
    /// `built` holds the masks of the layers already planned, in draw order;
    /// a [`MaskShape::Copy`] reads its source from there.
    pub(crate) fn build(
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

    pub(crate) fn sample(&self, p: [f32; 2]) -> f32 {
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
pub(crate) fn grid_dims(aspect: f32) -> (usize, usize) {
    let w = 256usize;
    let h = ((256.0 / aspect).round() as usize).clamp(32, 512);
    (w, h)
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MaskGridCacheKey {
    pub(crate) source: u64,
    pub(crate) geometry: Geometry,
    pub(crate) layers: Vec<(u64, Mask)>,
    pub(crate) has_guide: bool,
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

pub(crate) fn shape_value(
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

pub(crate) struct LayerPlan<'a> {
    pub(crate) layer: &'a Layer,
    pub(crate) grid: Option<Arc<MaskGrid>>,
    pub(crate) overlay: bool,
    pub(crate) wb_gains: Option<[f32; 3]>,
    /// The layer's curves, precomputed as 256-entry LUTs.
    pub(crate) curve_rgb: Option<[f32; 256]>,
    pub(crate) curve_r: Option<[f32; 256]>,
    pub(crate) curve_g: Option<[f32; 256]>,
    pub(crate) curve_b: Option<[f32; 256]>,
    pub(crate) curve_luma: Option<[f32; 256]>,
    pub(crate) levels: Option<Levels>,
    /// Constants a per-pixel `local()` would otherwise recompute every time.
    pub(crate) exposure_gain: f32,
    pub(crate) saturation: f32,
    pub(crate) simple_exposure: bool,
    pub(crate) opacity: f32,
    /// Spatial operators; they need neighbours, so they run on the whole
    /// frame after the layers have been mixed.
    pub(crate) moire: Moire,
    pub(crate) sharpening: Option<Sharpening>,
    pub(crate) noise_reduction: Option<NoiseReduction>,
}

pub(crate) fn layer_has_spatial(a: &LocalAdjustments) -> bool {
    a.moire.amount > 0.0
        || a.sharpening.as_ref().is_some_and(|sharp| sharp.amount > 0.0)
        || a.noise_reduction.as_ref().is_some_and(|nr| {
            nr.luminance > 0.0 || nr.color > 0.0 || nr.single_pixel > 0.0
        })
}

/// Builds the per-layer plan in draw order. Each mask is rasterized before the
/// next layer starts, so a copy mask can read the layers below it.
pub(crate) fn layer_plan<'a>(
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

/// Everything precomputed once per render.
/// GPU-resident pixel gains for one frame. `vignette` is indexed by output
/// pixel; `lcc` is a source-resolution gain plane sampled at the source point.
pub(crate) struct PixelGains {
    pub(crate) vignette: Option<Arc<Vec<f32>>>,
    pub(crate) lcc: Option<(Arc<Vec<f32>>, usize, usize)>,
}

pub(crate) struct Plan<'a> {
    pub(crate) uniform_light: Option<(crate::dust_lcc::UniformLight<'a>, &'a FrameMap, [f32; 2])>,
    pub(crate) pixel_gains: Option<PixelGains>,
    pub(crate) adj: &'a AdjustmentSet,
    pub(crate) source: u64,
    /// Frame width / height, for the circular vignette metric.
    pub(crate) aspect: f32,
    /// Source canvas in pixels, for the vignette's pixel-space centres.
    pub(crate) size: [f32; 2],
    #[cfg(all(windows, feature = "z6iii-raw-export"))]
    pub(crate) c1_color: Option<crate::z6iii::Z6iiiColorStage>,
    /// Output-domain BandingSuppression, applied to the developed codes.
    pub(crate) banding: Option<BandingPlan>,
    pub(crate) gamma: GammaLut,
    /// Exact sRGB instead of the preview LUT, so 16-bit export is not capped
    /// at the LUT's 8192 steps.
    pub(crate) fine: bool,
    pub(crate) wb: [f32; 3],
    pub(crate) exposure: f32,
    pub(crate) curve_rgb: Option<[f32; 256]>,
    pub(crate) curve_r: Option<[f32; 256]>,
    pub(crate) curve_g: Option<[f32; 256]>,
    pub(crate) curve_b: Option<[f32; 256]>,
    pub(crate) curve_luma: Option<[f32; 256]>,
    pub(crate) levels: bool,
    pub(crate) color_editor: bool,
    pub(crate) color_balance: bool,
    pub(crate) layers: Vec<LayerPlan<'a>>,
}

/// C1 queue stage 77 on developed output codes: the same kernel the verified
/// 16-bit export runs on Adobe RGB codes, evaluated per source pixel so the
/// interactive sampler can apply it out of raster order.
pub(crate) struct BandingPlan {
    pub(crate) params: crate::c1_banding::Params,
    pub(crate) tables: crate::c1_banding::Tables,
    /// Frame size in source pixels, used to turn a normalized sample position
    /// back into the device coordinate the pattern is keyed on.
    pub(crate) frame: [f32; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerOutputWeightKey {
    pub(crate) source: u64,
    pub(crate) geometry: Geometry,
    pub(crate) layers: Vec<(u64, Mask)>,
    pub(crate) region: NormRect,
    pub(crate) width: usize,
    pub(crate) height: usize,
}
