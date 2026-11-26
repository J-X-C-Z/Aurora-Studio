//! Layer-mask composition, moved out of the CPU renderer.
//!
//! The functions here take the compiled [`crate::render_plan_types::Plan`] and
//! plain pixel buffers; they do not own decoder state or GPU handles. Keeping
//! them outside `processing/` is deliberate: rule 8 forbids the standalone word
//! `Plan` in a pure pixel stage, and these functions must name the compiled
//! plan type. The renderer in `mock::process` calls into this module.

use rayon::prelude::*;

use std::sync::{Arc, Mutex, OnceLock};

use crate::processing::color::{black_white, color_balance, color_range_weight};
use crate::processing::layer_weight::luma_range_weight;
use crate::processing::color::{color_editor, luma, luma601, lut_get};
use crate::processing::frame_map::FrameMap;
use crate::processing::spots::apply_cover_patches;
use crate::processing::statistics::{box_blur3, luma_plane, mean3x3};
use crate::processing::tone::{apply_dehaze, apply_hdr, tone_user};
use crate::render_plan_types::{layer_has_spatial, LayerOutputWeightKey, LayerPlan, MaskGrid, Plan};
use crate::{ColorBalance, ColorEditor, CoverKind, CoverPatch, LayerKind, Mask, NormRect};

pub(crate) fn local(g: [f32; 3], lp: &LayerPlan, m: f32) -> [f32; 3] {
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
pub(crate) fn layer_weight(lp: &LayerPlan, p: [f32; 2], g: [f32; 3]) -> f32 {
    layer_mask_weight(&lp.layer.mask, lp.grid.as_deref(), lp.opacity, p, g)
}

/// One layer's contribution to a pixel: the colour adjustments, blended in by
/// the mask weight `m`, plus the mask overlay tint.
pub(crate) fn layer_pixel(lp: &LayerPlan, g: [f32; 3], m: f32) -> [f32; 3] {
    let mut c = local(g, lp, m);
    if lp.overlay {
        c = [c[0] + (0.95 - c[0]) * m * 0.55, c[1] * (1.0 - m * 0.55), c[2] * (1.0 - m * 0.55)];
    }
    c
}

pub(crate) fn apply_layers(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    plan: &Plan,
    map: &FrameMap,
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
            apply_cover_patches(
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
                apply_cover_patches(
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

pub(crate) fn static_mask_can_apply(lp: &LayerPlan<'_>) -> bool {
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
pub(crate) fn apply_layer_color_row(
    line: &mut [u8],
    static_weights: Option<&[f32]>,
    recorded_weights: Option<&mut [f32]>,
    n: usize,
    plan: &Plan,
    map: &FrameMap,
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

pub(crate) fn cached_layer_weights(
    plan: &Plan<'_>,
    map: &FrameMap,
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

pub(crate) fn layer_mask_weight(
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
