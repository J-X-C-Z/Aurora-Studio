//! Path helpers for widgets that paint plots.

use gpui_kit::{Bounds, Hsla, PathBuilder, Pixels, Window, point, px};

/// Fills the area under `values` (each 0..1, bottom to top) across `bounds`.
pub fn fill_area(window: &mut Window, bounds: Bounds<Pixels>, values: &[f32], color: Hsla) {
    if values.len() < 2 {
        return;
    }
    let (x0, y0) = (bounds.origin.x, bounds.origin.y);
    let (w, h) = (f32::from(bounds.size.width), f32::from(bounds.size.height));
    let n = values.len() - 1;
    let mut b = PathBuilder::fill();
    b.move_to(point(x0, y0 + px(h)));
    for (i, v) in values.iter().enumerate() {
        b.line_to(point(x0 + px(w * i as f32 / n as f32), y0 + px(h * (1.0 - v.clamp(0.0, 1.0)))));
    }
    b.line_to(point(x0 + px(w), y0 + px(h)));
    b.close();
    if let Ok(path) = b.build() {
        window.paint_path(path, color);
    }
}

/// Strokes a polyline of normalized points (x right, y up) across `bounds`.
pub fn stroke_line(window: &mut Window, bounds: Bounds<Pixels>, points: &[(f32, f32)], width: f32, color: Hsla) {
    if points.len() < 2 {
        return;
    }
    let (x0, y0) = (bounds.origin.x, bounds.origin.y);
    let (w, h) = (f32::from(bounds.size.width), f32::from(bounds.size.height));
    let mut b = PathBuilder::stroke(px(width));
    for (i, (x, y)) in points.iter().enumerate() {
        let pt = point(x0 + px(w * x), y0 + px(h * (1.0 - y)));
        if i == 0 { b.move_to(pt) } else { b.line_to(pt) }
    }
    if let Ok(path) = b.build() {
        window.paint_path(path, color);
    }
}

/// Strokes one segment in absolute pixels.
pub fn segment(window: &mut Window, a: gpui_kit::Point<Pixels>, b: gpui_kit::Point<Pixels>, width: f32, color: Hsla) {
    let mut p = PathBuilder::stroke(px(width));
    p.move_to(a);
    p.line_to(b);
    if let Ok(path) = p.build() {
        window.paint_path(path, color);
    }
}

/// A filled polygon in absolute pixels.
pub fn polygon(window: &mut Window, points: &[gpui_kit::Point<Pixels>], color: Hsla) {
    if points.len() < 3 {
        return;
    }
    let mut b = PathBuilder::fill();
    b.add_polygon(points, true);
    if let Ok(path) = b.build() {
        window.paint_path(path, color);
    }
}

/// Normalizes histogram bins for display: square-root scale, robust to spikes.
pub fn normalize_bins(bins: &[u32], points: usize) -> Vec<f32> {
    if bins.is_empty() {
        return vec![0.0; points];
    }
    let per = (bins.len() as f32 / points as f32).max(1.0);
    let mut out: Vec<f32> = (0..points)
        .map(|i| {
            let a = (i as f32 * per) as usize;
            let b = (((i + 1) as f32 * per) as usize).min(bins.len()).max(a + 1);
            bins[a..b].iter().map(|v| *v as f32).sum::<f32>() / (b - a) as f32
        })
        .map(f32::sqrt)
        .collect();
    let mut sorted = out.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let peak = sorted[(sorted.len() as f32 * 0.985) as usize].max(1e-3);
    for v in &mut out {
        *v = (*v / peak).min(1.0) * 0.92;
    }
    out
}
