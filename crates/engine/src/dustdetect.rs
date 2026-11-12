//! AI dust detection: the `dust.onnx` model that Capture One's Spot Removal
//! uses for automatic dust/spot picking.
//!
//! Input is one `[1,3,416,416]` float32 tensor of RGB values in 0..255 (the
//! model has its own normalisation; feeding 0..1 collapses to noise). Output is
//! `[1,3549,6]`: three YOLO anchor grids (52/26/13 cells at strides 8/16/32),
//! each row `cx, cy, w, h, objectness, class`. Centres are cell offsets in
//! `(cell + raw) * stride` pixels, sizes are `raw * stride` pixels.
//!
//! Nothing here panics: a missing model or runtime, a buffer smaller than
//! `w * h * 4`, or a bad tensor shape all come back as `Err`.

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Result, bail};

use crate::aimask::{Runtime, Session};
use crate::{Spot, SpotKind};

/// Longest side the model sees; inputs are stretched to this square.
pub const INPUT_SIDE: usize = 416;
/// Anchors per detection grid (52² + 26² + 13²).
pub const ANCHORS: usize = 52 * 52 + 26 * 26 + 13 * 13;
/// Most spots one call returns.
pub const MAX_SPOTS: usize = 200;
/// Smallest spot radius in source pixels; dust is rarely under 1.5 px.
const MIN_RADIUS_PX: f32 = 1.5;
/// Two boxes overlap when their intersection over union exceeds this.
const NMS_IOU: f32 = 0.45;
const GRIDS: [(usize, usize, f32); 3] = [(52, 52, 8.0), (26, 26, 16.0), (13, 13, 32.0)];

/// Detects dust spots in a developed RGBA frame.
///
/// `rgba` is tightly packed `w * h * 4`. Returned spots are normalised like
/// every other engine spot: `x`/`y` in `0..=1` of the frame, `radius` as a
/// fraction of its long side. Results are sorted by confidence, best first.
pub fn detect_dust(rgba: &[u8], w: u32, h: u32, min_confidence: f32) -> Result<Vec<Spot>> {
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let need = w as usize * h as usize * 4;
    if rgba.len() < need {
        bail!("dust detection: buffer has {} bytes, need {need} for {w}x{h}", rgba.len());
    }
    let session = &session()?.session;
    let runtime = Runtime::get()
        .ok_or_else(|| anyhow::anyhow!("ONNX Runtime is not available for dust detection"))?;
    let mut input = input_tensor(rgba, w as usize, h as usize);
    let planes = runtime.run(session, &mut input, &[1, 3, INPUT_SIDE as i64, INPUT_SIDE as i64])?;
    let rows = planes
        .first()
        .ok_or_else(|| anyhow::anyhow!("dust model returned no output"))?;
    decode(rows, min_confidence, w, h)
}

struct Loaded {
    session: Session,
}

fn session() -> Result<&'static Loaded> {
    static SESSION: OnceLock<Result<Loaded, String>> = OnceLock::new();
    match SESSION.get_or_init(|| {
        let path = model_path().map_err(|e| format!("{e:#}"))?;
        let runtime = Runtime::get().ok_or_else(|| "ONNX Runtime is not available".to_string())?;
        let session = runtime.open_session(&path).map_err(|e| format!("{e:#}"))?;
        let dims = &session.input.dims;
        if dims != &[1, 3, INPUT_SIDE as i64, INPUT_SIDE as i64] || session.input.element_type != 1 {
            return Err(format!("dust.onnx: unexpected input {dims:?}"));
        }
        if session.outputs.len() != 1 || session.outputs[0].name.is_empty() {
            return Err("dust.onnx: expected one output".into());
        }
        Ok(Loaded { session })
    }) {
        Ok(loaded) => Ok(loaded),
        Err(err) => bail!("dust detection unavailable: {err}"),
    }
}

/// `LUMEN_DUST_MODEL` wins, then `LUMEN_AIMODELS/dust.onnx`, then the Capture
/// One install the other AI models come from.
fn model_path() -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = std::env::var_os("LUMEN_DUST_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(dir) = std::env::var_os("LUMEN_AIMODELS") {
        candidates.push(PathBuf::from(&dir).join("dust.onnx"));
        candidates.push(PathBuf::from(&dir).join("AIModels/Common/dust.onnx"));
    }
    for root in crate::aimask::c1_roots() {
        candidates.push(root.join("AIModels/Common/dust.onnx"));
        candidates.push(root.join("dust.onnx"));
    }
    if let Some(path) = candidates.iter().find(|p| p.is_file()) {
        return Ok(path.clone());
    }
    bail!(
        "dust.onnx not found; set LUMEN_DUST_MODEL to the model file (looked in {})",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// Bilinear resize to 416×416, RGB, 0..255 — the range the model was trained on.
fn input_tensor(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let side = INPUT_SIDE;
    let mut tensor = vec![0.0f32; 3 * side * side];
    for y in 0..side {
        let sy = (y as f32 + 0.5) * h as f32 / side as f32 - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..side {
            let sx = (x as f32 + 0.5) * w as f32 / side as f32 - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            let at = |xx: usize, yy: usize, c: usize| rgba[(yy * w + xx) * 4 + c] as f32;
            let i = y * side + x;
            for c in 0..3 {
                let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
                let bottom = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
                tensor[c * side * side + i] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
    tensor
}

/// Decodes the raw `[3549,6]` head into normalised spots.
fn decode(rows: &[f32], min_confidence: f32, w: u32, h: u32) -> Result<Vec<Spot>> {
    if rows.len() < ANCHORS * 6 {
        bail!("dust model returned {} values, expected {}", rows.len(), ANCHORS * 6);
    }
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let long_side = w.max(h) as f32;
    let (sx, sy) = (w as f32 / INPUT_SIDE as f32, h as f32 / INPUT_SIDE as f32);
    let min_confidence = if min_confidence.is_finite() { min_confidence } else { 0.0 };

    let mut found: Vec<(f32, [f32; 4])> = Vec::new();
    let mut index = 0usize;
    for (gw, gh, stride) in GRIDS {
        for gy in 0..gh {
            for gx in 0..gw {
                let row = &rows[index * 6..index * 6 + 6];
                index += 1;
                let confidence = row[4] * row[5];
                if !confidence.is_finite() || confidence < min_confidence {
                    continue;
                }
                // A head can predict outside its cell; keep the centre on the canvas.
                let cx = ((gx as f32 + row[0]) * stride).clamp(0.0, INPUT_SIDE as f32);
                let cy = ((gy as f32 + row[1]) * stride).clamp(0.0, INPUT_SIDE as f32);
                let (bw, bh) = (row[2] * stride, row[3] * stride);
                if !cx.is_finite() || !cy.is_finite() || !(bw > 0.0) || !(bh > 0.0) {
                    continue;
                }
                let half_w = bw.min(INPUT_SIDE as f32 * 2.0) * 0.5;
                let half_h = bh.min(INPUT_SIDE as f32 * 2.0) * 0.5;
                let box_ = [
                    (cx - half_w).clamp(0.0, INPUT_SIDE as f32),
                    (cy - half_h).clamp(0.0, INPUT_SIDE as f32),
                    (cx + half_w).clamp(0.0, INPUT_SIDE as f32),
                    (cy + half_h).clamp(0.0, INPUT_SIDE as f32),
                ];
                if box_[2] <= box_[0] || box_[3] <= box_[1] {
                    continue;
                }
                found.push((confidence, box_));
            }
        }
    }
    found.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<(f32, [f32; 4])> = Vec::new();
    for (confidence, box_) in found {
        if kept.len() >= MAX_SPOTS {
            break;
        }
        if kept.iter().any(|(_, k)| iou(k, &box_) > NMS_IOU) {
            continue;
        }
        kept.push((confidence, box_));
    }

    let mut spots = Vec::with_capacity(kept.len());
    for (i, (_, b)) in kept.into_iter().enumerate() {
        let cx = (b[0] + b[2]) * 0.5;
        let cy = (b[1] + b[3]) * 0.5;
        let r_px = (0.5 * (b[2] - b[0]) * sx).max(0.5 * (b[3] - b[1]) * sy);
        spots.push(Spot {
            id: i as u64 + 1,
            kind: SpotKind::Dust,
            x: (cx / INPUT_SIDE as f32).clamp(0.0, 1.0),
            y: (cy / INPUT_SIDE as f32).clamp(0.0, 1.0),
            radius: r_px.max(MIN_RADIUS_PX) / long_side,
        });
    }
    Ok(spots)
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_with(detections: &[(usize, [f32; 6])]) -> Vec<f32> {
        let mut rows = vec![0.0f32; ANCHORS * 6];
        for (index, row) in detections {
            rows[index * 6..index * 6 + 6].copy_from_slice(row);
        }
        rows
    }

    #[test]
    fn empty_result_when_nothing_passes_the_threshold() {
        let rows = rows_with(&[(0, [0.5, 0.5, 0.2, 0.2, 0.1, 0.1])]);
        assert!(decode(&rows, 0.3, 416, 416).unwrap().is_empty());
        assert!(decode(&rows, 0.0, 416, 416).unwrap().len() == 1);
    }

    #[test]
    fn cell_offsets_become_normalized_coordinates() {
        // Grid 0, cell (10, 5), centre half a cell in, box 0.1 cell = 0.8 px.
        let rows = rows_with(&[(5 * 52 + 10, [0.5, 0.5, 0.1, 0.1, 0.9, 0.9])]);
        let spots = decode(&rows, 0.5, 416, 416).unwrap();
        assert_eq!(spots.len(), 1);
        let s = &spots[0];
        assert!((s.x - (10.5 * 8.0 / 416.0)).abs() < 1e-5, "x = {}", s.x);
        assert!((s.y - (5.5 * 8.0 / 416.0)).abs() < 1e-5, "y = {}", s.y);
        assert_eq!(s.kind, SpotKind::Dust);
    }

    #[test]
    fn radius_never_drops_below_the_floor() {
        let rows = rows_with(&[(0, [0.5, 0.5, 1e-6, 1e-6, 0.9, 0.9])]);
        let spots = decode(&rows, 0.5, 4000, 3000).unwrap();
        let radius_px = spots[0].radius * 4000.0;
        assert!(radius_px >= MIN_RADIUS_PX - 1e-3, "radius {radius_px} px");
    }

    #[test]
    fn out_of_range_centres_are_clamped() {
        let rows = rows_with(&[
            (0, [-3.0, -3.0, 0.5, 0.5, 0.9, 0.9]),
            (ANCHORS - 1, [9.0, 9.0, 0.5, 0.5, 0.9, 0.9]),
        ]);
        let spots = decode(&rows, 0.5, 416, 416).unwrap();
        assert_eq!(spots.len(), 2);
        for s in &spots {
            assert!((0.0..=1.0).contains(&s.x), "x = {}", s.x);
            assert!((0.0..=1.0).contains(&s.y), "y = {}", s.y);
            assert!(s.radius.is_finite() && s.radius > 0.0);
        }
    }

    #[test]
    fn non_positive_boxes_are_rejected() {
        let rows = rows_with(&[(0, [0.5, 0.5, -0.7, 0.2, 0.9, 0.9])]);
        assert!(decode(&rows, 0.0, 416, 416).unwrap().is_empty());
    }

    #[test]
    fn overlapping_grids_are_deduplicated() {
        // Same spot seen by the 8 px and 16 px heads: cell (12,12) and (6,6).
        let rows = rows_with(&[
            (12 * 52 + 12, [0.4, 0.4, 1.0, 1.0, 0.8, 0.8]),
            (52 * 52 + 6 * 26 + 6, [0.2, 0.2, 0.5, 0.5, 0.7, 0.7]),
        ]);
        let spots = decode(&rows, 0.3, 416, 416).unwrap();
        assert_eq!(spots.len(), 1);
    }

    #[test]
    fn results_are_sorted_and_capped() {
        let mut detections = Vec::new();
        for i in 0..300usize {
            let row = 40 + (i / 52) as usize;
            let col = i % 52;
            detections.push((row * 52 + col, [0.5, 0.5, 0.01, 0.01, (i as f32 / 300.0).min(0.99), 1.0]));
        }
        let rows = rows_with(&detections);
        let spots = decode(&rows, 0.0, 416, 416).unwrap();
        assert_eq!(spots.len(), MAX_SPOTS);
        let ids: Vec<u64> = spots.iter().map(|s| s.id).collect();
        assert_eq!(ids, (1..=MAX_SPOTS as u64).collect::<Vec<_>>());
        // Highest anchor (bottom/right) has the highest confidence, so it comes first.
        assert!(spots[0].x > 0.5);
    }

    #[test]
    fn wrong_output_size_is_an_error() {
        assert!(decode(&[0.0; 12], 0.0, 416, 416).is_err());
    }

    #[test]
    fn short_buffer_is_an_error() {
        assert!(detect_dust(&[0u8; 16], 8, 8, 0.3).is_err());
    }

    /// Real model run. Needs `LUMEN_DUST_MODEL` (and ONNX Runtime) on the host.
    #[test]
    #[ignore]
    fn detect_synthetic_dots_with_the_real_model() {
        let (w, h) = (416u32, 416u32);
        let mut buf = vec![153u8; (w * h * 4) as usize];
        for (cx, cy) in [(100i64, 100i64), (200, 100), (300, 300)] {
            for y in cy - 9..=cy + 9 {
                for x in cx - 9..=cx + 9 {
                    if (x - cx).pow(2) + (y - cy).pow(2) > 81 {
                        continue;
                    }
                    let p = ((y as u32 * w + x as u32) * 4) as usize;
                    buf[p..p + 3].fill(0);
                }
            }
        }
        let start = std::time::Instant::now();
        let spots = detect_dust(&buf, w, h, 0.3).expect("detection");
        let elapsed = start.elapsed();
        let warm = std::time::Instant::now();
        let again = detect_dust(&buf, w, h, 0.3).expect("detection");
        println!(
            "detected {} spots in {elapsed:?} (first call, includes model load); \
             {} in {:?} warm",
            spots.len(),
            again.len(),
            warm.elapsed()
        );
        for s in spots.iter().take(12) {
            println!(
                "  x={:.4} y={:.4} r={:.4} (px {:.1})",
                s.x,
                s.y,
                s.radius,
                s.radius * w.max(h) as f32
            );
        }
        assert!(!spots.is_empty(), "no dust found on a synthetic test pattern");
        assert!(spots.len() <= MAX_SPOTS);
        assert!(spots.iter().any(|s| (s.x - 100.0 / 416.0).abs() < 0.05 && (s.y - 100.0 / 416.0).abs() < 0.05));
    }

    /// Run where neither `LUMEN_DUST_MODEL` nor a Capture One install exists.
    #[test]
    #[ignore]
    fn a_missing_model_is_an_error_not_a_panic() {
        let buf = vec![0u8; 16 * 16 * 4];
        assert!(detect_dust(&buf, 16, 16, 0.3).is_err());
    }
}
