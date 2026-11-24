//! AI dust detection: `dust.onnx`, the model behind Capture One's
//! `IC_DetectDust` automatic dust picking.
//!
//! Contract recovered from `ImgCoreDll.dll` (see `analysis/ai_models.md`):
//! input `[1,3,416,416]` float32 RGB in 0..255; output `[1,3549,6]`.  The
//! three grids are strides 8/16/32, 52²/26²/13² cells.  A row is
//! `cx_in_cell, cy_in_cell, log(w/stride), log(h/stride), objectness, class`.
//! The image is scaled by `f = 416/min_side + 0.1` (floor 0.5 under 80 MP,
//! 0.25 above), tiled 416 with 100 px overlap, and the last tile is flush with
//! the right/bottom edge.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::aimask::{Runtime, Session};
use crate::{Spot, SpotKind};

pub const INPUT_SIDE: usize = 416;
pub const ANCHORS: usize = 52 * 52 + 26 * 26 + 13 * 13;
pub const MAX_SPOTS: usize = 200;
/// `DAT_182a2bc70`; the old 0.3 dropped faint dust.
pub const DEFAULT_CONFIDENCE: f32 = 0.15;
const TILE_STEP: usize = INPUT_SIDE - 100;
pub(crate) const MIN_RADIUS_PX: f32 = 1.5;
pub(crate) const MAX_RADIUS_PX: f32 = 100.0;
pub(crate) const NMS_IOU: f32 = 0.45;
const STRIDES: [usize; 3] = [8, 16, 32];

/// `AURORA_DUST_ONNX=1` turns automatic detection on.
pub fn enabled() -> bool {
    std::env::var("AURORA_DUST_ONNX").ok().as_deref() == Some("1")
}

pub fn min_confidence() -> f32 {
    std::env::var("AURORA_DUST_CONF")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(DEFAULT_CONFIDENCE)
}

/// Image-to-canvas scale.  `None` means C1 itself rejects the image as too
/// small (`Dust detection rejected image because it is too small`).
fn spot_scale(w: u32, h: u32) -> Option<f32> {
    let min_side = w.min(h) as f32;
    let floor = if (w as u64) * (h as u64) < 80_000_001 { 0.5 } else { 0.25 };
    let f = (INPUT_SIDE as f32 / min_side + 0.1).max(floor);
    (f <= 1.0).then_some(f)
}

pub(crate) fn spot_tiles(n: usize) -> Vec<usize> {
    if n <= INPUT_SIDE {
        return vec![0];
    }
    let mut out: Vec<usize> = (0..=n - INPUT_SIDE).step_by(TILE_STEP).collect();
    if out.last().copied() != Some(n - INPUT_SIDE) {
        out.push(n - INPUT_SIDE);
    }
    out
}

pub(crate) struct RawSpot {
    pub score: f32,
    pub cx: f32,
    pub cy: f32,
    pub side: f32,
}

#[derive(Clone, PartialEq)]
struct CacheKey {
    frame: u64,
    threshold_bits: u32,
    model: Vec<u8>,
}

struct CacheEntry {
    key: CacheKey,
    spots: Arc<Vec<Spot>>,
}

const DUST_CACHE_ENTRIES: usize = 8;

fn cache() -> &'static Mutex<Vec<CacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

/// Model identity: the file's metadata and first/middle/last KiB. A model
/// replacement changes this even when the path is reused, so stale inference
/// results cannot be served after an ONNX update.
fn model_fingerprint() -> Vec<u8> {
    let mut out = Vec::new();
    let Ok(path) = model_path() else { return out };
    let Ok(meta) = std::fs::metadata(&path) else { return out };
    out.extend_from_slice(&meta.len().to_le_bytes());
    out.extend_from_slice(&meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|t| t.as_secs().to_le_bytes()).unwrap_or([0; 8]));
    use std::io::{Read, Seek, SeekFrom};
    if let Ok(mut file) = std::fs::File::open(&path) {
        for at in [SeekFrom::Start(0), SeekFrom::End(-1024), SeekFrom::End(-2048)] {
            let mut bytes = [0u8; 1024];
            if file.seek(at).is_ok() && file.read(&mut bytes).unwrap_or(0) > 0 {
                out.extend_from_slice(&bytes);
            }
        }
    }
    out
}

/// Cached automatic dust detection. `frame` must identify the source snapshot,
/// geometry and output frame; colour sliders are deliberately absent. The
/// threshold and the model bytes are part of the key so changing either
/// re-detects, while dragging an unrelated slider hits the cache.
pub fn detect_dust_cached(
    frame: u64,
    rgba: &[u8],
    w: u32,
    h: u32,
    min_confidence: f32,
) -> Result<Arc<Vec<Spot>>> {
    let key = CacheKey {
        frame,
        threshold_bits: min_confidence.to_bits(),
        model: model_fingerprint(),
    };
    {
        let entries = cache().lock().unwrap_or_else(|error| error.into_inner());
        if let Some(found) = entries.iter().find(|entry| entry.key == key) {
            return Ok(Arc::clone(&found.spots));
        }
    }
    let spots = Arc::new(detect_dust(rgba, w, h, min_confidence)?);
    let mut entries = cache().lock().unwrap_or_else(|error| error.into_inner());
    entries.retain(|entry| entry.key != key);
    if entries.len() >= DUST_CACHE_ENTRIES {
        entries.remove(0);
    }
    entries.push(CacheEntry { key, spots: Arc::clone(&spots) });
    Ok(spots)
}

/// Detects dust spots in a developed RGBA frame.  Returned coordinates are
/// normalised like every other `Spot`; `radius` is a fraction of the long side.
pub fn detect_dust(rgba: &[u8], w: u32, h: u32, min_confidence: f32) -> Result<Vec<Spot>> {
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let need = w as usize * h as usize * 4;
    if rgba.len() < need {
        bail!("dust detection: buffer has {} bytes, need {need} for {w}x{h}", rgba.len());
    }
    let Some(scale) = spot_scale(w, h) else {
        bail!("Dust detection rejected image because it is too small");
    };
    let canvas_w = ((w as f32 * scale).ceil() as usize).max(INPUT_SIDE);
    let canvas_h = ((h as f32 * scale).ceil() as usize).max(INPUT_SIDE);
    let canvas = resize_canvas(rgba, w as usize, h as usize, canvas_w, canvas_h);
    let session = &session()?.session;
    let runtime = Runtime::get()
        .ok_or_else(|| anyhow::anyhow!("ONNX Runtime is not available for dust detection"))?;
    let tiles_x = spot_tiles(canvas_w);
    let tiles_y = spot_tiles(canvas_h);
    let tensor_len = 3 * INPUT_SIDE * INPUT_SIDE;
    let found: Result<Vec<Vec<RawSpot>>> = tiles_y
        .par_iter()
        .map(|&ty| {
            let mut tile_input = vec![0.0f32; tensor_len];
            let mut row = Vec::new();
            for &tx in &tiles_x {
                fill_tile_tensor(&mut tile_input, &canvas, canvas_w, canvas_h, tx, ty);
                let planes = runtime.run(
                    session,
                    &mut tile_input,
                    &[1, 3, INPUT_SIDE as i64, INPUT_SIDE as i64],
                )?;
                let rows = planes
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("dust model returned no output"))?;
                row.extend(decode_tile(rows, tx, ty, scale, min_confidence)?);
            }
            Ok(row)
        })
        .collect();
    let mut found: Vec<RawSpot> = found?.into_iter().flatten().collect();
    found.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<RawSpot> = Vec::new();
    for spot in found {
        if kept.len() >= MAX_SPOTS {
            break;
        }
        let radius = spot.side * 0.5;
        if !(MIN_RADIUS_PX..=MAX_RADIUS_PX).contains(&radius) {
            continue;
        }
        if kept.iter().any(|k| iou(&spot, k) > NMS_IOU) {
            continue;
        }
        kept.push(spot);
    }
    let long = w.max(h) as f32;
    Ok(kept
        .into_iter()
        .enumerate()
        .map(|(i, s)| Spot {
            id: i as u64 + 1,
            kind: SpotKind::Dust,
            x: (s.cx / w as f32).clamp(0.0, 1.0),
            y: (s.cy / h as f32).clamp(0.0, 1.0),
            radius: (s.side * 0.5 / long).clamp(0.0, 1.0),
        })
        .collect())
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

/// `AURORA_DUST_MODEL` wins, then `AURORA_AIMODELS`, then the Capture One
/// install.  The real model lives in `AIModels/Common/DustRemoval/dust.onnx`.
fn model_path() -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = std::env::var_os("AURORA_DUST_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(dir) = std::env::var_os("AURORA_AIMODELS") {
        candidates.push(PathBuf::from(&dir).join("dust.onnx"));
        candidates.push(PathBuf::from(&dir).join("Common/DustRemoval/dust.onnx"));
        candidates.push(PathBuf::from(&dir).join("AIModels/Common/DustRemoval/dust.onnx"));
    }
    for root in crate::aimask::c1_roots() {
        candidates.push(root.join("AIModels/Common/DustRemoval/dust.onnx"));
        candidates.push(root.join("AIModels/Common/dust.onnx"));
        candidates.push(root.join("dust.onnx"));
    }
    if let Some(path) = candidates.iter().find(|p| p.is_file()) {
        return Ok(path.clone());
    }
    bail!(
        "dust.onnx not found; set AURORA_DUST_MODEL to the model file (looked in {})",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// Bilinear resize to the C1 canvas, RGB 0..255 (the model's trained range).
pub(crate) fn resize_canvas(rgba: &[u8], w: usize, h: usize, cw: usize, ch: usize) -> Vec<f32> {
    let mut canvas = vec![0.0f32; cw * ch * 3];
    canvas
        .par_chunks_mut(cw * 3)
        .enumerate()
        .for_each(|(y, line)| {
            let sy = (y as f32 + 0.5) * h as f32 / ch as f32 - 0.5;
            let y0 = sy.floor().max(0.0) as usize;
            let y1 = (y0 + 1).min(h - 1);
            let fy = (sy - y0 as f32).clamp(0.0, 1.0);
            for (x, pixel) in line.chunks_exact_mut(3).enumerate() {
                let sx = (x as f32 + 0.5) * w as f32 / cw as f32 - 0.5;
                let x0 = sx.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(w - 1);
                let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                let at = |xx: usize, yy: usize, c: usize| rgba[(yy * w + xx) * 4 + c] as f32;
                for c in 0..3 {
                    let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
                    let bottom = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
                    pixel[c] = top * (1.0 - fy) + bottom * fy;
                }
            }
        });
    canvas
}

/// Fills a reusable tile tensor in place (the caller zero-fills once and keeps
/// it for the rest of the row), avoiding a 1.9 MiB allocation per inference.
fn fill_tile_tensor(out: &mut [f32], canvas: &[f32], cw: usize, ch: usize, tx: usize, ty: usize) {
    out.fill(0.0);
    for y in 0..INPUT_SIDE {
        let sy = ty + y;
        if sy >= ch {
            continue;
        }
        for x in 0..INPUT_SIDE {
            let sx = tx + x;
            if sx >= cw {
                continue;
            }
            let si = (sy * cw + sx) * 3;
            let di = y * INPUT_SIDE + x;
            for c in 0..3 {
                out[c * INPUT_SIDE * INPUT_SIDE + di] = canvas[si + c];
            }
        }
    }
}

pub(crate) fn tile_tensor(canvas: &[f32], cw: usize, ch: usize, tx: usize, ty: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * INPUT_SIDE * INPUT_SIDE];
    fill_tile_tensor(&mut out, canvas, cw, ch, tx, ty);
    out
}

/// `FUN_181435ef0`: anchors 8/16/32, `w=exp(col2)*stride`, `conf=col4*col5`.
pub(crate) fn decode_tile(
    rows: &[f32],
    x_off: usize,
    y_off: usize,
    scale: f32,
    min_confidence: f32,
) -> Result<Vec<RawSpot>> {
    if rows.len() < ANCHORS * 6 {
        bail!("dust model returned {} values, expected {}", rows.len(), ANCHORS * 6);
    }
    let threshold = if min_confidence.is_finite() { min_confidence } else { 0.0 };
    let mut found = Vec::new();
    let mut index = 0usize;
    for stride in STRIDES {
        let grid = INPUT_SIDE / stride;
        for gy in 0..grid {
            for gx in 0..grid {
                let p = &rows[index * 6..index * 6 + 6];
                index += 1;
                let score = p[4] * p[5];
                if !score.is_finite() || score <= threshold {
                    continue;
                }
                let bw = p[2].exp() * stride as f32;
                let bh = p[3].exp() * stride as f32;
                if !bw.is_finite() || !bh.is_finite() || bw <= 0.0 || bh <= 0.0 {
                    continue;
                }
                let cx = (x_off as f32 + (gx as f32 + p[0]) * stride as f32) / scale;
                let cy = (y_off as f32 + (gy as f32 + p[1]) * stride as f32) / scale;
                let side = bw.max(bh) / scale;
                if cx.is_finite() && cy.is_finite() && side.is_finite() && side > 0.0 {
                    found.push(RawSpot { score, cx, cy, side });
                }
            }
        }
    }
    Ok(found)
}

pub(crate) fn iou(a: &RawSpot, b: &RawSpot) -> f32 {
    let ah = a.side * 0.5;
    let bh = b.side * 0.5;
    let w = (a.cx + ah).min(b.cx + bh) - (a.cx - ah).max(b.cx - bh);
    let h = (a.cy + ah).min(b.cy + bh) - (a.cy - ah).max(b.cy - bh);
    if w <= 0.0 || h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    inter / (a.side * a.side + b.side * b.side - inter).max(f32::MIN_POSITIVE)
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
        let rows = rows_with(&[(0, [0.5, 0.5, 0.0, 0.0, 0.1, 0.1])]);
        assert!(decode_tile(&rows, 0, 0, 1.0, 0.15).unwrap().is_empty());
        assert_eq!(decode_tile(&rows, 0, 0, 1.0, 0.0).unwrap().len(), 1);
    }

    #[test]
    fn exp_size_and_cell_offsets_are_decoded() {
        let rows = rows_with(&[(5 * 52 + 10, [0.5, 0.5, 0.0, 0.0, 0.9, 0.9])]);
        let spots = decode_tile(&rows, 0, 0, 1.0, 0.15).unwrap();
        assert_eq!(spots.len(), 1);
        assert!((spots[0].cx - 10.5 * 8.0).abs() < 1e-4);
        assert!((spots[0].cy - 5.5 * 8.0).abs() < 1e-4);
        assert!((spots[0].side - 8.0).abs() < 1e-4);
    }

    #[test]
    fn tile_offsets_and_scale_return_to_image_pixels() {
        let rows = rows_with(&[(0, [0.5, 0.5, 0.0, 0.0, 0.9, 0.9])]);
        let spots = decode_tile(&rows, 316, 100, 0.5, 0.15).unwrap();
        assert!((spots[0].cx - (316.0 + 4.0) / 0.5).abs() < 1e-3);
        assert!((spots[0].cy - (100.0 + 4.0) / 0.5).abs() < 1e-3);
        assert!((spots[0].side - 16.0).abs() < 1e-3);
    }

    #[test]
    fn non_positive_boxes_are_rejected() {
        let rows = rows_with(&[(0, [0.5, 0.5, f32::NEG_INFINITY, 0.0, 0.9, 0.9])]);
        assert!(decode_tile(&rows, 0, 0, 1.0, 0.0).unwrap().is_empty());
    }

    #[test]
    fn overlapping_tiles_are_deduplicated() {
        let a = RawSpot { score: 0.9, cx: 100.0, cy: 100.0, side: 20.0 };
        let b = RawSpot { score: 0.8, cx: 102.0, cy: 99.0, side: 20.0 };
        assert!(iou(&a, &b) > NMS_IOU);
    }

    #[test]
    fn scale_and_tiles_match_c1() {
        assert!(spot_scale(416, 416).is_none());
        let f = spot_scale(6048, 4032).unwrap();
        assert!((f - 0.5).abs() < 1e-6);
        let tiles = spot_tiles(2016);
        assert_eq!(tiles.first(), Some(&0));
        assert_eq!(tiles.last(), Some(&(2016 - 416)));
    }

    #[test]
    fn short_buffer_is_an_error() {
        assert!(detect_dust(&[0u8; 16], 8, 8, 0.15).is_err());
    }

    /// Real model: 600×600 with three dark dots, C1 scale/tile path.
    #[test]
    #[ignore]
    fn detect_synthetic_dots_with_the_real_model() {
        let (w, h) = (600u32, 600u32);
        let mut buf = vec![153u8; (w * h * 4) as usize];
        let mut dots = Vec::new();
        for (cx, cy) in [(150i64, 150i64), (300, 150), (450, 450)] {
            dots.push((cx, cy));
            for y in cy - 12..=cy + 12 {
                for x in cx - 12..=cx + 12 {
                    if (x - cx).pow(2) + (y - cy).pow(2) <= 144 {
                        let p = ((y as u32 * w + x as u32) * 4) as usize;
                        buf[p..p + 3].fill(0);
                    }
                }
            }
        }
        let start = std::time::Instant::now();
        let spots = detect_dust(&buf, w, h, 0.15).expect("detection");
        println!("detected {} spots in {:?}", spots.len(), start.elapsed());
        for s in &spots {
            println!("  x={:.4} y={:.4} r={:.4} (px {:.1})", s.x, s.y, s.radius, s.radius * 600.0);
        }
        for (cx, cy) in dots {
            assert!(
                spots.iter().any(|s| (s.x * 600.0 - cx as f32).abs() < 25.0
                    && (s.y * 600.0 - cy as f32).abs() < 25.0),
                "missing dot at ({cx},{cy})"
            );
        }
    }

    #[test]
    #[ignore]
    fn flat_image_has_no_spots_with_the_real_model() {
        let buf = vec![153u8; 600 * 600 * 4];
        let spots = detect_dust(&buf, 600, 600, 0.15).expect("detection");
        assert!(spots.is_empty(), "flat frame produced {spots:?}");
    }

    #[test]
    #[ignore]
    fn detect_real_9868_frame() {
        let path = r"C:\WorkSpace\Aurora Studio\90_work\build-cache\z6iii-desktop-smoke-a88d98b0\output\DSC_9868.tif";
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let image = image::open(path).unwrap().to_rgba8();
        let (w, h) = image.dimensions();
        let spots = detect_dust(image.as_raw(), w, h, 0.15).expect("detection");
        println!("real 9868: {} spots in {w}x{h}", spots.len());
        for s in spots.iter().take(20) {
            println!("  x={:.4} y={:.4} r={:.4}", s.x, s.y, s.radius);
        }
    }
}
