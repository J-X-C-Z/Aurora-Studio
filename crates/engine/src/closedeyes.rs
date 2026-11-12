//! Capture One's closed-eye check (`AIModels/Common/ClosedEyesDetection`).
//!
//! Two 64×64 ONNX classifiers: `EyeDetector.onnx` (MobileNetV3, sigmoid) scores
//! whether a crop holds an eye, `DenseNet121_FP32.onnx` scores whether that eye
//! is closed (a logit: `> 0` is a blink). Neither model localizes anything, so
//! eyes are found by scanning a fixed pyramid and keeping only horizontal pairs
//! of similar size — the geometry a face always has and clutter usually does
//! not. Capture One itself feeds these crops from its own face detector; this
//! engine has none, so `detect_eyes` reports what it can prove from the pixels.
//!
//! Both models want RGB in `0..=1` with ImageNet mean/std; the graphs carry no
//! normalization of their own. Nothing here panics: a missing model is an
//! error, a bad crop is skipped.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use crate::aimask::{Runtime, Session};

/// One eye, in normalized `0..=1` image coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeBox {
    /// Center.
    pub x: f32,
    pub y: f32,
    /// Side of the square crop the detector fired on.
    pub size: f32,
    pub score: f32,
}

/// What one image's eyes turned out to be.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyeReport {
    pub evaluated: usize,
    pub closed: usize,
    /// Highest blink probability among the evaluated eyes; `0.0` when none.
    pub worst_confidence: f32,
}

const SIDE: usize = 64;
/// Long side the scan runs at; 1024 keeps a 50 MP frame cheap.
const WORK_LONG: u32 = 1024;
const SCAN_SIDES: [u32; 4] = [48, 64, 96, 128];
const DETECT_SCORE: f32 = 0.9;
/// Blink logit above this counts as closed.
const BLINK_LOGIT: f32 = 0.0;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

struct Net {
    session: Session,
}

fn net(name: &str) -> Result<Arc<Net>> {
    static NETS: OnceLock<Mutex<HashMap<String, Arc<Net>>>> = OnceLock::new();
    let nets = NETS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(net) = nets.lock().unwrap_or_else(|e| e.into_inner()).get(name) {
        return Ok(net.clone());
    }
    let path = crate::aimask::model_path(name)?;
    let runtime = Runtime::get().context("ONNX Runtime is not available")?;
    let session = runtime.open_session(&path)?;
    if session.input.element_type != 1 || session.input.dims.len() != 4 {
        bail!("{name}: expected one float32 NCHW input, got {:?}", session.input.dims);
    }
    let net = Arc::new(Net { session });
    nets.lock().unwrap_or_else(|e| e.into_inner()).insert(name.to_string(), net.clone());
    Ok(net)
}

struct Plane {
    w: usize,
    h: usize,
    rgba: Vec<u8>,
}

/// Bilinear RGBA resize, the same sampling the mask models use.
fn resize(rgba: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let mut out = vec![0u8; nw * nh * 4];
    for y in 0..nh {
        let sy = (y as f32 + 0.5) * h as f32 / nh as f32 - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..nw {
            let sx = (x as f32 + 0.5) * w as f32 / nw as f32 - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            for c in 0..4 {
                let at = |xx: usize, yy: usize| rgba[(yy * w + xx) * 4 + c] as f32;
                let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
                let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
                out[(y * nw + x) * 4 + c] = (top + (bottom - top) * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// One square crop resized to 64×64, RGB, ImageNet-normalized, NCHW.
fn crop_tensor(plane: &Plane, cx: f32, cy: f32, side: f32, out: &mut [f32]) {
    let half = side / 2.0;
    for y in 0..SIDE {
        let sy = cy - half + (y as f32 + 0.5) * side / SIDE as f32 - 0.5;
        let y0 = sy.floor().clamp(0.0, plane.h as f32 - 1.0) as usize;
        let y1 = (y0 + 1).min(plane.h - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..SIDE {
            let sx = cx - half + (x as f32 + 0.5) * side / SIDE as f32 - 0.5;
            let x0 = sx.floor().clamp(0.0, plane.w as f32 - 1.0) as usize;
            let x1 = (x0 + 1).min(plane.w - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            let i = y * SIDE + x;
            for c in 0..3 {
                let at = |xx: usize, yy: usize| plane.rgba[(yy * plane.w + xx) * 4 + c] as f32;
                let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
                let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
                let value = (top + (bottom - top) * fy) / 255.0;
                out[c * SIDE * SIDE + i] = (value - MEAN[c]) / STD[c];
            }
        }
    }
}

/// Runs the model over `crops` and returns one score per crop.
fn infer(net: &Net, plane: &Plane, crops: &[(f32, f32, f32)]) -> Result<Vec<f32>> {
    let mut scores = Vec::with_capacity(crops.len());
    for batch in crops.chunks(256) {
        let mut input = vec![0.0f32; batch.len() * 3 * SIDE * SIDE];
        for (i, &(cx, cy, side)) in batch.iter().enumerate() {
            crop_tensor(plane, cx, cy, side, &mut input[i * 3 * SIDE * SIDE..][..3 * SIDE * SIDE]);
        }
        let shape = [batch.len() as i64, 3, SIDE as i64, SIDE as i64];
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let mut outputs = runtime.run(&net.session, &mut input, &shape)?;
        let out = outputs.first_mut().context("model returned no output")?;
        if out.len() < batch.len() {
            bail!("model returned {} values for {} crops", out.len(), batch.len());
        }
        scores.extend(out.iter().take(batch.len()).copied());
    }
    Ok(scores)
}

fn work_plane(rgba: &[u8], w: u32, h: u32) -> Plane {
    let (w, h) = (w as usize, h as usize);
    let long = w.max(h) as u32;
    if long <= WORK_LONG {
        return Plane { w, h, rgba: rgba.to_vec() };
    }
    let scale = WORK_LONG as f32 / long as f32;
    let (nw, nh) = ((w as f32 * scale).round() as usize, (h as f32 * scale).round() as usize);
    Plane { w: nw, h: nh, rgba: resize(rgba, w, h, nw.max(1), nh.max(1)) }
}

/// Eyes the detector found, plus the blink logit of each.
fn analyze(rgba: &[u8], w: u32, h: u32) -> Result<(Vec<EyeBox>, Vec<f32>)> {
    if rgba.len() != (w as usize) * (h as usize) * 4 {
        bail!("expected {} RGBA bytes for {w}x{h}, got {}", w as usize * h as usize * 4, rgba.len());
    }
    if w == 0 || h == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let start = std::time::Instant::now();
    let plane = work_plane(rgba, w, h);
    let (pw, ph) = (plane.w as f32, plane.h as f32);
    if pw < SIDE as f32 || ph < SIDE as f32 {
        return Ok((Vec::new(), Vec::new()));
    }
    let eye_net = net("EyeDetector.onnx")?;

    // Coarse pyramid.
    let mut candidates: Vec<EyeBox> = Vec::new();
    for side in SCAN_SIDES {
        let side_f = side as f32;
        if side_f > pw || side_f > ph {
            continue;
        }
        let stride = (side / 2).max(8) as f32;
        let mut crops = Vec::new();
        let mut y = side_f / 2.0;
        while y < ph {
            let mut x = side_f / 2.0;
            while x < pw {
                crops.push((x, y, side_f));
                x += stride;
            }
            y += stride;
        }
        for ((x, y, size), score) in crops.iter().zip(infer(&eye_net, &plane, &crops)?) {
            if score >= DETECT_SCORE {
                candidates.push(EyeBox { x: *x, y: *y, size: *size, score });
            }
        }
    }

    // Refine the strongest hits on a finer grid in one batch, then drop duplicates.
    // A hit is at most `size/4` from the coarse grid's peak, so that box is enough.
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(12);
    let mut crops = Vec::new();
    let mut spans = Vec::new();
    for c in &candidates {
        let step = (c.size / 8.0).max(2.0);
        let reach = c.size / 4.0;
        let start = crops.len();
        let mut dy = -reach;
        while dy <= reach {
            let mut dx = -reach;
            while dx <= reach {
                let (x, y) = (c.x + dx, c.y + dy);
                if x >= c.size / 2.0 && y >= c.size / 2.0 && x + c.size / 2.0 <= pw && y + c.size / 2.0 <= ph {
                    crops.push((x, y, c.size));
                }
                dx += step;
            }
            dy += step;
        }
        spans.push(start..crops.len());
    }
    let scores = infer(&eye_net, &plane, &crops)?;
    log::debug!(
        "closed eyes: {}×{}, {} candidates, {} refine crops in {:.0} ms",
        pw,
        ph,
        candidates.len(),
        crops.len(),
        start.elapsed().as_secs_f32() * 1000.0
    );
    let mut refined: Vec<EyeBox> = Vec::new();
    for span in spans {
        let Some((i, score)) = span
            .clone()
            .map(|i| (i, scores[i]))
            .max_by(|a, b| a.1.total_cmp(&b.1))
        else {
            continue;
        };
        let crop = crops[i];
        refined.push(EyeBox { x: crop.0, y: crop.1, size: crop.2, score });
    }
    refined.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<EyeBox> = Vec::new();
    for c in refined {
        if kept.iter().all(|k| {
            let d = (c.x - k.x).powi(2) + (c.y - k.y).powi(2);
            d > (0.6 * c.size.max(k.size)).powi(2)
        }) {
            kept.push(c);
        }
    }

    // A face's eyes are one horizontal pair; clutter is not. Keep the best pair
    // only — two eye widths apart, level, same scale.
    let mut best: Option<(f32, EyeBox, EyeBox)> = None;
    for (i, a) in kept.iter().enumerate() {
        for b in &kept[i + 1..] {
            let side = (a.size + b.size) / 2.0;
            let dx = (a.x - b.x).abs() / side;
            let dy = (a.y - b.y).abs() / side;
            if dy > 0.3 || !(1.5..=3.0).contains(&dx) || a.size.max(b.size) > 1.5 * a.size.min(b.size) {
                continue;
            }
            let cost = dy + (dx - 2.2).abs() + (2.0 - a.score - b.score);
            if best.as_ref().is_none_or(|(c, _, _)| cost < *c) {
                best = Some((cost, *a, *b));
            }
        }
    }
    let mut eyes = Vec::new();
    if let Some((_, a, b)) = best {
        eyes.extend([a, b]);
    }

    let blink_net = net("DenseNet121_FP32.onnx")?;
    let crops: Vec<(f32, f32, f32)> = eyes.iter().map(|e| (e.x, e.y, 2.0 * e.size)).collect();
    let logits = infer(&blink_net, &plane, &crops)?;
    let long = pw.max(ph);
    let eyes = eyes
        .into_iter()
        .map(|e| EyeBox { x: e.x / pw, y: e.y / ph, size: e.size / long, score: e.score })
        .collect();
    Ok((eyes, logits))
}

/// Eye boxes in normalized coordinates; empty when no paired eyes are found.
pub fn detect_eyes(rgba: &[u8], w: u32, h: u32) -> Result<Vec<EyeBox>> {
    let (eyes, _) = analyze(rgba, w, h)?;
    Ok(eyes)
}

/// Runs both models once per eye and summarizes the frame.
pub fn any_eye_closed(rgba: &[u8], w: u32, h: u32) -> Result<EyeReport> {
    let (eyes, logits) = analyze(rgba, w, h)?;
    let mut report = EyeReport::default();
    for logit in logits.iter().take(eyes.len()) {
        let p = 1.0 / (1.0 + (-logit).exp());
        report.evaluated += 1;
        report.worst_confidence = report.worst_confidence.max(p);
        if *logit > BLINK_LOGIT {
            report.closed += 1;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_image_has_no_eyes() {
        let report = any_eye_closed(&[], 0, 0).unwrap();
        assert_eq!(report, EyeReport::default());
        assert!(detect_eyes(&[], 0, 0).unwrap().is_empty());
    }

    #[test]
    fn flat_image_has_no_eyes() {
        // Finding "no eyes" still means running the detector once.
        if crate::aimask::model_path("EyeDetector.onnx").is_err() {
            return;
        }
        let (w, h) = (640u32, 480u32);
        let rgba: Vec<u8> = (0..w * h).flat_map(|_| [120u8, 130, 140, 255]).collect();
        let report = any_eye_closed(&rgba, w, h).unwrap();
        assert_eq!(report.evaluated, 0);
        assert_eq!(report.closed, 0);
        assert!(detect_eyes(&rgba, w, h).unwrap().is_empty());
    }

    #[test]
    fn wrong_buffer_size_is_an_error() {
        assert!(any_eye_closed(&[0; 7], 4, 4).is_err());
        assert!(detect_eyes(&[0; 7], 4, 4).is_err());
    }

    #[test]
    #[ignore = "needs LUMEN_AIMODELS pointing at C1's ClosedEyesDetection models"]
    fn real_face() {
        let path = std::env::var("LUMEN_EYE_PHOTO").expect("set LUMEN_EYE_PHOTO to a portrait");
        let image = image::open(&path).unwrap().to_rgba8();
        let (w, h) = image.dimensions();
        let start = std::time::Instant::now();
        let eyes = detect_eyes(image.as_raw(), w, h).unwrap();
        let report = any_eye_closed(image.as_raw(), w, h).unwrap();
        println!(
            "{path}: {w}x{h}, {} eyes, evaluated={} closed={} worst_p={:.3}, {:.0} ms",
            eyes.len(),
            report.evaluated,
            report.closed,
            report.worst_confidence,
            start.elapsed().as_secs_f32() * 1000.0
        );
        for e in &eyes {
            println!("  eye at ({:.3},{:.3}) size {:.3} score {:.3}", e.x, e.y, e.size, e.score);
        }
        assert!(report.evaluated <= eyes.len());
    }

    #[test]
    #[ignore = "needs LUMEN_AIMODELS and LUMEN_EYE_PHOTO"]
    fn file_engine_reports_once_and_caches() {
        use crate::{FileEngine, ImageEngine};

        let path = std::env::var("LUMEN_EYE_PHOTO").expect("set LUMEN_EYE_PHOTO to a portrait");
        let dir = std::env::temp_dir().join(format!("lumen-closedeyes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let copy = dir.join("portrait.png");
        image::open(&path).unwrap().to_rgba8().save(&copy).unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let image = engine.images()[0].id;
        let first = std::time::Instant::now();
        let report = engine.eye_report(image).unwrap();
        let cold_ms = first.elapsed().as_secs_f32() * 1000.0;
        let again = std::time::Instant::now();
        assert_eq!(engine.eye_report(image).unwrap(), report);
        println!(
            "file engine: evaluated={} closed={} worst_p={:.3}, {cold_ms:.0} ms cold, {:.1} ms cached",
            report.evaluated,
            report.closed,
            report.worst_confidence,
            again.elapsed().as_secs_f32() * 1000.0
        );
        assert!(again.elapsed().as_secs_f32() < 0.05);

        std::fs::remove_dir_all(&dir).ok();
    }
}
