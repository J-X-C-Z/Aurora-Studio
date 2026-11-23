//! AI beauty-mark detection: `BeautyMarks/beautyMarks.onnx`.
//!
//! Contract recovered from `ImgCoreDll.dll`:
//!
//! ```text
//! IC_DetectBeautyMarks            0x1813a0670
//!   FUN_1814306b0                 0x1814306b0  canvas scale + tile loop
//!     FUN_1814835e0               0x1814835e0  shared 416/overlap-100 walker
//!       FUN_181430cf0             0x181430cf0  per-tile decode -> rect records
//! ```
//!
//! * Input `[1,3,416,416]` float32 RGB, raw bytes widened to 0..255 (the same
//!   conversion as dust, `FUN_1814350b0`).
//! * Output `[1,3549,6]`; rows are the same anchor-free grid as dust, with
//!   `conf = col4 * col5 >= 0.15`.
//! * The canvas is `f = 2 * 416 / min_side` (not dust's `416/min + 0.1`);
//!   `f > 1` is refused, so the short side must be at least 832 px.
//! * Tiles are 416 px with 100 px overlap, the last one flush.
//! * `IC_DetectBeautyMarks` gates on `IC_DetectFacesV2` first and reports
//!   "Problem detecting face" without one; the returned items are rectangles
//!   `IC_BeautyMark { int type = 1; float score; int x0, y0, x1, y1; }` in raw
//!   image pixels.
//!
//! The face-gated canvas (`local_128 = 2 * face / min(canvas * ratio)`) is only
//! partly recovered; this module uses the shorter, documented `2 * 416 /
//! min_side` form and therefore matches the reference implementation, not a
//! per-face canvas that has not been pinned down yet.

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Result, bail};

use crate::aimask::{Runtime, Session};
use crate::dustdetect::{
    self, ANCHORS, INPUT_SIDE, MAX_SPOTS, MIN_RADIUS_PX, NMS_IOU, RawSpot,
};

/// `DAT_182a2bc70`, the same 0.15 threshold dust uses.
pub const DEFAULT_CONFIDENCE: f32 = 0.15;
/// Rectangles shorter than this are dropped when converting the box to a rect.
const MIN_MARK_SIDE: f32 = 3.0;

/// `AURORA_BEAUTYMARKS_ONNX=1` protects detected marks from Skin Evening.
pub fn enabled() -> bool {
    std::env::var("AURORA_BEAUTYMARKS_ONNX").ok().as_deref() == Some("1")
}

pub fn min_confidence() -> f32 {
    std::env::var("AURORA_BEAUTYMARKS_CONF")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(DEFAULT_CONFIDENCE)
}

/// One detected mark, in source-image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeautyMark {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

impl BeautyMark {
    pub fn centre(&self) -> (f32, f32) {
        ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)
    }

    pub fn contains(&self, x: f32, y: f32, grow: f32) -> bool {
        x >= self.x0 - grow && x <= self.x1 + grow && y >= self.y0 - grow && y <= self.y1 + grow
    }
}

/// `f = 2 * 416 / min_side`; `None` is C1's refusal for too-small images.
pub fn mark_scale(w: u32, h: u32) -> Option<f32> {
    if w == 0 || h == 0 {
        return None;
    }
    let f = 2.0 * INPUT_SIDE as f32 / w.min(h) as f32;
    (f <= 1.0).then_some(f)
}

/// Runs the model over the whole developed frame and returns mark rectangles.
pub fn detect_marks(rgba: &[u8], w: u32, h: u32, min_confidence: f32) -> Result<Vec<BeautyMark>> {
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let need = w as usize * h as usize * 4;
    if rgba.len() < need {
        bail!("beauty marks: buffer has {} bytes, need {need} for {w}x{h}", rgba.len());
    }
    let Some(scale) = mark_scale(w, h) else {
        bail!("Beauty marks rejected image because it is too small");
    };
    let cw = ((w as f32 * scale).ceil() as usize).max(INPUT_SIDE);
    let ch = ((h as f32 * scale).ceil() as usize).max(INPUT_SIDE);
    let canvas = dustdetect::resize_canvas(rgba, w as usize, h as usize, cw, ch);
    let session = &session()?.session;
    let runtime = Runtime::get()
        .ok_or_else(|| anyhow::anyhow!("ONNX Runtime is not available for beauty marks"))?;
    let mut found = Vec::new();
    for ty in dustdetect::spot_tiles(ch) {
        for tx in dustdetect::spot_tiles(cw) {
            let mut input = dustdetect::tile_tensor(&canvas, cw, ch, tx, ty);
            let planes = runtime.run(
                session,
                &mut input,
                &[1, 3, INPUT_SIDE as i64, INPUT_SIDE as i64],
            )?;
            let rows = planes
                .first()
                .ok_or_else(|| anyhow::anyhow!("beauty model returned no output"))?;
            if rows.len() < ANCHORS * 6 {
                bail!("beauty model returned {} values, expected {}", rows.len(), ANCHORS * 6);
            }
            found.extend(dustdetect::decode_tile(rows, tx, ty, scale, min_confidence)?);
        }
    }
    found.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<RawSpot> = Vec::new();
    for mark in found {
        if kept.len() >= MAX_SPOTS {
            break;
        }
        if (mark.side * 0.5) < MIN_RADIUS_PX || mark.side > 200.0 {
            continue;
        }
        if kept.iter().any(|k| dustdetect::iou(&mark, k) > NMS_IOU) {
            continue;
        }
        kept.push(mark);
    }
    Ok(kept
        .into_iter()
        .filter_map(|s| {
            if s.side < MIN_MARK_SIDE {
                return None;
            }
            let half = s.side * 0.5;
            Some(BeautyMark {
                x0: s.cx - half,
                y0: s.cy - half,
                x1: s.cx + half,
                y1: s.cy + half,
                score: s.score,
            })
        })
        .collect())
}

/// [`detect_marks`] on a face rectangle (`[x, y, w, h]`, normalised), which is
/// how C1 feeds the detector: the canvas is derived from the face, not the
/// whole frame.  The crop is copied, detected at its own `2*416/min_side`, and
/// the rectangles are moved back into frame pixels.
pub fn detect_marks_in(
    rgba: &[u8],
    w: u32,
    h: u32,
    face: [f32; 4],
    min_confidence: f32,
) -> Result<Vec<BeautyMark>> {
    if w == 0 || h == 0 || bad_face_rect(face) {
        return Ok(Vec::new());
    }
    let x0 = (face[0] * w as f32).floor().clamp(0.0, w as f32 - 1.0) as u32;
    let y0 = (face[1] * h as f32).floor().clamp(0.0, h as f32 - 1.0) as u32;
    let x1 = ((face[0] + face[2]) * w as f32).ceil().clamp(1.0, w as f32) as u32;
    let y1 = ((face[1] + face[3]) * h as f32).ceil().clamp(1.0, h as f32) as u32;
    if x1 <= x0 || y1 <= y0 {
        return Ok(Vec::new());
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    let mut crop = Vec::with_capacity((cw * ch * 4) as usize);
    for y in y0..y1 {
        let start = ((y * w + x0) * 4) as usize;
        crop.extend_from_slice(&rgba[start..start + (cw * 4) as usize]);
    }
    let mut marks = detect_marks(&crop, cw, ch, min_confidence)?;
    for mark in &mut marks {
        mark.x0 += x0 as f32;
        mark.x1 += x0 as f32;
        mark.y0 += y0 as f32;
        mark.y1 += y0 as f32;
    }
    Ok(marks)
}

fn bad_face_rect(face: [f32; 4]) -> bool {
    !(face[2] > 0.0 && face[3] > 0.0) || !face.iter().all(|v| v.is_finite())
}

/// Zeroes a Skin Evening mask under the marks (plus `grow` px) so smoothing
/// cannot smear them. Marks are in frame pixels, the same space as `mask`.
pub fn protect_mask(mask: &mut [u8], width: u32, height: u32, marks: &[BeautyMark], grow: f32) {
    if marks.is_empty() || width == 0 || height == 0 {
        return;
    }
    let (w, h) = (width as usize, height as usize);
    if mask.len() != w * h {
        return;
    }
    for mark in marks {
        let x0 = (mark.x0 - grow).floor().max(0.0) as usize;
        let y0 = (mark.y0 - grow).floor().max(0.0) as usize;
        let x1 = ((mark.x1 + grow).ceil() as usize).min(w);
        let y1 = ((mark.y1 + grow).ceil() as usize).min(h);
        for y in y0..y1 {
            mask[y * w + x0..y * w + x1].fill(0);
        }
    }
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
            return Err(format!("beautyMarks.onnx: unexpected input {dims:?}"));
        }
        if session.outputs.len() != 1 || session.outputs[0].name.is_empty() {
            return Err("beautyMarks.onnx: expected one output".into());
        }
        Ok(Loaded { session })
    }) {
        Ok(loaded) => Ok(loaded),
        Err(err) => bail!("beauty marks unavailable: {err}"),
    }
}

/// `AURORA_BEAUTYMARKS_MODEL`, then `AURORA_AIMODELS`, then the Capture One
/// install (`AIModels/Common/BeautyMarks/beautyMarks.onnx`).
fn model_path() -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = std::env::var_os("AURORA_BEAUTYMARKS_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(dir) = std::env::var_os("AURORA_AIMODELS") {
        let dir = PathBuf::from(&dir);
        candidates.push(dir.join("beautyMarks.onnx"));
        candidates.push(dir.join("Common/BeautyMarks/beautyMarks.onnx"));
        candidates.push(dir.join("AIModels/Common/BeautyMarks/beautyMarks.onnx"));
    }
    for root in crate::aimask::c1_roots() {
        candidates.push(root.join("AIModels/Common/BeautyMarks/beautyMarks.onnx"));
        candidates.push(root.join("AIModels/Common/beautyMarks.onnx"));
        candidates.push(root.join("beautyMarks.onnx"));
    }
    if let Some(path) = candidates.iter().find(|p| p.is_file()) {
        return Ok(path.clone());
    }
    bail!(
        "beautyMarks.onnx not found; set AURORA_BEAUTYMARKS_MODEL to the model file (looked in {})",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_matches_the_binary_and_rejects_small_images() {
        assert!(mark_scale(800, 600).is_none());
        let f = mark_scale(2400, 1600).unwrap();
        assert!((f - 0.52).abs() < 1e-6, "f={f}");
        assert_eq!(mark_scale(0, 10), None);
    }

    #[test]
    fn protection_zeroes_only_the_mark_rectangle() {
        let (w, h) = (8u32, 8u32);
        let mut mask = vec![255u8; (w * h) as usize];
        let mark = BeautyMark { x0: 2.0, y0: 2.0, x1: 3.0, y1: 3.0, score: 1.0 };
        protect_mask(&mut mask, w, h, &[mark], 1.0);
        assert_eq!(mask[2 * 8 + 2], 0);
        assert_eq!(mask[1 * 8 + 1], 0);
        assert_eq!(mask[0], 255);
    }

    /// Real face (`PF90002-stage-before.png`), 900² face crop: clean negative
    /// control, then six stamped moles. Dumps marks and the Skin Evening result
    /// with/without protection for the Python oracle.
    #[test]
    #[ignore]
    fn real_face_marks_and_protection() {
        let path = r"C:\WorkSpace\Aurora Studio\90_work\lanes\portrait-parse\o834_final_index\parse\s815_moisture_producer\PF90002-stage-before.png";
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let full = image::open(path).unwrap().to_rgba8();
        let crop = image::imageops::crop_imm(&full, 300, 120, 900, 900).to_image();
        let clean = detect_marks(crop.as_raw(), 900, 900, DEFAULT_CONFIDENCE).expect("clean");
        println!("clean 900x900 face: {} mark(s)", clean.len());

        let mut stamped = crop.clone();
        let moles = [(200u32, 200u32), (450, 150), (350, 480), (250, 700), (300, 260), (520, 620)];
        for (cx, cy) in moles {
            for y in cy - 11..=cy + 11 {
                for x in cx - 11..=cx + 11 {
                    if (x as i32 - cx as i32).pow(2) + (y as i32 - cy as i32).pow(2) > 121 {
                        continue;
                    }
                    let p = ((y * 900 + x) * 4) as usize;
                    for k in 0..3 {
                        stamped.as_mut()[p + k] = (stamped.as_ref()[p + k] as f32 * 0.35) as u8;
                    }
                }
            }
        }
        stamped.save(r"C:\WorkSpace\Aurora Studio\90_work\build-cache\beautymarks_stamped.png").unwrap();
        let image = stamped;
        let (w, h) = image.dimensions();
        let started = std::time::Instant::now();
        let marks = detect_marks(image.as_raw(), w, h, DEFAULT_CONFIDENCE).expect("detection");
        println!("stamped beauty marks on {w}x{h}: {} in {:.0} ms", marks.len(), started.elapsed().as_secs_f32() * 1000.0);
        for mark in marks.iter().take(12) {
            let (cx, cy) = mark.centre();
            println!(
                "  c=({cx:.1},{cy:.1}) box=({:.1},{:.1})-({:.1},{:.1}) score={:.3}",
                mark.x0, mark.y0, mark.x1, mark.y1, mark.score
            );
        }
        let json: Vec<String> = marks
            .iter()
            .map(|m| format!("[{:.4},{:.4},{:.4},{:.4},{:.6}]", m.x0, m.y0, m.x1, m.y1, m.score))
            .collect();
        let dump = format!("{{\"width\":{w},\"height\":{h},\"marks\":[{}]}}", json.join(","));
        std::fs::write(r"C:\WorkSpace\Aurora Studio\90_work\build-cache\beautymarks_rust.json", dump).unwrap();

        let mut mask = vec![255u8; (w * h) as usize];
        let plain = mask.clone();
        let protected = {
            let mut m = std::mem::take(&mut mask);
            protect_mask(&mut m, w, h, &marks, 1.0);
            m
        };
        let face = [0.05, 0.05, 0.95, 0.95];
        let mut without = image.clone().into_raw();
        crate::retouch_c1::skin_evening(&mut without, w, h, &plain, face, 50.0, 25.0);
        let mut with = image.into_raw();
        crate::retouch_c1::skin_evening(&mut with, w, h, &protected, face, 50.0, 25.0);
        let diff: Vec<u8> = without.iter().zip(&with).map(|(a, b)| a.abs_diff(*b)).collect();
        let max = diff.iter().copied().max().unwrap_or(0);
        let changed = diff.iter().filter(|d| **d != 0).count();
        println!("protection diff: max={max} changed={changed}/{}", diff.len());
        assert!(changed > 0, "protection changed nothing; masks may be empty");
        assert!(
            moles.iter().all(|(cx, cy)| marks.iter().any(|m| {
                let (mx, my) = m.centre();
                (mx - *cx as f32).abs() <= 6.0 && (my - *cy as f32).abs() <= 6.0
            })),
            "a stamped mole was missed: {marks:?}"
        );
        let outp = image::RgbaImage::from_raw(w, h, with).unwrap();
        outp.save(r"C:\WorkSpace\Aurora Studio\90_work\build-cache\beautymarks_protected.png").unwrap();
        let outu = image::RgbaImage::from_raw(w, h, without).unwrap();
        outu.save(r"C:\WorkSpace\Aurora Studio\90_work\build-cache\beautymarks_plain.png").unwrap();
    }

    /// Obama acceptance: 0.3/0.6 × protection on/off on the real portrait.
    #[test]
    #[ignore]
    fn obama_skin_evening_acceptance() {
        let path = r"C:\WorkSpace\Aurora Studio\04_reference\codex-c1-task\outputs\maskcats\obama.jpg";
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let image = image::open(path).unwrap().to_rgba8();
        let (w, h) = image.dimensions();
        let face = [0.275, 0.030, 0.345, 0.310];
        let started = std::time::Instant::now();
        let marks = detect_marks_in(image.as_raw(), w, h, face, DEFAULT_CONFIDENCE).expect("marks");
        println!(
            "obama {w}x{h}: {} mark(s), {:.0} ms",
            marks.len(),
            started.elapsed().as_secs_f32() * 1000.0
        );
        let protected: f64 = marks
            .iter()
            .map(|m| ((m.x1 - m.x0).max(0.0) * (m.y1 - m.y0).max(0.0)) as f64)
            .sum();
        println!("protected area {protected:.0} px");
        for amount in [30.0f32, 60.0] {
            let mut plain_mask = vec![255u8; (w * h) as usize];
            let mut plain = image.clone().into_raw();
            crate::retouch_c1::skin_evening(&mut plain, w, h, &plain_mask, face, amount, 0.0);
            let mut with = image.clone().into_raw();
            protect_mask(&mut plain_mask, w, h, &marks, 1.0);
            crate::retouch_c1::skin_evening(&mut with, w, h, &plain_mask, face, amount, 0.0);
            let diff: Vec<u8> = plain.iter().zip(&with).map(|(a, b)| a.abs_diff(*b)).collect();
            let max = diff.iter().copied().max().unwrap_or(0);
            let changed = diff.iter().filter(|d| **d != 0).count();
            println!("amount={amount:>4.0} protect={} max={max} changed={changed}", !marks.is_empty());
        }
        println!("(real-portrait mark count is reported, not asserted: the shipped model may find none)");
    }
}
