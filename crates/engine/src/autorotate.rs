//! AI auto-rotation on top of Capture One's `autoRotation.onnx`.
//!
//! The model is not a four-class classifier: it is a *straighten* head with 25
//! softmax classes, one per degree from +12° to −12° (class 12 is level). C1
//! uses it to level the horizon. A quarter-turn can still be recovered from it:
//! the upright orientation is the one where that tilt distribution is centred,
//! and sideways frames score far lower. The 0°/180° twin is settled with the
//! usual sky-on-top cue, because a perfectly level horizon looks identical
//! upside down.
//!
//! The result is only a suggestion. Nothing here touches the user's
//! adjustments; the interface decides whether to write the turn.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result, bail};

use crate::aimask::{Runtime, Session};

const MODEL: &str = "autoRotation.onnx";
const MAX_CACHED: usize = 32;
/// Level classes: +1°, 0°, −1°.
const LEVEL: std::ops::RangeInclusive<usize> = 11..=13;
/// Below this the image has no usable horizon, so any answer would be a guess.
const MIN_SCORE: f32 = 0.15;
/// The upright orientation must beat every other quarter turn by this much.
const MIN_MARGIN: f32 = 0.05;

/// Clockwise turn that makes the image upright.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Quadrant {
    R0 = 0,
    R90 = 90,
    R180 = 180,
    R270 = 270,
}

impl Quadrant {
    pub fn degrees(self) -> u16 {
        self as u16
    }
}

/// Adds a suggested clockwise quarter turn to the orientation already applied.
pub fn add_quarter(orientation: u8, suggested_degrees: u16) -> u8 {
    (orientation + (suggested_degrees / 90) as u8) % 4
}

struct Rotator {
    session: Session,
    path: PathBuf,
    in_w: usize,
    in_h: usize,
    nhwc: bool,
    channels: usize,
}

impl Rotator {
    fn load(path: &Path) -> Result<Arc<Rotator>> {
        static MODELS: OnceLock<Mutex<HashMap<PathBuf, Arc<Rotator>>>> = OnceLock::new();
        let models = MODELS.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(model) = models.lock().unwrap_or_else(|e| e.into_inner()).get(path) {
            return Ok(model.clone());
        }
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let session = runtime.open_session(path)?;
        let dims = &session.input.dims;
        if dims.len() != 4 || session.input.element_type != 1 {
            bail!("{MODEL}: expected one float32 NCHW/NHWC input, got {dims:?}");
        }
        let nhwc = dims[3] == 1 || dims[3] == 3;
        let channels = if nhwc { dims[3] } else { dims[1] } as usize;
        if channels != 1 {
            bail!("{MODEL}: expected a single grayscale input channel, got {channels}");
        }
        let side = |v: i64| if v > 0 { v as usize } else { 384 };
        let (in_h, in_w) = if nhwc { (side(dims[1]), side(dims[2])) } else { (side(dims[2]), side(dims[3])) };
        if session.outputs.len() != 1 {
            bail!("{MODEL}: expected one output, got {}", session.outputs.len());
        }
        let model = Arc::new(Rotator { session, path: path.to_path_buf(), in_w, in_h, nhwc, channels });
        models.lock().unwrap_or_else(|e| e.into_inner()).insert(path.to_path_buf(), model.clone());
        Ok(model)
    }

    /// Softmax over the 25 tilt classes.
    fn tilt(&self, rgba: &[u8], w: usize, h: usize) -> Result<Vec<f32>> {
        let mut input = grayscale_tensor(rgba, w, h, self.in_w, self.in_h, self.nhwc, self.channels);
        let shape: [i64; 4] = if self.nhwc {
            [1, self.in_h as i64, self.in_w as i64, self.channels as i64]
        } else {
            [1, self.channels as i64, self.in_h as i64, self.in_w as i64]
        };
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let mut outputs = runtime.run(&self.session, &mut input, &shape)?;
        if outputs.is_empty() {
            bail!("{}: model returned no outputs", self.path.display());
        }
        Ok(outputs.swap_remove(0))
    }

    fn quadrant(&self, rgba: &[u8], w: u32, h: u32) -> Result<Quadrant> {
        let mut best = (0u8, f32::MIN);
        let mut second = f32::MIN;
        let (mut upright, mut uw, mut uh) = (rgba.to_vec(), w, h);
        for turn in 0..4u8 {
            if turn > 0 {
                (upright, uw, uh) = rotate_quarter(&upright, uw, uh);
            }
            let tilt = self.tilt(&upright, uw as usize, uh as usize)?;
            let level: f32 = LEVEL.filter_map(|i| tilt.get(i)).sum();
            let score = level * (0.2 + 0.8 * sky_on_top(&upright, uw, uh));
            if score > best.1 {
                second = best.1;
                best = (turn, score);
            } else if score > second {
                second = score;
            }
        }
        if best.1 < MIN_SCORE || best.1 - second < MIN_MARGIN {
            bail!(
                "{MODEL}: no confident upright orientation (score {:.2}, runner-up {:.2}); \
                 the image has no usable horizon",
                best.1,
                second
            );
        }
        Ok(match best.0 {
            0 => Quadrant::R0,
            1 => Quadrant::R90,
            2 => Quadrant::R180,
            _ => Quadrant::R270,
        })
    }
}

/// Bilinear resize to the model input, luma, 0..1, in the model's layout.
fn grayscale_tensor(
    rgba: &[u8],
    w: usize,
    h: usize,
    in_w: usize,
    in_h: usize,
    nhwc: bool,
    channels: usize,
) -> Vec<f32> {
    let mut tensor = vec![0.0f32; channels * in_w * in_h];
    for y in 0..in_h {
        let sy = (y as f32 + 0.5) * h as f32 / in_h as f32 - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(h - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..in_w {
            let sx = (x as f32 + 0.5) * w as f32 / in_w as f32 - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(w - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            let at = |xx: usize, yy: usize, c: usize| rgba[(yy * w + xx) * 4 + c] as f32 / 255.0;
            let top = |c: usize| at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
            let bottom = |c: usize| at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
            let luma = |c: usize| top(c) * (1.0 - fy) + bottom(c) * fy;
            let value = 0.299 * luma(0) + 0.587 * luma(1) + 0.114 * luma(2);
            let i = y * in_w + x;
            for c in 0..channels {
                if nhwc {
                    tensor[i * channels + c] = value;
                } else {
                    tensor[c * in_w * in_h + i] = value;
                }
            }
        }
    }
    tensor
}

fn rotate_quarter(rgba: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let (nw, nh) = (h, w);
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let src = ((y * w + x) * 4) as usize;
            let nx = h - 1 - y;
            let dst = ((x * nw + nx) * 4) as usize;
            out[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
        }
    }
    (out, nw, nh)
}

/// How much brighter the top fifth is than the bottom fifth, 0..1.
fn sky_on_top(rgba: &[u8], w: u32, h: u32) -> f32 {
    let band = (h / 5).max(1);
    let mut sum = [0.0f32; 2];
    for (slot, rows) in [0..band, (h - band)..h].into_iter().enumerate() {
        let mut total = 0.0;
        for y in rows {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                total += 0.299 * rgba[i] as f32 + 0.587 * rgba[i + 1] as f32 + 0.114 * rgba[i + 2] as f32;
            }
        }
        sum[slot] = total / (band * w) as f32 / 255.0;
    }
    ((sum[0] - sum[1]) * 3.0).clamp(0.0, 1.0)
}

type CacheKey = (u32, u32, u64);

fn cache() -> &'static Mutex<HashMap<CacheKey, Quadrant>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, Quadrant>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Cheap content hash over a 32×32 sample, so repeated requests for the same
/// image skip the model.
fn cache_key(rgba: &[u8], w: u32, h: u32) -> CacheKey {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    w.hash(&mut hasher);
    h.hash(&mut hasher);
    let step_x = (w as usize / 32).max(1);
    let step_y = (h as usize / 32).max(1);
    let mut y = 0usize;
    while y < h as usize {
        let mut x = 0usize;
        while x < w as usize {
            let i = (y * w as usize + x) * 4;
            rgba[i..i + 3].hash(&mut hasher);
            x += step_x;
        }
        y += step_y;
    }
    (w, h, hasher.finish())
}

/// Clockwise quarter turn that would make `rgba` upright. Errors when the model
/// is missing or the image has no horizon to be confident about.
pub fn detect_quadrant(rgba: &[u8], w: u32, h: u32) -> Result<Quadrant> {
    detect_with(&crate::aimask::model_path(MODEL)?, rgba, w, h)
}

fn detect_with(path: &Path, rgba: &[u8], w: u32, h: u32) -> Result<Quadrant> {
    if w == 0 || h == 0 {
        bail!("auto rotation needs a non-empty image");
    }
    if rgba.len() != (w as usize * h as usize * 4) {
        bail!("expected {} RGBA bytes for {w}x{h}, got {}", w as usize * h as usize * 4, rgba.len());
    }
    let key = cache_key(rgba, w, h);
    if let Some(found) = cache().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(*found);
    }
    let start = Instant::now();
    let quadrant = Rotator::load(path)?.quadrant(rgba, w, h)?;
    log::info!("auto rotation suggestion {:?} in {:.1} ms", quadrant, start.elapsed().as_secs_f32() * 1000.0);
    let mut map = cache().lock().unwrap_or_else(|e| e.into_inner());
    // shortcut: whole-cache eviction; plenty for a browsing session.
    if map.len() >= MAX_CACHED {
        map.clear();
    }
    map.insert(key, quadrant);
    Ok(quadrant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::rotate90;

    /// Sky, sun and a ridge: a scene the straighten head has something to say about.
    fn scene(w: u32, h: u32) -> image::RgbaImage {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        let horizon = 0.62 * h as f32;
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                let px = if (y as f32) < horizon {
                    let t = fy / 0.62;
                    let sun = ((fx - 0.74).powi(2) + (fy - 0.18).powi(2)).sqrt() < 0.06;
                    if sun {
                        [255, 245, 220]
                    } else {
                        [(60.0 + 170.0 * t) as u8, (110.0 + 105.0 * t) as u8, (200.0 + 20.0 * t) as u8]
                    }
                } else {
                    let ridge = (fx * 35.0).sin() * 3.0;
                    let dark = (fy - 0.62) / 0.38;
                    [
                        (70.0 - 45.0 * dark + ridge) as u8,
                        (80.0 - 50.0 * dark + ridge * 0.5) as u8,
                        (60.0 - 35.0 * dark) as u8,
                    ]
                };
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        image::RgbaImage::from_raw(w, h, rgba).unwrap()
    }

    #[test]
    fn rejects_bad_sizes() {
        assert!(detect_quadrant(&[], 0, 0).is_err());
        assert!(detect_quadrant(&[0; 12], 2, 2).is_err());
    }

    #[test]
    fn quarter_turns_accumulate() {
        assert_eq!(add_quarter(0, 90), 1);
        assert_eq!(add_quarter(1, 270), 0);
        assert_eq!(add_quarter(2, 180), 0);
        assert_eq!(add_quarter(3, 90), 0);
        assert_eq!(add_quarter(2, 0), 2);
    }

    #[test]
    fn missing_model_errors() {
        assert!(detect_with(Path::new("definitely-not-a-model.onnx"), &[0; 16], 2, 2).is_err());
    }

    /// `LUMEN_AIMODELS=~/agents/evidence cargo test -p lumen-engine autorotate -- --ignored --nocapture`
    #[test]
    #[ignore = "needs Capture One's autoRotation.onnx"]
    fn model_io() {
        let path = crate::aimask::model_path(MODEL).expect("model");
        let model = Rotator::load(&path).expect("session");
        let dims = |d: &[i64]| d.iter().map(i64::to_string).collect::<Vec<_>>().join("x");
        println!("path {}", path.display());
        println!(
            "input  {:?} [{}] elem_type {}",
            model.session.input.name, dims(&model.session.input.dims), model.session.input.element_type
        );
        for o in &model.session.outputs {
            println!("output {:?} [{}] elem_type {}", o.name, dims(&o.dims), o.element_type);
        }
    }

    #[test]
    #[ignore = "needs Capture One's autoRotation.onnx"]
    fn recovers_quarter_turns() {
        let path = crate::aimask::model_path(MODEL).expect("model");
        let model = Rotator::load(&path).expect("session");
        let (w, h) = (640u32, 427u32);
        let source = scene(w, h);
        let expected = [Quadrant::R0, Quadrant::R270, Quadrant::R180, Quadrant::R90];
        for (turns, want) in expected.into_iter().enumerate() {
            let mut img = source.clone();
            for _ in 0..turns {
                img = rotate90(&img);
            }
            let (iw, ih) = img.dimensions();
            let start = Instant::now();
            let got = detect_with(&path, img.as_raw(), iw, ih).expect("detect");
            println!("rotated {turns}x90 -> {got:?} (want {want:?}), {:.1} ms", start.elapsed().as_secs_f32() * 1000.0);
            assert_eq!(got, want);
        }
    }

    #[test]
    #[ignore = "needs Capture One's autoRotation.onnx"]
    fn flat_image_is_an_error() {
        let path = crate::aimask::model_path(MODEL).expect("model");
        let rgba = vec![128u8; 320 * 240 * 4];
        assert!(detect_with(&path, &rgba, 320, 240).is_err());
    }

    /// Real photos, with per-turn scores: `LUMEN_PROBE_IMAGE=/path/photo.jpg`.
    #[test]
    #[ignore = "needs a real photo"]
    fn real_photo_round_trip() {
        let Ok(photo) = std::env::var("LUMEN_PROBE_IMAGE") else {
            println!("set LUMEN_PROBE_IMAGE to a real photo; skipping");
            return;
        };
        let path = crate::aimask::model_path(MODEL).expect("model");
        let model = Rotator::load(&path).expect("session");
        let source = image::open(&photo).expect("photo").to_rgba8();
        for turns in 0..4u32 {
            let mut img = source.clone();
            for _ in 0..turns {
                img = rotate90(&img);
            }
            let (iw, ih) = img.dimensions();
            let (mut upright, mut uw, mut uh) = (img.as_raw().clone(), iw, ih);
            let mut scores = Vec::new();
            for k in 0..4u32 {
                if k > 0 {
                    (upright, uw, uh) = rotate_quarter(&upright, uw, uh);
                }
                let tilt = model.tilt(&upright, uw as usize, uh as usize).expect("tilt");
                let level: f32 = LEVEL.filter_map(|i| tilt.get(i)).sum();
                scores.push((k, level, sky_on_top(&upright, uw, uh)));
            }
            let start = Instant::now();
            let got = detect_with(&path, img.as_raw(), iw, ih);
            println!(
                "input {turns}x90 -> {got:?} in {:.1} ms, per-turn (k, level, sky) {scores:?}",
                start.elapsed().as_secs_f32() * 1000.0
            );
        }
    }
}
