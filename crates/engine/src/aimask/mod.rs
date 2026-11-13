//! Capture One's own AI segmentation models (`subjectMasking.onnx`,
//! `skinMasking.onnx`), run through ONNX Runtime.
//!
//! The models are the ones a Capture One install ships under
//! `AIModels/Common`: a 320×320 subject masker and a 512×512 six-class people
//! parser. The parser's classes are background, hair, body skin, face skin,
//! clothing and a held object; there is no eye or lip class. Sky has no model
//! at all, so it is a colour heuristic over the subject mask.
//! `LUMEN_AIMODELS` points at a folder holding copies of the two files, for
//! machines without Capture One.
//!
//! `LUMEN_ORT_LIB` overrides the ONNX Runtime library. Everything degrades to a
//! warning when either is missing; nothing here panics.

mod ort;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, bail};

pub(crate) use ort::{Runtime, Session};

/// Loaded ONNX Runtime: library path, reported version, API table version.
/// `None` when no usable library was found.
pub fn runtime_info() -> Option<(String, String, u32)> {
    let (path, version, api) = Runtime::get()?.describe();
    Some((path.display().to_string(), version.to_string(), api))
}

/// The model's output for one image, sampled in normalized image coordinates.
pub struct MaskPlane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl MaskPlane {
    pub fn new(w: usize, h: usize, data: Vec<f32>) -> Self {
        Self { w, h, data }
    }

    /// Bilinear sample at normalized `p`; `0.0` outside the plane.
    pub fn sample(&self, p: [f32; 2]) -> f32 {
        if self.w == 0 || self.h == 0 || !(0.0..=1.0).contains(&p[0]) || !(0.0..=1.0).contains(&p[1]) {
            return 0.0;
        }
        let x = (p[0] * self.w as f32 - 0.5).clamp(0.0, (self.w as f32 - 1.001).max(0.0));
        let y = (p[1] * self.h as f32 - 0.5).clamp(0.0, (self.h as f32 - 1.001).max(0.0));
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let at = |xx: usize, yy: usize| self.data[yy * self.w + xx];
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
        (top + (bottom - top) * ty).clamp(0.0, 1.0)
    }

    /// Share of the mask above 0.5, for logging and tests.
    pub fn foreground_fraction(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().filter(|v| **v > 0.5).count() as f32 / self.data.len() as f32
    }
}

const MAX_CACHED_MASKS: usize = 16;

fn masks() -> &'static Mutex<HashMap<String, Arc<MaskPlane>>> {
    static MASKS: OnceLock<Mutex<HashMap<String, Arc<MaskPlane>>>> = OnceLock::new();
    MASKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Caches one computed mask under the key `ImageEngine::compute_ai_mask` hands
/// back to the interface.
pub fn store(key: &str, plane: Arc<MaskPlane>) {
    let mut masks = masks().lock().unwrap_or_else(|e| e.into_inner());
    // shortcut: whole-cache eviction, enough for a few hundred megapixels;
    // switch to LRU if sessions ever hold many masked images at once.
    if masks.len() >= MAX_CACHED_MASKS {
        masks.clear();
    }
    masks.insert(key.to_string(), plane);
}

pub fn lookup(key: &str) -> Option<Arc<MaskPlane>> {
    masks().lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned()
}

/// Folders a Capture One install keeps its models and `onnxruntime.dll` in.
pub(crate) fn c1_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(dir) = std::env::var_os("LUMEN_C1_DIR") {
        roots.push(PathBuf::from(dir));
    }
    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        roots.push(PathBuf::from(program_files).join("Capture One").join("Capture One"));
    }
    roots
}

fn model_candidates(name: &str) -> Vec<PathBuf> {
    let subfolder = match name {
        "subjectMasking.onnx" => "AIModels/Common/SubjectMasking",
        "skinMasking.onnx" => "AIModels/Common/PeopleMasking",
        "EyeDetector.onnx" | "DenseNet121_FP32.onnx" => "AIModels/Common/ClosedEyesDetection",
        "autoRotation.onnx" => "AIModels/Common/Rotation",
        _ => "AIModels/Common",
    };
    let mut paths = Vec::new();
    if let Some(dir) = std::env::var_os("LUMEN_AIMODELS") {
        let dir = PathBuf::from(dir);
        paths.push(dir.join(name));
        paths.push(dir.join(subfolder).join(name));
        paths.push(dir.join(subfolder.rsplit('/').next().unwrap_or("")).join(name));
    }
    // The subfolder is also a valid relative location, which keeps the search
    // independent of where Capture One is installed (or whether it is at all).
    paths.push(PathBuf::from(subfolder).join(name));
    for root in c1_roots() {
        paths.push(root.join(subfolder).join(name));
        paths.push(root.join(name));
    }
    paths
}

pub(crate) fn model_path(name: &str) -> Result<PathBuf> {
    let candidates = model_candidates(name);
    if let Some(path) = candidates.iter().find(|p| p.is_file()) {
        return Ok(path.clone());
    }
    bail!(
        "AI model {name} not found; set LUMEN_AIMODELS to a folder with the file (looked in {})",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// Whether a model file exists anywhere [`model_path`] would look, without
/// opening the ONNX session.
pub fn model_available(name: &str) -> bool {
    model_candidates(name).iter().any(|path| path.is_file())
}

/// A loaded model plus what its metadata says about the tensors.
pub(crate) struct Segmenter {
    pub(crate) session: Session,
    pub(crate) path: PathBuf,
    pub(crate) in_w: usize,
    pub(crate) in_h: usize,
    pub(crate) nchw: bool,
    pub(crate) channels: usize,
    pub(crate) out_nchw: bool,
}

impl Segmenter {
    fn load(name: &str) -> Result<Arc<Segmenter>> {
        static MODELS: OnceLock<Mutex<HashMap<String, Arc<Segmenter>>>> = OnceLock::new();
        let models = MODELS.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(model) = models.lock().unwrap_or_else(|e| e.into_inner()).get(name) {
            return Ok(model.clone());
        }
        let path = model_path(name)?;
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let session = runtime.open_session(&path)?;
        let dims = &session.input.dims;
        if dims.len() != 4 || session.input.element_type != 1 {
            bail!("{name}: expected one float32 NCHW/NHWC input, got {dims:?}");
        }
        // [1,3,H,W] is NCHW, [1,H,W,3] is NHWC.
        let nchw = dims[1] == 3;
        let side = |v: i64, fallback: usize| if v > 0 { v as usize } else { fallback };
        let (in_h, in_w) = if nchw {
            (side(dims[2], 320), side(dims[3], 320))
        } else {
            (side(dims[1], 320), side(dims[2], 320))
        };
        let out_dims = &session.outputs[0].dims;
        let channels = out_dims
            .iter()
            .copied()
            .find(|d| *d > 1 && *d < 64)
            .unwrap_or(1) as usize;
        let out_nchw = channels > 1 && out_dims.get(1) == Some(&(channels as i64));
        let model = Arc::new(Segmenter { session, path: path.clone(), in_w, in_h, nchw, channels, out_nchw });
        models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string(), model.clone());
        Ok(model)
    }

    fn input_tensor(&self, rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
        let mut tensor = vec![0.0f32; 3 * self.in_w * self.in_h];
        for y in 0..self.in_h {
            let sy = (y as f32 + 0.5) * h as f32 / self.in_h as f32 - 0.5;
            let y0 = sy.floor().max(0.0) as usize;
            let y1 = (y0 + 1).min(h - 1);
            let fy = (sy - y0 as f32).clamp(0.0, 1.0);
            for x in 0..self.in_w {
                let sx = (x as f32 + 0.5) * w as f32 / self.in_w as f32 - 0.5;
                let x0 = sx.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(w - 1);
                let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                let at = |xx: usize, yy: usize, c: usize| rgba[(yy * w + xx) * 4 + c] as f32 / 255.0;
                let i = y * self.in_w + x;
                for c in 0..3 {
                    let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
                    let bottom = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
                    let value = top * (1.0 - fy) + bottom * fy;
                    if self.nchw {
                        tensor[c * self.in_w * self.in_h + i] = value;
                    } else {
                        tensor[i * 3 + c] = value;
                    }
                }
            }
        }
        tensor
    }

    /// Runs the model and returns its first output at model resolution.
    fn run(&self, input: &mut [f32]) -> Result<Vec<f32>> {
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let shape: [i64; 4] = if self.nchw {
            [1, 3, self.in_h as i64, self.in_w as i64]
        } else {
            [1, self.in_h as i64, self.in_w as i64, 3]
        };
        let mut outputs = runtime.run(&self.session, input, &shape)?;
        if outputs.is_empty() {
            bail!("{}: model returned no outputs", self.path.display());
        }
        Ok(outputs.swap_remove(0))
    }
}

fn check_rgba(rgba: &[u8], w: usize, h: usize) -> Result<()> {
    if rgba.len() != w.saturating_mul(h).saturating_mul(4) {
        bail!("expected {} RGBA bytes for {w}x{h}, got {}", w * h * 4, rgba.len());
    }
    Ok(())
}

/// Bilinear resample of a single-channel plane.
fn resample(src: &[f32], sw: usize, sh: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let sy = (y as f32 + 0.5) * sh as f32 / h as f32 - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..w {
            let sx = (x as f32 + 0.5) * sw as f32 / w as f32 - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            let at = |xx: usize, yy: usize| src[yy * sw + xx];
            let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
            let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
            out[y * w + x] = (top + (bottom - top) * fy).clamp(0.0, 1.0);
        }
    }
    out
}

/// Subject/people/clothing mask: 0..1 per pixel of the input image.
pub fn subject_mask(rgba: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
    let (w, h) = (w as usize, h as usize);
    check_rgba(rgba, w, h)?;
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let model = Segmenter::load("subjectMasking.onnx")?;
    let mut input = model.input_tensor(rgba, w, h);
    let out = model.run(&mut input)?;
    let expected = model.in_w * model.in_h;
    if out.len() < expected {
        bail!("subjectMasking.onnx returned {} values, expected {expected}", out.len());
    }
    Ok(resample(&out[..expected], model.in_w, model.in_h, w, h))
}

/// The six classes the people parser scores per pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeopleClass {
    Background = 0,
    Hair = 1,
    BodySkin = 2,
    FaceSkin = 3,
    Clothing = 4,
    HeldObject = 5,
}

/// Softmax probability of one parser class, resampled to the input size.
pub fn people_class_mask(rgba: &[u8], w: u32, h: u32, class: PeopleClass) -> Result<Vec<f32>> {
    let (w, h) = (w as usize, h as usize);
    check_rgba(rgba, w, h)?;
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let model = Segmenter::load("skinMasking.onnx")?;
    let mut input = model.input_tensor(rgba, w, h);
    let out = model.run(&mut input)?;
    let pixels = model.in_w * model.in_h;
    let classes = model.channels.max(1);
    if out.len() < pixels * classes {
        bail!("skinMasking.onnx returned {} values, expected {}", out.len(), pixels * classes);
    }
    let plane = class_plane(&out, pixels, classes, model.out_nchw, class as usize);
    Ok(resample(&plane, model.in_w, model.in_h, w, h))
}

/// Per-pixel softmax probability of `class`, in model resolution.
fn class_plane(out: &[f32], pixels: usize, classes: usize, out_nchw: bool, class: usize) -> Vec<f32> {
    let class = class.min(classes.saturating_sub(1));
    (0..pixels)
        .map(|i| {
            let at = |c: usize| {
                if out_nchw { out[c * pixels + i] } else { out[i * classes + c] }
            };
            let max = (0..classes).fold(f32::MIN, |m, c| m.max(at(c)));
            let sum: f32 = (0..classes).map(|c| (at(c) - max).exp()).sum();
            if sum > 0.0 { (at(class) - max).exp() / sum } else { 0.0 }
        })
        .collect()
}

/// Face skin from the people parser.
pub fn skin_mask(rgba: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
    people_class_mask(rgba, w, h, PeopleClass::FaceSkin)
}

/// Hair from the people parser. It expects a portrait-scale face; a small face
/// in a wide frame collapses to background, like C1's own parser inputs.
pub fn hair_mask(rgba: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
    people_class_mask(rgba, w, h, PeopleClass::Hair)
}

/// How much a pixel's colour looks like sky: blue dominance over red and
/// green. Overcast white sky scores 0, which is why this is only a starting
/// mask. `0..1`.
fn sky_prior(r: f32, g: f32, b: f32) -> f32 {
    if b < 0.35 {
        return 0.0; // dark blue shadows are not sky
    }
    let max = r.max(g).max(b);
    let blue = (b - r.max(g)) / max.max(0.05);
    ((blue - 0.04) / 0.12).clamp(0.0, 1.0)
}

/// Sky has no model in `AIModels/Common`. This is a heuristic: the inverse of
/// the subject mask, weighted by a colour prior and faded out towards the
/// bottom of the frame. It will miss a coloured or overcast sky and can keep
/// blue clothing, so it is a starting mask, not a segmentation.
pub fn sky_mask(rgba: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
    let (w, h) = (w as usize, h as usize);
    check_rgba(rgba, w, h)?;
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let subject = subject_mask(rgba, w as u32, h as u32)?;
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        // Sky is above the horizon: full weight down to 60%, gone by 90%.
        let vertical = ((0.9 - y as f32 / h as f32) / 0.3).clamp(0.0, 1.0);
        for x in 0..w {
            let i = y * w + x;
            let p = &rgba[i * 4..i * 4 + 4];
            let prior = sky_prior(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
            out[i] = (1.0 - subject[i]) * prior * vertical;
        }
    }
    Ok(out)
}

/// One line per tensor, for the probe binary.
pub fn model_summary(name: &str) -> Result<Vec<String>> {
    let model = Segmenter::load(name)?;
    let dims = |d: &[i64]| d.iter().map(i64::to_string).collect::<Vec<_>>().join("x");
    let mut lines = vec![format!(
        "{} input {:?} [{}] float32",
        model.path.display(),
        model.session.input.name,
        dims(&model.session.input.dims)
    )];
    lines.extend(model.session.outputs.iter().map(|o| {
        format!("  output {:?} [{}]", o.name, dims(&o.dims))
    }));
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_empty() {
        assert!(subject_mask(&[], 0, 0).unwrap().is_empty());
        assert!(skin_mask(&[], 0, 0).unwrap().is_empty());
        assert!(hair_mask(&[], 0, 0).unwrap().is_empty());
        assert!(sky_mask(&[], 0, 0).unwrap().is_empty());
    }

    #[test]
    fn size_mismatch_errors() {
        assert!(subject_mask(&[0u8; 12], 2, 2).is_err());
        assert!(skin_mask(&[0u8; 16], 300, 300).is_err());
        assert!(hair_mask(&[0u8; 16], 300, 300).is_err());
        assert!(sky_mask(&[0u8; 16], 300, 300).is_err());
    }

    #[test]
    fn class_indices_match_the_people_parser() {
        assert_eq!(PeopleClass::Background as usize, 0);
        assert_eq!(PeopleClass::Hair as usize, 1);
        assert_eq!(PeopleClass::BodySkin as usize, 2);
        assert_eq!(PeopleClass::FaceSkin as usize, 3);
        assert_eq!(PeopleClass::Clothing as usize, 4);
        assert_eq!(PeopleClass::HeldObject as usize, 5);
    }

    #[test]
    fn class_plane_reads_every_channel_at_model_resolution() {
        // Two pixels, six logits each, NHWC: pixel 0 is class 4, pixel 1 class 1.
        let out = [0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0];
        for class in 0..6 {
            assert_eq!(class_plane(&out, 2, 6, false, class).len(), 2);
        }
        assert!(class_plane(&out, 2, 6, false, 4)[0] > 0.99);
        assert!(class_plane(&out, 2, 6, false, 1)[1] > 0.99);
        // NCHW stores the same two pixels channel-major.
        let mut nchw = [0.0f32; 12];
        nchw[4 * 2] = 8.0;
        nchw[1 * 2 + 1] = 8.0;
        assert!(class_plane(&nchw, 2, 6, true, 4)[0] > 0.99);
        assert!(class_plane(&nchw, 2, 6, true, 1)[1] > 0.99);
    }

    #[test]
    fn sky_prior_prefers_blue_and_rejects_ground() {
        assert!(sky_prior(0.35, 0.55, 0.85) > 0.8);
        assert_eq!(sky_prior(0.35, 0.30, 0.22), 0.0);
        assert_eq!(sky_prior(0.92, 0.93, 0.95), 0.0); // overcast white sky is not detected
        assert_eq!(sky_prior(0.75, 0.52, 0.42), 0.0); // skin
        assert_eq!(sky_prior(0.20, 0.24, 0.29), 0.0); // dark blue shadow
    }

    #[test]
    fn missing_model_is_an_error_not_a_panic() {
        assert!(model_path("not-a-model.onnx").is_err());
        assert!(Segmenter::load("not-a-model.onnx").is_err());
    }

    #[test]
    fn cache_returns_the_same_plane() {
        let key = "test:cache:subject";
        assert!(lookup(key).is_none());
        let plane = Arc::new(MaskPlane::new(2, 1, vec![0.0, 1.0]));
        store(key, plane.clone());
        let cached = lookup(key).expect("cached");
        assert!(Arc::ptr_eq(&plane, &cached));
        assert_eq!(cached.sample([0.25, 0.5]), 0.0);
        assert!(cached.sample([0.75, 0.5]) > 0.99);
        assert_eq!(cached.foreground_fraction(), 0.5);
    }

    #[test]
    fn windows_install_root_is_searched() {
        if !cfg!(windows) {
            return;
        }
        assert!(
            c1_roots().iter().any(|root| root.ends_with("Capture One/Capture One")),
            "c1_roots must include the Capture One install root: {:?}",
            c1_roots()
        );
    }

    #[test]
    fn rotation_model_is_searched_in_its_subfolder() {
        let candidates = model_candidates("autoRotation.onnx");
        assert!(
            candidates
                .iter()
                .any(|path| path.ends_with("AIModels/Common/Rotation/autoRotation.onnx")),
            "{candidates:?}"
        );
    }

    /// `LUMEN_AIMODELS=~/agents/evidence cargo test -p lumen-engine -- --ignored`
    #[test]
    #[ignore = "needs Capture One's ONNX models"]
    fn real_models_produce_a_mask() {
        let (w, h) = (640u32, 480u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let sky = y < h / 3;
                let subject = (x as i32 - w as i32 / 2).abs() < 120 && y > h / 3;
                let px = if sky { [90, 140, 220] } else if subject { [200, 150, 120] } else { [40, 40, 45] };
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        let start = std::time::Instant::now();
        let subject = subject_mask(&rgba, w, h).expect("subject model");
        let subject_ms = start.elapsed().as_secs_f32() * 1000.0;
        let start = std::time::Instant::now();
        let skin = skin_mask(&rgba, w, h).expect("skin model");
        let skin_ms = start.elapsed().as_secs_f32() * 1000.0;
        for (name, mask, ms) in [("subject", subject, subject_ms), ("skin", skin, skin_ms)] {
            let plane = MaskPlane::new(w as usize, h as usize, mask);
            println!(
                "{name}: {}x{}, {:.0} ms, foreground {:.1}%",
                plane.w,
                plane.h,
                ms,
                plane.foreground_fraction() * 100.0
            );
            assert_eq!(plane.data.len(), (w * h) as usize);
            assert!(plane.data.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }

    /// The same models through `FileEngine`: compute, cache, and the kinds no
    /// model covers.
    #[test]
    #[ignore = "needs Capture One's ONNX models"]
    fn file_engine_stores_and_reuses_the_mask() {
        use crate::{AiMaskKind, FileEngine, ImageEngine};

        let (w, h) = (320u32, 240u32);
        let rgba: Vec<u8> = (0..w * h).flat_map(|_| [180u8, 160, 140, 255]).collect();
        let dir = std::env::temp_dir().join(format!("lumen-aimask-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::from_raw(w, h, rgba).unwrap().save(dir.join("frame.png")).unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let image = engine.images()[0].id;
        let key = engine.compute_ai_mask(image, AiMaskKind::Subject).unwrap();
        let plane = lookup(&key).expect("stored under the returned key");
        assert_eq!(plane.w, 1024);
        assert_eq!(engine.compute_ai_mask(image, AiMaskKind::Subject).unwrap(), key);
        assert!(engine.compute_ai_mask(image, AiMaskKind::Background).is_ok());
        assert!(engine.compute_ai_mask(image, AiMaskKind::Sky).is_ok());
        assert!(engine.compute_ai_mask(image, AiMaskKind::Hair).is_ok());
        let err = engine.compute_ai_mask(image, AiMaskKind::Eyes).unwrap_err();
        assert!(err.to_string().contains("no Eyes class"), "{err}");
        let err = engine.compute_ai_mask(image, AiMaskKind::Lips).unwrap_err();
        assert!(err.to_string().contains("no Lips class"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
