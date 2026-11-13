//! An in-memory engine with procedural images and a CPU preview pipeline.
//!
//! Replace it with the real engine by implementing [`crate::ImageEngine`]; the
//! interface only ever talks to the trait.

pub mod process;
pub(crate) mod scenes;
pub(crate) mod spots;
pub(crate) mod detail;
pub(crate) mod geometry;
pub(crate) mod lens_profile;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use crate::*;
use crate::preview_cache::{self, CachedFrame, PreviewCache};
use process::Source;
use scenes::Scene;

enum Origin {
    Scene(Scene, u32),
    File(PathBuf),
}

struct Entry {
    info: ImageInfo,
    origin: Origin,
    adjustments: AdjustmentSet,
}

#[derive(Default)]
struct State {
    entries: Vec<Entry>,
    next_id: u64,
    sources: HashMap<ImageId, Arc<Source>>,
    /// Most recently used last; bounds memory to a few full-size sources.
    lru: Vec<ImageId>,
    jobs: HashMap<u64, JobStatus>,
    preview: PreviewCache,
    profiled: crate::engine::ProfileCache,
}

pub struct MockEngine {
    state: Arc<Mutex<State>>,
}

const SOURCE_LONG_SIDE: u32 = 2400;
const CACHED_SOURCES: usize = 6;

const CAMERAS: [(&str, &str, &str); 4] = [
    ("Phase One", "IQ4 150MP", "Schneider Kreuznach 80mm LS f/2.8"),
    ("Fujifilm", "GFX100 II", "GF 110mm F2 R LM WR"),
    ("Sony", "ILCE-1", "FE 24-70mm F2.8 GM II"),
    ("Nikon", "Z 8", "NIKKOR Z 50mm f/1.2 S"),
];

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MockEngine {
    pub fn new() -> Self {
        let mut state = State { next_id: 1, ..Default::default() };
        for (i, scene) in Scene::ALL.iter().cycle().take(24).enumerate() {
            let id = ImageId(state.next_id);
            state.next_id += 1;
            let variant_seed = i as u32 / Scene::ALL.len() as u32;
            let (w, h) = if scene.portrait() { (1600, 2400) } else { (2400, 1600) };
            let (make, model, lens) = CAMERAS[i % CAMERAS.len()];
            let info = ImageInfo {
                id,
                path: PathBuf::from(format!("C:/Photos/2026/Session/LUM_{:04}.IIQ", 1000 + i)),
                file_name: format!("LUM_{:04} {}.IIQ", 1000 + i, scene.title()),
                kind: FileKind::Raw,
                heif: None,
                width: w,
                height: h,
                file_size: 92_000_000 + (i as u64 * 1_234_567) % 30_000_000,
                exif: Exif {
                    camera_make: Some(make.into()),
                    camera_model: Some(model.into()),
                    lens: Some(lens.into()),
                    focal_length_mm: Some([80.0, 110.0, 35.0, 50.0][i % 4]),
                    aperture: Some([2.8, 4.0, 5.6, 8.0, 11.0][i % 5]),
                    shutter: Some([1.0 / 125.0, 1.0 / 250.0, 1.0 / 60.0, 1.0 / 1000.0, 2.0][i % 5]),
                    iso: Some([64, 100, 200, 400, 1600][i % 5]),
                    exposure_bias: Some(0.0),
                    flash: Some(matches!(scene, Scene::Portrait)),
                    captured_at: Some(format!("2026-10-{:02}T{:02}:{:02}:00+08:00", 1 + i % 28, 6 + i % 12, (i * 7) % 60)),
                    gps: Some((46.5 + i as f64 * 0.01, 8.0 + i as f64 * 0.02)),
                },
                iptc: Iptc { creator: "Lumen Studio".into(), copyright: "© 2026".into(), ..Default::default() },
                keywords: vec![scene.title().to_lowercase()],
                rating: [0, 3, 5, 4, 0, 2, 1, 0, -1, 3][i % 10],
                color_tag: ColorTag::ALL[[0, 0, 4, 0, 1, 0, 5, 0, 0, 3][i % 10]],
                variant: 1,
                has_adjustments: false,
                trashed: false,
            };
            state.entries.push(Entry {
                info,
                origin: Origin::Scene(*scene, 7 + variant_seed * 101),
                adjustments: AdjustmentSet::default(),
            });
        }
        Self { state: Arc::new(Mutex::new(state)) }
    }

    fn source(&self, image: ImageId) -> anyhow::Result<Arc<Source>> {
        let origin = {
            let mut st = self.state.lock();
            if let Some(src) = st.sources.get(&image).cloned() {
                st.lru.retain(|i| *i != image);
                st.lru.push(image);
                return Ok(src);
            }
            let entry = st.entries.iter().find(|e| e.info.id == image).ok_or_else(|| anyhow::anyhow!("unknown image {image:?}"))?;
            match &entry.origin {
                Origin::Scene(scene, seed) => Origin::Scene(*scene, *seed),
                Origin::File(path) => Origin::File(path.clone()),
            }
        };
        let source = Arc::new(match origin {
            Origin::Scene(scene, seed) => {
                let (w, h) = if scene.portrait() {
                    (SOURCE_LONG_SIDE * 2 / 3, SOURCE_LONG_SIDE)
                } else {
                    (SOURCE_LONG_SIDE, SOURCE_LONG_SIDE * 2 / 3)
                };
                Source::from_rgba(w, h, &scenes::render(scene, w, h, seed))
            }
            Origin::File(path) => {
                let img = image::open(&path)?;
                let img = if img.width().max(img.height()) > 4096 {
                    img.resize(4096, 4096, image::imageops::FilterType::Triangle)
                } else {
                    img
                };
                let rgba = img.to_rgba8();
                Source::from_rgba(rgba.width(), rgba.height(), rgba.as_raw())
            }
        });
        let mut st = self.state.lock();
        st.sources.insert(image, source.clone());
        st.lru.retain(|i| *i != image);
        st.lru.push(image);
        while st.lru.len() > CACHED_SOURCES {
            let old = st.lru.remove(0);
            st.sources.remove(&old);
        }
        Ok(source)
    }

    /// Cached preview frame at full requested quality, including during a drag.
    fn preview_frame(&self, request: &PreviewRequest, _fast: bool) -> anyhow::Result<PreviewFrame> {
        let start = Instant::now();
        let src = self.source(request.image)?;
        let max = request.max_size;
        let fast = false;
        let adj = crate::styles::resolve(&request.adjustments);
        let key = preview_cache::key(request.image, &adj, request.region, max, request.bypass, request.mask_overlay, fast);
        let profile_key = crate::engine::ProfileKey::new(key, request.proof_profile.as_deref(), request.display_profile.as_deref(), (request.proof_paper_white, request.proof_black_point));
        let hit = self.state.lock().preview.get(key);
        if let Some(hit) = hit {
            let histogram = hit.histogram.clone();
            let source = hit.into_image();
            let gamut_warning = crate::engine::gamut_warning(&source, request.proof_profile.as_deref());
            let (image, degraded) = self.state.lock().profiled.converted(profile_key, source, request.proof_profile.as_deref(), request.display_profile.as_deref(), (request.proof_paper_white, request.proof_black_point));
            return Ok(PreviewFrame {
                histogram,
                image,
                generation: request.generation,
                elapsed_ms: start.elapsed().as_secs_f32() * 1000.0,
                degraded,
                gamut_warning,
            });
        }
        let out = process::render_at(&src, &adj, request.region, max, request.bypass, request.mask_overlay, fast);
        let frame = CachedFrame {
            width: out.width,
            height: out.height,
            pixels: Arc::new(out.rgba),
            histogram: out.histogram.clone(),
        };
        self.state.lock().preview.insert(key, frame.clone());
        let source = frame.into_image();
        let gamut_warning = crate::engine::gamut_warning(&source, request.proof_profile.as_deref());
        let (image, degraded) = self.state.lock().profiled.converted(profile_key, source, request.proof_profile.as_deref(), request.display_profile.as_deref(), (request.proof_paper_white, request.proof_black_point));
        Ok(PreviewFrame {
            image,
            histogram: out.histogram,
            generation: request.generation,
            elapsed_ms: start.elapsed().as_secs_f32() * 1000.0,
            degraded,
            gamut_warning,
        })
    }

    fn add_file(&self, path: &Path) -> anyhow::Result<ImageInfo> {
        let (w, h) = image::image_dimensions(path)?;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        let kind = match ext.as_str() {
            "jpg" | "jpeg" => FileKind::Jpeg,
            "png" => FileKind::Png,
            "tif" | "tiff" => FileKind::Tiff,
            _ => FileKind::Jpeg,
        };
        let mut st = self.state.lock();
        let id = ImageId(st.next_id);
        st.next_id += 1;
        let info = ImageInfo {
            id,
            path: path.to_path_buf(),
            file_name: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            kind,
            heif: None,
            width: w,
            height: h,
            file_size: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            exif: Exif::default(),
            iptc: Iptc::default(),
            keywords: Vec::new(),
            rating: 0,
            color_tag: ColorTag::None,
            variant: 1,
            has_adjustments: false,
            trashed: false,
        };
        st.entries.push(Entry { info: info.clone(), origin: Origin::File(path.to_path_buf()), adjustments: AdjustmentSet::default() });
        Ok(info)
    }
}

fn is_image(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp")
    )
}

impl ImageEngine for MockEngine {
    fn name(&self) -> String {
        "Lumen Mock Engine 0.1 (CPU preview)".into()
    }

    fn images(&self) -> Vec<ImageInfo> {
        self.state.lock().entries.iter().map(|e| e.info.clone()).collect()
    }

    fn import(&self, paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
        let mut out = Vec::new();
        for path in paths {
            if path.is_dir() {
                let mut files: Vec<_> = std::fs::read_dir(path)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| is_image(p)).collect();
                files.sort();
                for f in files {
                    match self.add_file(&f) {
                        Ok(info) => out.push(info),
                        Err(err) => log::warn!("import skipped {}: {err}", f.display()),
                    }
                }
            } else if is_image(path) {
                out.push(self.add_file(path)?);
            }
        }
        Ok(out)
    }

    fn collections(&self) -> Vec<Collection> {
        let count = self.state.lock().entries.len();
        let node = |id, name: &str, kind, count, children| Collection { id, name: name.into(), kind, count, children };
        vec![
            node(1, "All Images", CollectionKind::AllImages, count, vec![]),
            node(2, "Recent Imports", CollectionKind::RecentImports, count.min(12), vec![]),
            node(3, "Recent Captures", CollectionKind::RecentCaptures, 0, vec![]),
            node(4, "Trash", CollectionKind::Trash, 0, vec![]),
            node(
                10,
                "Wedding · Aster & Lin",
                CollectionKind::Project,
                9,
                vec![
                    node(11, "Ceremony", CollectionKind::Album, 4, vec![]),
                    node(12, "Portraits", CollectionKind::Album, 5, vec![]),
                    node(13, "Picks ★4+", CollectionKind::SmartAlbum, 3, vec![]),
                ],
            ),
            node(
                20,
                "Landscapes",
                CollectionKind::Group,
                14,
                vec![
                    node(21, "Alps 2026", CollectionKind::Album, 6, vec![]),
                    node(22, "Desert", CollectionKind::Album, 4, vec![]),
                    node(23, "Red tagged", CollectionKind::SmartAlbum, 2, vec![]),
                ],
            ),
            node(
                30,
                "C:/Photos",
                CollectionKind::Folder,
                count,
                vec![node(31, "2026", CollectionKind::Folder, count, vec![node(32, "Session", CollectionKind::Folder, count, vec![])])],
            ),
        ]
    }

    fn update_info(&self, info: &ImageInfo) -> anyhow::Result<()> {
        let mut st = self.state.lock();
        if let Some(e) = st.entries.iter_mut().find(|e| e.info.id == info.id) {
            e.info = info.clone();
        }
        Ok(())
    }

    fn set_trashed(&self, ids: &[ImageId], trashed: bool) -> anyhow::Result<()> {
        let mut st = self.state.lock();
        for e in st.entries.iter_mut().filter(|e| ids.contains(&e.info.id)) {
            e.info.trashed = trashed;
        }
        Ok(())
    }

    fn load_adjustments(&self, image: ImageId) -> AdjustmentSet {
        self.state.lock().entries.iter().find(|e| e.info.id == image).map(|e| e.adjustments.clone()).unwrap_or_default()
    }

    fn save_adjustments(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<()> {
        let mut st = self.state.lock();
        if let Some(e) = st.entries.iter_mut().find(|e| e.info.id == image) {
            e.adjustments = adjustments.clone();
            e.info.has_adjustments = !adjustments.is_default();
        }
        Ok(())
    }

    fn new_variant(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo> {
        let mut st = self.state.lock();
        let next = st.next_id;
        let entry = st.entries.iter().find(|e| e.info.id == image).ok_or_else(|| anyhow::anyhow!("unknown image"))?;
        let variants = st.entries.iter().filter(|e| e.info.path == entry.info.path).count() as u32;
        let mut info = entry.info.clone();
        info.id = ImageId(next);
        info.variant = variants + 1;
        info.has_adjustments = !adjustments.is_default();
        let origin = match &entry.origin {
            Origin::Scene(s, seed) => Origin::Scene(*s, *seed),
            Origin::File(p) => Origin::File(p.clone()),
        };
        let pos = st.entries.iter().position(|e| e.info.id == image).unwrap_or(0);
        st.next_id += 1;
        st.entries.insert(pos + 1, Entry { info: info.clone(), origin, adjustments: adjustments.clone() });
        Ok(info)
    }

    fn thumbnail(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<RgbaImage> {
        let src = self.source(image)?;
        let out = process::render(&src, &crate::styles::resolve(adjustments), NormRect::FULL, max_size, false, None);
        Ok(RgbaImage::new(out.width, out.height, out.rgba))
    }

    fn render_preview(&self, request: &PreviewRequest) -> anyhow::Result<PreviewFrame> {
        self.preview_frame(request, false)
    }

    fn render_preview_fast(&self, request: &PreviewRequest, fast: bool) -> anyhow::Result<PreviewFrame> {
        self.preview_frame(request, fast)
    }

    fn sample_color(&self, image: ImageId, adjustments: &AdjustmentSet, point: [f32; 2]) -> anyhow::Result<[f32; 3]> {
        let src = self.source(image)?;
        Ok(process::sample_point(&src, &crate::styles::resolve(adjustments), point))
    }

    fn auto_adjust(&self, image: ImageId, current: &AdjustmentSet, target: AutoTarget) -> anyhow::Result<AutoResult> {
        let src = self.source(image)?;
        let probe = process::render(&src, &AdjustmentSet { geometry: current.geometry.clone(), ..Default::default() }, NormRect::FULL, 256, false, None);
        let h = &probe.histogram;
        let total: u32 = h.luma.iter().sum();
        let mean = h.luma.iter().enumerate().map(|(i, c)| i as f32 * *c as f32).sum::<f32>() / total.max(1) as f32 / 255.0;
        let mut adj = current.clone();
        let all = target == AutoTarget::All;
        if all || target == AutoTarget::Exposure {
            adj.exposure.exposure = ((0.46 / mean.max(0.02)).log2() * 0.8).clamp(-2.0, 2.0);
            adj.exposure.contrast = 8.0;
        }
        if all || target == AutoTarget::HighDynamicRange {
            adj.hdr.highlights = (-h.clipped_highlights * 800.0).clamp(-60.0, 0.0);
            adj.hdr.shadows = (h.clipped_shadows * 600.0 + 10.0).clamp(0.0, 50.0);
        }
        if all || target == AutoTarget::WhiteBalance {
            let avg = |v: &Vec<u32>| v.iter().enumerate().map(|(i, c)| i as f32 * *c as f32).sum::<f32>() / total.max(1) as f32;
            let (r, b) = (avg(&h.red), avg(&h.blue));
            adj.white_balance.mode = WhiteBalanceMode::Auto;
            adj.white_balance.kelvin = (5600.0 + (r - b) * -18.0).clamp(2500.0, 12000.0);
            adj.white_balance.tint = 0.0;
        }
        if all || target == AutoTarget::Levels {
            let cut = total / 500;
            let mut acc = 0;
            let lo = h.luma.iter().position(|c| {
                acc += c;
                acc > cut
            });
            acc = 0;
            let hi = h.luma.iter().rposition(|c| {
                acc += c;
                acc > cut
            });
            adj.levels.rgb.input_black = lo.unwrap_or(0) as f32 / 255.0;
            adj.levels.rgb.input_white = hi.unwrap_or(255) as f32 / 255.0;
        }
        if target == AutoTarget::Rotation {
            adj.geometry.rotation = 0.0;
        }
        if target == AutoTarget::Keystone {
            adj.geometry.keystone = Keystone { amount: 100.0, ..Default::default() };
        }
        Ok(AutoResult { adjustments: adj })
    }

    fn pick_white_balance(&self, image: ImageId, point: [f32; 2]) -> anyhow::Result<(f32, f32)> {
        let src = self.source(image)?;
        let c = process::sample_point(&src, &AdjustmentSet::default(), point);
        let warm = (c[0] - c[2]) / (c[0] + c[1] + c[2]).max(1e-3);
        let green = (c[1] - (c[0] + c[2]) / 2.0) / (c[0] + c[1] + c[2]).max(1e-3);
        Ok(((5600.0 - warm * 9000.0).clamp(2000.0, 15000.0), (green * 120.0).clamp(-50.0, 50.0)))
    }

    fn compute_ai_mask(&self, image: ImageId, kind: AiMaskKind) -> anyhow::Result<String> {
        std::thread::sleep(std::time::Duration::from_millis(600));
        Ok(format!("mock-{}-{:?}", image.0, kind))
    }

    fn icc_profiles(&self) -> Vec<String> {
        let found = crate::icc::profile_names();
        if found.is_empty() {
            return ["sRGB Color Space Profile", "Adobe RGB (1998)"]
                .into_iter()
                .map(String::from)
                .collect();
        }
        found
    }

    fn lens_profiles(&self) -> Vec<String> {
        ["Generic", "Schneider Kreuznach 80mm LS f/2.8", "GF 110mm F2 R LM WR", "FE 24-70mm F2.8 GM II", "NIKKOR Z 50mm f/1.2 S"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn styles(&self) -> Vec<StyleInfo> {
        crate::styles::all()
    }

    fn start_export(&self, job: ExportJob) -> anyhow::Result<()> {
        job.recipe.format.ensure_supported()?;
        job.recipe.format.supports_bit_depth(job.recipe.bit_depth)?;
        let state = self.state.clone();
        let job_id = job.job_id;
        state.lock().jobs.insert(job_id, JobStatus::Queued);
        let src = self.source(job.image)?;
        let adjustments = self.load_adjustments(job.image);
        let info = self
            .state
            .lock()
            .entries
            .iter()
            .find(|e| e.info.id == job.image)
            .map(|e| e.info.clone())
            .ok_or_else(|| anyhow::anyhow!("export: unknown image {}", job.image.0))?;
        std::thread::spawn(move || {
            state.lock().jobs.insert(job_id, JobStatus::Running { progress: 0.05 });
            let (tw, th) = crate::recipe::output_size(&job.recipe, (info.width, info.height));
            let dir = job.recipe.destination.clone().unwrap_or_else(|| std::env::temp_dir().join("lumen-output"));
            let wide = job.recipe.bit_depth == 16;
            let result = if wide {
                let out = process::render16(
                    &src,
                    &crate::styles::resolve(&adjustments),
                    NormRect::FULL,
                    tw.max(th).max(1),
                    false,
                );
                crate::recipe::write_export16(
                    &dir,
                    &info,
                    &job.recipe,
                    0,
                    out.rgba,
                    out.width,
                    out.height,
                    |p| {
                        state.lock().jobs.insert(job_id, JobStatus::Running { progress: 0.1 + 0.9 * p });
                    },
                )
            } else {
                let out = process::render(
                    &src,
                    &crate::styles::resolve(&adjustments),
                    NormRect::FULL,
                    tw.max(th).max(1),
                    false,
                    None,
                );
                crate::recipe::write_export(
                    &dir,
                    &info,
                    &job.recipe,
                    0,
                    out.rgba,
                    out.width,
                    out.height,
                    |p| {
                        state.lock().jobs.insert(job_id, JobStatus::Running { progress: 0.1 + 0.9 * p });
                    },
                )
            };
            state.lock().jobs.insert(
                job_id,
                match result {
                    Ok(output) => JobStatus::Done { output },
                    Err(err) => JobStatus::Failed { message: err.to_string() },
                },
            );
        });
        Ok(())
    }

    fn job_status(&self, job_id: u64) -> JobStatus {
        self.state.lock().jobs.get(&job_id).cloned().unwrap_or(JobStatus::Failed { message: "unknown job".into() })
    }

    fn camera_status(&self) -> CameraStatus {
        CameraStatus::Connected { model: "Phase One IQ4 150MP".into(), battery: 0.82, shots_remaining: 1248 }
    }

    fn camera_settings(&self) -> Option<CameraSettings> {
        Some(CameraSettings {
            iso: "ISO 100".into(),
            shutter: "1/125".into(),
            aperture: "f/8".into(),
            exposure_comp: 0.0,
            white_balance: "Daylight".into(),
            drive_mode: "Single".into(),
            focus_mode: "Auto".into(),
            focus_distance: "2.4 m".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft_request(image: ImageId) -> PreviewRequest {
        PreviewRequest {
            image,
            adjustments: AdjustmentSet::default(),
            region: NormRect::FULL,
            max_size: 1400,
            quality: RenderQuality::Draft,
            bypass: false,
            mask_overlay: None,
            generation: 1,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
        }
    }

    #[test]
    fn a_repeated_request_hits_the_cache() {
        let engine = MockEngine::new();
        let mut request = draft_request(engine.images()[0].id);
        let first = engine.render_preview(&request).unwrap();
        // A repaint arrives with a newer generation; content is unchanged.
        request.generation = 7;
        let second = engine.render_preview(&request).unwrap();
        assert!(Arc::ptr_eq(&first.image.pixels, &second.image.pixels));
        assert_eq!(second.generation, 7);
        assert_eq!(first.histogram, second.histogram);
    }

    #[test]
    fn profiled_thumbnail_defaults_to_plain_pixels() {
        let engine = MockEngine::new();
        let id = engine.images()[0].id;
        let adj = AdjustmentSet::default();
        let plain = engine.thumbnail(id, &adj, 64).unwrap();
        let same = engine.thumbnail_profiled(id, &adj, 64, None, None, (1.0, 0.0)).unwrap();
        assert_eq!(plain.pixels, same.pixels);
        let clamped = engine.thumbnail_profiled(id, &adj, 64, None, None, (2.0, -1.0)).unwrap();
        assert_eq!(plain.pixels, clamped.pixels);
        if crate::icc::profile("NikonZ6III-Generic").is_some() {
            let neutral = engine.thumbnail_profiled(id, &adj, 64, None, Some("NikonZ6III-Generic"), (1.0, 0.0)).unwrap();
            let level = engine.thumbnail_profiled(id, &adj, 64, None, Some("NikonZ6III-Generic"), (0.8, 0.1)).unwrap();
            assert_ne!(neutral.pixels, level.pixels, "non-neutral paper levels must change a proofed thumbnail");
        }
        let srgb = std::path::Path::new(r"C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm");
        if srgb.exists() {
            let skipped = engine.thumbnail_profiled(id, &adj, 64, Some(srgb), None, (1.0, 0.0)).unwrap();
            assert_eq!(plain.pixels, skipped.pixels);
        }
    }

    #[test]
    fn start_export_rejects_an_unimplemented_format_before_queueing() {
        let engine = MockEngine::new();
        let mut recipe = crate::output::ProcessRecipe::default();
        recipe.format = crate::output::ExportFormat::JpegXl;
        let job = ExportJob { job_id: 9001, image: engine.images()[0].id, recipe };

        let err = engine.start_export(job).unwrap_err();

        assert!(err.to_string().contains("JPEG XL"), "{err}");
        assert!(engine.state.lock().jobs.is_empty(), "the job was queued anyway");
    }

    #[test]
    fn changing_one_parameter_misses_the_cache() {
        let engine = MockEngine::new();
        let mut request = draft_request(engine.images()[0].id);
        let first = engine.render_preview(&request).unwrap();
        request.adjustments.exposure.exposure = 0.01;
        let second = engine.render_preview(&request).unwrap();
        assert!(!Arc::ptr_eq(&first.image.pixels, &second.image.pixels));
        assert_ne!(first.image.pixels, second.image.pixels);
    }

    #[test]
    fn draft_final_and_fast_preserve_full_pixels() {
        let mut request = draft_request(MockEngine::new().images()[0].id);
        request.max_size = 1600;
        let engine = MockEngine::new();
        let draft = engine.render_preview(&request).unwrap();
        let fast = engine.render_preview_fast(&request, true).unwrap();
        assert_eq!((draft.image.width, draft.image.height), (fast.image.width, fast.image.height));

        assert_eq!(draft.image.width.max(draft.image.height), 1600);
        request.quality = RenderQuality::Final;
        let final_frame = engine.render_preview(&request).unwrap();
        assert_eq!(draft.image.pixels, final_frame.image.pixels);
        assert_eq!(draft.image.pixels, fast.image.pixels);

        // A separate engine must produce identical pixels without a cache hit.
        let other = MockEngine::new().render_preview_fast(&request, true).unwrap();
        assert_eq!(fast.image.pixels, other.image.pixels);
    }

    #[test]
    #[ignore]
    fn bench_preview() {
        let engine = MockEngine::new();
        let mut request = draft_request(engine.images()[0].id);
        engine.render_preview(&request).unwrap();

        let hits: Vec<f32> = (0..5)
            .map(|i| {
                request.generation = i;
                engine.render_preview(&request).unwrap().elapsed_ms
            })
            .collect();

        let drafts: Vec<f32> = (0..5)
            .map(|i| {
                request.adjustments.exposure.exposure = i as f32 * 0.001;
                engine.render_preview(&request).unwrap().elapsed_ms
            })
            .collect();

        let fasts: Vec<f32> = (0..5)
            .map(|i| {
                request.adjustments.exposure.exposure = i as f32 * 0.001 + 0.5;
                engine.render_preview_fast(&request, true).unwrap().elapsed_ms
            })
            .collect();

        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        println!("cache hit : {:.3} ms", mean(&hits));
        println!("draft 1400: {:.1} ms", mean(&drafts));
        println!("fast 1400 : {:.1} ms", mean(&fasts));
    }

    #[test]
    fn set_trashed_marks_images_but_keeps_them_listed() {
        let engine = MockEngine::new();
        let before = engine.images().len();
        let ids: Vec<ImageId> = engine.images().iter().take(3).map(|i| i.id).collect();
        engine.set_trashed(&ids, true).unwrap();
        let images = engine.images();
        assert_eq!(images.len(), before);
        assert!(images.iter().filter(|i| i.trashed).all(|i| ids.contains(&i.id)));
        assert_eq!(images.iter().filter(|i| i.trashed).count(), 3);
        engine.set_trashed(&ids, false).unwrap();
        assert!(engine.images().iter().all(|i| !i.trashed));
    }

    #[test]
    fn live_view_is_absent_without_a_camera_backend() {
        let engine = MockEngine::new();
        assert!(engine.live_view_frame(1280).is_none());
        engine.live_view_focus(3);
    }

    #[test]
    fn a_sixteen_bit_jpeg_job_is_rejected_before_it_starts() {
        let engine = MockEngine::new();
        let recipe = ProcessRecipe { format: ExportFormat::Jpeg, bit_depth: 16, ..Default::default() };
        let job = ExportJob { job_id: 99, image: engine.images()[0].id, recipe };
        let err = engine.start_export(job).unwrap_err().to_string();
        assert!(err.contains("16-bit"), "{err}");
        assert!(matches!(engine.job_status(99), JobStatus::Failed { message } if message == "unknown job"));
    }
}
