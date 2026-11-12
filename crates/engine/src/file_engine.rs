//! A catalog backed by real files on disk, with edits persisted in `.lumen`.
//!
//! [`crate::MockEngine`] serves procedural scenes so the interface can be built
//! before an engine exists. This one reads an actual folder: it scans it,
//! reads EXIF, keeps ratings and keywords, and writes adjustments back.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use rayon::prelude::*;

use crate::adjust::{AdjustmentSet, WhiteBalanceMode};
use crate::catalog::{ColorTag, Exif, FileKind, ImageId, ImageInfo, Iptc};
use crate::engine::{AutoResult, AutoTarget, CameraStatus, ImageEngine, PreviewFrame, PreviewRequest, RenderQuality, StyleInfo};
use crate::mock::process::{self, Source};
use crate::preview_cache::{self, CachedFrame, PreviewCache};
use crate::store::{Catalog, Entry, Store, StoreError};
use crate::tether::{self, TetherStatus};
use crate::{AiMaskKind, Collection, CollectionKind, ExportJob, JobStatus, NormRect, RgbaImage};

/// Longest side kept for a decoded source; larger files are downscaled.
const SOURCE_LONG_SIDE: u32 = 4096;
/// How many decoded sources stay in memory at once.
const CACHED_SOURCES: usize = 8;
/// Folder collection ids start here; the index into `folder_dirs()` is added.
const FOLDER_BASE: u64 = 1000;

fn folder_id(index: usize) -> u64 {
    FOLDER_BASE + index as u64
}

/// A decoded frame, capped to [`SOURCE_LONG_SIDE`] with its aspect ratio kept.
fn fit_long_side(size: (u32, u32)) -> (u32, u32) {
    let (w, h) = size;
    if w.max(h) <= SOURCE_LONG_SIDE {
        return size;
    }
    let scale = SOURCE_LONG_SIDE as f32 / w.max(h) as f32;
    let short = ((w.min(h) as f32) * scale).round().max(1.0) as u32;
    if w >= h { (SOURCE_LONG_SIDE, short) } else { (short, SOURCE_LONG_SIDE) }
}

/// Area-average downscale, parallel over output rows.
/// `image`'s resize is single-threaded and costs seconds on a 24 MP photo;
/// this is ~1.5×1.5 source pixels per output pixel, so a box average is both
/// cheap and free of aliasing. Callers only ever shrink (see [`fit_long_side`]).
fn downscale_rgba(img: &image::RgbaImage, w: u32, h: u32) -> image::RgbaImage {
    let (sw, sh) = img.dimensions();
    if (w, h) == (sw, sh) || w == 0 || h == 0 {
        return img.clone();
    }
    let src = img.as_raw();
    let mut out = vec![0u8; (w * h * 4) as usize];
    out.par_chunks_mut((w * 4) as usize).enumerate().for_each(|(y, row)| {
        let y0 = (y as u64 * sh as u64 / h as u64) as u32;
        let y1 = (((y as u64 + 1) * sh as u64 / h as u64) as u32).clamp(y0 + 1, sh);
        for x in 0..w {
            let x0 = (x as u64 * sw as u64 / w as u64) as u32;
            let x1 = (((x as u64 + 1) * sw as u64 / w as u64) as u32).clamp(x0 + 1, sw);
            let mut acc = [0u32; 4];
            let mut n = 0u32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let i = ((yy * sw + xx) * 4) as usize;
                    for k in 0..4 {
                        acc[k] += src[i + k] as u32;
                    }
                    n += 1;
                }
            }
            let o = (x * 4) as usize;
            for k in 0..4 {
                row[o + k] = (acc[k] / n) as u8;
            }
        }
    });
    image::RgbaImage::from_raw(w, h, out).expect("downscale output matches its size")
}

/// A RAW at libraw's own 16-bit linear precision, before any transfer encoding.
/// `None` when the file is not a RAW or libraw cannot demosaic it.
fn raw16_source(path: &Path) -> Option<Source> {
    if !crate::raw::is_raw(path) {
        return None;
    }
    let (w, h, px) = crate::raw::decode16(path)?;
    let img: image::ImageBuffer<image::Rgba<u16>, Vec<u16>> =
        image::ImageBuffer::from_raw(w, h, px)?;
    let (w, h) = fit_long_side((w, h));
    let img = if img.dimensions() != (w, h) {
        image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    Some(Source::from_rgba16(w, h, img.as_raw()))
}

struct FileEntry {
    info: ImageInfo,
    path: PathBuf,
    rel_path: String,
    adjustments: AdjustmentSet,
}

struct State {
    entries: Vec<FileEntry>,
    next_id: u64,
    sources: HashMap<ImageId, Arc<Source>>,
    /// Most recently used last.
    lru: Vec<ImageId>,
    jobs: HashMap<u64, JobStatus>,
    imported_styles: Vec<StyleInfo>,
    catalog: Catalog,
    preview: PreviewCache,
}

/// An engine over a real folder of images.
pub struct FileEngine {
    root: PathBuf,
    state: Arc<Mutex<State>>,
    store: Option<Store>,
    /// Camera defaults from Capture One's `coreConfig.cdx`, read once.
    camera_defaults: std::sync::OnceLock<Option<crate::CameraDefaults>>,
    /// Closed-eye results, computed once per image.
    eyes: Mutex<HashMap<ImageId, crate::EyeReport>>,
}

impl FileEngine {
    /// Opens `root` and imports every supported image below it.
    pub fn open(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        let store = match Store::open(&root) {
            Ok(store) => Some(store),
            Err(err) => {
                match err.downcast_ref::<StoreError>() {
                    Some(StoreError::Newer { found }) => {
                        log::warn!("catalog schema v{found} is newer than this build; opening read-only")
                    }
                    _ => log::warn!("catalog unavailable: {err}"),
                }
                None
            }
        };
        let catalog = store.as_ref().and_then(|s| s.load_index().ok()).unwrap_or_default();
        let engine = Self {
            root,
            state: Arc::new(Mutex::new(State {
                entries: Vec::new(),
                next_id: 1,
                sources: HashMap::new(),
                lru: Vec::new(),
                jobs: HashMap::new(),
                imported_styles: Vec::new(),
                catalog,
                preview: PreviewCache::default(),
            })),
            store,
            camera_defaults: std::sync::OnceLock::new(),
            eyes: Mutex::new(HashMap::new()),
        };
        engine.rescan()?;
        engine.preload_ai_masks();
        Ok(engine)
    }

    /// The folder this engine reads.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Opens a Capture One catalog or session read-only and imports its images
    /// as Lumen entries. Edits are saved to `.lumen` beside the database; the C1
    /// database itself is never written to.
    pub fn open_c1_catalog(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let import = crate::c1import::C1Import::open(path.as_ref())?;
        // A bundle is `X.cocatalogdb/X.cocatalogdb` (or `.cosessiondb`, or a
        // `.cos` session folder); `.lumen` must not land inside it.
        let mut root = import.catalog_path.parent().unwrap_or(Path::new(".")).to_path_buf();
        // A bundle is a folder named after the database it holds; step outside it so
        // our own `.lumen` never lands inside Capture One's package.
        let stem = import.catalog_path.file_stem().map(|s| s.to_os_string());
        while stem.is_some() && root.file_name().map(|n| n.to_os_string()) == stem {
            root = root.parent().unwrap_or(Path::new(".")).to_path_buf();
        }
        while root.extension().is_some_and(|e| {
            ["cocatalogdb", "cosessiondb", "cos"]
                .iter()
                .any(|want| e.eq_ignore_ascii_case(want))
        }) {
            root = root.parent().unwrap_or(Path::new(".")).to_path_buf();
        }
        let store = Store::open(&root).ok();
        let catalog = store.as_ref().and_then(|s| s.load_index().ok()).unwrap_or_default();
        let engine = Self {
            root,
            state: Arc::new(Mutex::new(State {
                entries: Vec::new(),
                next_id: 1,
                sources: HashMap::new(),
                lru: Vec::new(),
                jobs: HashMap::new(),
                imported_styles: Vec::new(),
                catalog,
                preview: PreviewCache::default(),
            })),
            store,
            camera_defaults: std::sync::OnceLock::new(),
            eyes: Mutex::new(HashMap::new()),
        };
        engine.import_c1(&import);
        engine.preload_ai_masks();
        Ok(engine)
    }

    /// Puts masks referenced by saved layers back into the in-memory cache, so
    /// a reopened folder renders the segmentation instead of the ellipse.
    fn preload_ai_masks(&self) {
        let keys: Vec<String> = {
            let st = self.state.lock();
            st.entries
                .iter()
                .flat_map(|e| e.adjustments.layers.iter())
                .flat_map(|layer| layer.mask.shapes.iter())
                .filter_map(|shape| match shape {
                    crate::MaskShape::Ai(ai) => ai.cache_key.clone(),
                    _ => None,
                })
                .collect()
        };
        for key in keys {
            if crate::aimask::lookup(&key).is_none() {
                if let Some(plane) = crate::mask_store::load(&self.root, &key) {
                    crate::aimask::store(&key, plane);
                }
            }
        }
    }

    fn import_c1(&self, import: &crate::c1import::C1Import) {
        let mut imported: Vec<(String, AdjustmentSet, ImageInfo)> = Vec::new();
        let mut st = self.state.lock();
        for image in import.images() {
            let path = PathBuf::from(&image.abs_path);
            for variant in import.variants_of(image.pk) {
                let number = variant.index.max(0) as u32 + 1;
                let rel_path = crate::c1import::store_key(&path, number);
                // Our store wins over C1's copy, so reopening the catalog keeps the
                // edits made in Lumen instead of resetting them to C1's.
                let stored = st.catalog.entry(&rel_path).cloned();
                let adjustments = match (&self.store, &stored) {
                    (Some(store), Some(_)) => store.load_adjustments(&rel_path).unwrap_or_default(),
                    _ => {
                        let mut adjustments = variant.adjustments.clone();
                        // C1 stores the display rotation in ZROTATION and libraw
                        // already applied the camera's EXIF rotation to the pixels,
                        // so only the difference is left for the renderer.
                        if crate::raw::is_raw(&path) {
                            // Nothing in this pipeline turns the sensor data: libraw is
                            // asked for the stored orientation and the embedded preview
                            // is stored the same way. So the full turn is still needed.
                            if adjustments.geometry.orientation == 0 && path.is_file() {
                                adjustments.geometry.orientation =
                                    ((crate::raw::exif_rotation_deg(&path) / 90) % 4) as u8;
                            }
                        }
                        adjustments
                    }
                };
                let id = ImageId(st.next_id);
                st.next_id += 1;
                let fresh = stored.is_none();
                let info = ImageInfo {
                    id,
                    path: path.clone(),
                    file_name: image.file_name.clone(),
                    kind: file_kind(&path),
                    width: image.width,
                    height: image.height,
                    file_size: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
                    exif: Exif {
                        camera_make: Some(image.camera_make.clone()).filter(|s| !s.is_empty()),
                        camera_model: Some(image.camera_model.clone()).filter(|s| !s.is_empty()),
                        lens: Some(image.camera_lens.clone()).filter(|s| !s.is_empty()),
                        focal_length_mm: image.focal_length,
                        aperture: image.aperture,
                        shutter: image.shutter,
                        iso: image.iso.map(|v| v as u32),
                        ..Exif::default()
                    },
                    iptc: Iptc::default(),
                    keywords: stored.as_ref().map(|e| e.keywords.clone()).unwrap_or_else(|| variant.keywords.clone()),
                    rating: stored.as_ref().map(|e| e.rating).unwrap_or((variant.rating as i8).clamp(-1, 5)),
                    color_tag: match &stored {
                        Some(e) => ColorTag::ALL[(e.color_tag.clamp(0, 7)) as usize],
                        None => ColorTag::ALL[(variant.color_tag.clamp(0, 7)) as usize],
                    },
                    variant: number,
                    has_adjustments: !adjustments.is_default(),
                    trashed: stored.as_ref().is_some_and(|e| e.trashed),
                };
                st.entries.push(FileEntry {
                    info: info.clone(),
                    path: path.clone(),
                    rel_path: rel_path.clone(),
                    adjustments: adjustments.clone(),
                });
                if fresh {
                    imported.push((rel_path, adjustments, info));
                }
            }
        }
        // Persist everything in one catalog write: one save per image would rewrite
        // catalog.json once per file on a large catalog.
        if let Some(store) = &self.store {
            for (rel_path, adjustments, _) in &imported {
                if let Err(err) = store.save_adjustments(rel_path, adjustments) {
                    log::warn!("save adjustments {rel_path}: {err}");
                }
            }
        }
        for (rel_path, _, info) in &imported {
            st.catalog.upsert(Entry {
                rel_path: rel_path.clone(),
                rating: info.rating,
                color_tag: ColorTag::ALL.iter().position(|c| *c == info.color_tag).unwrap_or(0) as i8,
                keywords: info.keywords.clone(),
                variant: info.variant,
                updated_at: now_secs(),
                trashed: info.trashed,
            });
        }
        let catalog = st.catalog.clone();
        drop(st);
        if let Some(store) = &self.store {
            if let Err(err) = store.save_index(&catalog) {
                log::warn!("save catalog: {err}");
            }
        }
    }

    /// Re-reads the folder, keeping edits already made.
    pub fn rescan(&self) -> anyhow::Result<()> {
        let mut found = Vec::new();
        collect(&self.root, &self.root, 0, &mut found)?;
        found.sort_by(|a, b| a.0.cmp(&b.0));

        let mut st = self.state.lock();
        let previous: HashMap<String, (AdjustmentSet, i8, ColorTag, Vec<String>, u32)> = st
            .entries
            .iter()
            .map(|e| {
                (
                    e.rel_path.clone(),
                    (e.adjustments.clone(), e.info.rating, e.info.color_tag, e.info.keywords.clone(), e.info.variant),
                )
            })
            .collect();

        st.entries.clear();
        for (rel, abs) in found {
            let (w, h) = dimensions(&abs);
            let rel_path = rel.replace('\\', "/");
            let stored = previous.get(&rel_path);
            let mut adjustments = stored.map(|s| s.0.clone()).unwrap_or_default();
            // Restore edits saved in a previous process; keep unsaved in-memory
            // edits made before this rescan.
            if adjustments.is_default() {
                if let Some(store) = &self.store {
                    if let Ok(saved) = store.load_adjustments(&rel_path) {
                        adjustments = saved;
                    }
                }
            }
            let mut rating = stored.map(|s| s.1).unwrap_or(0);
            let mut color_tag = stored.map(|s| s.2).unwrap_or(ColorTag::None);
            let mut keywords = stored.map(|s| s.3.clone()).unwrap_or_default();
            let mut variant = stored.map(|s| s.4).unwrap_or(1);
            let mut trashed = false;

            // The catalog wins over the in-memory copy: it survives restarts.
            if let Some(entry) = st.catalog.entry(&rel_path) {
                rating = entry.rating;
                color_tag = ColorTag::ALL[(entry.color_tag.clamp(0, 7)) as usize];
                keywords = entry.keywords.clone();
                variant = variant.max(entry.variant);
                trashed = entry.trashed;
            }

            let (exif, iptc) = crate::exif::read_metadata(&abs);
            if keywords.is_empty() {
                keywords = iptc.keywords.clone();
            }
            let id = ImageId(st.next_id);
            st.next_id += 1;
            st.entries.push(FileEntry {
                info: ImageInfo {
                    id,
                    path: abs.clone(),
                    file_name: abs.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                    kind: file_kind(&abs),
                    width: w,
                    height: h,
                    file_size: std::fs::metadata(&abs).map(|m| m.len()).unwrap_or(0),
                    exif,
                    iptc,
                    keywords,
                    rating,
                    color_tag,
                    variant,
                    has_adjustments: !adjustments.is_default(),
                    trashed,
                },
                path: abs,
                rel_path,
                adjustments,
            });
        }
        Ok(())
    }

    fn add_file(&self, path: &Path) -> anyhow::Result<ImageInfo> {
        let (w, h) = dimensions(path);
        let rel = path.strip_prefix(&self.root).unwrap_or(path).to_string_lossy().replace('\\', "/");
        let mut st = self.state.lock();
        let id = ImageId(st.next_id);
        st.next_id += 1;
        let (exif, iptc) = crate::exif::read_metadata(path);
        let keywords = iptc.keywords.clone();
        let info = ImageInfo {
            id,
            path: path.to_path_buf(),
            file_name: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            kind: file_kind(path),
            width: w,
            height: h,
            file_size: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            exif,
            iptc,
            keywords,
            rating: 0,
            color_tag: ColorTag::None,
            variant: 1,
            has_adjustments: false,
            trashed: false,
        };
        st.entries.push(FileEntry {
            info: info.clone(),
            path: path.to_path_buf(),
            rel_path: rel,
            adjustments: AdjustmentSet::default(),
        });
        Ok(info)
    }

    fn source(&self, image: ImageId) -> anyhow::Result<Arc<Source>> {
        {
            let mut st = self.state.lock();
            if let Some(src) = st.sources.get(&image).cloned() {
                st.lru.retain(|i| *i != image);
                st.lru.push(image);
                return Ok(src);
            }
        }
        let (path, lens) = {
            let st = self.state.lock();
            let entry = st
                .entries
                .iter()
                .find(|e| e.info.id == image)
                .ok_or_else(|| anyhow::anyhow!("unknown image {image:?}"))?;
            (entry.path.clone(), crate::mock::lens_profile::LensShot::from_exif(&entry.info.exif))
        };
        // Decoding happens outside the lock so a slow file cannot block the UI.
        // A RAW keeps libraw's 16-bit linear samples; every other file and any
        // RAW libraw cannot demosaic arrives as 8-bit sRGB.
        let source = match raw16_source(&path) {
            Some(source) => source,
            None => {
                let img = if crate::raw::is_raw(&path) {
                    crate::raw::decode(&path)?
                } else {
                    image::open(&path)?.to_rgba8()
                };
                let (w, h) = fit_long_side(img.dimensions());
                let img = if img.dimensions() != (w, h) {
                    downscale_rgba(&img, w, h)
                } else {
                    img
                };
                Source::from_rgba(w, h, img.as_raw())
            }
        };
        let camera_preview =
            crate::raw::is_raw(&path) && crate::raw::from_sensor(&path) == Some(false);
        let source = Arc::new(source.with_lens_shot(lens).with_camera_preview(camera_preview));

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

    /// Cached preview frame at full requested pixel quality.
    fn preview_frame(&self, request: &PreviewRequest, _fast: bool) -> anyhow::Result<PreviewFrame> {
        let start = Instant::now();
        let src = self.source(request.image)?;
        let max = request.max_size;
        let fast = false;
        let adj = crate::styles::resolve(&request.adjustments);
        let key = preview_cache::key(request.image, &adj, request.region, max, request.bypass, request.mask_overlay, fast);
        if let Some(hit) = self.state.lock().preview.get(key) {
            return Ok(PreviewFrame {
                histogram: hit.histogram.clone(),
                image: hit.into_image(),
                generation: request.generation,
                elapsed_ms: start.elapsed().as_secs_f32() * 1000.0,
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
        Ok(PreviewFrame {
            image: frame.into_image(),
            histogram: out.histogram,
            generation: request.generation,
            elapsed_ms: start.elapsed().as_secs_f32() * 1000.0,
        })
    }

    fn persist(&self, rel_path: &str, adjustments: &AdjustmentSet, info: &ImageInfo) {
        let Some(store) = &self.store else { return };
        if let Err(err) = store.save_adjustments(rel_path, adjustments) {
            log::warn!("save adjustments {rel_path}: {err}");
            return;
        }
        let mut st = self.state.lock();
        st.catalog.upsert(Entry {
            rel_path: rel_path.to_string(),
            rating: info.rating,
            color_tag: ColorTag::ALL.iter().position(|c| *c == info.color_tag).unwrap_or(0) as i8,
            keywords: info.keywords.clone(),
            variant: info.variant,
            updated_at: now_secs(),
            trashed: info.trashed,
        });
        let catalog = st.catalog.clone();
        drop(st);
        if let Err(err) = store.save_index(&catalog) {
            log::warn!("save catalog: {err}");
        }
    }

    fn collection_tree(&self) -> Vec<Collection> {
        let (count, trashed) = {
            let st = self.state.lock();
            (st.entries.len(), st.entries.iter().filter(|e| e.info.trashed).count())
        };
        let folders: Vec<Collection> = self
            .folder_dirs()
            .iter()
            .enumerate()
            .map(|(index, (dir, n))| Collection {
                id: folder_id(index),
                name: if dir.is_empty() { self.root.display().to_string() } else { dir.clone() },
                kind: CollectionKind::Folder,
                count: *n,
                children: Vec::new(),
            })
            .collect();
        let node = |id, name: &str, kind, count, children| Collection { id, name: name.into(), kind, count, children };
        vec![
            node(1, "All Images", CollectionKind::AllImages, count, vec![]),
            node(2, "Recent Imports", CollectionKind::RecentImports, count.min(12), vec![]),
            node(3, "Recent Captures", CollectionKind::RecentCaptures, 0, vec![]),
            node(4, "Trash", CollectionKind::Trash, trashed, vec![]),
            node(10, "Folders", CollectionKind::Group, folders.len(), folders),
        ]
    }

    fn collection_images(&self, collection: u64) -> Option<Vec<ImageId>> {
        let index = collection.checked_sub(FOLDER_BASE)? as usize;
        let dir = self.folder_dirs().get(index)?.0.clone();
        let st = self.state.lock();
        Some(
            st.entries
                .iter()
                .filter(|e| match e.rel_path.rfind('/') {
                    Some(idx) => e.rel_path[..idx] == dir,
                    None => dir.is_empty(),
                })
                .map(|e| e.info.id)
                .collect(),
        )
    }

    /// Every immediate directory of the catalog with its image count, sorted.
    fn folder_dirs(&self) -> Vec<(String, usize)> {
        let st = self.state.lock();
        let mut by_dir: HashMap<String, usize> = HashMap::new();
        for entry in &st.entries {
            let dir = match entry.rel_path.rfind('/') {
                Some(idx) => entry.rel_path[..idx].to_string(),
                None => String::new(),
            };
            *by_dir.entry(dir).or_insert(0) += 1;
        }
        let mut dirs: Vec<(String, usize)> = by_dir.into_iter().collect();
        dirs.sort();
        dirs
    }
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn file_kind(path: &Path) -> FileKind {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("jpg" | "jpeg") => FileKind::Jpeg,
        Some("png") => FileKind::Png,
        Some("tif" | "tiff") => FileKind::Tiff,
        Some("webp") => FileKind::Raw,
        Some("heic" | "heif") => FileKind::Heif,
        Some("dng") => FileKind::Dng,
        _ => FileKind::Raw,
    }
}

/// Depth-limited so a symlink loop cannot hang the scan.
fn collect(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(String, PathBuf)>) -> anyhow::Result<()> {
    if depth > 12 {
        return Ok(());
    }
    let Ok(read) = std::fs::read_dir(dir) else { return Ok(()) };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "lumen-output" {
                continue;
            }
            collect(root, &path, depth + 1, out)?;
        } else if path.is_file() && is_supported(&path) {
            let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().into_owned();
            out.push((rel, path));
        }
    }
    Ok(())
}

fn is_supported(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp")
    ) || crate::raw::is_raw(path)
}

/// Pixel dimensions of a file: the sensor size for RAW, else the header size.
fn dimensions(path: &Path) -> (u32, u32) {
    if crate::raw::is_raw(path) {
        return crate::raw::sensor_dimensions(path).unwrap_or((0, 0));
    }
    image::image_dimensions(path).unwrap_or((0, 0))
}

fn rd_u16(b: &[u8], at: usize, le: bool) -> Option<u16> {
    let s = b.get(at..at + 2)?;
    Some(if le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
}

fn rd_u32(b: &[u8], at: usize, le: bool) -> Option<u32> {
    let s = b.get(at..at + 4)?;
    Some(if le {
        u32::from_le_bytes([s[0], s[1], s[2], s[3]])
    } else {
        u32::from_be_bytes([s[0], s[1], s[2], s[3]])
    })
}

/// ICC bytes embedded in a JPEG (APP2 `ICC_PROFILE` chunks) or a TIFF
/// (`ICCProfile`, tag 34675). Reads the head first; a TIFF profile that sits
/// past it falls back to a seek instead of reading the whole file.
fn embedded_icc(path: &Path) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = Vec::new();
    file.by_ref().take(1 << 20).read_to_end(&mut head).ok()?;

    if head.starts_with(&[0xFF, 0xD8]) {
        return jpeg_icc(&head);
    }
    if !(head.starts_with(b"II\x2a\x00") || head.starts_with(b"MM\x00\x2a")) {
        return None;
    }

    let le = head.starts_with(b"II");
    let ifd = rd_u32(&head, 4, le)? as usize;
    let count = rd_u16(&head, ifd, le)? as usize;
    for i in 0..count {
        let entry = ifd + 2 + i * 12;
        if rd_u16(&head, entry, le)? != 34675 {
            continue;
        }
        if rd_u16(&head, entry + 2, le)? != 7 {
            return None;
        }
        let len = rd_u32(&head, entry + 4, le)? as usize;
        if len <= 4 {
            return Some(head.get(entry + 8..entry + 8 + len)?.to_vec());
        }
        let at = rd_u32(&head, entry + 8, le)? as usize;
        if let Some(bytes) = head.get(at..at.checked_add(len)?) {
            return Some(bytes.to_vec());
        }
        file.seek(SeekFrom::Start(at as u64)).ok()?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).ok()?;
        return Some(bytes);
    }
    None
}

/// Concatenate a JPEG's APP2 `ICC_PROFILE` chunks in sequence order.
fn jpeg_icc(data: &[u8]) -> Option<Vec<u8>> {
    let mut at = 2;
    let mut chunks: Vec<(u8, &[u8])> = Vec::new();
    while at + 4 <= data.len() && data[at] == 0xFF {
        let marker = data[at + 1];
        at += 2;
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        if marker == 0xDA || marker == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([data[at], data[at + 1]]) as usize;
        let end = at.checked_add(len)?;
        if len < 2 || end > data.len() {
            break;
        }
        let seg = &data[at + 2..end];
        if marker == 0xE2 && seg.len() > 14 && seg.starts_with(b"ICC_PROFILE\0") {
            chunks.push((seg[12], &seg[14..]));
        }
        at = end;
    }
    if chunks.is_empty() {
        return None;
    }
    chunks.sort_by_key(|(seq, _)| *seq);
    Some(chunks.into_iter().flat_map(|(_, b)| b.iter().copied()).collect())
}

/// The model keyword of a mapped profile name: the part before its first '-'
/// ("NikonZ6III-ProStandard" -> "NIKONZ6III").
fn profile_keyword(name: &str) -> String {
    name.split('-')
        .next()
        .unwrap_or(name)
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

/// A camera profile is only re-applied when the file is still in the camera's
/// colour space: the embedded ICC description names the model that
/// `for_camera` maps to. An output profile (sRGB, Adobe RGB) never matches, so
/// those files are left untouched and are not converted twice.
fn camera_icc(path: &Path, model: Option<&str>) -> Option<&'static str> {
    let mapped = model.and_then(crate::camera_profile::for_camera)?;
    let keyword = profile_keyword(mapped);
    if keyword.is_empty() {
        return None;
    }
    let description = crate::icc::parse_icc(&embedded_icc(path)?).ok()?.description;
    let description: String = description
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase();
    description.contains(&keyword).then_some(mapped)
}

impl ImageEngine for FileEngine {
    fn name(&self) -> String {
        format!("Lumen File Engine ({})", self.root.display())
    }

    fn images(&self) -> Vec<ImageInfo> {
        self.state.lock().entries.iter().map(|e| e.info.clone()).collect()
    }

    fn import(&self, paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
        let mut added = Vec::new();
        for path in paths {
            if path.is_dir() && !path.starts_with(&self.root) {
                let mut files = Vec::new();
                collect(path, path, 0, &mut files)?;
                for (_, abs) in files {
                    if let Ok(info) = self.add_file(&abs) {
                        added.push(info);
                    }
                }
            } else if path.is_file() && is_supported(path) {
                added.push(self.add_file(path)?);
            }
        }
        Ok(added)
    }

    fn collections(&self) -> Vec<Collection> {
        self.collection_tree()
    }

    // Without this the trait default (`None`) hides the folder membership from the
    // interface, which only sees `dyn ImageEngine`.
    fn collection_images(&self, collection: u64) -> Option<Vec<ImageId>> {
        FileEngine::collection_images(self, collection)
    }

    fn save_user_collections(&self, collections: &[crate::UserCollection]) -> anyhow::Result<()> {
        match &self.store {
            Some(store) => store.save_user_collections(collections).map_err(Into::into),
            None => Ok(()),
        }
    }

    fn set_trashed(&self, ids: &[ImageId], trashed: bool) -> anyhow::Result<()> {
        let catalog = {
            let mut st = self.state.lock();
            let mut changed = Vec::new();
            for entry in st.entries.iter_mut().filter(|e| ids.contains(&e.info.id)) {
                entry.info.trashed = trashed;
                changed.push((entry.rel_path.clone(), entry.info.clone()));
            }
            for (rel, info) in changed {
                st.catalog.upsert(Entry {
                    rel_path: rel,
                    rating: info.rating,
                    color_tag: ColorTag::ALL.iter().position(|c| *c == info.color_tag).unwrap_or(0) as i8,
                    keywords: info.keywords.clone(),
                    variant: info.variant,
                    updated_at: now_secs(),
                    trashed,
                });
            }
            st.catalog.clone()
        };
        if let Some(store) = &self.store {
            store.save_index(&catalog)?;
        }
        Ok(())
    }

    fn update_info(&self, info: &ImageInfo) -> anyhow::Result<()> {
        let (rel, catalog) = {
            let mut st = self.state.lock();
            let entry = st
                .entries
                .iter_mut()
                .find(|e| e.info.id == info.id)
                .ok_or_else(|| anyhow::anyhow!("unknown image"))?;
            entry.info = info.clone();
            let rel = entry.rel_path.clone();
            st.catalog.upsert(Entry {
                rel_path: rel.clone(),
                rating: info.rating,
                color_tag: ColorTag::ALL.iter().position(|c| *c == info.color_tag).unwrap_or(0) as i8,
                keywords: info.keywords.clone(),
                variant: info.variant,
                updated_at: now_secs(),
                trashed: info.trashed,
            });
            (rel.clone(), st.catalog.clone())
        };
        let _ = rel;
        if let Some(store) = &self.store {
            if let Err(err) = store.save_index(&catalog) {
                log::warn!("save catalog: {err}");
            }
        }
        Ok(())
    }

    fn load_adjustments(&self, image: ImageId) -> AdjustmentSet {
        let st = self.state.lock();
        let Some(entry) = st.entries.iter().find(|e| e.info.id == image) else {
            return AdjustmentSet::default();
        };
        let mut adjustments = entry.adjustments.clone();
        // RAW defaults follow Capture One: its own camera profile and film curve,
        // plus the noise reduction it would start at for this ISO. JPEG/TIFF frames
        // are already display-referred, so they get none of this.
        // Only sensor-decoded RAW gets the camera profile and film curve; the
        // embedded preview is already processed by the camera.
        if crate::raw::is_raw(&entry.info.path)
            && crate::raw::from_sensor(&entry.info.path) != Some(false)
        {
            let model = entry.info.exif.camera_model.as_deref();
            if adjustments.base.film_curve.is_none() {
                adjustments.base.film_curve = model.and_then(crate::camera_curve::default_curve);
            }
            if adjustments.base.icc_profile.is_none() {
                adjustments.base.icc_profile = model
                    .and_then(crate::camera_profile::for_camera)
                    .filter(|name| crate::camera_profile::is_available(name))
                    .map(str::to_string);
            }
            if adjustments.noise_reduction == crate::NoiseReduction::default() {
                if let Some(iso) = entry.info.exif.iso {
                    adjustments.noise_reduction = crate::noise_defaults::for_iso(iso).into();
                }
            }
        } else if !crate::raw::is_raw(&entry.info.path) && adjustments.base.icc_profile.is_none() {
            // Display-referred JPEG/TIFF keeps its embedded profile. Only a file
            // still carrying the camera's own colour space gets the camera
            // profile added here (see `camera_icc`).
            adjustments.base.icc_profile = camera_icc(
                &entry.info.path,
                entry.info.exif.camera_model.as_deref(),
            )
            .filter(|name| crate::camera_profile::is_available(name))
            .map(str::to_string);
        }
        adjustments
    }

    fn save_adjustments(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<()> {
        let (rel, info) = {
            let mut st = self.state.lock();
            let entry = st
                .entries
                .iter_mut()
                .find(|e| e.info.id == image)
                .ok_or_else(|| anyhow::anyhow!("unknown image"))?;
            entry.adjustments = adjustments.clone();
            entry.info.has_adjustments = !adjustments.is_default();
            (entry.rel_path.clone(), entry.info.clone())
        };
        self.persist(&rel, adjustments, &info);
        Ok(())
    }

    fn new_variant(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo> {
        let mut st = self.state.lock();
        let next = st.next_id;
        let pos = st
            .entries
            .iter()
            .position(|e| e.info.id == image)
            .ok_or_else(|| anyhow::anyhow!("unknown image"))?;
        let mut info = st.entries[pos].info.clone();
        info.id = ImageId(next);
        info.variant += 1;
        info.has_adjustments = !adjustments.is_default();
        st.next_id += 1;
        let entry = FileEntry {
            info: info.clone(),
            path: st.entries[pos].path.clone(),
            rel_path: st.entries[pos].rel_path.clone(),
            adjustments: adjustments.clone(),
        };
        st.entries.insert(pos + 1, entry);
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
        let probe = process::render(
            &src,
            &AdjustmentSet { geometry: current.geometry.clone(), ..Default::default() },
            NormRect::FULL,
            256,
            false,
            None,
        );
        let h = &probe.histogram;
        let total: u32 = h.luma.iter().sum();
        let mean = h.luma.iter().enumerate().map(|(i, c)| i as f32 * *c as f32).sum::<f32>() / total.max(1) as f32 / 255.0;
        let mut adj = current.clone();
        let all = target == AutoTarget::All;
        if all || target == AutoTarget::Exposure {
            adj.exposure.exposure = (((0.46f32 / mean.max(0.02f32)).log2()) * 0.8f32).clamp(-2.0f32, 2.0f32);
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
            let turn = self.auto_rotate(image)?;
            adj.geometry.orientation = crate::autorotate::add_quarter(adj.geometry.orientation, turn);
            if turn != 0 {
                adj.geometry.rotation = 0.0;
            }
        }
        if target == AutoTarget::Keystone {
            // Existing keystone is excluded so the suggestion is absolute.
            let mut geometry = current.geometry.clone();
            geometry.keystone = crate::Keystone::default();
            let frame = process::render(
                &src,
                &AdjustmentSet { geometry, ..Default::default() },
                NormRect::FULL,
                768,
                false,
                None,
            );
            let suggestion = crate::keystone::detect(&frame.rgba, frame.width, frame.height);
            adj.geometry.keystone = crate::Keystone {
                vertical: suggestion.vertical,
                horizontal: suggestion.horizontal,
                amount: 100.0,
                aspect: current.geometry.keystone.aspect,
            };
        }
        Ok(AutoResult { adjustments: adj })
    }

    fn pick_white_balance(&self, image: ImageId, point: [f32; 2]) -> anyhow::Result<(f32, f32)> {
        let src = self.source(image)?;
        let c = process::sample_point(&src, &AdjustmentSet::default(), point);
        let warm = (c[0] - c[2]) / (c[0] + c[1] + c[2]).max(1e-3);
        let green = (c[1] - (c[0] + c[2]) / 2.0) / (c[0] + c[1] + c[2]).max(1e-3);
        Ok((((5600.0f32 - warm * 9000.0f32).clamp(2000.0f32, 15000.0f32)), ((green * 120.0f32).clamp(-50.0f32, 50.0f32))))
    }

    fn match_look(&self, reference: ImageId, target: ImageId) -> anyhow::Result<crate::MatchLook> {
        // 512 px is plenty for histogram statistics and keeps the second decode cheap.
        let probe = |id| -> anyhow::Result<process::Rendered> {
            let src = self.source(id)?;
            Ok(process::render(&src, &AdjustmentSet::default(), NormRect::FULL, 512, false, None))
        };
        let r = probe(reference)?;
        let t = probe(target)?;
        crate::matchlook::match_look(&r.rgba, r.width, r.height, &t.rgba, t.width, t.height)
    }

    fn compute_ai_mask(&self, image: ImageId, kind: AiMaskKind) -> anyhow::Result<String> {
        let (rel_path, variant) = {
            let st = self.state.lock();
            let entry = st
                .entries
                .iter()
                .find(|e| e.info.id == image)
                .ok_or_else(|| anyhow::anyhow!("unknown image {image:?}"))?;
            (entry.rel_path.clone(), entry.info.variant)
        };
        let key = crate::mask_store::key(&rel_path, variant, kind);
        if crate::aimask::lookup(&key).is_some() {
            return Ok(key);
        }
        if let Some(plane) = crate::mask_store::load(&self.root, &key) {
            crate::aimask::store(&key, plane);
            return Ok(key);
        }
        let src = self.source(image)?;
        let geometry = self.load_adjustments(image).geometry;
        // The models are small; 1024 px on the long side is plenty and keeps
        // the mask under a few megabytes for a 50 MP frame.
        let (rgba, w, h) = crate::mock::process::ai_input_rgba(&src, &geometry, 1024);
        let start = Instant::now();
        let mask = match kind {
            AiMaskKind::Subject | AiMaskKind::Person | AiMaskKind::Clothing => {
                crate::aimask::subject_mask(&rgba, w, h)
            }
            AiMaskKind::Skin => crate::aimask::skin_mask(&rgba, w, h),
            AiMaskKind::Hair => crate::aimask::hair_mask(&rgba, w, h),
            AiMaskKind::Sky => crate::aimask::sky_mask(&rgba, w, h),
            AiMaskKind::Background => crate::aimask::subject_mask(&rgba, w, h)
                .map(|mut mask| {
                    for v in &mut mask {
                        *v = 1.0 - *v;
                    }
                    mask
                }),
            other => anyhow::bail!(
                "Capture One's six-class people parser has no {other:?} class (background, hair, \
                 body skin, face skin, clothing, held object); no model segments it"
            ),
        };
        let mask = match mask {
            Ok(mask) => mask,
            Err(err) => {
                log::warn!("AI mask {kind:?} unavailable: {err:#}");
                return Err(err);
            }
        };
        let plane = std::sync::Arc::new(crate::aimask::MaskPlane::new(w as usize, h as usize, mask));
        log::info!(
            "AI mask {kind:?} for image {}: {}x{}, {:.0} ms, foreground {:.1}%",
            image.0,
            plane.w,
            plane.h,
            start.elapsed().as_secs_f32() * 1000.0,
            plane.foreground_fraction() * 100.0
        );
        crate::aimask::store(&key, plane.clone());
        if let Err(err) = crate::mask_store::save(&self.root, &key, &plane) {
            log::warn!("AI mask {kind:?} for image {} not saved: {err:#}", image.0);
        }
        Ok(key)
    }

    fn eye_report(&self, image: ImageId) -> anyhow::Result<crate::EyeReport> {
        if let Some(report) = self.eyes.lock().get(&image) {
            return Ok(*report);
        }
        let src = self.source(image)?;
        let geometry = self.load_adjustments(image).geometry;
        let (rgba, w, h) = crate::mock::process::ai_input_rgba(&src, &geometry, 1024);
        let start = Instant::now();
        let report = match crate::any_eye_closed(&rgba, w, h) {
            Ok(report) => report,
            Err(err) => {
                log::warn!("closed-eye check for image {} unavailable: {err:#}", image.0);
                return Err(err);
            }
        };
        log::info!(
            "closed-eye check for image {}: {} eyes, {} closed, worst p={:.2}, {:.0} ms",
            image.0,
            report.evaluated,
            report.closed,
            report.worst_confidence,
            start.elapsed().as_secs_f32() * 1000.0
        );
        self.eyes.lock().insert(image, report);
        Ok(report)
    }
    fn auto_rotate(&self, image: ImageId) -> anyhow::Result<u16> {
        let src = self.source(image)?;
        // The classifier reads the frame as the user currently sees it, so an
        // orientation already applied is baked in and the answer is the extra turn.
        let geometry = self.load_adjustments(image).geometry;
        let (rgba, w, h) = crate::mock::process::ai_input_rgba(&src, &geometry, 512);
        let start = Instant::now();
        let quadrant = crate::autorotate::detect_quadrant(&rgba, w, h)?;
        log::info!(
            "auto rotation for image {}: {:?}, {:.0} ms",
            image.0,
            quadrant,
            start.elapsed().as_secs_f32() * 1000.0
        );
        Ok(quadrant.degrees())
    }

    fn styles(&self) -> Vec<StyleInfo> {
        // A real Capture One install ships its own presets and styles; fall back
        // to the built-in demo list only when none are found.
        let installed = crate::presets::styles();
        let mut out = if installed.is_empty() { crate::styles::all() } else { installed };
        for style in &self.state.lock().imported_styles {
            out.retain(|s| s.name != style.name);
            out.push(style.clone());
        }
        out
    }

    fn film_curves(&self) -> Vec<String> {
        static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        NAMES.get_or_init(|| crate::fcrv_index::scan(&crate::curve_dir()).rows.into_iter().map(|r| r.name).collect()).clone()
    }

    fn import_style(&self, path: &Path) -> anyhow::Result<StyleInfo> {
        let parsed = crate::load_style(path)?;
        let style = StyleInfo { name: parsed.name, group: "Imported".into(), adjustments: parsed.adjustments };
        let mut st = self.state.lock();
        st.imported_styles.retain(|s| s.name != style.name);
        st.imported_styles.push(style.clone());
        Ok(style)
    }

    fn icc_profiles(&self) -> Vec<String> {
        ["sRGB IEC61966-2.1", "Adobe RGB (1998)", "Display P3", "ProPhoto RGB"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Camera ranges and defaults from `coreConfig.cdx`, plus the frame's own
    /// values (EXIF exposure, Nikon MakerNote white balance).
    fn param_overrides(&self, image: ImageId) -> crate::ParamOverrides {
        let defaults = self
            .camera_defaults
            .get_or_init(|| {
                let path = std::env::var_os("LUMEN_CORE_CONFIG")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from(r"C:\Program Files\Capture One\Capture One\coreConfig.cdx"));
                match crate::CameraDefaults::load(&path) {
                    Ok(d) => Some(d),
                    Err(err) => {
                        log::info!("coreConfig unavailable ({}): {err}", path.display());
                        None
                    }
                }
            })
            .as_ref();

        let (model, iso, path) = {
            let st = self.state.lock();
            let Some(entry) = st.entries.iter().find(|e| e.info.id == image) else {
                return crate::ParamOverrides::default();
            };
            (entry.info.exif.camera_model.clone(), entry.info.exif.iso, entry.path.clone())
        };

        let mut overrides = defaults.map_or_else(Default::default, |d| d.overrides(model.as_deref(), iso));
        if let Ok(frame) = crate::read_nikon_info_block(&path) {
            frame.apply_to(&mut overrides);
        }
        overrides
    }

    fn lens_profiles(&self) -> Vec<String> {
        Vec::new()
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
                    info.variant as usize,
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
                    info.variant as usize,
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

    fn tether_status(&self) -> TetherStatus {
        tether::probe()
    }
    fn camera_status(&self) -> CameraStatus {
        CameraStatus::Disconnected
    }
    fn capture(&self) -> anyhow::Result<ImageInfo> {
        let frame = tether::capture_frame()?;
        let dir = self.root.join("Capture");
        std::fs::create_dir_all(&dir)?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = dir.join(format!("CAP_{stamp}.jpg"));
        std::fs::write(&path, &frame.jpeg)?;
        self.add_file(&path)
    }
    fn live_view_frame(&self, width: u32) -> Option<RgbaImage> {
        match tether::preview_frame(width) {
            Ok(frame) => Some(frame),
            Err(err) => {
                log::debug!("live view: {err:#}");
                None
            }
        }
    }
    /// libgphoto2 exposes focus only through the camera's config tree, which this
    /// binding does not read yet, so a focus step is a no-op.
    fn live_view_focus(&self, _steps: i32) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_scales_the_long_side_and_keeps_the_aspect_ratio() {
        // 6064x4040 is a Nikon Z6_3 frame: downscaling must keep 3:2, not square it.
        let (w, h) = (6064u32, 4040u32);
        let long = w.max(h);
        assert!(long > SOURCE_LONG_SIDE);
        let scale = SOURCE_LONG_SIDE as f32 / long as f32;
        let short = ((w.min(h) as f32) * scale).round().max(1.0) as u32;
        let size = if w >= h { (SOURCE_LONG_SIDE, short) } else { (short, SOURCE_LONG_SIDE) };
        assert_eq!(size, (4096, 2729));
        let ratio = size.0 as f32 / size.1 as f32;
        assert!((ratio - 1.5).abs() < 0.01, "{ratio}");
    }
    #[test]
    fn raw_files_are_supported() {
        for ext in ["nef", "cr3", "arw", "dng"] {
            assert!(is_supported(Path::new(&format!("shot.{ext}"))), "{ext}");
        }
        for ext in ["jpg", "png", "tif", "webp"] {
            assert!(is_supported(Path::new(&format!("shot.{ext}"))), "{ext}");
        }
        assert!(!is_supported(Path::new("notes.txt")));
    }

    #[test]
    fn downscale_averages_source_blocks_and_keeps_the_size_when_equal() {
        // Left half dark, right half bright: a 2x box average must split the difference.
        let mut img = image::RgbaImage::new(4, 4);
        for (x, _, p) in img.enumerate_pixels_mut() {
            *p = if x < 2 { image::Rgba([0, 0, 0, 255]) } else { image::Rgba([200, 100, 40, 255]) };
        }
        let small = downscale_rgba(&img, 2, 2);
        assert_eq!(small.dimensions(), (2, 2));
        assert_eq!(small.get_pixel(0, 0), &image::Rgba([0, 0, 0, 255]));
        assert_eq!(small.get_pixel(1, 1), &image::Rgba([200, 100, 40, 255]));
        assert_eq!(downscale_rgba(&img, 4, 4).as_raw(), img.as_raw());
    }

    #[test]
    fn integrated_preview_quality_and_style_import() {
        let dir = std::env::temp_dir().join(format!("aurora-integration-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::from_fn(1600, 32, |x, y| image::Rgba([(x % 251) as u8, (y * 7) as u8, 90, 255]))
            .save(dir.join("frame.png")).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let image = engine.images()[0].id;
        let mut request = PreviewRequest { image, adjustments: AdjustmentSet::default(), region: NormRect::FULL,
            max_size: 1600, quality: RenderQuality::Draft, bypass: false, mask_overlay: None, generation: 1 };
        let draft = engine.render_preview(&request).unwrap();
        request.quality = RenderQuality::Final;
        let final_frame = engine.render_preview(&request).unwrap();
        let fast = engine.render_preview_fast(&request, true).unwrap();
        assert_eq!(draft.image.width, 1600);
        assert_eq!(draft.image.pixels, final_frame.image.pixels);
        assert_eq!(draft.image.pixels, fast.image.pixels);
        let path = dir.join("test.costyle");
        std::fs::write(&path, r#"<SL><E K="Name" V="Integration"/><E K="Exposure" V="0.5"/></SL>"#).unwrap();
        let style = engine.import_style(&path).unwrap();
        assert_eq!(style.name, "Integration");
        assert_eq!(style.adjustments.exposure.exposure, 0.5);
        engine.import_style(&path).unwrap();
        assert_eq!(engine.styles().iter().filter(|s| s.name == style.name).count(), 1);
        std::fs::write(&path, "invalid style").unwrap();
        assert!(engine.import_style(&path).is_err());
        drop(engine);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn opening_an_empty_folder_yields_no_images() {
        let dir = std::env::temp_dir().join(format!("lumen-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        assert!(engine.images().is_empty());
        assert!(engine.thumbnail(ImageId(1), &AdjustmentSet::default(), 256).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn categories_without_a_parser_class_return_an_error() {
        use crate::{AiMaskKind, ImageEngine};

        let dir = std::env::temp_dir().join(format!("lumen-maskcat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::from_pixel(32, 24, image::Rgba([120, 90, 70, 255]))
            .save(dir.join("frame.png"))
            .unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let image = engine.images()[0].id;
        for kind in [AiMaskKind::Eyes, AiMaskKind::Lips] {
            let err = engine.compute_ai_mask(image, kind).unwrap_err();
            assert!(err.to_string().contains(&format!("no {kind:?} class")), "{err}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_folder_mixing_formats_lists_only_supported_files() {
        let dir = std::env::temp_dir().join(format!("lumen-mixed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        image::RgbImage::from_pixel(8, 6, image::Rgb([10, 20, 30])).save(dir.join("a.jpg")).unwrap();
        image::RgbaImage::from_pixel(3, 5, image::Rgba([40, 50, 60, 255])).save(dir.join("sub/b.png")).unwrap();
        std::fs::write(dir.join("notes.txt"), b"not an image").unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let images = engine.images();
        let names: Vec<&str> = images.iter().map(|i| i.file_name.as_str()).collect();
        assert_eq!(names, vec!["a.jpg", "b.png"]);
        assert_eq!((images[0].kind, images[1].kind), (FileKind::Jpeg, FileKind::Png));
        assert_eq!((images[0].width, images[0].height), (8, 6));
        assert_eq!((images[1].width, images[1].height), (3, 5));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn saved_ai_masks_are_back_in_memory_after_reopening() {
        use crate::{AiMask, Layer, Mask, MaskOp, MaskShape};

        let dir = std::env::temp_dir().join(format!("lumen-mask-reopen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::from_pixel(8, 6, image::Rgba([50, 60, 70, 255]))
            .save(dir.join("frame.png"))
            .unwrap();
        let key = crate::mask_store::key("frame.png", 1, AiMaskKind::Subject);
        crate::mask_store::save(&dir, &key, &crate::aimask::MaskPlane::new(4, 3, vec![1.0; 12])).unwrap();

        {
            let engine = FileEngine::open(&dir).unwrap();
            let id = engine.images()[0].id;
            let mut adjustments = AdjustmentSet::default();
            adjustments.layers.push(Layer {
                mask: Mask {
                    shapes: vec![MaskShape::Ai(AiMask {
                        op: MaskOp::Add,
                        kind: AiMaskKind::Subject,
                        cache_key: Some(key.clone()),
                    })],
                    ..Default::default()
                },
                ..Default::default()
            });
            engine.save_adjustments(id, &adjustments).unwrap();
        }

        // A fresh engine has none of the previous process's memory cache.
        let reopened = FileEngine::open(&dir).unwrap();
        assert!(reopened.images()[0].has_adjustments);
        let plane = crate::aimask::lookup(&key).expect("preloaded from .lumen/masks");
        assert_eq!((plane.w, plane.h), (4, 3));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn trash_survives_reopening_the_folder() {
        let dir = std::env::temp_dir().join(format!("lumen-trash-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([20, 40, 60, 255])).save(dir.join("a.png")).unwrap();

        let id = {
            let engine = FileEngine::open(&dir).unwrap();
            let id = engine.images()[0].id;
            engine.set_trashed(&[id], true).unwrap();
            assert!(engine.images()[0].trashed);
            id
        };

        let reopened = FileEngine::open(&dir).unwrap();
        assert!(reopened.images()[0].trashed, "trash is persisted in .lumen/catalog.json");
        let trash = reopened.collections().into_iter().find(|c| c.kind == CollectionKind::Trash).unwrap();
        assert_eq!(trash.count, 1);
        let folders: Vec<u64> = reopened
            .collections()
            .into_iter()
            .flat_map(|c| c.children)
            .map(|c| c.id)
            .collect();
        assert_eq!(folders.len(), 1);
        assert_eq!(reopened.collection_images(folders[0]).unwrap(), vec![id]);
        std::fs::remove_dir_all(&dir).unwrap();
    }


    fn icc_with_description(description: &str) -> Vec<u8> {
        let text = description.as_bytes();
        let mut desc = Vec::new();
        desc.extend_from_slice(b"desc");
        desc.extend_from_slice(&[0; 4]);
        desc.extend_from_slice(&((text.len() + 1) as u32).to_be_bytes());
        desc.extend_from_slice(text);
        desc.push(0);
        let xyz = |x: i32, y: i32, z: i32| {
            let mut d = Vec::new();
            d.extend_from_slice(b"XYZ ");
            d.extend_from_slice(&[0; 4]);
            for v in [x, y, z] {
                d.extend_from_slice(&(v * 65536).to_be_bytes());
            }
            d
        };
        let mut trc = Vec::new();
        trc.extend_from_slice(b"curv");
        trc.extend_from_slice(&[0; 4]);
        trc.extend_from_slice(&1u32.to_be_bytes());
        trc.extend_from_slice(&0x0100u16.to_be_bytes());
        let payloads: Vec<(&[u8; 4], Vec<u8>)> = vec![
            (b"desc", desc),
            (b"rXYZ", xyz(1, 0, 0)),
            (b"gXYZ", xyz(0, 1, 0)),
            (b"bXYZ", xyz(0, 0, 1)),
            (b"rTRC", trc),
        ];
        let mut out = vec![0u8; 132];
        out[12..16].copy_from_slice(b"mntr");
        out[16..20].copy_from_slice(b"RGB ");
        out[20..24].copy_from_slice(b"XYZ ");
        out[36..40].copy_from_slice(b"acsp");
        out[128..132].copy_from_slice(&(payloads.len() as u32).to_be_bytes());
        let mut offset = 128 + 4 + payloads.len() * 12;
        for (sig, data) in &payloads {
            out.extend_from_slice(*sig);
            out.extend_from_slice(&(offset as u32).to_be_bytes());
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            offset += (data.len() + 3) & !3;
        }
        for (_, data) in &payloads {
            out.extend_from_slice(data);
            while out.len() % 4 != 0 {
                out.push(0);
            }
        }
        let total = out.len() as u32;
        out[0..4].copy_from_slice(&total.to_be_bytes());
        out
    }

    /// Append tags to a TIFF by pointing IFD0 at a new table at the end; the
    /// original entries and their data offsets stay valid.
    fn tiff_with_tags(mut tiff: Vec<u8>, tags: &[(u16, u16, u32, Vec<u8>)]) -> Vec<u8> {
        let le = tiff.starts_with(b"II");
        let u16b = |v: u16| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let u32b = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let r16 = |b: &[u8], at: usize| {
            if le { u16::from_le_bytes([b[at], b[at + 1]]) } else { u16::from_be_bytes([b[at], b[at + 1]]) }
        };
        let r32 = |b: &[u8], at: usize| {
            if le {
                u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
            } else {
                u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
            }
        };
        let ifd = r32(&tiff, 4) as usize;
        let old_count = r16(&tiff, ifd) as usize;
        let mut entries: Vec<(u16, [u8; 12])> = (0..old_count)
            .map(|i| {
                let at = ifd + 2 + i * 12;
                let mut entry = [0u8; 12];
                entry.copy_from_slice(&tiff[at..at + 12]);
                (r16(&tiff, at), entry)
            })
            .collect();
        if tiff.len() % 2 == 1 {
            tiff.push(0);
        }
        let new_ifd = tiff.len();
        let count = old_count + tags.len();
        let data_at = new_ifd + 2 + count * 12 + 4;
        let mut data: Vec<u8> = Vec::new();
        for (tag, typ, field_count, value) in tags {
            let mut entry = [0u8; 12];
            entry[0..2].copy_from_slice(&u16b(*tag));
            entry[2..4].copy_from_slice(&u16b(*typ));
            entry[4..8].copy_from_slice(&u32b(*field_count));
            if value.len() <= 4 {
                entry[8..8 + value.len()].copy_from_slice(value);
            } else {
                entry[8..12].copy_from_slice(&u32b((data_at + data.len()) as u32));
                data.extend_from_slice(value);
                if data.len() % 2 == 1 {
                    data.push(0);
                }
            }
            entries.push((*tag, entry));
        }
        entries.sort_by_key(|(tag, _)| *tag);
        let mut out = tiff;
        out[4..8].copy_from_slice(&u32b(new_ifd as u32));
        out.extend_from_slice(&u16b(count as u16));
        for (_, entry) in &entries {
            out.extend_from_slice(entry);
        }
        out.extend_from_slice(&u32b(0));
        out.extend_from_slice(&data);
        out
    }

    fn app1_exif(model: &str) -> Vec<u8> {
        let text = format!("{model}\0");
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II\x2a\x00");
        tiff.extend_from_slice(&8u32.to_le_bytes());
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x0110u16.to_le_bytes());
        tiff.extend_from_slice(&2u16.to_le_bytes());
        tiff.extend_from_slice(&(text.len() as u32).to_le_bytes());
        tiff.extend_from_slice(&26u32.to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());
        tiff.extend_from_slice(text.as_bytes());
        let mut segment = vec![0xFF, 0xE1];
        segment.extend_from_slice(&((2 + 6 + tiff.len()) as u16).to_be_bytes());
        segment.extend_from_slice(b"Exif\0\0");
        segment.extend_from_slice(&tiff);
        segment
    }

    fn app2_icc(icc: &[u8]) -> Vec<u8> {
        let mut segment = vec![0xFF, 0xE2];
        segment.extend_from_slice(&((2 + 14 + icc.len()) as u16).to_be_bytes());
        segment.extend_from_slice(b"ICC_PROFILE\0");
        segment.push(1);
        segment.push(1);
        segment.extend_from_slice(icc);
        segment
    }

    fn jpeg_with_segments(mut jpeg: Vec<u8>, segments: &[Vec<u8>]) -> Vec<u8> {
        let mut out: Vec<u8> = jpeg.drain(..2).collect();
        for segment in segments {
            out.extend_from_slice(segment);
        }
        out.extend_from_slice(&jpeg);
        out
    }

    fn icc_temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-icc-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_tiff_still_in_the_camera_colour_space_gets_the_camera_profile() {
        let dir = icc_temp_dir("tiff-camera");
        let path = dir.join("cam.tif");
        image::RgbImage::from_pixel(2, 2, image::Rgb([90, 80, 70])).save(&path).unwrap();
        let icc = icc_with_description("Nikon Z 6III ProStandard");
        let tiff = tiff_with_tags(
            std::fs::read(&path).unwrap(),
            &[(0x0110, 2, 10, b"NIKON Z6_3\0".to_vec()), (34675, 7, icc.len() as u32, icc)],
        );
        std::fs::write(&path, tiff).unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let images = engine.images();
        assert_eq!(images.len(), 1);
        assert_eq!(
            engine.load_adjustments(images[0].id).base.icc_profile.as_deref(),
            Some("NikonZ6III-ProStandard"),
            "the embedded ICC names the camera, so the camera profile applies"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_jpeg_already_in_srgb_keeps_its_own_profile() {
        let dir = icc_temp_dir("jpeg-srgb");
        let path = dir.join("out.jpg");
        image::RgbImage::from_pixel(2, 2, image::Rgb([90, 80, 70])).save(&path).unwrap();
        let jpg = jpeg_with_segments(
            std::fs::read(&path).unwrap(),
            &[
                app1_exif("NIKON Z6_3"),
                app2_icc(&icc_with_description("sRGB IEC61966-2.1")),
            ],
        );
        std::fs::write(&path, jpg).unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let images = engine.images();
        assert_eq!(images.len(), 1);
        assert_eq!(
            engine.load_adjustments(images[0].id).base.icc_profile, None,
            "an sRGB output profile must not be converted again"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_jpeg_without_an_icc_profile_gets_no_camera_profile() {
        let dir = icc_temp_dir("jpeg-none");
        let path = dir.join("out.jpg");
        image::RgbImage::from_pixel(2, 2, image::Rgb([90, 80, 70])).save(&path).unwrap();
        let jpg = jpeg_with_segments(std::fs::read(&path).unwrap(), &[app1_exif("NIKON Z6_3")]);
        std::fs::write(&path, jpg).unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let images = engine.images();
        assert_eq!(images.len(), 1);
        assert_eq!(engine.load_adjustments(images[0].id).base.icc_profile, None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn raw_files_still_get_their_camera_defaults() {
        let dir = icc_temp_dir("raw");
        let tiff_path = dir.join("shot.tif");
        image::RgbImage::from_pixel(2, 2, image::Rgb([90, 80, 70])).save(&tiff_path).unwrap();
        let icc = icc_with_description("sRGB IEC61966-2.1");
        let tiff = tiff_with_tags(
            std::fs::read(&tiff_path).unwrap(),
            &[(0x0110, 2, 10, b"NIKON Z6_3\0".to_vec()), (34675, 7, icc.len() as u32, icc)],
        );
        std::fs::write(dir.join("shot.nef"), tiff).unwrap();
        std::fs::remove_file(&tiff_path).unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let images = engine.images();
        assert_eq!(images.len(), 1);
        assert_eq!(
            engine.load_adjustments(images[0].id).base.icc_profile.as_deref(),
            Some("NikonZ6III-ProStandard"),
            "RAW keeps the old sensor path: the embedded sRGB profile is irrelevant"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/file_engine-tests.rs");
}
