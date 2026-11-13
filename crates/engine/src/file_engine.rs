//! A catalog backed by real files on disk, with edits persisted in `.lumen`.
//!
//! [`crate::MockEngine`] serves procedural scenes so the interface can be built
//! before an engine exists. This one reads an actual folder: it scans it,
//! reads EXIF, keeps ratings and keywords, and writes adjustments back.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;

use crate::adjust::{AdjustmentSet, WhiteBalanceMode};
#[cfg(test)]
use crate::RenderQuality;
use crate::catalog::{ColorTag, Exif, FileKind, ImageId, ImageInfo, Iptc};
use crate::engine::{AutoResult, AutoTarget, CameraSettings, CameraStatus, ImageEngine, PreviewFrame, PreviewRequest, StyleInfo};
use crate::mock::process::{self, Source};
use crate::preview_cache::{self, CachedFrame, PreviewCache};
use crate::store::{Catalog, Entry, Store, StoreError};
use crate::tether::{self, TetherStatus};
use crate::{AiMaskKind, Collection, CollectionKind, ExportJob, JobStatus, NormRect, RgbaImage};

/// Longest side kept for a decoded source; larger files are downscaled.
const SOURCE_LONG_SIDE: u32 = 4096;
/// How many decoded sources stay in memory at once.
const CACHED_SOURCES: usize = 8;
/// Test probes: how many decodes really ran, and an optional slow-down that
/// makes ten concurrent requests provably overlap.
#[cfg(test)]
static DECODE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static DECODE_DELAY_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Persistent folder identities start here.
const FOLDER_BASE: u64 = 1000;

#[path = "library_fs.rs"]
mod library_fs;

/// Stable category for an import/decode failure, so the interface can show a
/// translated message. Raw error text still goes to the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportErrorKind {
    /// libraw is not installed or cannot be loaded.
    NoLibRaw,
    /// libraw or the image decoder could not read the file.
    DecodeFailed,
    /// The file or its pixels exceed what this build can hold.
    TooLarge,
    /// The format or compression is not supported.
    Unsupported,
    /// Filesystem error (missing file, permissions).
    Io,
    Other,
}

impl ImportErrorKind {
    pub fn of(err: &anyhow::Error) -> Self {
        let lower = format!("{err:#}").to_ascii_lowercase();
        if lower.contains("too large") || lower.contains("out of memory") || lower.contains("memory") {
            Self::TooLarge
        } else if lower.contains("libraw") && (lower.contains("not loadable") || lower.contains("not found")) {
            Self::NoLibRaw
        } else if lower.contains("unsupported") || lower.contains("not supported") {
            Self::Unsupported
        } else if lower.contains("decode") || lower.contains("demosaic") || lower.contains("not decodable") {
            Self::DecodeFailed
        } else if lower.contains("os error") || lower.contains("no such file") || lower.contains("permission denied") {
            Self::Io
        } else {
            Self::Other
        }
    }
}

/// Why a preview is not full sensor data, for a translated viewer notice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceStatusKind {
    /// libraw could not decode the sensor data at all.
    DecodeFailed,
    /// libraw decoded the embedded preview because the compression is unsupported.
    UnsupportedCompression,
    Other,
}

impl SourceStatusKind {
    pub fn of(reason: &str) -> Self {
        let lower = reason.to_ascii_lowercase();
        if lower.contains("decode failed") {
            Self::DecodeFailed
        } else if lower.contains("unsupported raw compression") || lower.contains("embedded preview") {
            Self::UnsupportedCompression
        } else {
            Self::Other
        }
    }
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

#[derive(Clone)]
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
    /// Jobs whose worker must not overwrite the cancelled status.
    cancelled: std::collections::HashSet<u64>,
    /// Why a source is not full sensor data; consumed by `source_status`.
    source_reasons: HashMap<ImageId, String>,
    catalog: Catalog,
    preview: PreviewCache,
    profiled: crate::engine::ProfileCache,
}

/// An engine over a real folder of images.
pub struct FileEngine {
    root: PathBuf,
    state: Arc<Mutex<State>>,
    /// One decode at a time. Waiters re-check the source cache after taking it,
    /// so concurrent requests for the same image decode it once.
    decode: Mutex<()>,
    store: Option<Store>,
    /// Camera defaults from Capture One's `coreConfig.cdx`, read once.
    camera_defaults: std::sync::OnceLock<Option<crate::CameraDefaults>>,
    /// Closed-eye results, computed once per image.
    eyes: Mutex<HashMap<ImageId, crate::EyeReport>>,
    imported_styles: RwLock<Vec<StyleInfo>>,
}

impl FileEngine {
    /// Opens `root` and imports every supported image below it.
    pub fn open(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = library_fs::canonical_path(root.as_ref())?;
        anyhow::ensure!(root.is_dir(), "library root is not a directory");
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
        let catalog = match &store {
            Some(store) => store.load_index()?,
            None => Catalog::default(),
        };
        let engine = Self {
            root: root.clone(),
            state: Arc::new(Mutex::new(State {
                entries: Self::restore_entries(&root, store.as_ref(), &catalog)?,
                next_id: catalog.next_id.max(catalog.images.iter().filter_map(|e| e.info.as_ref()).map(|i| i.id.0 + 1).max().unwrap_or(1)),
                sources: HashMap::new(),
                lru: Vec::new(),
                jobs: HashMap::new(),
                cancelled: Default::default(),
                source_reasons: HashMap::new(),
                catalog,
                preview: PreviewCache::default(),
                profiled: Default::default(),
            })),
            decode: Mutex::new(()),
            store,
            camera_defaults: std::sync::OnceLock::new(),
            eyes: Mutex::new(HashMap::new()),
            imported_styles: RwLock::new(Vec::new()),
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
        // Step outside every Capture One package component so our own `.lumen`
        // never lands inside one. A plain `Shoot/Shoot.cocatalogdb` file keeps
        // `Shoot` as the root; only extension-named folders are packages.
        while root.extension().is_some_and(|e| {
            ["cocatalogdb", "cosessiondb", "cos"]
                .iter()
                .any(|want| e.eq_ignore_ascii_case(want))
        }) {
            root = root.parent().unwrap_or(Path::new(".")).to_path_buf();
        }
        let root = library_fs::canonical_path(&root)?;
        let store = Store::open(&root).ok();
        let catalog = match &store {
            Some(store) => store.load_index()?,
            None => Catalog::default(),
        };
        let engine = Self {
            root: root.clone(),
            state: Arc::new(Mutex::new(State {
                entries: Self::restore_entries(&root, store.as_ref(), &catalog)?,
                next_id: catalog.next_id.max(catalog.images.iter().filter_map(|e| e.info.as_ref()).map(|i| i.id.0 + 1).max().unwrap_or(1)),
                sources: HashMap::new(),
                lru: Vec::new(),
                jobs: HashMap::new(),
                cancelled: Default::default(),
                source_reasons: HashMap::new(),
                catalog,
                preview: PreviewCache::default(),
                profiled: Default::default(),
            })),
            decode: Mutex::new(()),
            store,
            camera_defaults: std::sync::OnceLock::new(),
            eyes: Mutex::new(HashMap::new()),
            imported_styles: RwLock::new(Vec::new()),
        };
        engine.import_c1(&import)?;
        engine.rescan()?;
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

    fn import_c1(&self, import: &crate::c1import::C1Import) -> anyhow::Result<()> {
        let mut st = self.state.lock();
        let mut entries = st.entries.clone();
        let catalog = st.catalog.clone();
        let mut next_id = st.next_id;
        for image in import.images() {
            let path = PathBuf::from(&image.abs_path);
            if catalog.excluded.contains(&path) { continue; }
            for variant in import.variants_of(image.pk) {
                let number = variant.index.max(0) as u32 + 1;
                let rel_path = crate::c1import::store_key(&path, number);
                if catalog.excluded_variants.iter().any(|v| v.key == rel_path || (v.path == path && v.variant == number)) { continue; }
                if entries.iter().any(|e| e.rel_path == rel_path || (e.path == path && e.info.variant == number)) { continue; }
                // Our store wins over C1's copy, so reopening the catalog keeps the
                // edits made in Lumen instead of resetting them to C1's.
                let stored = catalog.entry(&rel_path).cloned();
                let adjustments = match (&self.store, &stored) {
                    (Some(store), Some(_)) => store.load_adjustments(&rel_path).unwrap_or_default(),
                    _ => {
                        let mut adjustments = variant.adjustments.clone();
                        // C1 stores the display rotation in ZROTATION and libraw
                        // already applied the camera's EXIF rotation to the pixels,
                        // so only the difference is left for the renderer.
                        if crate::raw::is_raw(&path) {
                            adjustments.geometry.orientation =
                                ((adjustments.geometry.orientation as u16 + 4
                                    - crate::raw::exif_rotation_deg(&path) / 90) % 4) as u8;
                        }
                        adjustments
                    }
                };
                let id = ImageId(next_id);
                next_id += 1;
                let info = ImageInfo {
                    id,
                    path: path.clone(),
                    file_name: image.file_name.clone(),
                    kind: file_kind(&path),
                    heif: None,
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
                entries.push(FileEntry {
                    info: info.clone(),
                    path: path.clone(),
                    rel_path: rel_path.clone(),
                    adjustments: adjustments.clone(),
                });
            }
        }
        self.commit_library(&mut st, entries, catalog, &[], None)?;
        Ok(())
    }

    /// Re-reads the folder, preserving identities, variants, edits and offline records.
    pub fn rescan(&self) -> anyhow::Result<()> {
        self.synchronize_folder(&self.root, true, false).map(|_| ())
    }

    fn add_file(&self, path: &Path) -> anyhow::Result<ImageInfo> {
        let path = library_fs::canonical_path(path)?;
        anyhow::ensure!(path.is_file() && is_supported(&path), "unsupported image {}", path.display());
        let mut st = self.state.lock();
        if let Some(entry) = st.entries.iter().find(|e| e.path == path) {
            return Ok(entry.info.clone());
        }
        let mut catalog = st.catalog.clone();
        catalog.excluded.retain(|p| p != &path);
        catalog.excluded_variants.retain(|v| v.path != path);
        let mut entries = st.entries.clone();
        let id = ImageId(st.next_id);
        let key = self.unique_store_key(&path, 1, id, &entries);
        let entry = self.discovered_entry(&path, id, key.clone(), catalog.entry(&key))?;
        let info = entry.info.clone();
        entries.push(entry);
        self.commit_library(&mut st, entries, catalog, &[], None)?;
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
        // Single flight: the first request decodes while the rest wait here,
        // then find the finished source in the cache instead of decoding again.
        let _decoding = self.decode.lock();
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
        #[cfg(test)]
        {
            DECODE_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let delay = DECODE_DELAY_MS.load(std::sync::atomic::Ordering::SeqCst);
            if delay > 0 {
                std::thread::sleep(std::time::Duration::from_millis(delay));
            }
        }
        // Decoding happens outside the state lock; the decode lock is the only
        // thing a concurrent caller waits on.
        // A RAW keeps libraw's 16-bit linear samples; every other file and any
        // RAW libraw cannot demosaic arrives as 8-bit sRGB.
        let mut decoded_reason = None;
        let source = match raw16_source(&path) {
            Some(source) => source,
            None => {
                let img = if crate::raw::is_raw(&path) {
                    match crate::raw::decode(&path) {
                        Ok(img) => img,
                        Err(err) => {
                            let reason = format!("RAW decode failed: {err:#}");
                            self.state.lock().source_reasons.insert(image, reason);
                            return Err(err);
                        }
                    }
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
        if camera_preview {
            decoded_reason = Some("unsupported RAW compression; showing the camera's embedded preview".into());
        }
        let source = Arc::new(source.with_lens_shot(lens).with_camera_preview(camera_preview));

        let mut st = self.state.lock();
        match decoded_reason {
            Some(reason) => {
                st.source_reasons.insert(image, reason);
            }
            None => {
                st.source_reasons.remove(&image);
            }
        }
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
        if let Some(image) = self.offline_library_preview(request.image, request.max_size)? {
            return Ok(PreviewFrame {
                histogram: process::histogram(image.pixels.as_ref()), image,
                generation: request.generation, elapsed_ms: start.elapsed().as_secs_f32() * 1000.0,
             degraded: None, gamut_warning: None, });
        }
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
pub(crate) fn embedded_icc(path: &Path) -> Option<Vec<u8>> {
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
        // image's TIFF encoder writes BYTE(1); the TIFF spec uses UNDEFINED(7).
        if !matches!(rd_u16(&head, entry + 2, le)?, 1 | 7) {
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
fn camera_icc(path: &Path, model: Option<&str>) -> Option<String> {
    let mapped = model.and_then(crate::camera_profile::for_camera)?;
    let keyword = profile_keyword(mapped.as_str());
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
            if path.is_dir() {
                let before: Vec<_> = self.images().iter().map(|i| i.id).collect();
                added.extend(self.synchronize_folder(path, true, true)?.into_iter().filter(|i| !before.contains(&i.id)));
            } else if is_supported(path) {
                added.push(self.add_file(path)?);
            }
        }
        Ok(added)
    }

    fn collections(&self) -> Vec<Collection> {
        self.collection_tree()
    }

    fn library_root(&self) -> Option<PathBuf> {
        Some(self.root.clone())
    }

    fn collection_path(&self, id: u64) -> Option<PathBuf> {
        self.state.lock().catalog.folders.iter().find(|f| f.id == id).map(|f| f.path.clone())
    }

    fn user_collections(&self) -> Vec<crate::UserCollection> {
        self.store.as_ref().map(|store| store.load_user_collections()).unwrap_or_default()
    }

    fn manage_files(&self, action: crate::LibraryAction) -> anyhow::Result<Vec<ImageInfo>> {
        self.manage_library(action)
    }

    fn collection_images(&self, collection: u64) -> Option<Vec<ImageId>> {
        let path = self.collection_path(collection)?;
        Some(self.state.lock().entries.iter().filter(|e| e.path.parent() == Some(path.as_path())).map(|e| e.info.id).collect())
    }

    fn save_user_collections(&self, collections: &[crate::UserCollection]) -> anyhow::Result<()> {
        let _st = self.state.lock();
        let store = self.writable_store()?;
        store.save_user_collections(collections)
    }

    fn set_trashed(&self, ids: &[ImageId], trashed: bool) -> anyhow::Result<()> {
        let mut st = self.state.lock();
        let mut entries = st.entries.clone();
        for entry in entries.iter_mut().filter(|e| ids.contains(&e.info.id)) {
            entry.info.trashed = trashed;
        }
        let catalog = st.catalog.clone();
        self.commit_library(&mut st, entries, catalog, &[], None)
    }

    fn update_info(&self, info: &ImageInfo) -> anyhow::Result<()> {
        let mut st = self.state.lock();
        let mut entries = st.entries.clone();
        let entry = entries.iter_mut().find(|e| e.info.id == info.id)
            .ok_or_else(|| anyhow::anyhow!("unknown image"))?;
        anyhow::ensure!(entry.path == info.path && entry.info.variant == info.variant, "use manage_files to change image paths");
        entry.info = info.clone();
        let catalog = st.catalog.clone();
        self.commit_library(&mut st, entries, catalog, &[], None)
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
                    .map(|name| name.as_str().to_string());
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
            .filter(|name| crate::camera_profile::is_available(name));
        }
        adjustments
    }

    fn save_adjustments(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<()> {
        let mut st = self.state.lock();
        let mut entries = st.entries.clone();
        let entry = entries.iter_mut().find(|e| e.info.id == image)
            .ok_or_else(|| anyhow::anyhow!("unknown image"))?;
        entry.adjustments = adjustments.clone();
        entry.info.has_adjustments = !adjustments.is_default();
        let catalog = st.catalog.clone();
        self.commit_library(&mut st, entries, catalog, &[], None)
    }

    fn new_variant(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo> {
        let mut st = self.state.lock();
        let pos = st.entries.iter().position(|e| e.info.id == image)
            .ok_or_else(|| anyhow::anyhow!("unknown image"))?;
        let mut entries = st.entries.clone();
        let mut entry = entries[pos].clone();
        entry.info.id = ImageId(st.next_id);
        entry.info.variant = entries.iter().filter(|e| e.path == entry.path).map(|e| e.info.variant)
            .chain(st.catalog.excluded_variants.iter().filter(|v| v.path == entry.path).map(|v| v.variant))
            .max().unwrap_or(0) + 1;
        entry.info.has_adjustments = !adjustments.is_default();
        entry.rel_path = self.unique_store_key(&entry.path, entry.info.variant, entry.info.id, &entries);
        entry.adjustments = adjustments.clone();
        let info = entry.info.clone();
        entries.insert(pos + 1, entry);
        let catalog = st.catalog.clone();
        self.commit_library(&mut st, entries, catalog, &[], None)?;
        Ok(info)
    }

    fn thumbnail(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<RgbaImage> {
        if let Some(preview) = self.offline_library_preview(image, max_size)? { return Ok(preview); }
        let src = self.source(image)?;
        let out = process::render(&src, &crate::styles::resolve(adjustments), NormRect::FULL, max_size, false, None);
        let frame = RgbaImage::new(out.width, out.height, out.rgba);
        let st = self.state.lock();
        if st.entries.iter().any(|e| e.info.id == image) {
            if let Some(store) = &self.store {
                if let Err(err) = store.save_library_preview(image, &frame) { log::warn!("save offline preview: {err}"); }
            }
        }
        Ok(frame)
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
        let needs_exposure = all || target == AutoTarget::Exposure;
        let needs_hdr = all || target == AutoTarget::HighDynamicRange;
        let suggestions = if needs_exposure || needs_hdr {
            Some(crate::auto_exposure::analyze(&probe.rgba, probe.width, probe.height))
        } else {
            None
        };
        if let (true, Some(s)) = (needs_exposure, suggestions.as_ref()) {
            adj.exposure.exposure = s.exposure;
            adj.exposure.contrast = s.contrast;
            adj.exposure.brightness = s.brightness;
            adj.exposure.saturation = s.saturation;
        }
        if let (true, Some(s)) = (needs_hdr, suggestions.as_ref()) {
            adj.hdr = s.hdr.clone();
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
        let mut all = crate::styles::all();
        all.extend(crate::c1_style_index::installed_styles());
        all.extend(film_curve_styles());
        all.extend(self.imported_styles.read().clone());
        dedupe_styles(all)
    }

    fn film_curves(&self) -> Vec<String> {
        crate::filmcurve::names()
    }

    fn import_style(&self, path: &Path) -> anyhow::Result<StyleInfo> {
        let style = crate::c1_styles::load_style(path)?;
        let info = StyleInfo { name: style.name, group: "Imported".into(), adjustments: style.adjustments };
        let mut imported = self.imported_styles.write();
        imported.retain(|s| !(s.group == info.group && s.name == info.name));
        imported.push(info.clone());
        Ok(info)
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
            let cancelled = |state: &Mutex<State>| state.lock().cancelled.contains(&job_id);
            if cancelled(&state) {
                return;
            }
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
                        if !cancelled(&state) {
                            state.lock().jobs.insert(job_id, JobStatus::Running { progress: 0.1 + 0.9 * p });
                        }
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
                        if !cancelled(&state) {
                            state.lock().jobs.insert(job_id, JobStatus::Running { progress: 0.1 + 0.9 * p });
                        }
                    },
                )
            };
            let mut state = state.lock();
            if !state.cancelled.contains(&job_id) {
                state.jobs.insert(
                    job_id,
                    match result {
                        Ok(output) => JobStatus::Done { output },
                        Err(err) => JobStatus::Failed { message: err.to_string() },
                    },
                );
            }
        });
        Ok(())
    }

    fn job_status(&self, job_id: u64) -> JobStatus {
        self.state.lock().jobs.get(&job_id).cloned().unwrap_or(JobStatus::Failed { message: "unknown job".into() })
    }

    fn cancel_job(&self, job_id: u64) -> bool {
        let mut st = self.state.lock();
        match st.jobs.get(&job_id) {
            Some(JobStatus::Queued) | Some(JobStatus::Running { .. }) => {
                st.cancelled.insert(job_id);
                st.jobs.insert(job_id, JobStatus::Failed { message: "cancelled".into() });
                true
            }
            _ => false,
        }
    }

    fn source_status(&self, image: ImageId) -> Option<String> {
        self.state.lock().source_reasons.get(&image).cloned()
    }

    fn tether_status(&self) -> TetherStatus {
        tether::probe()
    }
    fn capabilities(&self) -> crate::engine::EngineCapabilities {
        crate::engine::EngineCapabilities {
            ai_mask_kinds: vec![
                AiMaskKind::Subject,
                AiMaskKind::Person,
                AiMaskKind::Clothing,
                AiMaskKind::Skin,
                AiMaskKind::Hair,
                AiMaskKind::Sky,
                AiMaskKind::Background,
            ],
            eye_report: true,
            auto_rotate: true,
            match_look: true,
            tether: cfg!(windows),
            export_formats: crate::output::ExportFormat::ALL
                .into_iter()
                .filter(|format| format.is_supported())
                .collect(),
        }
    }
    fn camera_status(&self) -> CameraStatus {
        tether::camera_status()
    }
    fn camera_settings(&self) -> Option<CameraSettings> {
        tether::camera_settings()
    }
    fn capture(&self) -> anyhow::Result<ImageInfo> {
        // Do not create a capture folder unless the camera produced a frame.
        let frame = tether::capture_frame()?;
        let dir = self.capture_directory()?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = dir.join(format!("CAP_{stamp}.jpg"));
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
        file.write_all(&frame.jpeg)?;
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

/// Sorted by `(group, name)`; of a duplicate pair the first one wins.
fn dedupe_styles(mut styles: Vec<StyleInfo>) -> Vec<StyleInfo> {
    styles.sort_by(|a, b| a.group.cmp(&b.group).then_with(|| a.name.cmp(&b.name)));
    styles.dedup_by(|a, b| a.group == b.group && a.name == b.name);
    styles
}

/// One representative curve per kind the installer ships, keyed by the file
/// stem `curve_lut` reads. Missing kinds are skipped.
fn film_curve_styles() -> Vec<StyleInfo> {
    const KINDS: [&str; 3] = ["-Auto", "-Linear Response", "-Film Standard"];
    let names = crate::filmcurve::names();
    KINDS
        .iter()
        .filter_map(|kind| {
            let name = names
                .iter()
                .find(|n| n.ends_with(kind) && crate::filmcurve::curve_lut(n).is_some())?;
            let mut adjustments = AdjustmentSet::default();
            adjustments.base.film_curve = Some(name.clone());
            Some(StyleInfo { name: name.clone(), group: "Film Curve".into(), adjustments })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ten previews for the same cold image must decode it once: the first
    /// request decodes, the other nine wait on the decode lock and then use the
    /// cached source. A test-only delay makes the overlap deterministic.
    #[test]
    fn ten_concurrent_previews_decode_the_source_once() {
        let (dir, synthetic) = match std::env::var_os("LUMEN_PREVIEW_BENCH_DIR") {
            Some(dir) => (PathBuf::from(dir), false),
            None => {
                let dir = std::env::temp_dir().join(format!("lumen-single-flight-{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir);
                std::fs::create_dir_all(&dir).unwrap();
                let img = image::RgbaImage::from_fn(3000, 2000, |x, y| {
                    let v = ((x * 7 + y * 13) % 256) as u8;
                    image::Rgba([v, v / 3, 255 - v, 255])
                });
                img.save(dir.join("frame.png")).unwrap();
                (dir, true)
            }
        };
        let engine = Arc::new(FileEngine::open(&dir).unwrap());
        let image = engine.images()[0].id;
        let request = PreviewRequest {
            image,
            adjustments: AdjustmentSet::default(),
            region: NormRect::FULL,
            max_size: 1400,
            quality: crate::RenderQuality::Draft,
            bypass: false,
            mask_overlay: None,
            generation: 1,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
        };
        let before = DECODE_COUNT.load(std::sync::atomic::Ordering::SeqCst);
        DECODE_DELAY_MS.store(150, std::sync::atomic::Ordering::SeqCst);
        let frames = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for step in 0..10u64 {
                let engine = engine.clone();
                let frames = &frames;
                let mut request = request.clone();
                request.generation = step + 1;
                request.adjustments.exposure.exposure = step as f32 * 0.1;
                scope.spawn(move || {
                    let frame = engine.render_preview(&request).unwrap();
                    frames.lock().push((frame.generation, frame.image.width));
                });
            }
        });
        DECODE_DELAY_MS.store(0, std::sync::atomic::Ordering::SeqCst);
        let frames = frames.into_inner();
        assert_eq!(frames.len(), 10, "every preview must succeed");
        let decodes = DECODE_COUNT.load(std::sync::atomic::Ordering::SeqCst) - before;
        assert_eq!(decodes, 1, "ten concurrent previews must decode the source exactly once");
        println!("single-flight: 10 previews, decodes={decodes}, frames={frames:?}");
        if synthetic {
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    /// Draft, Final and the internal half-size fast path must all return the
    /// same dimensions; only the pixels computed internally may differ.
    #[test]
    fn draft_final_and_fast_keep_the_same_output_size() {
        let dir = std::env::temp_dir().join(format!("lumen-same-size-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::from_fn(2400, 1600, |x, y| image::Rgba([(x % 255) as u8, (y % 255) as u8, 80, 255]))
            .save(dir.join("frame.png"))
            .unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let image = engine.images()[0].id;
        let request = PreviewRequest {
            image,
            adjustments: AdjustmentSet::default(),
            region: NormRect::FULL,
            max_size: 2000,
            quality: crate::RenderQuality::Draft,
            bypass: false,
            mask_overlay: None,
            generation: 1,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
        };
        let draft = engine.render_preview(&request).unwrap();
        let mut final_request = request.clone();
        final_request.quality = crate::RenderQuality::Final;
        final_request.adjustments.exposure.exposure = 0.25;
        let final_frame = engine.render_preview(&final_request).unwrap();
        let mut fast_request = request.clone();
        fast_request.adjustments.exposure.exposure = -0.25;
        let fast = engine.render_preview_fast(&fast_request, true).unwrap();
        let size = |f: &PreviewFrame| (f.image.width, f.image.height);
        assert_eq!(size(&draft), size(&final_frame), "Draft cap changed the output size");
        assert_eq!(size(&fast), size(&final_frame), "fast changed the output size");
        assert_eq!(final_frame.image.width, 2000);
        println!("same-size: draft={:?} final={:?} fast={:?}", size(&draft), size(&final_frame), size(&fast));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A missing camera must fail before anything is written or catalogued; the
    /// mock engine is never consulted from this path.
    #[test]
    fn capture_without_a_camera_leaves_the_catalog_unchanged() {
        if !matches!(tether::probe(), TetherStatus::NoCamera) {
            return;
        }
        let dir = std::env::temp_dir().join(format!("lumen-nocam-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let before = engine.images().len();
        let err = engine.capture().expect_err("capture must fail without a camera").to_string();
        assert!(err.contains("NoCamera"), "{err}");
        assert_eq!(engine.images().len(), before);
        assert!(!dir.join("Capture").exists(), "an empty capture folder was created");
    }

    /// The interface decides which panels and formats to show from this, so it
    /// must list only what the file engine really produces.
    #[test]
    fn capabilities_match_the_engine_and_the_encoders() {
        let dir = std::env::temp_dir().join(format!("lumen-caps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let caps = FileEngine::open(&dir).unwrap().capabilities();
        assert!(caps.eye_report && caps.auto_rotate && caps.match_look);
        assert!(caps.ai_mask_kinds.contains(&AiMaskKind::Subject));
        assert!(!caps.ai_mask_kinds.contains(&AiMaskKind::Eyes));
        let expected: Vec<_> = crate::output::ExportFormat::ALL
            .into_iter()
            .filter(|f| f.is_supported())
            .collect();
        assert_eq!(caps.export_formats, expected);
    }

    #[test]
    fn import_errors_map_to_stable_kinds() {
        use anyhow::anyhow;
        let kind = |text: &str| ImportErrorKind::of(&anyhow!("{text}"));
        assert_eq!(kind("libraw is not loadable on this machine"), ImportErrorKind::NoLibRaw);
        assert_eq!(kind("RAW decode failed: bad CFA data"), ImportErrorKind::DecodeFailed);
        assert_eq!(kind("image is too large (90000 x 90000)"), ImportErrorKind::TooLarge);
        assert_eq!(kind("Unsupported: no encoder for PSD"), ImportErrorKind::Unsupported);
        assert_eq!(kind("No such file or directory (os error 2)"), ImportErrorKind::Io);
        assert_eq!(kind("collect failed"), ImportErrorKind::Other);
    }

    #[test]
    fn importing_an_unsupported_file_adds_nothing() {
        let dir = std::env::temp_dir().join(format!("lumen-import-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let note = dir.join("note.txt");
        std::fs::write(&note, b"not an image").unwrap();
        let added = engine.import(&[note]).unwrap();
        assert!(added.is_empty());
        assert_eq!(engine.images().len(), 0);
    }

    #[test]
    fn source_status_reasons_map_to_stable_kinds() {
        assert_eq!(
            SourceStatusKind::of("RAW decode failed: bad data"),
            SourceStatusKind::DecodeFailed
        );
        assert_eq!(
            SourceStatusKind::of("unsupported RAW compression; showing the camera's embedded preview"),
            SourceStatusKind::UnsupportedCompression
        );
        assert_eq!(SourceStatusKind::of("other"), SourceStatusKind::Other);
    }

    #[test]
    fn cancelling_a_queued_job_sticks() {
        let dir = std::env::temp_dir().join(format!("lumen-cancel-job-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        engine.state.lock().jobs.insert(77, JobStatus::Queued);
        assert!(engine.cancel_job(77));
        assert_eq!(engine.job_status(77), JobStatus::Failed { message: "cancelled".into() });
        assert!(!engine.cancel_job(77));
        assert_eq!(engine.source_status(ImageId(1)), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_imported_style_appears_once_in_styles() {
        let dir = std::env::temp_dir().join(format!("lumen-import-style-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let style = dir.join("Imported.costyle");
        std::fs::write(&style, b"<SL><E K=\"Name\" V=\"Synthetic\"/><E K=\"Exposure\" V=\"1\"/></SL>").unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        assert_eq!(engine.import_style(&style).unwrap().name, "Synthetic");
        assert_eq!(engine.import_style(&style).unwrap().name, "Synthetic");
        let count = engine.styles().iter().filter(|s| s.group == "Imported" && s.name == "Synthetic").count();
        assert_eq!(count, 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

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
            max_size: 1600, quality: RenderQuality::Draft, bypass: false, mask_overlay: None, generation: 1 , display_profile: None, proof_profile: None, proof_paper_white: 1.0, proof_black_point: 0.0, };
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

    #[cfg(feature = "external-fixtures")]
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

    #[cfg(feature = "external-fixtures")]
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
