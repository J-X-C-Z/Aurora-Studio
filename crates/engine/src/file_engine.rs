//! A catalog backed by real files on disk, with edits persisted in `.portrait`.
//!
//! [`crate::MockEngine`] serves procedural scenes so the interface can be built
//! before an engine exists. This one reads an actual folder: it scans it,
//! reads EXIF, keeps ratings and keywords, and writes adjustments back.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, RwLock};
use rayon::prelude::*;

use crate::adjust::{AdjustmentSet, Curves, Geometry, Levels, NoiseReduction, Sharpening, WhiteBalance, WhiteBalanceMode};
use crate::catalog::{ColorTag, Exif, FileKind, ImageId, ImageInfo, Iptc};
use crate::document::DocumentRevision;
use crate::engine::{AutoResult, AutoTarget, CameraSettings, CameraStatus, ImageEngine, PreviewFrame, PreviewRequest, RenderQuality, StyleInfo};
use crate::job_lifecycle::{CancelToken, JobId, JobRegistry, JobScope, JobState};
use crate::layers::{Layer, LinearGradient, LocalAdjustments, Mask, MaskOp, MaskShape};
use crate::mock::process::{self, Source};
use crate::preview_cache::{self, CachedFrame, PreviewCache};
use crate::save_queue::{FlushReport, QueueStatus, SaveQueue, SaveWriter};
use crate::save_report::SaveReport;
use crate::store::{Catalog, Entry, Store, StoreError};
use crate::tether::{self, TetherStatus};
use crate::{AiMaskKind, Collection, CollectionKind, ExportJob, HdrOptions, JobStatus, NormRect, PanoramaOptions, RgbaImage};

#[path = "library_ops.rs"]
mod library_ops;
#[path = "file_engine_services.rs"]
mod services;

/// Longest side kept for a decoded source; larger files are downscaled.
const SOURCE_LONG_SIDE: u32 = 4096;
/// Test-only count of the one-shot full-resolution export decodes. Exporting
/// at or below [`SOURCE_LONG_SIDE`] must never take this path.
#[cfg(test)]
static FULL_RES_EXPORT_DECODES: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
fn full_res_export_decode_count() -> u64 {
    FULL_RES_EXPORT_DECODES.load(Ordering::Relaxed)
}

/// The decode counter is process-global, so tests that assert on it must not
/// run concurrently with another full-resolution export test (the default
/// harness runs tests in parallel; the repository's CI passes
/// `--test-threads=1`, but `cargo test <filter>` may not).
#[cfg(test)]
fn full_res_export_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
/// Bounds for the decoded-source working set. A source is charged for every
/// mip level it really owns, so one 4096-long-side RAW is ~170 MiB while a
/// thumbnail-sized JPEG is ~1 MiB. The default budget is derived from this
/// machine's memory at engine start, never from a fixed image count.
const SOURCE_CACHE_MIN_BYTES: usize = 256 * 1024 * 1024;
const SOURCE_CACHE_MAX_BYTES: usize = 4 * 1024 * 1024 * 1024;
/// Used only when the platform cannot report memory and no env override is set.
const SOURCE_CACHE_FALLBACK_BYTES: usize = 768 * 1024 * 1024;
/// One admission unit for a decode. A request reserves as many units as its
/// estimated working set needs, so large frames reduce concurrency while small
/// JPEGs still run several at a time.
const DECODE_UNIT_BYTES: usize = 256 * 1024 * 1024;
/// Interactive sources kept pinned around the current image (current plus the
/// two most recently visited neighbours). Pins survive byte-budget eviction.
const RECENT_SOURCE_PINS: usize = 3;
/// Folder collection ids start here; the index into `folder_dirs()` is added.
const FOLDER_BASE: u64 = 1000;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse::<usize>().ok()
}

/// Total and currently available physical memory, in bytes.
#[cfg(windows)]
fn system_memory_bytes() -> Option<(u64, u64)> {
    #[repr(C)]
    #[derive(Default)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx { length: std::mem::size_of::<MemoryStatusEx>() as u32, ..Default::default() };
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0 && status.total_phys > 0).then_some((status.total_phys, status.avail_phys))
}

/// Linux fallback so the engine still derives a budget off Windows.
#[cfg(not(windows))]
fn system_memory_bytes() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|value| value.parse::<u64>().ok())
            .map(|kib| kib * 1024)
    };
    Some((field("MemTotal:")?, field("MemAvailable:")?))
}

/// Source-cache budget in bytes. Env overrides:
/// `AURORA_SOURCE_CACHE_BYTES` (exact), `AURORA_SOURCE_CACHE_MB` (MiB).
fn default_source_cache_bytes() -> usize {
    if let Some(bytes) = env_usize("AURORA_SOURCE_CACHE_BYTES") {
        return bytes.max(64 * 1024 * 1024);
    }
    if let Some(mib) = env_usize("AURORA_SOURCE_CACHE_MB") {
        return mib.saturating_mul(1024 * 1024).max(64 * 1024 * 1024);
    }
    system_memory_bytes().map_or(SOURCE_CACHE_FALLBACK_BYTES, |(total, available)| {
        source_cache_budget_from_memory(total, available)
    })
}

/// ~8.3% of installed RAM, and never more than half of what is free:
/// 16 GiB → ~1.33 GiB, 32 GiB → ~2.67 GiB. The installed size keeps the budget
/// predictable; the available-memory term only shrinks it under real pressure.
fn source_cache_budget_from_memory(total: u64, available: u64) -> usize {
    let bytes = (total / 12).min(available / 2);
    usize::try_from(bytes)
        .map(|bytes| bytes.clamp(SOURCE_CACHE_MIN_BYTES, SOURCE_CACHE_MAX_BYTES))
        .unwrap_or(SOURCE_CACHE_FALLBACK_BYTES)
}

fn recent_source_pins() -> usize {
    env_usize("AURORA_SOURCE_CACHE_RECENT_PINS").unwrap_or(RECENT_SOURCE_PINS).min(8)
}

/// A cached source's content token stays trusted while its metadata is
/// unchanged for this long before it is re-hashed.
const SOURCE_IDENTITY_RECHECK: std::time::Duration = std::time::Duration::from_secs(2);
/// Hits allowed between content-token re-hashes while the metadata is unchanged.
const SOURCE_IDENTITY_RECHECK_HITS: u32 = 8;

/// Bytes a decoded source really owns or has reserved: every mip level's
/// allocated capacity (or the full payload a deferred native level will need)
/// plus the lazily built portrait derivatives (RGBA8 face input and masks).
fn source_cache_bytes(source: &Source) -> usize {
    source
        .levels
        .iter()
        .fold(0usize, |total, plane| total.saturating_add(plane.allocated_bytes()))
        .saturating_add(source.cached_derived_bytes())
}

/// Bytes one AI input keeps alive: its RGBA frame plus the struct itself.
fn ai_input_bytes(input: &AiInput) -> usize {
    input.rgba.capacity().saturating_add(std::mem::size_of::<AiInput>())
}

/// `(len, mtime nanos)`; `None` when the file is gone or unreadable.
fn file_metadata_stamp(path: &Path) -> Option<(u64, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX));
    Some((meta.len(), mtime_nanos))
}

/// Identity of the file a cached source was decoded from. The metadata is
/// checked on every cache hit; the content token is taken at decode time and
/// re-taken when the metadata changes or the recheck interval elapses, so a
/// same-size replacement that preserved the timestamp still invalidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SourceFileIdentity {
    len: u64,
    mtime_nanos: u64,
    token: Option<u64>,
}

impl SourceFileIdentity {
    fn read(path: &Path) -> Option<Self> {
        let (len, mtime_nanos) = file_metadata_stamp(path)?;
        Some(Self { len, mtime_nanos, token: None })
    }

    /// Adds the content fingerprint from `raw_cache`; a decode-time cost.
    fn with_token(mut self, path: &Path) -> Self {
        self.token = crate::raw_cache::source_identity_token(path);
        self
    }

    /// `true` when the stored identity holds no usable provenance.
    fn is_unknown(&self) -> bool {
        self.len == 0 && self.mtime_nanos == 0 && self.token.is_none()
    }
}

/// The source one export was queued for: its catalog path and the lightweight
/// file identity (length + mtime) read at queue time.
///
/// The worker checks both again before it decodes and once more afterwards, so
/// a file that was deleted, moved or replaced while the job waited fails the
/// job instead of exporting different pixels with the queued recipe. A
/// replacement that keeps the exact length and mtime is not distinguishable by
/// this lightweight check; catching that needs the content token, which costs
/// a file read and is only taken when a source is stored for the cache.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExportSourceSnapshot {
    path: PathBuf,
    identity: Option<SourceFileIdentity>,
}

impl ExportSourceSnapshot {
    fn read(path: &Path) -> Self {
        Self { path: path.to_path_buf(), identity: SourceFileIdentity::read(path) }
    }

    /// Fails unless the catalog still points at the queued path and the file
    /// still has the length and mtime it had when the job was queued.
    fn ensure_unchanged(
        &self,
        engine: &FileEngine,
        image: ImageId,
        phase: &str,
    ) -> anyhow::Result<()> {
        if engine.image_path(image).as_deref() != Some(self.path.as_path()) {
            anyhow::bail!("export source path changed {phase}");
        }
        if self.identity.is_none() || SourceFileIdentity::read(&self.path) != self.identity {
            anyhow::bail!("export source changed {phase}");
        }
        Ok(())
    }
}

/// How a file is decoded, for working-set estimation only. Proxy/native RAW and
/// raster decoders allocate very different intermediates at the *source*
/// resolution, so admission must not be based on the retained 4096 proxy alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeFormat {
    /// RAW/EIP decoded at source resolution, then downscaled to the 4096 proxy.
    RawProxy,
    /// RAW/EIP decoded at full source resolution and retained 1:1.
    RawNative,
    /// JPEG/PNG/TIFF/HEIF/COP and any other raster decoded through `image`.
    Raster,
}

impl DecodeFormat {
    fn is_native(self) -> bool {
        matches!(self, DecodeFormat::RawNative)
    }

    /// Transient bytes per *source* pixel while decoding, from the buffers each
    /// pipeline holds at peak. RAW: Bayer16 plane 2 B/px + demosaiced RGB16
    /// 6 B/px + demosaic intermediates 4 B/px = 12 B/px. Raster: `DynamicImage`
    /// 4 B/px + RGBA8/RGBA16 copy 8 B/px + downscaled frame 4 B/px = 16 B/px.
    /// These cover the whole decode, not the retained proxy. They are the
    /// admission envelope; re-check against Release peak RSS whenever the
    /// decoder pipeline or libraw/image versions change.
    fn transient_bytes_per_px(self) -> usize {
        match self {
            DecodeFormat::RawProxy | DecodeFormat::RawNative => RAW_DECODE_TRANSIENT_BYTES_PER_PX,
            DecodeFormat::Raster => RASTER_DECODE_TRANSIENT_BYTES_PER_PX,
        }
    }
}

/// F32 RGB bytes per retained proxy pixel (the mip pyramid stores `[f32; 3]`).
const RETAINED_BYTES_PER_PX: usize = std::mem::size_of::<[f32; 3]>();
/// Decoder intermediates per source pixel for RAW/EIP (see `DecodeFormat`).
const RAW_DECODE_TRANSIENT_BYTES_PER_PX: usize = 12;
/// Decoder intermediates per source pixel for raster formats.
const RASTER_DECODE_TRANSIENT_BYTES_PER_PX: usize = 16;

/// Rough working set of one decode: the retained mip pyramid plus the decoder's
/// transient frame at the source resolution. Used only for admission control,
/// not for cache accounting. The proxy is capped to 4096, but the decoder still
/// touches every source pixel, so only the retained part uses `fit_long_side`.
fn estimated_decode_bytes(width: u32, height: u32, format: DecodeFormat) -> usize {
    let source_px = (width as usize).saturating_mul(height as usize);
    let (kept_w, kept_h) = if format.is_native() {
        (width, height)
    } else {
        fit_long_side((width, height))
    };
    let kept_px = (kept_w as usize).saturating_mul(kept_h as usize);
    let retained = kept_px.saturating_mul(RETAINED_BYTES_PER_PX).saturating_mul(4) / 3;
    let transient = source_px.saturating_mul(format.transient_bytes_per_px());
    retained.saturating_add(transient)
}

/// Which RAW decoder the route selects, for the *observed* working-set
/// estimate. Deliberately separate from [`DecodeFormat`]: it does not feed
/// admission until real peak-RSS data lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum RawDecodeShape {
    LibRaw,
    AuroraGeneric,
    AuroraZ6iii,
}

#[allow(dead_code)]
impl RawDecodeShape {
    fn of(route: &crate::raw::DecodeRoute) -> Self {
        if !route.c1_raw_enabled {
            Self::LibRaw
        } else if route.z6iii_nef && route.z6iii_feature {
            Self::AuroraZ6iii
        } else {
            Self::AuroraGeneric
        }
    }

    /// Decoder buffers per source pixel at peak, excluding the container file.
    fn transient_bytes_per_px(self) -> usize {
        match self {
            // Bayer16 2 + LibRaw imgdata.image (4 x u16) 8 + RGB16 6 +
            // demosaic intermediates 4.
            Self::LibRaw => 20,
            // Bayer16 2 + LibRaw image 8 + stage1 [f32; 4] 16 + stage2/3
            // [f32; 3] 12 + RGBA16 output 8.
            Self::AuroraGeneric => 46,
            // Roadmap estimate: halo/adaptive Bayer + stage1 [f32; 4] +
            // RGBA16 output + reconstruction scaffolding. Probe value, not a
            // measured peak.
            Self::AuroraZ6iii => 40,
        }
    }
}

/// Working set a decode is expected to hold, including the container bytes and
/// the per-backend buffers the 12/16 B/px admission constants omit.
///
/// **Estimation only.** Admission still uses [`estimated_decode_bytes`] with
/// its unchanged constants; switching the gate to this value needs real
/// peak-RSS measurements per format (Z6III HE/HE★, lossless NEF, ARW, CR3,
/// RAF, DNG).
#[allow(dead_code)]
fn estimated_decode_bytes_observed(
    width: u32,
    height: u32,
    format: DecodeFormat,
    file_bytes: usize,
    route: &crate::raw::DecodeRoute,
) -> usize {
    let source_px = (width as usize).saturating_mul(height as usize);
    let (kept_w, kept_h) = if format.is_native() {
        (width, height)
    } else {
        fit_long_side((width, height))
    };
    let kept_px = (kept_w as usize).saturating_mul(kept_h as usize);
    let retained = kept_px.saturating_mul(RETAINED_BYTES_PER_PX).saturating_mul(4) / 3;
    let per_px = match format {
        DecodeFormat::RawProxy | DecodeFormat::RawNative => {
            RawDecodeShape::of(route).transient_bytes_per_px()
        }
        DecodeFormat::Raster => RASTER_DECODE_TRANSIENT_BYTES_PER_PX,
    };
    file_bytes
        .saturating_add(retained)
        .saturating_add(source_px.saturating_mul(per_px))
}

/// Decode-format class for a proxy decode of `path`; native RAW decodes pass
/// `RawNative` explicitly.
fn proxy_decode_format(path: &Path) -> DecodeFormat {
    if crate::raw::is_raw(path) || is_eip(path) {
        DecodeFormat::RawProxy
    } else {
        DecodeFormat::Raster
    }
}

/// Admission units both decode gates are sized to for a byte budget.
fn decode_gate_units(cache_bytes: usize) -> usize {
    (cache_bytes / DECODE_UNIT_BYTES).clamp(2, 16)
}

/// Admission units one decode asks for: its estimated working set in 256 MiB
/// units. This stays the caller's true ask — the gate clamps the charge to its
/// own total when the request is registered — so a budget retune before the
/// grant cannot shrink what the caller originally needed.
fn decode_units(bytes: usize) -> usize {
    bytes.div_ceil(DECODE_UNIT_BYTES).max(1)
}

/// Removes a cached source and everything charged alongside it. Returns the
/// bytes released from `current_bytes`.
fn evict_cached_source(st: &mut State, image: ImageId) -> usize {
    let mut released = 0usize;
    if let Some(removed) = st.sources.remove(&image) {
        released = released.saturating_add(removed.bytes);
        st.source_cache.evictions += 1;
    }
    if let Some(input) = st.ai_inputs.remove(&image) {
        released = released.saturating_add(ai_input_bytes(&input));
    }
    st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(released);
    released
}

/// Evicts under budget pressure in a fixed order, never looping forever:
/// 1. AI inputs are pure derived data, so unpinned ones go first (no Source
///    is dropped for them);
/// 2. then the LRU of unpinned, unshared sources and the native slot;
/// 3. then the implicit "recent" pins are degraded oldest-first, keeping
///    `protect` and explicit pins;
/// 4. anything still over budget is protected by an explicit pin or a caller
///    holding the `Arc`; the peak is recorded and logged.
/// `protect` is the image that was just requested.
fn evict_sources_to_budget(st: &mut State, protect: Option<ImageId>) {
    if st.source_cache.current_bytes <= st.source_cache.max_bytes {
        return;
    }
    let active = st.source_cache.active_preview;
    let is_protected = |image: ImageId| Some(image) == protect || Some(image) == active;
    // 1) Drop unpinned AI inputs. They are recomputable and never a Source.
    let ai_victims: Vec<ImageId> = st
        .ai_inputs
        .keys()
        .copied()
        .filter(|image| !is_protected(*image) && !st.source_cache.is_pinned(*image))
        .collect();
    for image in ai_victims {
        if st.source_cache.current_bytes <= st.source_cache.max_bytes {
            break;
        }
        if let Some(input) = st.ai_inputs.remove(&image) {
            let bytes = ai_input_bytes(&input);
            st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(bytes);
            log::debug!("dropped AI input {image:?} under budget pressure: {bytes} bytes");
        }
    }
    // 2) LRU over unpinned sources and the native slot.
    while st.source_cache.current_bytes > st.source_cache.max_bytes {
        let victim = {
            let cache = &st.source_cache;
            let mut best: Option<(u64, bool, ImageId)> = None;
            for (id, entry) in st.sources.iter() {
                if is_protected(*id) || cache.is_pinned(*id) || Arc::strong_count(&entry.source) > 1 {
                    continue;
                }
                if best.is_none_or(|(last, _, _)| entry.last_used < last) {
                    best = Some((entry.last_used, false, *id));
                }
            }
            if let Some(slot) = &st.native_raw {
                if slot.bytes > 0
                    && !is_protected(slot.image)
                    && !cache.is_pinned(slot.image)
                    && slot.source.as_ref().is_none_or(|source| Arc::strong_count(source) <= 1)
                    && best.is_none_or(|(last, _, _)| slot.last_used < last)
                {
                    best = Some((slot.last_used, true, slot.image));
                }
            }
            best
        };
        let Some((_, is_native, image)) = victim else { break };
        if is_native {
            if let Some(slot) = st.native_raw.take() {
                st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(slot.bytes);
                st.source_cache.evictions += 1;
                log::debug!("evicted native RAW {image:?}: {} bytes", slot.bytes);
            } else {
                break;
            }
        } else {
            let released = evict_cached_source(st, image);
            if released == 0 {
                break;
            }
            log::debug!("evicted source {image:?}: {released} bytes");
        }
    }
    // 3) Degrade the oldest implicit recent pins; explicit pins, `protect`, and
    //    the active preview image stay. Each candidate is visited at most once.
    if st.source_cache.current_bytes > st.source_cache.max_bytes {
        let candidates: Vec<ImageId> = st.source_cache.recent.iter().rev().copied().collect();
        for image in candidates {
            if st.source_cache.current_bytes <= st.source_cache.max_bytes {
                break;
            }
            if is_protected(image) || st.source_cache.pins.get(&image).is_some_and(|count| *count > 0) {
                continue;
            }
            let evictable_source = st
                .sources
                .get(&image)
                .is_some_and(|entry| Arc::strong_count(&entry.source) <= 1);
            let evictable_native = st.native_raw.as_ref().is_some_and(|slot| {
                slot.image == image && slot.source.as_ref().is_none_or(|source| Arc::strong_count(source) <= 1)
            });
            st.source_cache.recent.retain(|id| *id != image);
            if evictable_source {
                let released = evict_cached_source(st, image);
                log::debug!("dropped recent pin and evicted source {image:?} under pressure: {released} bytes");
            } else if evictable_native {
                if let Some(slot) = st.native_raw.take() {
                    st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(slot.bytes);
                    st.source_cache.evictions += 1;
                    log::debug!("dropped recent pin and evicted native RAW {image:?}: {} bytes", slot.bytes);
                }
            }
        }
    }
    // 4) Still over: explicit pins or caller-held Arcs are hard protection.
    let over = st.source_cache.current_bytes.saturating_sub(st.source_cache.max_bytes);
    if over > 0 {
        st.source_cache.over_budget_peak = st.source_cache.over_budget_peak.max(over);
        log::debug!(
            "source cache over budget by {over} bytes after pressure eviction: {}/{} bytes, {} sources",
            st.source_cache.current_bytes,
            st.source_cache.max_bytes,
            st.sources.len()
        );
    }
}

#[derive(Clone)]
struct ScannedFile {
    rel_path: String,
    path: PathBuf,
    file_name: String,
    kind: FileKind,
    width: u32,
    height: u32,
    file_size: u64,
    exif: Exif,
    iptc: Iptc,
    packed: Option<EipAssets>,
    packed_document: Option<Arc<crate::c1_cos::CosDocument>>,
    packed_variant: u32,
}

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

fn geometry_fingerprint(geometry: &Geometry) -> u64 {
    let bytes = serde_json::to_vec(geometry).unwrap_or_default();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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
/// Capture One preview cache files are JXL, not RAW.
fn is_cop(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("cop"))
}

fn is_eip(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("eip"))
}

/// A C1 `.eip` package's RAW member, and its CoMask sidecar when present,
/// written under `<root>/.portrait/eip-cache` so the existing path-based
/// decoders read them unchanged.
#[derive(Clone)]
pub struct EipAssets {
    pub raw: PathBuf,
    pub settings: Option<PathBuf>,
    pub comask: Option<PathBuf>,
    /// `MaskUUID` of each local layer in the packed `.cos`.
    pub mask_uuids: Vec<String>,
}

/// A 60 MB RAW per package would pile up when a folder of EIPs is scanned in
/// parallel, so archive extraction stays single-flight.
static EIP_EXTRACT: Mutex<()> = Mutex::new(());

fn materialize_eip(root: &Path, package: &Path) -> anyhow::Result<EipAssets> {
    use std::hash::{Hash, Hasher};
    let _guard = EIP_EXTRACT.lock();
    let bytes = std::fs::read(package)?;
    let pkg = crate::c1_eip::EipPackage::parse(&bytes)?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    package.hash(&mut hasher);
    let stem = format!(
        "{}-{:016x}",
        package
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "package".into())
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect::<String>(),
        hasher.finish()
    );
    let dir = root.join(".portrait").join("eip-cache");
    std::fs::create_dir_all(&dir)?;

    let raw_name = &pkg.manifest.paths["RawPath"];
    let extension = Path::new(raw_name).extension().and_then(|e| e.to_str()).unwrap_or("raw");
    let raw = dir.join(format!("{stem}.{extension}"));
    write_stale(package, &raw, &pkg.member(raw_name)?)?;
    let settings = match pkg.manifest.paths.get("SettingsPath") {
        Some(name) => match pkg.member(name) {
            Ok(bytes) => {
                let path = dir.join(format!("{stem}.cos"));
                write_stale(package, &path, &bytes)?;
                Some((path, bytes))
            }
            Err(err) => {
                log::warn!("EIP {}: settings member {name}: {err}", package.display());
                None
            }
        },
        None => None,
    };
    let comask = match pkg.manifest.paths.get("CoMaskPath") {
        Some(name) => match pkg.member(name) {
            Ok(bytes) => {
                let path = dir.join(format!("{stem}.comask"));
                write_stale(package, &path, &bytes)?;
                Some(path)
            }
            Err(err) => {
                log::warn!("EIP {}: CoMask member {name}: {err}", package.display());
                None
            }
        },
        None => None,
    };
    let mask_uuids = settings
        .as_ref()
        .and_then(|(_, bytes)| crate::c1_cos::CosDocument::parse(bytes.as_ref()).ok())
        .and_then(|document| {
            document
                .variant()
                .map(|variant| variant.layers.iter().filter_map(|layer| layer.mask_uuid.clone()).collect())
        })
        .unwrap_or_default();
    Ok(EipAssets {
        raw,
        settings: settings.map(|(path, _)| path),
        comask,
        mask_uuids,
    })
}

/// Rewrites a cached member only when the package is newer than the copy.
fn write_stale(package: &Path, dest: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let stale = match (std::fs::metadata(dest), std::fs::metadata(package)) {
        (Ok(cached), Ok(source)) => match (cached.modified(), source.modified()) {
            (Ok(cached), Ok(source)) => cached < source,
            _ => true,
        },
        _ => true,
    };
    if stale {
        std::fs::write(dest, bytes)?;
    }
    Ok(())
}

fn raw16_source(path: &Path) -> Option<(Source, crate::raw::DecodeProvenance)> {
    if is_cop(path) {
        return None;
    }
    if !crate::raw::is_raw(path) {
        return None;
    }
    crate::raw_cache::get_or_decode_source16(path, SOURCE_LONG_SIDE)
}

/// [`raw16_source`] plus the bytes a failed 16-bit decode already read.
///
/// The 8-bit fallback consumes them instead of reading the container and
/// reopening LibRaw, so a cold load pays the file read once per attempt.
pub(crate) fn raw16_source_with_bytes(
    path: &Path,
) -> (
    Option<(Source, crate::raw::DecodeProvenance)>,
    Option<crate::raw::FailedRaw16>,
) {
    if is_cop(path) || !crate::raw::is_raw(path) {
        return (None, None);
    }
    crate::raw_cache::get_or_decode_source16_with_bytes(path, SOURCE_LONG_SIDE)
}

/// Decode the original file at its full resolution for an export whose
/// requested output is larger than the cached preview proxy.
///
/// Unlike the preview `source()` path this never applies [`fit_long_side`], so
/// a 100% export of a raster above [`SOURCE_LONG_SIDE`] is not an upscaled
/// proxy. The decode is one-shot: the result is not stored in the source cache
/// and is dropped together with the export job.
///
/// Admission note: [`FullResExportRequest::decode`] calls this from the export
/// worker after the `ExportScheduler` permit is taken and before the render.
/// The permit is sized from the output dimensions only, so this full-resolution
/// working set is *not* yet part of export admission accounting and adds to the
/// scheduler's estimate rather than being budgeted by it. The caller applies a
/// conservative memory cap before decoding; that cap is a safety refusal, not
/// scheduler accounting.
///
/// A RAW with no native decode has no full-resolution raster fallback: libraw's
/// 8-bit path may return only the camera's embedded preview, so refusing is safer
/// than silently upsampling. HEIF follows the same libheif → `image` fallback as
/// the preview source; when neither can read it, the export fails.
fn decode_full_res_export_source(
    root: &Path,
    path: &Path,
    lens: Option<crate::mock::lens_profile::LensShot>,
) -> anyhow::Result<Source> {
    #[cfg(test)]
    FULL_RES_EXPORT_DECODES.fetch_add(1, Ordering::Relaxed);
    let name = crate::output::log_name(path);
    let format = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_uppercase())
        .unwrap_or_else(|| "unknown".into());
    let decoded_path = if is_eip(path) {
        materialize_eip(root, path).map_err(|err| {
            anyhow::anyhow!(
                "export: {name} ({format}) cannot be decoded at full resolution: EIP package: {err:#}"
            )
        })?.raw
    } else {
        path.to_path_buf()
    };
    if crate::raw::is_raw(&decoded_path) {
        anyhow::bail!(
            "export: {name} ({format}) cannot be decoded at full resolution: native RAW decode \
             failed and no full-resolution raster fallback exists"
        );
    }
    let decoded = if is_cop(&decoded_path) {
        crate::jxl::decode_cop(&decoded_path).map_err(|err| {
            anyhow::anyhow!("export: {name} ({format}) cannot be decoded at full resolution: {err:#}")
        })?
    } else if file_kind(&decoded_path) == FileKind::Heif {
        match crate::heif::decode(&decoded_path) {
            Ok(image) => image,
            Err(err) => match image::open(&decoded_path) {
                Ok(image) => image.to_rgba8(),
                Err(fallback) => anyhow::bail!(
                    "export: {name} ({format}) cannot be decoded at full resolution: \
                     libheif: {err:#}; image crate: {fallback:#}"
                ),
            },
        }
    } else {
        image::open(&decoded_path).map_err(|err| {
            anyhow::anyhow!("export: {name} ({format}) cannot be decoded at full resolution: {err:#}")
        })?.to_rgba8()
    };
    let (width, height) = decoded.dimensions();
    if width == 0 || height == 0 {
        anyhow::bail!(
            "export: {name} ({format}) cannot be decoded at full resolution: \
             decoder returned an empty {width}x{height} frame"
        );
    }
    Ok(Source::from_rgba(width, height, decoded.as_raw()).with_lens_shot(lens))
}

/// What the standard export will render from.
///
/// `start_export` only decides this on the calling thread. A full-resolution
/// fallback is deferred to the export worker so a multi-megapixel original is
/// never decoded on the UI/caller thread.
enum StandardExportSource {
    /// Native RAW, or the cached preview source at or below the proxy cap.
    Ready(Arc<Source>),
    /// Decode the original at full resolution inside the export worker.
    FullRes(FullResExportRequest),
}

/// Estimated peak working set of one full-resolution export decode, per source
/// pixel: `Source::from_rgba` holds a linear f32 RGB plane (12 B/px) plus its
/// mip chain (×4/3 ≈ 16 B/px) while `image::open`'s RGBA8 raster (4 B/px) is
/// still alive. The extra 4 B/px over 20 is allocator/portrait headroom.
const FULL_RES_EXPORT_BYTES_PER_PX: u64 = 24;

fn full_res_export_estimate_bytes(width: u32, height: u32) -> u64 {
    (width as u64).saturating_mul(height as u64).saturating_mul(FULL_RES_EXPORT_BYTES_PER_PX)
}

/// Safety cap for one full-resolution export decode: a quarter of installed
/// RAM and at most half of what is free; 2 GiB when the platform cannot report
/// memory. This is a refusal threshold, *not* `ExportScheduler` accounting.
fn full_res_export_decode_cap() -> u64 {
    system_memory_bytes().map_or(2 * 1024 * 1024 * 1024, |(total, available)| {
        (total / 4).min(available / 2)
    })
}

/// Pure threshold test: `Err(estimate)` when the decode's estimated working set
/// exceeds `cap` bytes. Kept separate so the refusal is unit-testable without a
/// real 100 MP image.
fn full_res_export_decode_fits(width: u32, height: u32, cap: u64) -> Result<(), u64> {
    let estimate = full_res_export_estimate_bytes(width, height);
    if estimate > cap { Err(estimate) } else { Ok(()) }
}

/// Cheap, owned state for the deferred full-resolution decode. Paths stay
/// private to the worker; the decode itself is not cached.
struct FullResExportRequest {
    root: PathBuf,
    path: PathBuf,
    lens: Option<crate::mock::lens_profile::LensShot>,
    original_width: u32,
    original_height: u32,
    original_long: u32,
    requested_long: u32,
}

impl FullResExportRequest {
    /// Runs on the export worker after `ExportScheduler` admission and before
    /// the render. First refuses a decode whose estimated working set exceeds
    /// [`full_res_export_decode_cap`], so a huge raster cannot OOM the process
    /// (and is never silently downgraded to the proxy). Then a decoder that
    /// returned less than the file reported is rejected too: using it would
    /// upsample again.
    fn decode(&self) -> anyhow::Result<Source> {
        let cap = full_res_export_decode_cap();
        if let Err(estimate) =
            full_res_export_decode_fits(self.original_width, self.original_height, cap)
        {
            anyhow::bail!(
                "export: {} cannot be decoded at full resolution: the {}x{} frame needs about \
                 {} MiB (cap {} MiB); refusing rather than risking an out-of-memory failure",
                crate::output::log_name(&self.path),
                self.original_width,
                self.original_height,
                estimate.div_ceil(1024 * 1024),
                cap.div_ceil(1024 * 1024)
            );
        }
        let source = decode_full_res_export_source(&self.root, &self.path, self.lens.clone())?;
        let floor = self.requested_long.min(self.original_long).max(1);
        if source.size().0.max(source.size().1) < floor {
            anyhow::bail!(
                "export: {} decoded to {}x{} at full resolution, below the {} px long side the \
                 original reported",
                crate::output::log_name(&self.path),
                source.size().0,
                source.size().1,
                floor
            );
        }
        Ok(source)
    }
}

/// Capture ISO and body model for the output-domain BandingSuppression.
/// Proxy and native RAW sources must use the same fields; when either is
/// missing the stage stays off instead of guessing an amount.
fn raw_banding_identity(exif: &crate::Exif) -> (Option<u32>, Option<String>) {
    (exif.iso, exif.camera_model.clone())
}

/// Switch slightly before native 1:1 (rounding and minimum request sizes may
/// differ between the viewer and the engine). Keep the proxy for fit views.
fn wants_native_raw(
    sensor: (u32, u32), geometry: &crate::Geometry, region: NormRect, max_size: u32,
) -> bool {
    if sensor.0.max(sensor.1) <= SOURCE_LONG_SIDE || sensor.0 == 0 || sensor.1 == 0 {
        return false;
    }
    let (w, h) = crate::mock::geometry::FrameMap::new(sensor, geometry).frame_size();
    let region = process::clean_region(region);
    let footprint = (w * region.width).max(h * region.height).max(1.0);
    max_size as f32 >= 0.9 * footprint
}

/// Indexes `<source file name>.comask` sidecars below `Adjustments/LAM`.
fn collect_comask(dir: &Path, out: &mut HashMap<String, Vec<PathBuf>>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_comask(&path, out);
        } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("comask")) {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                out.entry(stem.to_string()).or_default().push(path);
            }
        }
    }
}

/// Bilinear sample of a decoded `.comask` plane in source coordinates.
fn sample_comask(plane: &crate::comask::MaskPlane, s: [f32; 2]) -> f32 {
    if plane.width == 0 || plane.height == 0 {
        return 0.0;
    }
    let x = (s[0] * plane.width as f32 - 0.5).clamp(0.0, (plane.width - 1) as f32);
    let y = (s[1] * plane.height as f32 - 0.5).clamp(0.0, (plane.height - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(plane.width - 1), (y0 + 1).min(plane.height - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let at = |xx: usize, yy: usize| plane.data[yy * plane.width + xx];
    let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
    let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
    top + (bottom - top) * ty
}

/// Rotates a source-space `.comask` plane into the renderer's oriented space.
///
// shortcut: orientation and flips are baked in; geometry.rotation and keystone
// are not (both zero in the sample catalog). The plane is capped at 2048 px
// because the render grid samples 256 cells, so this stays 8x above it.
fn oriented_mask_plane(
    plane: &crate::comask::MaskPlane,
    g: &crate::Geometry,
) -> crate::aimask::MaskPlane {
    let (ow, oh) = if g.orientation % 2 == 1 {
        (plane.height, plane.width)
    } else {
        (plane.width, plane.height)
    };
    let scale = (2048.0 / ow.max(oh).max(1) as f32).min(1.0);
    let w = ((ow as f32 * scale).round() as usize).max(1);
    let h = ((oh as f32 * scale).round() as usize).max(1);
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let o = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32];
            data.push(sample_comask(plane, crate::mock::process::oriented_to_source(o, g)));
        }
    }
    crate::aimask::MaskPlane::new(w, h, data)
}

/// C1 skews the ramp around a midpoint other than 0.5. That curve has no C1
/// reference sample yet, so say so instead of silently drawing a smoothstep.
fn warn_unverified_midpoint(layer: &str, gradient: [f32; 5]) {
    if !((gradient[4] - 0.5).abs() <= 1e-6) {
        log::warn!(
            "C1 layer {layer:?}: gradient midpoint {:.3} != 0.5; the skewed ramp is not matched to C1, drawing smoothstep",
            gradient[4]
        );
    }
}

/// Source point → oriented point, the inverse of
/// [`crate::mock::process::oriented_to_source`].
fn source_to_oriented(s: [f32; 2], g: &crate::Geometry) -> [f32; 2] {
    let q = match g.orientation % 4 {
        1 => [1.0 - s[1], s[0]],
        2 => [1.0 - s[0], 1.0 - s[1]],
        3 => [s[1], 1.0 - s[0]],
        _ => s,
    };
    [
        if g.flip_horizontal { 1.0 - q[0] } else { q[0] },
        if g.flip_vertical { 1.0 - q[1] } else { q[1] },
    ]
}

/// Builds one render layer per C1 local layer whose `.comask` mask is present.
fn attach_c1_layers(
    adjustments: &mut AdjustmentSet,
    variant: &crate::c1catalog::C1Variant,
    image: &crate::c1catalog::C1Image,
    comask_files: &HashMap<String, Vec<PathBuf>>,
) {
    let Some(candidates) = comask_files.get(&image.file_name) else {
        for layer in &variant.local_layers {
            log::warn!("C1 layer {:?}: no .comask under Adjustments/LAM; not attached", layer.name);
        }
        return;
    };
    let uuids: Vec<&str> = variant.local_layers.iter().filter_map(|l| l.mask_uuid.as_deref()).collect();
    for path in candidates {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let Ok(document) = crate::comask::ComaskDocument::parse(&bytes) else { continue };
        // Two variants can share a source name; take the file that owns them all.
        let owns_all = !uuids.is_empty()
            && uuids.iter().all(|uuid| document.masks.iter().any(|m| m.uuid.as_deref() == Some(*uuid)));
        if !owns_all {
            continue;
        }
        let Ok(planes) = crate::comask::read_comask_masks_for(&bytes, &uuids) else { continue };
        let by_uuid: HashMap<&str, &crate::comask::MaskPlane> = planes
            .iter()
            .filter_map(|p| p.uuid.as_deref().map(|u| (u, p)))
            .collect();
        let mut attached = Vec::new();
        for layer in &variant.local_layers {
            let Some(uuid) = layer.mask_uuid.as_deref() else {
                log::warn!("C1 layer {:?}: no ZMASKUUID; not attached", layer.name);
                continue;
            };
            let Some(record) = document.masks.iter().find(|m| m.uuid.as_deref() == Some(uuid)) else {
                log::warn!("C1 layer {:?}: UUID {uuid} missing from {}; not attached", layer.name, path.display());
                continue;
            };
            // A parameter gradient stays editable and costs no raster memory;
            // prefer it over the decoded plane of the same record.
            let shape = if let Some(g) = record.linear_gradient {
                warn_unverified_midpoint(layer.name.as_deref().unwrap_or("<unnamed>"), g);
                let w = image.width.max(1) as f32;
                let h = image.height.max(1) as f32;
                let zero_line = source_to_oriented([g[0] / w, g[1] / h], &adjustments.geometry);
                let full_line = source_to_oriented([g[2] / w, g[3] / h], &adjustments.geometry);
                // C1's ramp is zero at P0 and full at P1 (`comask::fill_gradient`,
                // matched to IC_MaskSampleInt), while `LinearGradient` is full at
                // `start`, so the C1 pair has to be fed reversed.
                MaskShape::Linear(LinearGradient { op: MaskOp::Add, start: full_line, end: zero_line })
            } else if let Some(plane) = by_uuid.get(uuid) {
                let oriented = oriented_mask_plane(plane, &adjustments.geometry);
                // The decoded bitmap is the layer's own geometry, so it rides
                // in the adjustment set instead of an engine-side cache key.
                let raster = crate::c1_masks::raster_shape_from_plane(
                    oriented.w,
                    oriented.h,
                    &oriented.data,
                    MaskOp::Add,
                )
                .unwrap_or_else(|| crate::RasterMask::empty(MaskOp::Add));
                MaskShape::Raster(raster)
            } else {
                // A mask record without pixels (an empty layer): a zero plane
                // keeps the layer attached without changing the frame.
                MaskShape::Raster(crate::RasterMask::empty(MaskOp::Add))
            };
            attached.push(Layer {
                id: layer.index.max(0) as u64 + 1,
                name: layer.name.clone().unwrap_or_else(|| format!("Layer {}", layer.index)),
                kind: crate::layers::LayerKind::Adjustment,
                enabled: layer.enabled,
                opacity: layer.opacity.clamp(0.0, 100.0),
                select_point: layer.select_point,
                mask_uuid: Some(uuid.to_owned()),
                style: None,
                cover: Vec::new(),
                mask: Mask {
                    shapes: vec![shape],
                    refine: layer.surface_smoothness.clamp(0.0, 100.0),
                    ..Default::default()
                },
                adjustments: layer.adjustments.clone(),
            });
        }
        adjustments.layers.extend(attached);
        return;
    }
    for layer in &variant.local_layers {
        log::warn!("C1 layer {:?}: no .comask mask decoded; not attached", layer.name);
    }
}

/// C1's `SessionFileLocationsProvider` puts a per-image `.cos` in
/// `<raw dir>/CaptureOne/Settings1680/<raw file name>.cos`. Session raws sit in
/// `Capture/`, so the session root is tried as well.
fn cos_settings_file(raw: &Path) -> Option<PathBuf> {
    let dir = raw.parent()?;
    let mut bases = vec![dir.join("CaptureOne")];
    if dir.file_name().is_some_and(|name| name.eq_ignore_ascii_case("Capture")) {
        if let Some(session) = dir.parent() {
            bases.push(session.join("CaptureOne"));
        }
    }
    let name = raw.file_name()?.to_string_lossy();
    let stem = raw.file_stem()?.to_string_lossy();
    for base in bases {
        for settings in settings_dirs(&base) {
            // Newest build first; some tools drop the RAW extension.
            for candidate in [
                settings.join(format!("{name}.cos")),
                settings.join(format!("{stem}.cos")),
            ] {
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

fn lcc_id_for(grid: &crate::dust_lcc::LightFalloff) -> String {
    use std::hash::{Hash, Hasher};
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    grid.header.hash(&mut hash);
    grid.columns.hash(&mut hash);
    grid.rows.hash(&mut hash);
    for node in &grid.nodes {
        node.luminance.hash(&mut hash);
        node.dx.hash(&mut hash);
        node.dy.hash(&mut hash);
    }
    format!("{:016x}", hash.finish())
}

/// `CaptureOne/Settings<build>` folders, newest build first.
fn settings_dirs(capture_one: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<(u32, PathBuf)> = std::fs::read_dir(capture_one)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let build = name.strip_prefix("Settings")?.parse().ok()?;
            entry.path().is_dir().then_some((build, entry.path()))
        })
        .collect();
    dirs.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    dirs.into_iter().map(|(_, path)| path).collect()
}

/// Applies a C1 per-image `.cos`: the `<DL>` global adjustments plus its local
/// `<AL>` layers that carry a `MaskUUID`. The mask comes from the `.comask`
/// sidecar beside the settings file, looked up by that UUID.
fn cos_adjustments(raw: &Path) -> Option<AdjustmentSet> {
    let settings = cos_settings_file(raw)?;
    read_cos_adjustments(raw, &settings, Some(&settings.with_extension("comask")), true)
}

impl ScannedFile {
    fn library_variants(self) -> impl Iterator<Item = Self> {
        let count = self.packed_document.as_ref().map_or(1, |document| document.variants.len().max(1));
        let additional = (2..=count as u32).map({
            let first = self.clone();
            move |number| {
                let mut scanned = first.clone();
                scanned.rel_path = format!("{}#eip-v{number}", first.rel_path);
                scanned.packed_variant = number;
                scanned
            }
        });
        std::iter::once(self).chain(additional)
    }

    fn c1_adjustments(&self) -> Option<AdjustmentSet> {
        if let Some(assets) = &self.packed {
            let document = self.packed_document.as_ref()?;
            let variant = document.variants.get(self.packed_variant as usize - 1)?;
            return Some(cos_variant_adjustments(
                &assets.raw,
                assets.settings.as_deref()?,
                assets.comask.as_deref(),
                false,
                variant,
            ));
        }
        if is_eip(&self.path) {
            return None;
        }
        cos_adjustments(&self.path)
    }
}

fn read_cos_adjustments(
    raw: &Path,
    settings: &Path,
    mask_path: Option<&Path>,
    session_lcc: bool,
) -> Option<AdjustmentSet> {
    let bytes = std::fs::read(settings).ok()?;
    let document = crate::c1_cos::CosDocument::parse(&bytes).ok()?;
    Some(cos_variant_adjustments(raw, settings, mask_path, session_lcc, document.variant()?))
}

fn cos_variant_adjustments(
    raw: &Path,
    settings: &Path,
    mask_path: Option<&Path>,
    session_lcc: bool,
    variant: &crate::c1_cos::CosVariant,
) -> AdjustmentSet {
    let comask_bytes = mask_path.and_then(|path| std::fs::read(path).ok());
    let comask =
        comask_bytes.as_deref().and_then(|bytes| crate::comask::ComaskDocument::parse(bytes).ok());
    let mut adjustments = variant.adjustments.clone();
    let uuids: Vec<&str> =
        variant.layers.iter().filter_map(|layer| layer.mask_uuid.as_deref()).collect();
    let planes = match comask_bytes.as_deref() {
        Some(bytes) if !uuids.is_empty() => {
            crate::comask::read_comask_masks_for(bytes, &uuids).unwrap_or_default()
        }
        _ => Vec::new(),
    };
    let by_uuid: HashMap<&str, &crate::comask::MaskPlane> =
        planes.iter().filter_map(|plane| plane.uuid.as_deref().map(|uuid| (uuid, plane))).collect();
    let (w, h) = dimensions(raw);
    let (w, h) = (w.max(1) as f32, h.max(1) as f32);

    // `Lccid` on the global layer names `<Settings>/<raw name>.<id>.lcc`.
    if let Some(lcc_id) = variant.lcc_id.as_deref().filter(|_| session_lcc) {
        let raw_name = raw.file_name().and_then(|name| name.to_str()).unwrap_or_default();
        let settings_dir = settings.parent().unwrap_or_else(|| Path::new(""));
        let path = crate::dust_lcc::sidecar_path(settings_dir, raw_name, lcc_id);
        match std::fs::read(&path) {
            Ok(bytes) => {
                if let Err(error) = adjustments.lens.load_dust_lcc(&bytes) {
                    log::warn!("C1 `.lcc` {} ignored: {error}", path.display());
                }
            }
            Err(_) => log::warn!("C1 `.lcc` {} missing", path.display()),
        }
    }

    let mut layers = Vec::new();
    for (index, layer) in variant.layers.iter().enumerate() {
        let Some(uuid) = layer.mask_uuid.as_deref() else { continue };
        let record = comask
            .as_ref()
            .and_then(|document| document.masks.iter().find(|mask| mask.uuid.as_deref() == Some(uuid)));
        // Same shape choice as a catalog import: a gradient stays editable and
        // needs no raster; anything else uses the decoded plane.
        let shape = if let Some(gradient) = record.and_then(|record| record.linear_gradient) {
            warn_unverified_midpoint(&format!("Layer {}", index + 1), gradient);
            let zero_line =
                source_to_oriented([gradient[0] / w, gradient[1] / h], &adjustments.geometry);
            let full_line =
                source_to_oriented([gradient[2] / w, gradient[3] / h], &adjustments.geometry);
            MaskShape::Linear(LinearGradient { op: MaskOp::Add, start: full_line, end: zero_line })
        } else if let Some(plane) = by_uuid.get(uuid) {
            let oriented = oriented_mask_plane(plane, &adjustments.geometry);
            let raster = crate::c1_masks::raster_shape_from_plane(
                oriented.w,
                oriented.h,
                &oriented.data,
                MaskOp::Add,
            )
            .unwrap_or_else(|| crate::RasterMask::empty(MaskOp::Add));
            MaskShape::Raster(raster)
        } else {
            log::warn!(
                "C1 `.cos` layer {uuid} in {}: no `.comask` mask; attached without pixels",
                settings.display()
            );
            MaskShape::Raster(crate::RasterMask::empty(MaskOp::Add))
        };
        layers.push(Layer {
            id: index as u64 + 1,
            name: format!("Layer {}", index + 1),
            kind: crate::layers::LayerKind::Adjustment,
            enabled: true,
            opacity: 100.0,
            select_point: None,
            mask: Mask { shapes: vec![shape], ..Default::default() },
            adjustments: local_adjustments(&layer.adjustments),
            ..Default::default()
        });
    }
    adjustments.layers.extend(layers);
    adjustments
}

/// A `.cos` layer body is a full [`AdjustmentSet`]; a render layer keeps only
/// the tools Capture One allows locally.
fn local_adjustments(a: &AdjustmentSet) -> LocalAdjustments {
    LocalAdjustments {
        white_balance: (a.white_balance != WhiteBalance::default()).then(|| a.white_balance.clone()),
        exposure: a.exposure.clone(),
        hdr: a.hdr.clone(),
        clarity: a.clarity.clone(),
        dehaze: a.dehaze.clone(),
        curves: (a.curves != Curves::default()).then(|| a.curves.clone()),
        levels: (a.levels != Levels::default()).then(|| a.levels.clone()),
        black_white: a.black_white.enabled.then(|| a.black_white.clone()),
        color_balance: a.color_balance.clone(),
        color_editor: a.color_editor.clone(),
        sharpening: (a.sharpening != Sharpening::default()).then(|| a.sharpening.clone()),
        noise_reduction: (a.noise_reduction != NoiseReduction::default())
            .then(|| a.noise_reduction.clone()),
        moire: a.moire.clone(),
        fill_color: None,
        ..Default::default()
    }
}

/// [`local_adjustments`] in reverse: a render layer's tools as a full set.
fn layer_adjustments(layer: &crate::Layer) -> AdjustmentSet {
    let local = &layer.adjustments;
    AdjustmentSet {
        white_balance: local.white_balance.clone().unwrap_or_default(),
        exposure: local.exposure.clone(),
        hdr: local.hdr.clone(),
        clarity: local.clarity.clone(),
        dehaze: local.dehaze.clone(),
        curves: local.curves.clone().unwrap_or_default(),
        levels: local.levels.clone().unwrap_or_default(),
        black_white: local.black_white.clone().unwrap_or_default(),
        color_balance: local.color_balance.clone(),
        color_editor: local.color_editor.clone(),
        sharpening: local.sharpening.clone().unwrap_or_default(),
        noise_reduction: local.noise_reduction.clone().unwrap_or_default(),
        moire: local.moire.clone(),
        ..Default::default()
    }
}

#[derive(Clone)]
struct FileEntry {
    info: ImageInfo,
    path: PathBuf,
    rel_path: String,
    adjustments: AdjustmentSet,
}

/// Batch size above which one library-wide id index is cheaper than a scan of
/// the entry list per id. Reuses the prior batch-lookup patch's fixed 8, so a
/// small batch does not build a whole-library map.
const BATCH_INDEX_MIN: usize = 8;

/// First slot of every id, so a duplicated id resolves to its first entry
/// exactly like `entries.iter().position(...)`.
fn image_indices(entries: &[FileEntry]) -> HashMap<ImageId, usize> {
    let mut indices = HashMap::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        indices.entry(entry.info.id).or_insert(index);
    }
    indices
}

/// Positions for `ids`, resolved without an `entries * ids` scan on a large
/// batch. Every id is checked before the caller mutates anything, so an
/// unknown id still leaves the entries untouched.
fn image_positions(entries: &[FileEntry], ids: &[ImageId]) -> anyhow::Result<Vec<usize>> {
    let indices = (ids.len() > BATCH_INDEX_MIN).then(|| image_indices(entries));
    ids.iter()
        .map(|id| match &indices {
            Some(indices) => indices.get(id).copied(),
            None => entries.iter().position(|entry| entry.info.id == *id),
        })
        .map(|position| position.ok_or_else(|| anyhow::anyhow!("unknown image")))
        .collect()
}

/// First catalog slot per wanted `rel_path`, filled by one catalog pass. A
/// duplicated path keeps its first slot, matching `Catalog::upsert`.
fn catalog_positions(catalog: &Catalog, wanted: &[&str]) -> HashMap<String, usize> {
    let wanted: HashSet<&str> = wanted.iter().copied().collect();
    let mut positions = HashMap::with_capacity(wanted.len());
    for (slot, entry) in catalog.images.iter().enumerate() {
        if wanted.contains(entry.rel_path.as_str()) {
            positions.entry(entry.rel_path.clone()).or_insert(slot);
        }
    }
    positions
}

/// Applies `entries` with `Catalog::upsert`'s exact replace-first / append-new
/// result. A large batch indexes the wanted paths once instead of scanning the
/// whole catalog per record.
fn upsert_entries(catalog: &mut Catalog, entries: Vec<Entry>) {
    if entries.len() <= BATCH_INDEX_MIN {
        for entry in entries {
            catalog.upsert(entry);
        }
        return;
    }
    let mut positions = {
        let wanted: Vec<&str> = entries.iter().map(|entry| entry.rel_path.as_str()).collect();
        catalog_positions(catalog, &wanted)
    };
    for entry in entries {
        let slot = positions.get(entry.rel_path.as_str()).copied();
        match slot {
            Some(slot) => catalog.images[slot] = entry,
            None => {
                positions.insert(entry.rel_path.clone(), catalog.images.len());
                catalog.images.push(entry);
            }
        }
    }
}

struct AiInput {
    source_id: u64,
    geometry: u64,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

/// One decoded image in the source cache, charged by its real mip bytes plus
/// any portrait derivatives it has built.
struct CachedSource {
    source: Arc<Source>,
    bytes: usize,
    last_used: u64,
    /// Library file this source was decoded from, for identity checks.
    path: PathBuf,
    identity: SourceFileIdentity,
    last_verified: Instant,
    hits_since_verify: u32,
}

/// At most one full-size RAW; a `None` source remembers a failure for that image.
struct NativeRawSlot {
    image: ImageId,
    source: Option<Arc<Source>>,
    bytes: usize,
    last_used: u64,
    identity: SourceFileIdentity,
    last_verified: Instant,
    hits_since_verify: u32,
}

/// Byte-budgeted LRU for decoded sources. `native_raw` and `ai_inputs` share
/// the same bytes.
struct SourceCacheBudget {
    max_bytes: usize,
    current_bytes: usize,
    peak_bytes: usize,
    /// Largest observed `current_bytes - max_bytes` while pinned data held the
    /// cache over budget.
    over_budget_peak: usize,
    hits: u64,
    native_hits: u64,
    misses: u64,
    evictions: u64,
    /// Cached sources dropped because the file behind them changed.
    identity_invalidations: u64,
    single_flight_waits: u64,
    recent_pins: usize,
    /// Explicit pins, e.g. images an export is still reading.
    pins: HashMap<ImageId, u32>,
    /// Currently previewed image: protected from background thumbnail eviction
    /// even when implicit `recent` pins degrade under memory pressure.
    active_preview: Option<ImageId>,
    /// Current image and its recently visited neighbours: implicit pins.
    recent: VecDeque<ImageId>,
    tick: u64,
}

impl SourceCacheBudget {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current_bytes: 0,
            peak_bytes: 0,
            over_budget_peak: 0,
            hits: 0,
            native_hits: 0,
            misses: 0,
            evictions: 0,
            identity_invalidations: 0,
            single_flight_waits: 0,
            recent_pins: recent_source_pins(),
            pins: HashMap::new(),
            active_preview: None,
            recent: VecDeque::new(),
            tick: 0,
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }

    fn note_use(&mut self, image: ImageId) {
        if self.recent_pins == 0 {
            return;
        }
        self.recent.retain(|id| *id != image);
        self.recent.push_front(image);
        while self.recent.len() > self.recent_pins {
            self.recent.pop_back();
        }
    }

    fn is_pinned(&self, image: ImageId) -> bool {
        self.pins.get(&image).is_some_and(|count| *count > 0) || self.recent.contains(&image)
    }
}

/// Snapshot of the decoded-source cache for the performance log and for the
/// unified memory budget in S7.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceCacheStats {
    pub max_bytes: usize,
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub cached_sources: usize,
    pub pinned_sources: usize,
    pub native_bytes: usize,
    /// Bytes charged to `State::ai_inputs`.
    pub ai_input_bytes: usize,
    /// Portrait RGBA/mask buffers currently held by cached sources.
    pub derived_bytes: usize,
    /// Bytes of cached sources an explicit or recent pin protects.
    pub pinned_bytes: usize,
    pub over_budget_bytes: usize,
    pub over_budget_peak: usize,
    pub identity_invalidations: u64,
    pub hits: u64,
    pub native_hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub single_flight_waits: u64,
    pub decode_waits: u64,
    pub decode_wait_ms: u64,
    pub decode_peak_concurrency: usize,
    pub heavy_units: usize,
    pub light_units: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DecodePriority {
    /// The visible preview, 1:1 zoom, sampling and export.
    Interactive = 0,
    /// Grid and filmstrip thumbnails: never ahead of the visible frame.
    Thumbnail = 1,
}

struct GateWaiter {
    ticket: u64,
    priority: DecodePriority,
    /// Units the caller asked for. A queued request is charged
    /// `requested_units.min(total_units)` and gets the original ask back when
    /// the budget grows again.
    requested_units: usize,
    /// Units this request is charged once it is granted. Retuning only ever
    /// touches requests that have not been granted yet.
    units: usize,
    granted: bool,
}

struct GateInner {
    total_units: usize,
    /// Units held by granted requests, including grants whose caller has not
    /// returned from `acquire` yet. `free_units` is derived from this and
    /// `total_units` so no separate free counter can drift when the budget
    /// shrinks below what is already in flight.
    occupied_units: usize,
    next_ticket: u64,
    waiters: Vec<GateWaiter>,
    active: usize,
    peak_active: usize,
}

impl GateInner {
    fn free_units(&self) -> usize {
        self.total_units.saturating_sub(self.occupied_units)
    }
}

/// A priority-aware counting semaphore. A decode reserves units proportional to
/// its estimated working set; callers with `Interactive` priority are admitted
/// before thumbnails, and large RAW frames simply take more units.
struct DecodeGate {
    inner: Mutex<GateInner>,
    released: Condvar,
    waits: AtomicU64,
    wait_nanos: AtomicU64,
}

struct GatePermit<'a> {
    gate: &'a DecodeGate,
    /// Exactly the units charged at grant time; returned on drop.
    units: usize,
}

/// Grants every waiter that fits, highest priority first, then oldest ticket.
fn dispatch_gate(inner: &mut GateInner) -> bool {
    let mut granted = false;
    loop {
        let free = inner.free_units();
        let Some(index) = inner
            .waiters
            .iter()
            .enumerate()
            .filter(|(_, waiter)| !waiter.granted && waiter.units <= free)
            .min_by_key(|(_, waiter)| (waiter.priority, waiter.ticket))
            .map(|(index, _)| index)
        else {
            break;
        };
        let units = inner.waiters[index].units;
        inner.waiters[index].granted = true;
        inner.occupied_units = inner.occupied_units.saturating_add(units);
        inner.active += 1;
        inner.peak_active = inner.peak_active.max(inner.active);
        granted = true;
    }
    granted
}

impl DecodeGate {
    fn new(total_units: usize) -> Self {
        let total_units = total_units.max(1);
        Self {
            inner: Mutex::new(GateInner {
                total_units,
                occupied_units: 0,
                next_ticket: 0,
                waiters: Vec::new(),
                active: 0,
                peak_active: 0,
            }),
            released: Condvar::new(),
            waits: AtomicU64::new(0),
            wait_nanos: AtomicU64::new(0),
        }
    }

    fn total_units(&self) -> usize {
        self.inner.lock().total_units
    }

    /// Retunes admission when the RAM budget changes. Granted requests keep the
    /// units they were charged because they are already in flight; only queued
    /// requests are reclamped to the new total, and a request that was clamped
    /// down gets its original ask back when the budget grows.
    fn set_total_units(&self, total_units: usize) {
        let total_units = total_units.max(1);
        let mut inner = self.inner.lock();
        inner.total_units = total_units;
        for waiter in &mut inner.waiters {
            if !waiter.granted {
                waiter.units = waiter.requested_units.clamp(1, total_units);
            }
        }
        if dispatch_gate(&mut inner) {
            self.released.notify_all();
        }
    }

    /// Registers a request without charging it; units are charged at grant.
    fn enqueue(inner: &mut GateInner, priority: DecodePriority, units: usize) -> u64 {
        let requested_units = units.max(1);
        let ticket = inner.next_ticket;
        inner.next_ticket = inner.next_ticket.wrapping_add(1);
        inner.waiters.push(GateWaiter {
            ticket,
            priority,
            requested_units,
            units: requested_units.min(inner.total_units.max(1)),
            granted: false,
        });
        ticket
    }

    fn is_granted(inner: &GateInner, ticket: u64) -> bool {
        inner.waiters.iter().find(|waiter| waiter.ticket == ticket).is_some_and(|waiter| waiter.granted)
    }

    /// Removes a granted request and returns the units it was charged. A granted
    /// request's charge never changes, so this is exactly what the permit holds
    /// until drop.
    fn claim(inner: &mut GateInner, ticket: u64) -> usize {
        let index = inner
            .waiters
            .iter()
            .position(|waiter| waiter.ticket == ticket)
            .expect("granted decode request must still be registered");
        inner.waiters.remove(index).units
    }

    fn acquire(&self, priority: DecodePriority, units: usize) -> GatePermit<'_> {
        let mut inner = self.inner.lock();
        let ticket = Self::enqueue(&mut inner, priority, units);
        dispatch_gate(&mut inner);
        if !Self::is_granted(&inner, ticket) {
            self.waits.fetch_add(1, Ordering::Relaxed);
            let start = Instant::now();
            while !Self::is_granted(&inner, ticket) {
                self.released.wait(&mut inner);
            }
            let waited = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            self.wait_nanos.fetch_add(waited, Ordering::Relaxed);
        }
        let units = Self::claim(&mut inner, ticket);
        GatePermit { gate: self, units }
    }

    fn peak_active(&self) -> usize {
        self.inner.lock().peak_active
    }

    fn waits(&self) -> u64 {
        self.waits.load(Ordering::Relaxed)
    }

    fn wait_nanos(&self) -> u64 {
        self.wait_nanos.load(Ordering::Relaxed)
    }
}

impl Drop for GatePermit<'_> {
    fn drop(&mut self) {
        let mut inner = self.gate.inner.lock();
        inner.occupied_units = inner.occupied_units.saturating_sub(self.units);
        inner.active = inner.active.saturating_sub(1);
        if dispatch_gate(&mut inner) {
            self.gate.released.notify_all();
        }
    }
}

/// RAW proxy/native decodes and light JPEG/PNG decodes have separate admission,
/// so a 45MP native frame cannot block a filmstrip thumbnail behind it. Within
/// each gate, preview work outranks thumbnails. Both gates are sized from the
/// byte budget and every request reserves units for its estimated working set,
/// so neither channel can bypass the memory budget.
struct DecodeGates {
    heavy: DecodeGate,
    light: DecodeGate,
}

impl DecodeGates {
    fn new(cache_bytes: usize) -> Self {
        let units = decode_gate_units(cache_bytes);
        Self { heavy: DecodeGate::new(units), light: DecodeGate::new(units) }
    }
}

fn default_export_concurrency() -> usize {
    if let Some(n) = env_usize("AURORA_EXPORT_CONCURRENCY") {
        return n.max(1);
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .clamp(1, 4)
        .min(2)
}

fn default_export_memory_budget() -> usize {
    match system_memory_bytes() {
        Some((total, available)) => export_memory_budget_from_memory(total, available),
        None => EXPORT_BUDGET_FALLBACK_BYTES,
    }
}

/// Smallest budget. A job larger than this still runs, alone, through the
/// scheduler's `active_slots == 0` door, so the floor only decides how many
/// *small* jobs pair, never whether a large one starts.
const EXPORT_BUDGET_MIN_BYTES: usize = 512 * 1024 * 1024;

/// Only used when the platform cannot report installed memory.
const EXPORT_BUDGET_FALLBACK_BYTES: usize = 1024 * 1024 * 1024;

/// What the export budget must leave for everything that is not an export: the
/// 3 GiB foreground viewer, the decoded-source cache this engine keeps (2.67 GiB
/// at 32 GiB installed, from [`source_cache_budget_from_memory`]) and the
/// engine's own steady state. The last term is measured, not guessed: the
/// `r10-real-strict-v1` solo arm ended a 24 MP 16-bit batch at 817 MiB working
/// set with 372 MiB of native source cached, so everything but the source and
/// the export buffers is about 445 MiB.
const EXPORT_BUDGET_RESERVE: u64 = 6 * 1024 * 1024 * 1024;

/// Frame the ceiling is sized from: the 24 MP frame a batch is usually built
/// from. Two of these must pair; three must not.
const EXPORT_CEILING_REFERENCE: (u32, u32) = (6000, 4000);

/// Hard ceiling for the export admission budget, derived from the charge the
/// scheduler already applies rather than from a round number.
///
/// One 24 MP 16-bit native-RAW job charges
/// `estimated_decode_bytes(6000,4000,RawNative)` (672 MB: 384 MB retained +
/// 288 MB decode transient) + `estimated_export_bytes(6000,4000,16)` (864 MB) =
/// 1,536 MB, so two jobs need 3,072 MB. The ceiling is 2.4 jobs
/// (1,536 x 12 / 5 = 3,686 MB = 3,516 MiB): a 26-30 MP frame can still pair
/// with a 24 MP one, while a third 24 MP job at the real 6,048x4,032 size
/// (3 x 1,561 MB = 4,683 MB) can never fit, and two 45 MP jobs
/// (2 x 2,908 MB = 5,816 MB) never fit either.
fn export_budget_ceiling() -> usize {
    let reference = estimated_decode_bytes(
        EXPORT_CEILING_REFERENCE.0,
        EXPORT_CEILING_REFERENCE.1,
        DecodeFormat::RawNative,
    )
    .saturating_add(estimated_export_bytes(
        EXPORT_CEILING_REFERENCE.0,
        EXPORT_CEILING_REFERENCE.1,
        16,
    ));
    reference.saturating_mul(12) / 5
}

/// Export budget: an eighth of installed memory, shrunk by whatever the machine
/// does not actually have free, then clamped into [min, ceiling].
///
/// An eighth is the point where a 32 GiB machine reaches the ceiling (two full
/// 24 MP jobs) and a 16 GiB machine stays at one (1,489 MiB fits, two do not).
/// The available-memory term is what keeps the budget honest under pressure: a
/// 32 GiB machine with 5 GiB free gets the 512 MiB floor, not 3,516 MiB.
fn export_memory_budget_from_memory(total: u64, available: u64) -> usize {
    let by_installation = usize::try_from(total / 8).unwrap_or(usize::MAX);
    let by_availability =
        usize::try_from(available.saturating_sub(EXPORT_BUDGET_RESERVE)).unwrap_or(0);
    by_installation
        .min(by_availability)
        .clamp(EXPORT_BUDGET_MIN_BYTES, export_budget_ceiling())
}

/// Rough peak working-set estimate for one export job based on output dimensions
/// and bit depth, used to serialize large-frame (e.g. 24MP/45MP) exports.
fn estimated_export_bytes(width: u32, height: u32, bit_depth: u8) -> usize {
    let pixels = (width as usize).saturating_mul(height as usize);
    let bytes_per_px = if bit_depth == 16 { 36usize } else { 24usize };
    pixels.saturating_mul(bytes_per_px).max(1)
}

/// Pixel size charged for a source whose real dimensions could not be read.
/// This is a conservative floor, not an upper bound: it only keeps an
/// unreadable size from being priced as free, while a frame that really is
/// larger is charged through its real (catalog) dimensions instead.
const UNKNOWN_SOURCE_SIZE: (u32, u32) = (6000, 4000);

/// Bytes one export is charged: the source frame it decodes plus the frame it
/// writes.
///
/// Both are derived from the catalog (the scan already read the dimensions) and
/// the recipe's geometry, so the charge is known before a pixel is decoded. The
/// source side uses the decode envelope of the format the job will read — the
/// native frame for a RAW or EIP, the proxy for everything else — and an
/// unreadable size is charged at [`UNKNOWN_SOURCE_SIZE`], a documented floor
/// rather than an upper bound, so a missing measurement is never priced as
/// free.
fn estimated_export_charge(recipe: &crate::ProcessRecipe, info: &ImageInfo, geometry: &Geometry) -> usize {
    let source = if info.width == 0 || info.height == 0 {
        UNKNOWN_SOURCE_SIZE
    } else {
        (info.width, info.height)
    };
    let format = if crate::raw::is_raw(&info.path) || is_eip(&info.path) {
        DecodeFormat::RawNative
    } else {
        DecodeFormat::Raster
    };
    let input_bytes = estimated_decode_bytes(source.0, source.1, format);
    let (width, height) = crate::mock::geometry::FrameMap::new(source, geometry).frame_size();
    let edited_size = (width.round().max(1.0) as u32, height.round().max(1.0) as u32);
    let (width, height) = crate::recipe::output_size(recipe, edited_size);
    input_bytes.saturating_add(estimated_export_bytes(width, height, recipe.bit_depth))
}

/// Whether the queued request is one the Windows exact desktop path can serve,
/// judged only from the catalog path, the queued adjustments and the recipe.
///
/// The specialization itself also proves the file is a Z6III HE NEF before it
/// runs, which needs the bytes; admission cannot wait for that, so it mirrors
/// the two pure gates the export opens with — the `.nef` extension and
/// [`crate::z6iii::desktop::compatible`] — and charges the exact envelope only
/// for requests that can actually reach that path. A non-NEF or a recipe the
/// specialised export rejects keeps the standard estimate, so an ordinary JPEG
/// is never charged the full RAW working set.
#[cfg(all(windows, feature = "z6iii-raw-export"))]
fn exact_export_candidate(
    info: &ImageInfo,
    adjustments: &crate::AdjustmentSet,
    recipe: &crate::ProcessRecipe,
) -> bool {
    info.path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("nef"))
        && crate::z6iii::desktop::compatible(adjustments, recipe)
}

/// Working set of an exact desktop (Z6III) export: it reads and writes the RAW
/// at its own size, so the geometry-scaled standard estimate is not an upper
/// bound for it and the two must be charged with the larger one. Only a request
/// [`exact_export_candidate`] admits pays this; everything else is charged the
/// standard estimate alone.
#[cfg(all(windows, feature = "z6iii-raw-export"))]
fn estimated_exact_export_charge(recipe: &crate::ProcessRecipe, info: &ImageInfo) -> usize {
    let source = if info.width == 0 || info.height == 0 {
        UNKNOWN_SOURCE_SIZE
    } else {
        (info.width, info.height)
    };
    estimated_decode_bytes(source.0, source.1, DecodeFormat::RawNative)
        .saturating_add(estimated_export_bytes(source.0, source.1, recipe.bit_depth))
}

/// Bytes one job is admitted with: the standard source-plus-output charge, and
/// for an [`exact_export_candidate`] the larger of that and the exact RAW
/// envelope. Pure and catalog-only, so it runs before the scheduler admits the
/// job and before any pixel is decoded.
#[cfg(all(windows, feature = "z6iii-raw-export"))]
fn estimated_job_charge(
    recipe: &crate::ProcessRecipe,
    info: &ImageInfo,
    adjustments: &crate::AdjustmentSet,
) -> usize {
    let standard = estimated_export_charge(recipe, info, &adjustments.geometry);
    if exact_export_candidate(info, adjustments, recipe) {
        standard.max(estimated_exact_export_charge(recipe, info))
    } else {
        standard
    }
}

struct ExportSchedulerInner {
    max_slots: usize,
    memory_budget_bytes: usize,
    active_slots: usize,
    used_bytes: usize,
    next_ticket: u64,
    queue: VecDeque<u64>,
    peak_active: usize,
}

/// Global bounded-concurrency and memory-budget admission gate for exports.
struct ExportScheduler {
    inner: Mutex<ExportSchedulerInner>,
    cv: Condvar,
}

/// RAII permit that releases an export slot and memory quota on drop/panic.
struct ExportPermit<'a> {
    scheduler: &'a ExportScheduler,
    charge_bytes: usize,
}

impl ExportScheduler {
    fn with_limits(max_slots: usize, memory_budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(ExportSchedulerInner {
                max_slots: max_slots.max(1),
                memory_budget_bytes: memory_budget_bytes.max(1),
                active_slots: 0,
                used_bytes: 0,
                next_ticket: 0,
                queue: VecDeque::new(),
                peak_active: 0,
            }),
            cv: Condvar::new(),
        }
    }

    fn acquire_cancellable(
        &self,
        estimated_bytes: usize,
        mut is_cancelled: impl FnMut() -> bool,
    ) -> Option<ExportPermit<'_>> {
        if is_cancelled() {
            return None;
        }
        let mut inner = self.inner.lock();
        let charge_bytes = estimated_bytes.clamp(1, inner.memory_budget_bytes);
        let ticket = inner.next_ticket;
        inner.next_ticket = inner.next_ticket.wrapping_add(1);
        inner.queue.push_back(ticket);
        loop {
            if is_cancelled() {
                inner.queue.retain(|&t| t != ticket);
                self.cv.notify_all();
                return None;
            }
            let at_front = inner.queue.front() == Some(&ticket);
            let fits_slots = inner.active_slots < inner.max_slots;
            let fits_mem = inner.active_slots == 0
                || inner.used_bytes.saturating_add(charge_bytes) <= inner.memory_budget_bytes;
            if at_front && fits_slots && fits_mem {
                inner.queue.pop_front();
                inner.active_slots += 1;
                inner.used_bytes = inner.used_bytes.saturating_add(charge_bytes);
                inner.peak_active = inner.peak_active.max(inner.active_slots);
                self.cv.notify_all();
                return Some(ExportPermit { scheduler: self, charge_bytes });
            }
            self.cv.wait_for(&mut inner, std::time::Duration::from_millis(20));
        }
    }

    fn notify_all(&self) {
        self.cv.notify_all();
    }

    #[cfg(test)]
    fn stats(&self) -> (usize, usize, usize) {
        let inner = self.inner.lock();
        (inner.active_slots, inner.used_bytes, inner.peak_active)
    }
}

impl Drop for ExportPermit<'_> {
    fn drop(&mut self) {
        let mut inner = self.scheduler.inner.lock();
        inner.active_slots = inner.active_slots.saturating_sub(1);
        inner.used_bytes = inner.used_bytes.saturating_sub(self.charge_bytes);
        self.scheduler.cv.notify_all();
    }
}

fn export_scheduler() -> &'static ExportScheduler {
    static SCHEDULER: std::sync::OnceLock<ExportScheduler> = std::sync::OnceLock::new();
    SCHEDULER.get_or_init(|| {
        ExportScheduler::with_limits(default_export_concurrency(), default_export_memory_budget())
    })
}

/// Single-flight key: the 4096 proxy and the full native frame decode separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SourceKey {
    Proxy(ImageId),
    Native(ImageId),
}

#[derive(Default)]
struct Flight {
    result: Mutex<Option<Result<Arc<Source>, String>>>,
    ready: Condvar,
}

impl Flight {
    fn publish(&self, result: Result<Arc<Source>, String>) {
        *self.result.lock() = Some(result);
        self.ready.notify_all();
    }

    fn wait(&self) -> Result<Arc<Source>, String> {
        let mut slot = self.result.lock();
        loop {
            if let Some(result) = slot.as_ref() {
                return result.clone();
            }
            self.ready.wait(&mut slot);
        }
    }
}

/// The decoding half of the engine.
///
/// The thread that owns a decode. If it unwinds, waiters are released with an
/// error instead of hanging on the flight forever.
struct FlightLeader<'a> {
    provider: &'a FileEngine,
    key: SourceKey,
    flight: Arc<Flight>,
    finished: bool,
}

impl FlightLeader<'_> {
    fn finish(mut self, result: Result<Arc<Source>, String>) -> Result<Arc<Source>, String> {
        self.flight.publish(result.clone());
        self.provider.flights.lock().remove(&self.key);
        self.finished = true;
        result
    }
}

impl Drop for FlightLeader<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.flight.publish(Err("decode aborted".into()));
            self.provider.flights.lock().remove(&self.key);
        }
    }
}

enum FlightRole<'a> {
    Leader(FlightLeader<'a>),
    Waiter(Arc<Flight>),
}

struct State {
    entries: Vec<FileEntry>,
    next_id: u64,
    sources: HashMap<ImageId, CachedSource>,
    /// At most one full-size RAW; remember failure for the current image too.
    native_raw: Option<NativeRawSlot>,
    ai_inputs: HashMap<ImageId, Arc<AiInput>>,
    /// Byte budget, LRU bookkeeping, pins and counters for `sources` + `native_raw`.
    source_cache: SourceCacheBudget,
    /// Why a source is not full sensor data; consumed by `source_status`.
    source_reasons: HashMap<ImageId, String>,
    /// Local layers read from a C1 catalog: listed in the panel, never rendered.
    layers: HashMap<ImageId, Vec<crate::engine::LayerSummary>>,
    /// Read-only retouching mask summaries keyed by image.
    retouch: HashMap<ImageId, Vec<crate::c1_retouch::RetouchMaskSummary>>,
    /// Collections C1 stores, and their members in this engine's ids.
    c1_collections: Vec<Collection>,
    c1_members: HashMap<u64, Vec<ImageId>>,
    catalog: Catalog,
    preview: PreviewCache,
}

/// One image's result from [`FileEngine::save_adjustments_queued`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueuedDocumentResult {
    /// The image variant the edit belongs to.
    pub image: ImageId,
    /// Path relative to the library root, or empty for an unknown image.
    pub path: String,
    pub outcome: QueuedSaveOutcome,
}

/// What happened to one document submitted to the bounded save queue.
///
/// The in-memory edit is applied before the outcome is known, so every variant
/// here except [`Self::Failed`] for an unknown image means the user's latest
/// content is already visible in this engine. Only [`Self::Accepted`] promises
/// the writer will try to persist it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueuedSaveOutcome {
    /// Queued with a strictly increasing revision. `flush_saves` reports what
    /// the writer did with it.
    Accepted { revision: DocumentRevision },
    /// The bounded queue is full for a new path. The edit is kept in memory but
    /// will not be written by this submission; flush and resubmit to retry.
    Full { capacity: usize },
    /// The queue is closed (for example the writer thread could not start).
    /// The edit stays in memory.
    Closed,
    /// An at-least-as-new revision already owns the path, so this submission was
    /// discarded. Only reachable in practice at revision saturation; the newer
    /// revision owns the bytes.
    Superseded { newest: DocumentRevision },
    /// The submission could not be made at all: an unknown image id or a
    /// library with no writable store.
    Failed { error: String },
}

/// Per-image results of one [`FileEngine::save_adjustments_queued`] call.
///
/// This is not a per-path queue snapshot; use [`FileEngine::flush_saves`] for
/// what actually reached disk and [`FileEngine::save_queue_status`] for the
/// bounded queue's current state.
#[derive(Clone, Debug, Default)]
pub struct QueuedSaveSummary {
    /// One result per input item, in input order.
    pub documents: Vec<QueuedDocumentResult>,
}

impl QueuedSaveSummary {
    /// Number of documents accepted by the queue.
    pub fn accepted(&self) -> usize {
        self.documents
            .iter()
            .filter(|document| matches!(document.outcome, QueuedSaveOutcome::Accepted { .. }))
            .count()
    }

    /// Number of new paths refused because the queue was at capacity.
    pub fn full(&self) -> usize {
        self.documents.iter().filter(|document| matches!(document.outcome, QueuedSaveOutcome::Full { .. })).count()
    }

    /// Number of submissions refused because the queue was closed.
    pub fn closed(&self) -> usize {
        self.documents.iter().filter(|document| matches!(document.outcome, QueuedSaveOutcome::Closed)).count()
    }

    /// Number of submissions discarded because a newer revision owned the path.
    pub fn superseded(&self) -> usize {
        self.documents
            .iter()
            .filter(|document| matches!(document.outcome, QueuedSaveOutcome::Superseded { .. }))
            .count()
    }

    /// Number of submissions that could not be made (unknown image / no store).
    pub fn failed(&self) -> usize {
        self.documents
            .iter()
            .filter(|document| matches!(document.outcome, QueuedSaveOutcome::Failed { .. }))
            .count()
    }

    /// The outcome recorded for `image`, if it was part of this call.
    pub fn outcome_of(&self, image: ImageId) -> Option<&QueuedSaveOutcome> {
        self.documents.iter().find(|document| document.image == image).map(|document| &document.outcome)
    }

    /// The first failure, for a concise diagnostic.
    pub fn first_failure(&self) -> Option<&QueuedDocumentResult> {
        self.documents
            .iter()
            .find(|document| matches!(document.outcome, QueuedSaveOutcome::Failed { .. }))
    }
}

/// An engine over a real folder of images.
pub struct FileEngine {
    root: PathBuf,
    state: Arc<Mutex<State>>,
    /// Per-document save ordering; see [`services::SaveOrder`].
    save_order: Mutex<services::SaveOrder>,
    /// ICC conversion has its own lock: a cold profile miss must not block the
    /// catalog, thumbnails, or source-cache readers.
    profiled: Mutex<crate::engine::ProfileCache>,
    /// Gamut counts are tiny compared with the frame; keep recent proof results
    /// so a repaint does not repeat the full round trip.
    gamut: Mutex<Vec<(preview_cache::PreviewKey, String, crate::GamutWarning)>>,
    /// The last preview's graph identity, so a request that differs only by a
    /// pruned-branch edit can reuse the frame on screen instead of rendering.
    graph_gate: Mutex<preview_cache::GraphGate>,
    /// One in-flight decode per `SourceKey`; later callers wait for that result
    /// instead of decoding the same image again.
    flights: Mutex<HashMap<SourceKey, Arc<Flight>>>,
    /// One full-size RAW at a time: the native slot holds a single frame, so a
    /// second native decode would only overwrite the first.
    native_decode: Mutex<()>,
    /// Bounded, priority-aware admission per decode class.
    decode_gates: DecodeGates,
    /// Serializes the final cancel check against export publication
    /// (`gate -> registry/maps`; see [`FileEngine::start_export`]).
    publish_gate: Arc<Mutex<()>>,
    /// Export job lifecycle. The registry owns the state and cancellation
    /// token, so a cancel never takes the document `state` lock.
    export_jobs: JobRegistry,
    /// UI job id (`ExportJob::job_id`) -> registry id. Shared with workers so a
    /// superseded worker can prove it still owns its id before publishing.
    export_job_ids: Arc<Mutex<HashMap<u64, JobId>>>,
    /// Published output path per UI job id, so `Succeeded` can map back to the
    /// `Done { output }` contract. Shared with workers; written under
    /// `publish_gate` just before the `Succeeded` transition.
    export_outputs: Arc<Mutex<HashMap<u64, PathBuf>>>,
    /// Lifecycle records for flight-leader decodes. Scheduling, single-flight
    /// and cache admission stay in `flights`/`decode_gates`; this only records
    /// that a decode started and how it ended.
    decode_jobs: JobRegistry,
    /// Lifecycle records for preview renders. One per call; terminal records
    /// are pruned so a drag cannot grow the map without bound.
    preview_jobs: JobRegistry,
    /// Lifecycle records for thumbnail renders. One per call, pruned the same
    /// way as preview records.
    thumbnail_jobs: JobRegistry,
    /// Lifecycle records for AI mask / closed-eye inference. One per call,
    /// pruned the same way as preview records.
    inference_jobs: JobRegistry,
    /// Lifecycle records for library import and folder rescan. One per public
    /// `import`/`rescan` call. The constructor's initial scan uses the
    /// unrecorded body so opening a library does not populate the snapshot.
    import_jobs: JobRegistry,
    store: Option<Store>,
    /// The [`ImageInfo`] captured when a document was last queued, shared with
    /// the single writer. The writer upserts the catalog from this under a
    /// short state lock after a successful file write, without the writer ever
    /// seeing the rest of the engine. Bounded: queued/in-flight paths plus the
    /// newest terminal entries are retained.
    save_infos: Arc<services::SaveInfos>,
    /// The newest queued revision per relative path. Only the engine generates
    /// revisions for the queued path, and it never reuses or lowers one.
    save_revisions: Mutex<HashMap<String, DocumentRevision>>,
    /// Bounded single-writer persistence. `None` when this library has no
    /// writable store; the synchronous save path stays available regardless.
    save_queue: Option<Arc<SaveQueue>>,
    /// A C1 catalog is read-only: opening it must not save the imported index.
    c1_catalog: bool,
    /// The database `open_c1_catalog` read, for the write-back entry point.
    c1_catalog_path: Option<PathBuf>,
    /// Camera defaults from Capture One's `coreConfig.cdx`, read once.
    camera_defaults: std::sync::OnceLock<Option<crate::CameraDefaults>>,
    /// Closed-eye results, computed once per image.
    eyes: Mutex<HashMap<ImageId, crate::EyeReport>>,
    imported_styles: RwLock<Vec<StyleInfo>>,
    merge_registry: Arc<crate::merge_async::MergeRegistry>,
    soft_proof: RwLock<crate::SoftProofSettings>,
}

/// Target white balance for Match Look: stored C1 gains when present, the
/// slider pair otherwise. The K/T table is sampled on the Z6III; other bodies
/// still use it as the same approximation the ceiling already uses.
struct CameraWhiteBalance {
    camera: Option<String>,
}

impl crate::matchlook::LookWhiteBalanceContext for CameraWhiteBalance {
    fn target_temperature(&self, target: &crate::WhiteBalance) -> (f32, f32) {
        let Some(gains) = target.gains else {
            return (target.kelvin, target.tint);
        };
        let (kelvin, tint, confidence) =
            crate::c1_model::gains_to_kelvin_tint_for_body(gains, self.camera.as_deref());
        if confidence == crate::c1_model::WbDisplayConfidence::Unknown {
            crate::c1_model::gains_to_kelvin_tint(gains)
        } else {
            (kelvin, tint)
        }
    }

    fn temperature_to_gains(&self, kelvin: f32, tint: f32) -> [f32; 3] {
        crate::c1_model::kelvin_tint_to_gains(kelvin, tint)
    }
}

impl FileEngine {
    fn lcc_files(&self, image: ImageId) -> Option<(PathBuf, PathBuf)> {
        Some((self.image_path(image)?, self.c1_settings_path(image)?))
    }

    /// The `.cos` sidecar that belongs to `image`: `<stem>.cos` beside the raw,
    /// or in a `Settings*` folder beside or above it (Capture One's layout).
    pub fn c1_settings_path(&self, image: ImageId) -> Option<PathBuf> {
        let raw = {
            let st = self.state.lock();
            st.entries
                .iter()
                .find(|entry| entry.info.id == image)
                .map(|entry| entry.path.clone())?
        };
        cos_settings_file(&raw)
    }

    /// Writes the image's adjustments back into its Capture One `.cos`.
    ///
    /// The global `<DL>` layer is compared to `adjustments`; every local layer
    /// is matched by `MaskUUID` and must have a `.comask` record in the same
    /// folder. Unmatched layers are reported, never silently dropped.
    /// Writes the image's adjustments into the catalog `open_c1_catalog` read.
    ///
    /// This is the database route; [`Self::save_to_c1_settings`] is the
    /// file-sidecar route and stays independent of it.
    pub fn save_to_c1_catalog(
        &self,
        image: ImageId,
        adjustments: &AdjustmentSet,
    ) -> anyhow::Result<crate::c1_writeback::CatalogSaveOutcome> {
        use crate::c1_writeback::{CatalogSaveOutcome, WriteOutcome};
        let (database, file_name, variant_index) = {
            let st = self.state.lock();
            let Some(database) = self.c1_catalog_path.clone() else {
                return Ok(CatalogSaveOutcome::NotACatalog);
            };
            let Some(entry) = st.entries.iter().find(|entry| entry.info.id == image) else {
                anyhow::bail!("image {} is not in this engine", image.0);
            };
            (database, entry.info.file_name.clone(), entry.info.variant)
        };
        let import = crate::c1import::C1Import::open(&database)?;
        let Some(c1_image) = import
            .catalog
            .images
            .iter()
            .find(|row| row.file_name == file_name)
        else {
            anyhow::bail!("{file_name} has no row in {}", database.display());
        };
        let mut variants: Vec<_> = import
            .catalog
            .variants
            .iter()
            .filter(|variant| variant.image_pk == c1_image.pk)
            .collect();
        variants.sort_by_key(|variant| variant.pk);
        let Some(variant) = variants
            .get((variant_index.max(1) as usize - 1).min(variants.len().saturating_sub(1)))
        else {
            anyhow::bail!("{file_name} has no variant in the catalog");
        };
        match crate::c1_writeback::write_variant_with_source(
            &database,
            variant.pk,
            adjustments,
            Some((c1_image.width, c1_image.height)),
        )? {
            WriteOutcome::Unchanged { warnings } => Ok(CatalogSaveOutcome::Unchanged { warnings }),
            WriteOutcome::Written { columns, .. } => {
                let mut name = database.as_os_str().to_os_string();
                name.push(".bak");
                Ok(CatalogSaveOutcome::Saved { columns, backup: PathBuf::from(name) })
            }
        }
    }

    pub fn save_to_c1_settings(
        &self,
        image: ImageId,
        adjustments: &AdjustmentSet,
    ) -> anyhow::Result<crate::c1_cos::CosSaveOutcome> {
        let path = self.c1_settings_path(image).ok_or_else(|| {
            anyhow::anyhow!("no `.cos` settings file was found for image {}", image.0)
        })?;
        let layers: Vec<crate::c1_cos::CosLayerWrite> = adjustments
            .layers
            .iter()
            .map(|layer| crate::c1_cos::CosLayerWrite {
                mask_uuid: None,
                adjustments: layer_adjustments(layer),
            })
            .collect();
        crate::c1_cos::save_adjustments(&path, adjustments, &layers)
    }

    /// Opens `root` and imports every supported image below it.
    pub fn open(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        let root = library_ops::canonical_root(root.as_ref())?;
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
        let soft_proof = store.as_ref().and_then(|s| s.load_soft_proof().ok()).unwrap_or_default();
        let cache_bytes = default_source_cache_bytes();
        log::debug!("source cache budget: {cache_bytes} bytes");
        let state = Arc::new(Mutex::new(State {
            entries: Vec::new(),
            next_id: 1,
            sources: HashMap::new(),
            native_raw: None,
            ai_inputs: HashMap::new(),
            source_cache: SourceCacheBudget::new(cache_bytes),
            source_reasons: HashMap::new(),
            layers: HashMap::new(),
            retouch: HashMap::new(),
            c1_collections: Vec::new(),
            c1_members: HashMap::new(),
            catalog,
            preview: PreviewCache::default(),
        }));
        let save_infos: Arc<services::SaveInfos> = Arc::new(services::SaveInfos::new());
        let save_queue = services::start_save_queue(store.as_ref(), &state, &save_infos);
        let engine = Self {
            root,
            state,
            save_order: Mutex::new(services::SaveOrder::default()),
            profiled: Mutex::new(Default::default()),
            gamut: Mutex::new(Vec::new()),
            graph_gate: Mutex::new(preview_cache::GraphGate::new()),
            flights: Mutex::new(HashMap::new()),
            native_decode: Mutex::new(()),
            decode_gates: DecodeGates::new(cache_bytes),
            publish_gate: Arc::new(Mutex::new(())),
            export_jobs: JobRegistry::new(),
            export_job_ids: Arc::new(Mutex::new(HashMap::new())),
            export_outputs: Arc::new(Mutex::new(HashMap::new())),
            decode_jobs: JobRegistry::new(),
            preview_jobs: JobRegistry::new(),
            thumbnail_jobs: JobRegistry::new(),
            inference_jobs: JobRegistry::new(),
            import_jobs: JobRegistry::new(),
            store,
            save_infos,
            save_revisions: Mutex::new(HashMap::new()),
            save_queue,
            c1_catalog: false,
            c1_catalog_path: None,
            camera_defaults: std::sync::OnceLock::new(),
            eyes: Mutex::new(HashMap::new()),
            imported_styles: RwLock::new(Vec::new()),
            merge_registry: Arc::new(crate::merge_async::MergeRegistry::default()),
            soft_proof: RwLock::new(soft_proof),
        };
        // The opening scan is not a user rescan: keep it out of the import job
        // snapshot so the records describe explicit `import`/`rescan` calls.
        engine.imports().rescan_inner()?;
        engine.inference().preload_ai_masks();
        Ok(engine)
    }

    /// The folder this engine reads.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read-only counts of the export jobs this engine's registry still tracks.
    ///
    /// The UI uses this for progress and close hooks; it is deliberately not
    /// part of [`ImageEngine`] and takes no lock the export worker needs.
    pub fn export_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.exports().export_job_snapshot()
    }

    /// Read-only counts of the decode jobs this engine's registry tracks.
    ///
    /// One record per flight-leader decode, so waiters that re-use the leader's
    /// result do not add records. Deliberately not part of [`ImageEngine`].
    pub fn decode_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.decode_jobs.snapshot()
    }

    /// Read-only counts of the preview render records this engine tracks.
    ///
    /// One record per `render_preview`/`render_preview_fast` call; terminal
    /// records are pruned periodically. Deliberately not part of
    /// [`ImageEngine`].
    pub fn preview_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.preview_jobs.snapshot()
    }

    /// Read-only counts of the thumbnail render records this engine tracks.
    ///
    /// One record per `thumbnail` call; terminal records are pruned
    /// periodically. Deliberately not part of [`ImageEngine`].
    pub fn thumbnail_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.renders().thumbnail_job_snapshot()
    }

    /// Read-only counts of the AI inference records this engine tracks.
    ///
    /// One record per `compute_ai_mask`/`eye_report` call, including cache
    /// hits; terminal records are pruned periodically. Deliberately not part of
    /// [`ImageEngine`].
    pub fn inference_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.inference().inference_job_snapshot()
    }

    /// Read-only counts of the import/rescan records this engine tracks.
    ///
    /// One record per public `import` or `rescan` call; terminal records are
    /// pruned periodically. The `open`/`open_c1_catalog` initial scan is not
    /// recorded. Deliberately not part of [`ImageEngine`].
    pub fn import_job_snapshot(&self) -> crate::job_lifecycle::RegistrySnapshot {
        self.imports().import_job_snapshot()
    }

    /// The packed RAW, settings and CoMask members of a C1 `.eip` entry,
    /// materialized on disk for the decoders that take a path. `mask_uuids`
    /// are what `comask::read_comask_masks_for` needs to find the layers.
    pub fn eip_assets(&self, image: ImageId) -> Option<EipAssets> {
        let path = self.state.lock().entries.iter().find(|e| e.info.id == image)?.path.clone();
        if !is_eip(&path) {
            return None;
        }
        materialize_eip(&self.root, &path).ok()
    }

    /// Opens a Capture One catalog or session read-only and imports its images
    /// as Portrait entries. Edits are saved to `.portrait` beside the database; the C1
    /// database itself is never written to.
    pub fn open_c1_catalog(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let import = crate::c1import::C1Import::open(path.as_ref())?;
        // A bundle is `X.cocatalogdb/X.cocatalogdb` (or `.cosessiondb`, or a
        // `.cos` session folder); `.portrait` must not land inside it.
        let mut root = import.catalog_path.parent().unwrap_or(Path::new(".")).to_path_buf();
        // The outer `X.cocatalogdb` folder is itself Capture One's package:
        // keep our sidecar in the folder that contains the package, never inside it.
        if root.extension().is_some_and(|e| {
            ["cocatalogdb", "cosessiondb", "cos"]
                .iter()
                .any(|want| e.eq_ignore_ascii_case(want))
        }) {
            root = root.parent().unwrap_or(Path::new(".")).to_path_buf();
        }
        let root = library_ops::canonical_root(&root)?;
        let store = Store::open(&root).ok();
        let catalog = store.as_ref().and_then(|s| s.load_index().ok()).unwrap_or_default();
        let soft_proof = store.as_ref().and_then(|s| s.load_soft_proof().ok()).unwrap_or_default();
        let cache_bytes = default_source_cache_bytes();
        log::debug!("source cache budget: {cache_bytes} bytes");
        let state = Arc::new(Mutex::new(State {
            entries: Vec::new(),
            next_id: 1,
            sources: HashMap::new(),
            native_raw: None,
            ai_inputs: HashMap::new(),
            source_cache: SourceCacheBudget::new(cache_bytes),
            source_reasons: HashMap::new(),
            layers: HashMap::new(),
            retouch: HashMap::new(),
            c1_collections: Vec::new(),
            c1_members: HashMap::new(),
            catalog,
            preview: PreviewCache::default(),
        }));
        let save_infos: Arc<services::SaveInfos> = Arc::new(services::SaveInfos::new());
        let save_queue = services::start_save_queue(store.as_ref(), &state, &save_infos);
        let engine = Self {
            root,
            state,
            save_order: Mutex::new(services::SaveOrder::default()),
            profiled: Mutex::new(Default::default()),
            gamut: Mutex::new(Vec::new()),
            graph_gate: Mutex::new(preview_cache::GraphGate::new()),
            flights: Mutex::new(HashMap::new()),
            native_decode: Mutex::new(()),
            decode_gates: DecodeGates::new(cache_bytes),
            publish_gate: Arc::new(Mutex::new(())),
            export_jobs: JobRegistry::new(),
            export_job_ids: Arc::new(Mutex::new(HashMap::new())),
            export_outputs: Arc::new(Mutex::new(HashMap::new())),
            decode_jobs: JobRegistry::new(),
            preview_jobs: JobRegistry::new(),
            thumbnail_jobs: JobRegistry::new(),
            inference_jobs: JobRegistry::new(),
            import_jobs: JobRegistry::new(),
            store,
            save_infos,
            save_revisions: Mutex::new(HashMap::new()),
            save_queue,
            c1_catalog: true,
            c1_catalog_path: Some(import.catalog_path.clone()),
            camera_defaults: std::sync::OnceLock::new(),
            eyes: Mutex::new(HashMap::new()),
            imported_styles: RwLock::new(Vec::new()),
            merge_registry: Arc::new(crate::merge_async::MergeRegistry::default()),
            soft_proof: RwLock::new(soft_proof),
        };
        engine.import_c1(&import);
        engine.inference().preload_ai_masks();
        Ok(engine)
    }

    fn import_c1(&self, import: &crate::c1import::C1Import) {
        let mut imported: Vec<(String, AdjustmentSet, ImageInfo)> = Vec::new();
        // `.comask` sidecars live under the catalog's `Adjustments/LAM` tree,
        // named after the source file. Index them once per import.
        let mut comask_files: HashMap<String, Vec<PathBuf>> = HashMap::new();
        if let Some(dir) = import.catalog_path.parent() {
            collect_comask(&dir.join("Adjustments").join("LAM"), &mut comask_files);
        }
        // C1 collections point at ZIMAGE rows; remember every variant's id.
        let mut pk_ids: HashMap<i64, Vec<ImageId>> = HashMap::new();
        let mut st = self.state.lock();
        for image in import.images() {
            let path = PathBuf::from(&image.abs_path);
            for variant in import.variants_of(image.pk) {
                let number = variant.index.max(0) as u32 + 1;
                let rel_path = crate::c1import::store_key(&path, number);
                // Our store wins over C1's copy, so reopening the catalog keeps the
                // edits made in Portrait instead of resetting them to C1's.
                let stored = st.catalog.entry(&rel_path).cloned();
                let adjustments = match (&self.store, &stored) {
                    (Some(store), Some(_)) => store.load_adjustments(&rel_path).unwrap_or_default(),
                    _ => {
                        let mut adjustments = variant.adjustments.clone();
                        adjustments
                    }
                };
                let mut render_adjustments = adjustments.clone();
                let mut retouch_documents = Vec::new();
                if !variant.local_layers.is_empty() {
                    attach_c1_layers(
                        &mut render_adjustments,
                        variant,
                        image,
                        &comask_files,
                    );
                    // Read the same sidecar once for retouching mask summaries.
                    if let Some(path) = comask_files.get(&image.file_name).and_then(|v| v.first()) {
                        if let Ok(bytes) = std::fs::read(path) {
                            if let Ok(document) = crate::comask::ComaskDocument::parse(&bytes) {
                                retouch_documents.push(document);
                            }
                        }
                    }
                }
                let id = ImageId(st.next_id);
                st.next_id += 1;
                pk_ids.entry(image.pk).or_default().push(id);
                if !variant.local_layers.is_empty() {
                    st.layers.insert(
                        id,
                        variant
                            .local_layers
                            .iter()
                            .map(|layer| {
                                let summary = layer.summary();
                                crate::engine::LayerSummary {
                                    name: summary.name,
                                    enabled: summary.enabled,
                                    mask_uuid: summary.mask_uuid,
                                }
                            })
                            .collect(),
                    );
                }
                if !variant.retouching.is_empty() {
                    let document = retouch_documents.first();
                    let masks: Vec<_> = variant
                        .retouching
                        .iter()
                        .flat_map(|row| {
                            row.mask_summaries(document.map(|d| d.masks.as_slice()))
                        })
                        .collect();
                    if !masks.is_empty() {
                        st.retouch.insert(id, masks);
                    }
                }
                let fresh = stored.is_none();
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
                st.entries.push(FileEntry {
                    info: info.clone(),
                    path: path.clone(),
                    rel_path: rel_path.clone(),
                    adjustments: render_adjustments,
                });
                if fresh {
                    imported.push((rel_path, adjustments, info));
                }
            }
        }
        // C1's own albums and folders become the collection tree; each member is
        // a ZIMAGE row, so every variant of that image belongs to it.
        let c1_collections = import.collections();
        if !c1_collections.is_empty() {
            let mut members = HashMap::new();
            for node in &import.catalog.collections {
                // C1 computes the system collections instead of storing rows.
                let ids: Vec<ImageId> = if node.kind == CollectionKind::AllImages {
                    pk_ids.values().flatten().copied().collect()
                } else {
                    node.images
                        .iter()
                        .flat_map(|pk| pk_ids.get(pk).cloned().unwrap_or_default())
                        .collect()
                };
                members.insert(crate::c1import::C1_COLLECTION_ID_BASE + node.pk as u64, ids);
            }
            st.c1_collections = c1_collections;
            st.c1_members = members;
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
        // Opening a read-only C1 catalog is not an edit: writing the index here
        // changes the source folder even when its bytes stay the same.
        if !self.c1_catalog {
            if let Some(store) = &self.store {
            if let Err(err) = store.save_index(&catalog) {
                log::warn!("save catalog: {err}");
            }
        }
        }
    }

    /// Re-reads the folder, keeping edits already made.
    pub fn rescan(&self) -> anyhow::Result<()> {
        self.imports().rescan()
    }

    fn add_file(&self, path: &Path) -> anyhow::Result<ImageInfo> {
        let rel = path.strip_prefix(&self.root).unwrap_or(path).to_string_lossy().replace('\\', "/");
        let scanned = scan_file(&self.root, rel, path.to_path_buf());
        let adjustments = scanned.c1_adjustments().unwrap_or_default();
        let mut st = self.state.lock();
        let id = ImageId(st.next_id);
        st.next_id += 1;
        let keywords = scanned.iptc.keywords.clone();
        let info = ImageInfo {
            id,
            path: scanned.path.clone(),
            file_name: scanned.file_name,
            kind: scanned.kind,
            heif: None,
            width: scanned.width,
            height: scanned.height,
            file_size: scanned.file_size,
            exif: scanned.exif,
            iptc: scanned.iptc,
            keywords,
            rating: 0,
            color_tag: ColorTag::None,
            variant: 1,
            has_adjustments: !adjustments.is_default(),
            trashed: false,
        };
        st.entries.push(FileEntry {
            info: info.clone(),
            path: scanned.path,
            rel_path: scanned.rel_path,
            adjustments,
        });
        Ok(info)
    }

    /// Verifies a cached identity against the current file. Returns `false`
    /// when the file is gone or its content changed. Metadata is read on every
    /// hit; the content token is re-read on metadata changes and after
    /// [`SOURCE_IDENTITY_RECHECK`] / [`SOURCE_IDENTITY_RECHECK_HITS`].
    fn refresh_source_identity(st: &mut State, image: ImageId, path: &Path, slot: bool) -> bool {
        let current = match SourceFileIdentity::read(path) {
            Some(current) => current,
            None => {
                Self::invalidate_cached_source(st, image, slot, "file stat failed");
                return false;
            }
        };
        let provenance = if slot {
            st.native_raw
                .as_ref()
                .map(|slot| (slot.identity, slot.last_verified, slot.hits_since_verify))
        } else {
            st.sources
                .get(&image)
                .map(|entry| (entry.identity, entry.last_verified, entry.hits_since_verify))
        };
        let Some((identity, last_verified, hits_since_verify)) = provenance else {
            return false;
        };
        if identity.is_unknown() {
            Self::invalidate_cached_source(st, image, slot, "no source identity");
            return false;
        }
        let metadata_changed = current.len != identity.len || current.mtime_nanos != identity.mtime_nanos;
        let due = metadata_changed
            || last_verified.elapsed() >= SOURCE_IDENTITY_RECHECK
            || hits_since_verify >= SOURCE_IDENTITY_RECHECK_HITS;
        let mut verified = current;
        if due {
            verified.token = crate::raw_cache::source_identity_token(path);
            if verified.token != identity.token {
                Self::invalidate_cached_source(st, image, slot, "source content changed");
                return false;
            }
        } else {
            verified.token = identity.token;
        }
        if slot {
            if let Some(slot) = st.native_raw.as_mut() {
                slot.identity = verified;
                if due {
                    slot.last_verified = Instant::now();
                    slot.hits_since_verify = 0;
                } else {
                    slot.hits_since_verify = slot.hits_since_verify.saturating_add(1);
                }
            }
        } else if let Some(entry) = st.sources.get_mut(&image) {
            entry.identity = verified;
            if due {
                entry.last_verified = Instant::now();
                entry.hits_since_verify = 0;
            } else {
                entry.hits_since_verify = entry.hits_since_verify.saturating_add(1);
            }
        }
        true
    }

    /// Removes a source/native slot (and the AI input that depends on it) after
    /// its file identity no longer matches.
    fn invalidate_cached_source(st: &mut State, image: ImageId, slot: bool, reason: &str) {
        if slot {
            let mut released = 0usize;
            if let Some(evicted) = st.native_raw.take() {
                released = released.saturating_add(evicted.bytes);
            }
            if let Some(input) = st.ai_inputs.remove(&image) {
                released = released.saturating_add(ai_input_bytes(&input));
            }
            st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(released);
        } else {
            evict_cached_source(st, image);
        }
        st.source_reasons.remove(&image);
        st.source_cache.identity_invalidations += 1;
        log::debug!("source {image:?} invalidated: {reason}");
    }

    fn image_path(&self, image: ImageId) -> Option<PathBuf> {
        self.state
            .lock()
            .entries
            .iter()
            .find(|entry| entry.info.id == image)
            .map(|entry| entry.path.clone())
    }

    /// A cached source, charged to the byte budget and refreshed in the LRU.
    /// Derived buffers are accounted again on every hit because they are built
    /// lazily after the source was stored.
    fn cached_source(&self, image: ImageId) -> Option<Arc<Source>> {
        let path = self.image_path(image)?;
        let mut st = self.state.lock();
        if !FileEngine::refresh_source_identity(&mut st, image, &path, false) {
            return None;
        }
        let tick = st.source_cache.next_tick();
        let (source, old_bytes, new_bytes) = {
            let entry = st.sources.get_mut(&image)?;
            entry.last_used = tick;
            (Arc::clone(&entry.source), entry.bytes, source_cache_bytes(&entry.source))
        };
        if new_bytes != old_bytes {
            if let Some(entry) = st.sources.get_mut(&image) {
                entry.bytes = new_bytes;
            }
            st.source_cache.current_bytes = st.source_cache.current_bytes
                .saturating_add(new_bytes)
                .saturating_sub(old_bytes);
            st.source_cache.peak_bytes = st.source_cache.peak_bytes.max(st.source_cache.current_bytes);
        }
        st.source_cache.hits += 1;
        st.source_cache.note_use(image);
        Some(source)
    }

    /// Stores a decoded source and evicts the oldest unpinned entries until the
    /// byte budget holds. The entry being stored, the current/neighbour images,
    /// and sources another caller still holds (strong_count > 1) are kept.
    fn store_cached_source(&self, image: ImageId, path: &Path, source: Arc<Source>) {
        let bytes = source_cache_bytes(&source);
        let identity = SourceFileIdentity::read(path).unwrap_or(SourceFileIdentity { len: 0, mtime_nanos: 0, token: None }).with_token(path);
        let mut st = self.state.lock();
        let tick = st.source_cache.next_tick();
        if let Some(old) = st.sources.remove(&image) {
            st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(old.bytes);
        }
        st.sources.insert(
            image,
            CachedSource {
                source,
                bytes,
                last_used: tick,
                path: path.to_path_buf(),
                identity,
                last_verified: Instant::now(),
                hits_since_verify: 0,
            },
        );
        st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_add(bytes);
        st.source_cache.peak_bytes = st.source_cache.peak_bytes.max(st.source_cache.current_bytes);
        st.source_cache.note_use(image);
        evict_sources_to_budget(&mut st, Some(image));
        log::debug!(
            "source {image:?}: {bytes} bytes; cache {}/{} bytes, {} sources",
            st.source_cache.current_bytes,
            st.source_cache.max_bytes,
            st.sources.len()
        );
    }

    /// A cached native RAW frame, or a remembered failure for that image.
    fn cached_native(&self, image: ImageId, path: &Path) -> Option<Option<Arc<Source>>> {
        let mut st = self.state.lock();
        if st.native_raw.as_ref().is_none_or(|slot| slot.image != image) {
            return None;
        }
        if !FileEngine::refresh_source_identity(&mut st, image, path, true) {
            return None;
        }
        let tick = st.source_cache.next_tick();
        let (cached, old_bytes, new_bytes) = {
            let slot = st.native_raw.as_mut()?;
            slot.last_used = tick;
            // A deferred mip may have been built since publication. Re-read the
            // reservation/actual bytes so the native slot cannot stay stale.
            let new_bytes = slot.source.as_ref().map_or(0, |source| source_cache_bytes(source));
            let old_bytes = slot.bytes;
            slot.bytes = new_bytes;
            (slot.source.clone(), old_bytes, new_bytes)
        };
        if new_bytes != old_bytes {
            st.source_cache.current_bytes = st
                .source_cache
                .current_bytes
                .saturating_add(new_bytes)
                .saturating_sub(old_bytes);
            st.source_cache.peak_bytes =
                st.source_cache.peak_bytes.max(st.source_cache.current_bytes);
        }
        st.source_cache.native_hits += 1;
        st.source_cache.note_use(image);
        Some(cached)
    }

    fn join_flight(&self, key: SourceKey) -> FlightRole<'_> {
        let mut flights = self.flights.lock();
        if let Some(flight) = flights.get(&key).cloned() {
            FlightRole::Waiter(flight)
        } else {
            let flight = Arc::new(Flight::default());
            flights.insert(key, Arc::clone(&flight));
            FlightRole::Leader(FlightLeader { provider: self, key, flight, finished: false })
        }
    }

    fn note_flight_wait(&self) {
        self.state.lock().source_cache.single_flight_waits += 1;
    }

    fn source(&self, image: ImageId) -> anyhow::Result<Arc<Source>> {
        self.sources().source(image)
    }

    /// One decode per image via the source service; see that module for the
    /// single-flight and admission contract.
    fn source_with_priority(&self, image: ImageId, priority: DecodePriority) -> anyhow::Result<Arc<Source>> {
        self.sources().source_with_priority(image, priority)
    }

    /// Full-size RAW for 1:1 zoom and export. The native slot holds one frame,
    /// so native decodes are serialized; same-image callers share one flight.
    fn native_raw_source(&self, image: ImageId, viewport: Option<(&crate::Geometry, NormRect, u32)>) -> Option<Arc<Source>> {
        let (path, lens, width, height, banding_identity) = {
            let st = self.state.lock();
            let entry = st.entries.iter().find(|e| e.info.id == image)?;
            if !(crate::raw::is_raw(&entry.path) || is_eip(&entry.path))
                || viewport.is_some_and(|(geometry, region, max)| {
                    !wants_native_raw((entry.info.width, entry.info.height), geometry, region, max)
                })
            {
                return None;
            }
            (
                entry.path.clone(),
                crate::mock::lens_profile::LensShot::from_exif(&entry.info.exif),
                entry.info.width,
                entry.info.height,
                raw_banding_identity(&entry.info.exif),
            )
        };
        if let Some(cached) = self.cached_native(image, &path) {
            return cached;
        }
        let key = SourceKey::Native(image);
        let leader = match self.join_flight(key) {
            FlightRole::Waiter(flight) => {
                self.note_flight_wait();
                return flight.wait().ok();
            }
            FlightRole::Leader(leader) => leader,
        };
        let decoded = (|| -> Option<Arc<Source>> {
            // Only one native frame is kept, so one native decode at a time.
            let _native = self.native_decode.lock();
            if let Some(cached) = self.cached_native(image, &path) {
                return cached;
            }
            // Drop the previous full frame before allocating a new one, so the
            // peak never holds two native frames at once.
            {
                let mut st = self.state.lock();
                if let Some(previous) = st.native_raw.take() {
                    st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_sub(previous.bytes);
                }
            }
            let units = decode_units(estimated_decode_bytes(width, height, DecodeFormat::RawNative));
            let _permit = self.decode_gates.heavy.acquire(DecodePriority::Interactive, units);
            self.state.lock().source_cache.misses += 1;
            let decoded = (|| {
                let path = if is_eip(&path) { materialize_eip(&self.root, &path).ok()?.raw } else { path.clone() };
                let (source, provenance) = crate::raw_cache::get_or_decode_source16(&path, 0)?;
                Some(Arc::new(
                    source
                        .with_lens_shot(lens)
                        .with_c1_z6iii_linear(provenance.unlocks_c1_z6iii_linear())
                        .with_banding_identity(
                            banding_identity.0,
                            banding_identity.1.as_deref(),
                        ),
                ))
            })();
            if decoded.is_none() {
                log::warn!("native RAW source unavailable for {image:?}");
            }
            let bytes = decoded.as_ref().map_or(0, |source| source_cache_bytes(source));
            let identity = SourceFileIdentity::read(&path)
                .unwrap_or(SourceFileIdentity { len: 0, mtime_nanos: 0, token: None })
                .with_token(&path);
            {
                let mut st = self.state.lock();
                let tick = st.source_cache.next_tick();
                st.native_raw = Some(NativeRawSlot {
                    image,
                    source: decoded.clone(),
                    bytes,
                    last_used: tick,
                    identity,
                    last_verified: Instant::now(),
                    hits_since_verify: 0,
                });
                st.source_cache.current_bytes = st.source_cache.current_bytes.saturating_add(bytes);
                st.source_cache.peak_bytes = st.source_cache.peak_bytes.max(st.source_cache.current_bytes);
                st.source_cache.note_use(image);
                evict_sources_to_budget(&mut st, Some(image));
            }
            decoded
        })();
        let flight_result = match decoded {
            Some(source) => Ok(source),
            None => Err(format!("native RAW source unavailable for {image:?}")),
        };
        leader.finish(flight_result).ok()
    }
    /// Decide the standard export source when no native RAW frame is available,
    /// without doing the expensive decode on the calling thread.
    ///
    /// At or below the cached proxy's long side this is exactly the preview
    /// [`Self::source`] path (unchanged). Above it, using the proxy would make
    /// the renderer upscale it (explicitly forbidden), so the plan hands the
    /// worker the cheap state it needs to decode the original at full
    /// resolution. That decode bypasses the source cache; see
    /// [`decode_full_res_export_source`] for the admission note.
    fn export_source_plan(&self, image: ImageId, requested_long: u32) -> anyhow::Result<StandardExportSource> {
        let (path, lens, original_width, original_height) = {
            let st = self.state.lock();
            let entry = st
                .entries
                .iter()
                .find(|entry| entry.info.id == image)
                .ok_or_else(|| anyhow::anyhow!("export: unknown image {}", image.0))?;
            (
                entry.path.clone(),
                crate::mock::lens_profile::LensShot::from_exif(&entry.info.exif),
                entry.info.width,
                entry.info.height,
            )
        };
        let original_long = original_width.max(original_height);
        // The preview source never keeps more than SOURCE_LONG_SIDE, so this is
        // the long side `self.source()` would return for this file. A source at
        // or below the cap keeps the existing `source()` path even for an
        // explicit enlargement: the proxy there is the full-resolution file.
        let proxy_long = original_long.min(SOURCE_LONG_SIDE);
        if original_long <= SOURCE_LONG_SIDE || requested_long <= proxy_long {
            return Ok(StandardExportSource::Ready(self.source(image)?));
        }
        Ok(StandardExportSource::FullRes(FullResExportRequest {
            root: self.root.clone(),
            path,
            lens,
            original_width,
            original_height,
            original_long,
            requested_long,
        }))
    }

    /// Current source-cache bytes, pins and counters. The app's performance log
    /// and the unified budget (S7) read this; it never decodes anything.
    pub fn source_cache_stats(&self) -> SourceCacheStats {
        // Gate locks are taken before the state lock: a decoder holds a gate
        // permit while it stores its result, so the reverse order could deadlock.
        let heavy_units = self.decode_gates.heavy.total_units();
        let light_units = self.decode_gates.light.total_units();
        let decode_waits = self.decode_gates.heavy.waits() + self.decode_gates.light.waits();
        let decode_wait_ms =
            (self.decode_gates.heavy.wait_nanos() + self.decode_gates.light.wait_nanos()) / 1_000_000;
        let decode_peak_concurrency =
            self.decode_gates.heavy.peak_active().max(self.decode_gates.light.peak_active());
        let st = self.state.lock();
        let cache = &st.source_cache;
        let ai_input_bytes_total = st.ai_inputs.values().fold(0usize, |total, input| {
            total.saturating_add(ai_input_bytes(input))
        });
        let derived_bytes = st.sources.values().fold(0usize, |total, entry| {
            total.saturating_add(entry.source.cached_derived_bytes())
        });
        let pinned_bytes = st.sources.iter().filter(|(image, _)| cache.is_pinned(**image)).fold(0usize, |total, (_, entry)| {
            total.saturating_add(entry.bytes)
        });
        SourceCacheStats {
            max_bytes: cache.max_bytes,
            current_bytes: cache.current_bytes,
            peak_bytes: cache.peak_bytes,
            cached_sources: st.sources.len(),
            pinned_sources: st.sources.keys().filter(|image| cache.is_pinned(**image)).count(),
            native_bytes: st.native_raw.as_ref().map_or(0, |slot| slot.bytes),
            ai_input_bytes: ai_input_bytes_total,
            derived_bytes,
            pinned_bytes,
            over_budget_bytes: cache.current_bytes.saturating_sub(cache.max_bytes),
            over_budget_peak: cache.over_budget_peak,
            identity_invalidations: cache.identity_invalidations,
            hits: cache.hits,
            native_hits: cache.native_hits,
            misses: cache.misses,
            evictions: cache.evictions,
            single_flight_waits: cache.single_flight_waits,
            decode_waits,
            decode_wait_ms,
            decode_peak_concurrency,
            heavy_units,
            light_units,
        }
    }

    /// Sets the byte budget; `0` recomputes the machine-derived default. Returns
    /// the budget in effect. Pinned sources can keep the cache above it.
    pub fn set_source_cache_budget_bytes(&self, max_bytes: usize) -> usize {
        let max_bytes = if max_bytes == 0 { default_source_cache_bytes() } else { max_bytes };
        // Concurrency follows the budget: a smaller cache admits fewer large
        // decodes at once. Gate before state, matching decoders' lock order.
        let units = decode_gate_units(max_bytes);
        self.decode_gates.heavy.set_total_units(units);
        self.decode_gates.light.set_total_units(units);
        let mut st = self.state.lock();
        st.source_cache.max_bytes = max_bytes;
        evict_sources_to_budget(&mut st, None);
        log::info!(
            "source cache budget {max_bytes} bytes; now {} bytes in {} sources",
            st.source_cache.current_bytes,
            st.sources.len()
        );
        max_bytes
    }

    pub fn source_cache_budget_bytes(&self) -> usize {
        self.state.lock().source_cache.max_bytes
    }

    /// Pins an image so budget eviction keeps it (exports hold sources this way).
    /// Counts are per image; repeat pins before each independent holder.
    pub fn pin_source(&self, image: ImageId) -> usize {
        let mut st = self.state.lock();
        let count = st.source_cache.pins.entry(image).or_insert(0);
        *count = count.saturating_add(1);
        *count as usize
    }

    pub fn unpin_source(&self, image: ImageId) -> usize {
        let mut st = self.state.lock();
        let remaining = match st.source_cache.pins.get_mut(&image) {
            Some(count) => {
                *count = count.saturating_sub(1);
                *count
            }
            None => 0,
        };
        if remaining == 0 {
            st.source_cache.pins.remove(&image);
        }
        remaining as usize
    }

    /// Drops every cached source now. Callers still holding an `Arc` keep their
    /// pixels; the cache simply stops charging for them. Used by diagnostics and
    /// the cold/warm benchmark harness, not by the render path.
    pub fn clear_source_cache(&self) {
        let mut st = self.state.lock();
        st.sources.clear();
        st.ai_inputs.clear();
        st.native_raw = None;
        st.source_cache.current_bytes = 0;
    }

    /// The cached model-sized input, for the engine's own tests. The inference
    /// service owns the production body.
    #[cfg(test)]
    fn ai_input(&self, image: ImageId) -> anyhow::Result<Arc<AiInput>> {
        self.inference().ai_input(image)
    }

    /// Cached preview frame; the render service owns the one-record-per-call
    /// lifecycle wrapper and the cache/latest-value logic.
    fn preview_frame(&self, request: &PreviewRequest, fast: bool) -> anyhow::Result<PreviewFrame> {
        self.renders().preview_frame(request, fast)
    }

    fn persist(&self, rel_path: &str, adjustments: &AdjustmentSet, info: &ImageInfo) -> SaveReport {
        self.persistence().persist(rel_path, adjustments, info)
    }

    /// Writes one batch of adjustment files and then the catalog index.
    ///
    /// Each document is written atomically on its own; this is not a multi-file
    /// transaction, so one failure does not roll back the others. Disk I/O runs
    /// without the state mutex. A document whose sequence was overtaken by a
    /// newer save for the same path is reported
    /// [`crate::save_report::SaveOutcome::Superseded`] and
    /// left untouched, so newer bytes are never replaced by older ones.
    // The persistence service owns the production call graph; this replay hook
    // is exercised directly by the save-ordering tests.
    #[allow(dead_code)]
    fn persist_batch(&self, items: &[(String, AdjustmentSet, ImageInfo)]) -> SaveReport {
        self.persistence().persist_batch(items)
    }

    /// [`Self::persist_batch`] with an explicit sequence, so tests can replay a
    /// stale write after a newer one.
    #[allow(dead_code)]
    fn persist_batch_at(&self, seq: u64, items: &[(String, AdjustmentSet, ImageInfo)]) -> SaveReport {
        self.persistence().persist_batch_at(seq, items)
    }

    /// Applies a multi-image edit in memory and returns what reached disk.
    ///
    /// The in-memory edits are always kept, even when their files failed: the
    /// user's latest content is never dropped just because the disk refused it.
    /// `save_adjustments_batch` is the `Result`-shaped wrapper for callers that
    /// only need pass/fail.
    pub fn save_adjustments_batch_report(&self, items: &[(ImageId, AdjustmentSet)]) -> SaveReport {
        self.persistence().save_adjustments_batch_report(items)
    }

    /// [`Self::save_adjustments_batch_report`] with an explicit sequence, so a
    /// test can replay an older call after a newer one and prove it neither
    /// restores stale in-memory edits nor overwrites newer bytes.
    #[allow(dead_code)]
    fn save_adjustments_batch_report_at(
        &self,
        seq: u64,
        items: &[(ImageId, AdjustmentSet)],
    ) -> SaveReport {
        self.persistence().save_adjustments_batch_report_at(seq, items)
    }

    /// Applies a multi-image edit in memory and hands each document to the
    /// bounded single-writer queue.
    ///
    /// The in-memory edit is applied **before** the queue is consulted, so the
    /// user's latest content is immediately visible; a refused submission
    /// ([`QueuedSaveOutcome::Full`], [`QueuedSaveOutcome::Closed`]) keeps that
    /// edit and only reports that it was not accepted for writing. A later
    /// flush and resubmit is the retry path. The synchronous
    /// [`Self::save_adjustments_batch_report`] path is independent and unchanged;
    /// callers should use one path at a time for a given document so their
    /// orderings cannot interleave.
    ///
    /// Revisions are allocated per relative path and strictly increase for the
    /// life of this engine, so the queue rejects any older revision. The
    /// captured [`ImageInfo`] is what the writer uses to upsert the catalog
    /// after a successful file write.
    pub fn save_adjustments_queued(&self, items: &[(ImageId, AdjustmentSet)]) -> QueuedSaveSummary {
        self.persistence().save_adjustments_queued(items)
    }

    /// Allocates the next queued revision for `rel_path`.
    ///
    /// The counter is per path and only moves forward. [`DocumentRevision::next`]
    /// saturates at `u64::MAX`; at that unreachable point further queued submits
    /// for the path report [`QueuedSaveOutcome::Superseded`] rather than reusing
    /// a revision.
    #[allow(dead_code)]
    fn next_save_revision(&self, rel_path: &str) -> DocumentRevision {
        self.persistence().next_save_revision(rel_path)
    }

    /// Waits for the save queue to drain (or `timeout`) and reports what
    /// actually reached disk. `None` when this library has no queue.
    ///
    /// This is a bounded wait, not a barrier command: a writer stuck in I/O
    /// still lets this return with `timed_out == true` and the affected paths
    /// reported [`crate::PathState::NotWritten`]. Queued work is never dropped.
    pub fn flush_saves(&self, timeout: Duration) -> Option<FlushReport> {
        self.persistence().flush_saves(timeout)
    }

    /// Cheap O(1) snapshot of the bounded queue. `None` when there is no queue.
    pub fn save_queue_status(&self) -> Option<QueueStatus> {
        self.persistence().save_queue_status()
    }

    /// Replaces the production queue with a caller-supplied writer and capacity.
    /// Test-only: the production capacity is fixed and its writer is the real
    /// `StoreSaveWriter`.
    #[cfg(test)]
    fn install_test_save_queue(&mut self, writer: Arc<dyn SaveWriter>, capacity: usize) {
        services::PersistenceService::install_test_save_queue(self, writer, capacity);
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
        self.catalog().collection_images(collection)
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
            if name.starts_with('.') || name == "portrait-output" {
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

fn scan_file(root: &Path, rel: String, path: PathBuf) -> ScannedFile {
    // EIP metadata and pixels live in the packed RAW, not in the ZIP header.
    let packed = if is_eip(&path) {
        match materialize_eip(root, &path) {
            Ok(assets) => Some(assets),
            Err(err) => {
                log::warn!("EIP {}: {err:#}", path.display());
                None
            }
        }
    } else {
        None
    };
    let media = packed.as_ref().map_or(path.as_path(), |assets| assets.raw.as_path());
    let (width, height) = dimensions(media);
    let (exif, iptc) = crate::exif::read_metadata(media);
    let packed_document = packed.as_ref()
        .and_then(|assets| std::fs::read(assets.settings.as_ref()?).ok())
        .and_then(|bytes| crate::c1_cos::CosDocument::parse(&bytes).ok())
        .map(Arc::new);
    ScannedFile {
        rel_path: rel.replace('\\', "/"),
        file_name: path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        kind: file_kind(&path),
        file_size: std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0),
        path,
        width,
        height,
        exif,
        iptc,
        packed,
        packed_document,
        packed_variant: 1,
    }
}

fn is_supported(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp" | "heic" | "heif")
    ) || is_cop(path)
        || is_eip(path)
        || crate::raw::is_raw(path)
}

/// Pixel dimensions of a file: the sensor size for RAW, else the header size.
fn dimensions(path: &Path) -> (u32, u32) {
    if is_cop(path) {
        return crate::jxl::dimensions(path).unwrap_or((0, 0));
    }
    if file_kind(path) == FileKind::Heif {
        return crate::heif::dimensions(path).unwrap_or((0, 0));
    }
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

/// EXIF lens name -> the `lensDatabase.cdx` entry's full name. This is Capture
/// One's automatic lens detection: both the name and the focal length must be
/// present, and nothing is guessed when the database has no entry.
fn auto_lens_profile(lenses: &[crate::LensProfile], exif: &Exif) -> Option<String> {
    let lens = exif.lens.as_deref()?;
    let focal = exif.focal_length_mm?;
    crate::lensdb::detect_lens(lenses, "", lens, focal).map(|p| p.full_name.clone())
}

/// Enables the calibrated warp when EXIF names a lens in the database. An
/// existing user choice is never replaced.
fn apply_auto_lens(lens: &mut crate::LensCorrection, lenses: &[crate::LensProfile], exif: &Exif) {
    if lens.use_profile || lens.profile.is_some() {
        return;
    }
    if let Some(name) = auto_lens_profile(lenses, exif) {
        lens.profile = Some(name);
        lens.use_profile = true;
    }
}

impl FileEngine {
    /// `coreConfig.cdx` ranges and defaults for one body and ISO; loaded once.
    fn catalog_overrides(&self, model: Option<&str>, iso: Option<u32>) -> crate::ParamOverrides {
        self.camera_defaults
            .get_or_init(|| {
                let path = std::env::var_os("AURORA_CORE_CONFIG")
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
            .as_ref()
            .map_or_else(Default::default, |d| d.overrides(model, iso))
    }

    /// Applies the import defaults Capture One starts a frame with to
    /// `adjustments` seeded from the catalog entry: sensor-RAW camera profile,
    /// film curve, noise reduction and sharpening, the display-referred ICC
    /// fallback, and automatic lens detection. Shared by the single-image and
    /// batch loads so both produce exactly the same set. Does not touch the
    /// engine state lock; callers snapshot the entry themselves.
    fn resolve_adjustments(&self, mut adjustments: AdjustmentSet, info: &ImageInfo) -> AdjustmentSet {
        // RAW defaults follow Capture One: its own camera profile and film curve,
        // plus the noise reduction and sharpening it starts a frame with for this
        // body and ISO. JPEG/TIFF frames are already display-referred, so they get
        // none of this.
        // Only sensor-decoded RAW gets the camera profile and film curve; the
        // embedded preview is already processed by the camera.
        if crate::raw::is_raw(&info.path)
            && crate::raw::from_sensor(&info.path) != Some(false)
        {
            let model = info.exif.camera_model.as_deref();
            if adjustments.base.film_curve.is_none() {
                adjustments.base.film_curve = model.and_then(crate::camera_curve::default_curve);
            }
            let icc_is_output_space = adjustments.base.icc_profile.as_deref().is_some_and(|name| {
                let lower = name.trim().to_ascii_lowercase();
                lower.contains("adobe rgb")
                    || lower.contains("srgb")
                    || lower.contains("display p3")
                    || lower.contains("prophoto")
            });
            if adjustments.base.icc_profile.is_none() || icc_is_output_space {
                adjustments.base.icc_profile = model
                    .and_then(crate::camera_profile::for_camera)
                    .filter(|name| crate::camera_profile::is_available(name))
                    .map(|name| name.as_str().to_string());
            }
            let catalog = self.catalog_overrides(model, info.exif.iso);
            if adjustments.noise_reduction == crate::NoiseReduction::default() {
                if let Some(iso) = info.exif.iso {
                    // C1's own values win; the ISO curve only fills gaps a body
                    // leaves unstated in coreConfig.
                    let heuristic: crate::NoiseReduction = crate::noise_defaults::for_iso(iso).into();
                    adjustments.noise_reduction = crate::NoiseReduction {
                        luminance: catalog.default_for("NoiseReductionAmount").unwrap_or(heuristic.luminance),
                        details: catalog.default_for("NoiseDetailsAmount").unwrap_or(heuristic.details),
                        color: catalog.default_for("ColorNoiseReductionAmount").unwrap_or(heuristic.color),
                        single_pixel: heuristic.single_pixel,
                    };
                }
            }
            if adjustments.sharpening == crate::Sharpening::default() {
                let default = crate::Sharpening::default();
                adjustments.sharpening = crate::Sharpening {
                    amount: catalog.default_for("USMAmount").unwrap_or(default.amount),
                    radius: catalog.default_for("USMRadius").unwrap_or(default.radius),
                    threshold: catalog.default_for("USMThreshold").unwrap_or(default.threshold),
                    halo_suppression: default.halo_suppression,
                };
            }
        } else if !crate::raw::is_raw(&info.path) && adjustments.base.icc_profile.is_none() {
            // Display-referred JPEG/TIFF keeps its embedded profile. Only a file
            // still carrying the camera's own colour space gets the camera
            // profile added here (see `camera_icc`).
            adjustments.base.icc_profile = camera_icc(
                &info.path,
                info.exif.camera_model.as_deref(),
            )
            .filter(|name| crate::camera_profile::is_available(name));
        }
        // C1's automatic lens detection is an import default like the ones
        // above: EXIF names the lens, the database supplies the profile, and
        // only a hit turns the calibrated warp on.
        apply_auto_lens(&mut adjustments.lens, crate::lensdb::default_database().unwrap_or(&[]), &info.exif);
        adjustments.lens.mount = info
            .exif
            .camera_model
            .as_deref()
            .and_then(crate::lensdb::mount_for_camera)
            .map(str::to_string);
        adjustments
    }

    /// [`Self::catalog_overrides`] plus the values the frame itself states
    /// (EXIF exposure, Nikon white balance). Used for the sliders; the render
    /// path only needs the catalog half, so it does not re-read the RAW.
    fn frame_defaults(&self, model: Option<&str>, iso: Option<u32>, path: &Path) -> crate::ParamOverrides {
        let catalog = self.catalog_overrides(model, iso);
        let catalog_loaded = matches!(self.camera_defaults.get(), Some(Some(_)));
        let frame = crate::read_nikon_info_block(path).ok();
        // `camera_defaults` encodes the reference Z6III's ladder. Use it only
        // when coreConfig is missing and the frame really is that body, so no
        // other camera gets defaults it never had.
        let reference = !catalog_loaded
            && model.is_some_and(|m| {
                crate::camera_profile::for_camera(m) == crate::camera_profile::for_camera("NIKON Z6_3")
            });
        let mut overrides = match (&frame, reference) {
            (Some(frame), true) => crate::overrides_for(frame),
            _ => catalog,
        };
        if let Some(frame) = &frame {
            frame.apply_to(&mut overrides);
        }
        overrides
    }

    fn camera_model(&self, image: ImageId) -> Option<String> {
        let st = self.state.lock();
        st.entries
            .iter()
            .find(|entry| entry.info.id == image)
            .and_then(|entry| entry.info.exif.camera_model.clone())
    }

    fn style_for_image_with(
        &self,
        image: ImageId,
        style: &StyleInfo,
        target: &AdjustmentSet,
        predict: impl FnOnce(&crate::matchlook::LookReference) -> Option<crate::matchlook::LookPrediction>,
    ) -> anyhow::Result<AdjustmentSet> {
        let Some(source) = style.source.as_deref().filter(|_| style.has_look) else {
            return Ok(style.adjustments.clone());
        };
        let white_balance = CameraWhiteBalance { camera: self.camera_model(image) };
        let mut context = crate::matchlook::LookToneContext {
            subtract_baseline: [0.0; 5],
            add_baseline: [0.0; 5],
            weights: [None; 4],
            auto_white_balance: false,
            white_balance: &white_balance,
        };
        let mut applied = false;
        match crate::c1_styles::load_style_for_target_with(
            source,
            target,
            &mut context,
            predict,
            &mut applied,
        ) {
            Ok(loaded) if applied => Ok(loaded.adjustments),
            Ok(_) => Ok(style.adjustments.clone()),
            Err(err) => {
                log::warn!(
                    "AICGR style {} for image {}: {err:#}; using the preloaded style",
                    source.display(),
                    image.0
                );
                Ok(style.adjustments.clone())
            }
        }
    }
}

impl FileEngine {
    /// Compositional catalog/metadata service (see `file_engine_services`).
    fn catalog(&self) -> services::CatalogService<'_> {
        services::CatalogService::new(self)
    }

    /// Compositional source-acquisition service (see `file_engine_services`).
    fn sources(&self) -> services::SourceService<'_> {
        services::SourceService::new(self)
    }

    /// Compositional preview/thumbnail render service.
    fn renders(&self) -> services::RenderService<'_> {
        services::RenderService::new(self)
    }

    /// Compositional export service (see `file_engine_services`).
    fn exports(&self) -> services::ExportService<'_> {
        services::ExportService::new(self)
    }

    /// Compositional merge service (see `file_engine_services`).
    fn merges(&self) -> services::MergeService<'_> {
        services::MergeService::new(self)
    }

    /// Compositional persistence service (see `file_engine_services`).
    fn persistence(&self) -> services::PersistenceService<'_> {
        services::PersistenceService::new(self)
    }

    /// Compositional AI inference service (see `file_engine_services`).
    fn inference(&self) -> services::InferenceService<'_> {
        services::InferenceService::new(self)
    }

    /// Compositional library import/rescan service (see `file_engine_services`).
    fn imports(&self) -> services::ImportService<'_> {
        services::ImportService::new(self)
    }

    /// Compositional library filesystem service (see `file_engine_services`).
    fn library(&self) -> services::LibraryService<'_> {
        services::LibraryService::new(self)
    }

    /// Compositional tethered-capture service (see `file_engine_services`).
    fn captures(&self) -> services::CaptureService<'_> {
        services::CaptureService::new(self)
    }
}

impl FileEngine {
    pub fn merge_hdr_frames(&self, ids: &[ImageId]) -> anyhow::Result<ImageInfo> {
        self.merges().merge_hdr_frames(ids)
    }

    /// [`Self::merge_hdr_frames`] with the interface's options. `align: false`
    /// is honoured by keeping the existing no-alignment path; `deghost` is
    /// consumed by `hdr::fuse_with_options`; `auto_tone` is only logged.
    pub fn merge_hdr_with_options(
        &self,
        ids: &[ImageId],
        options: &HdrOptions,
    ) -> anyhow::Result<ImageInfo> {
        self.merges().merge_hdr_with_options(ids, options)
    }

    fn start_merge(
        &self,
        kind: crate::merge_async::MergeKind,
        images: &[ImageId],
        options: &crate::MergeRequest,
    ) -> anyhow::Result<crate::MergeJobId> {
        self.merges().start_merge(kind, images, options)
    }

    /// Stitches frames left to right into one 8-bit TIFF beside the first frame
    /// and catalogs it. Frames must share one height; each neighbouring pair is
    /// aligned by a horizontal SAD search and the overlap is feathered.
    /// Distortion, perspective and exposure matching are not applied yet.
    pub fn merge_panorama_frames(&self, ids: &[ImageId]) -> anyhow::Result<ImageInfo> {
        self.merges().merge_panorama_frames(ids)
    }

    /// [`Self::merge_panorama_frames`] with the interface's options.
    /// `projection` is consumed by `pano::stitch_with_options`; `auto_crop:
    /// false` keeps the stitched canvas edges; boundary warp and auto crop are
    /// still not implemented.
    pub fn merge_panorama_with_options(
        &self,
        ids: &[ImageId],
        options: &PanoramaOptions,
    ) -> anyhow::Result<ImageInfo> {
        self.merges().merge_panorama_with_options(ids, options)
    }

}

impl ImageEngine for FileEngine {
    fn name(&self) -> String {
        format!("Portrait File Engine ({})", self.root.display())
    }

    fn save_c1_settings(
        &self,
        image: ImageId,
        adjustments: &AdjustmentSet,
    ) -> anyhow::Result<crate::c1_cos::CosSaveOutcome> {
        self.save_to_c1_settings(image, adjustments)
    }

    fn save_c1_catalog(
        &self,
        image: ImageId,
        adjustments: &AdjustmentSet,
    ) -> anyhow::Result<crate::c1_writeback::CatalogSaveOutcome> {
        self.save_to_c1_catalog(image, adjustments)
    }

    fn images(&self) -> Vec<ImageInfo> {
        self.catalog().images()
    }

    fn import(&self, paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
        self.imports().import(paths)
    }

    fn collections(&self) -> Vec<Collection> {
        self.catalog().collections()
    }

    fn library_root(&self) -> Option<PathBuf> {
        self.catalog().library_root()
    }

    fn collection_path(&self, id: u64) -> Option<PathBuf> {
        self.catalog().collection_path(id)
    }

    fn user_collections(&self) -> Vec<crate::UserCollection> {
        self.catalog().user_collections()
    }

    fn manage_files(&self, action: crate::LibraryAction) -> anyhow::Result<Vec<ImageInfo>> {
        self.manage_library(action)
    }

    fn set_lcc_calibration(
        &self,
        image: ImageId,
        calibration: Option<crate::dust_lcc::LightFalloff>,
    ) -> anyhow::Result<()> {
        let (raw, settings) = self
            .lcc_files(image)
            .ok_or_else(|| anyhow::anyhow!("image {} has no Capture One .cos settings file", image.0))?;
        let Some(calibration) = calibration else {
            return crate::c1_cos::write_global_lcc_id(&settings, None);
        };
        let existing = std::fs::read(&settings)
            .ok()
            .and_then(|bytes| crate::c1_cos::CosDocument::parse(&bytes).ok())
            .and_then(|document| document.variant().and_then(|variant| variant.lcc_id.clone()));
        let id = existing.unwrap_or_else(|| lcc_id_for(&calibration));
        let bytes = crate::dust_lcc::write_light_falloff(&calibration)?;
        let settings_dir = settings
            .parent()
            .ok_or_else(|| anyhow::anyhow!("settings path has no parent: {}", settings.display()))?;
        let raw_name = raw
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("raw path has no UTF-8 file name: {}", raw.display()))?;
        let path = crate::dust_lcc::sidecar_path(settings_dir, raw_name, &id);
        std::fs::write(&path, bytes)
            .map_err(|error| anyhow::anyhow!("writing {}: {error}", path.display()))?;
        crate::c1_cos::write_global_lcc_id(&settings, Some(&id))
    }

    fn lcc_calibration(&self, image: ImageId) -> Option<crate::dust_lcc::LightFalloff> {
        let (raw, settings) = self.lcc_files(image)?;
        let bytes = std::fs::read(&settings).ok()?;
        let document = crate::c1_cos::CosDocument::parse(&bytes).ok()?;
        let id = document.variant()?.lcc_id.as_deref()?;
        let settings_dir = settings.parent()?;
        let raw_name = raw.file_name()?.to_str()?;
        let path = crate::dust_lcc::sidecar_path(settings_dir, raw_name, id);
        crate::dust_lcc::load_light_falloff(&path).ok()
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
        ImageEngine::update_info_batch(self, std::slice::from_ref(info))
    }

    fn update_info_batch(&self, infos: &[ImageInfo]) -> anyhow::Result<()> {
        if infos.is_empty() {
            return Ok(());
        }
        let catalog = {
            let mut st = self.state.lock();
            for info in infos {
                if !st.entries.iter().any(|entry| entry.info.id == info.id) {
                    anyhow::bail!("unknown image");
                }
            }
            for info in infos {
                let rel_path = {
                    let entry = st
                        .entries
                        .iter_mut()
                        .find(|entry| entry.info.id == info.id)
                        .expect("validated image id");
                    entry.info = info.clone();
                    entry.rel_path.clone()
                };
                st.catalog.upsert(Entry {
                    rel_path,
                    rating: info.rating,
                    color_tag: ColorTag::ALL.iter().position(|c| *c == info.color_tag).unwrap_or(0) as i8,
                    keywords: info.keywords.clone(),
                    variant: info.variant,
                    updated_at: now_secs(),
                    trashed: info.trashed,
                });
            }
            st.catalog.clone()
        };
        if let Some(store) = &self.store {
            if let Err(err) = store.save_index(&catalog) {
                log::warn!("save catalog: {err}");
            }
        }
        Ok(())
    }

    fn load_adjustments(&self, image: ImageId) -> AdjustmentSet {
        let seeded = {
            let st = self.state.lock();
            st.entries
                .iter()
                .find(|e| e.info.id == image)
                .map(|e| (e.adjustments.clone(), e.info.clone()))
        };
        match seeded {
            Some((adjustments, info)) => self.resolve_adjustments(adjustments, &info),
            None => AdjustmentSet::default(),
        }
    }

    fn load_adjustments_batch(&self, images: &[ImageId]) -> Vec<(ImageId, AdjustmentSet)> {
        // One snapshot under the state lock: every requested entry is cloned
        // exactly once, then the import defaults resolve in parallel outside
        // it. This is what keeps a 20k-image startup from taking the lock and
        // scanning the entry list 20k times.
        let wanted: std::collections::HashSet<ImageId> = images.iter().copied().collect();
        let seeded: Vec<(ImageId, Option<(AdjustmentSet, ImageInfo)>)> = {
            let st = self.state.lock();
            let mut by_id: HashMap<ImageId, (AdjustmentSet, ImageInfo)> = st
                .entries
                .iter()
                .filter(|entry| wanted.contains(&entry.info.id))
                .map(|entry| (entry.info.id, (entry.adjustments.clone(), entry.info.clone())))
                .collect();
            let mut seen = std::collections::HashSet::with_capacity(images.len());
            images
                .iter()
                .map(|&image| {
                    if !seen.insert(image) {
                        // A repeated id gets another copy rather than falling
                        // through to the default, matching a per-image load.
                        let entry = st
                            .entries
                            .iter()
                            .find(|entry| entry.info.id == image)
                            .map(|entry| (entry.adjustments.clone(), entry.info.clone()));
                        return (image, entry);
                    }
                    (image, by_id.remove(&image))
                })
                .collect()
        };
        seeded
            .into_par_iter()
            .map(|(image, seeded)| {
                let adjustments = seeded.map_or_else(AdjustmentSet::default, |(adjustments, info)| {
                    self.resolve_adjustments(adjustments, &info)
                });
                (image, adjustments)
            })
            .collect()
    }

    fn save_adjustments(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<()> {
        ImageEngine::save_adjustments_batch(self, &[(image, adjustments.clone())])
    }

    fn save_adjustments_batch(
        &self,
        items: &[(ImageId, AdjustmentSet)],
    ) -> anyhow::Result<()> {
        {
            let st = self.state.lock();
            if items
                .iter()
                .any(|(image, _)| !st.entries.iter().any(|entry| entry.info.id == *image))
            {
                anyhow::bail!("unknown image");
            }
        }
        match FileEngine::save_adjustments_batch_report(self, items).into_error() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn save_adjustments_batch_report(
        &self,
        items: &[(ImageId, AdjustmentSet)],
    ) -> crate::SaveReport {
        FileEngine::save_adjustments_batch_report(self, items)
    }

    fn save_adjustments_queued(
        &self,
        items: &[(ImageId, AdjustmentSet)],
    ) -> Option<QueuedSaveSummary> {
        Some(FileEngine::save_adjustments_queued(self, items))
    }

    fn flush_pending_saves(&self, timeout: std::time::Duration) -> Option<FlushReport> {
        FileEngine::flush_saves(self, timeout)
    }

    fn pending_save_status(&self) -> Option<QueueStatus> {
        FileEngine::save_queue_status(self)
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

    fn merge_hdr_frames(&self, ids: &[ImageId]) -> anyhow::Result<ImageInfo> {
        FileEngine::merge_hdr_frames(self, ids)
    }

    fn merge_hdr_with_options(
        &self,
        ids: &[ImageId],
        options: &HdrOptions,
    ) -> anyhow::Result<ImageInfo> {
        FileEngine::merge_hdr_with_options(self, ids, options)
    }

    fn merge_panorama_frames(&self, ids: &[ImageId]) -> anyhow::Result<ImageInfo> {
        FileEngine::merge_panorama_frames(self, ids)
    }

    fn merge_panorama_with_options(
        &self,
        ids: &[ImageId],
        options: &PanoramaOptions,
    ) -> anyhow::Result<ImageInfo> {
        FileEngine::merge_panorama_with_options(self, ids, options)
    }

    fn merge_hdr(&self, images: &[ImageId], options: &crate::MergeRequest) -> anyhow::Result<crate::MergeJobId> {
        self.start_merge(crate::merge_async::MergeKind::Hdr, images, options)
    }

    fn merge_panorama(&self, images: &[ImageId], options: &crate::MergeRequest) -> anyhow::Result<crate::MergeJobId> {
        self.start_merge(crate::merge_async::MergeKind::Panorama, images, options)
    }

    fn merge_progress(&self, job_id: crate::MergeJobId) -> Option<crate::MergeProgress> {
        self.merge_registry.progress(job_id)
    }

    fn cancel_merge(&self, job_id: crate::MergeJobId) -> bool {
        self.merge_registry.cancel(job_id)
    }

    fn merge_result(&self, job_id: crate::MergeJobId) -> Option<std::result::Result<crate::MergeOutcome, crate::MergeError>> {
        self.merge_registry.result(job_id)
    }

    fn set_soft_proof_profile(&self, profile: Option<String>, intent: u8) -> anyhow::Result<()> {
        let settings = crate::merge_async::settings_with_profile(profile, intent)?;
        *self.soft_proof.write() = settings.clone();
        if let Some(store) = &self.store {
            store.save_soft_proof(&settings)?;
        }
        Ok(())
    }

    fn soft_proof_profile(&self) -> crate::SoftProofSettings {
        self.soft_proof.read().clone()
    }

    fn soft_proof_preview(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<PreviewFrame> {
        let settings = self.soft_proof.read().clone();
        let request = PreviewRequest {
            image,
            adjustments: adjustments.clone(),
            region: NormRect::FULL,
            max_size: max_size.max(1),
            quality: RenderQuality::Final,
            bypass: false,
            mask_overlay: None,
            generation: 1,
            display_profile: None,
            proof_profile: settings.profile,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
            rendering_intent: Some(settings.intent),
            black_point_compensation: false,
        };
        self.render_preview(&request)
    }

    fn soft_proof_gamut_warning(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<Option<crate::GamutWarning>> {
        Ok(self.soft_proof_preview(image, adjustments, max_size)?.gamut_warning)
    }

    fn thumbnail(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<RgbaImage> {
        self.renders().thumbnail(image, adjustments, max_size)
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
            // `auto_keystone` converts the detector's degrees into the
            // renderer's percentage controls; keep the user's aspect/skew/
            // focal_length/flags.
            let suggestion = crate::keystone::auto_keystone(
                current.geometry.keystone.aspect,
                &frame.rgba,
                frame.width,
                frame.height,
            );
            adj.geometry.keystone = crate::Keystone {
                aspect: current.geometry.keystone.aspect,
                skew: current.geometry.keystone.skew,
                focal_length: current.geometry.keystone.focal_length,
                flags: current.geometry.keystone.flags,
                ..suggestion
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
        self.inference().compute_ai_mask(image, kind)
    }

    fn eye_report(&self, image: ImageId) -> anyhow::Result<crate::EyeReport> {
        self.inference().eye_report(image)
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
        let ((c1, film), imported) = rayon::join(
            || {
                rayon::join(
                    crate::c1_style_index::installed_styles,
                    film_curve_styles,
                )
            },
            || self.imported_styles.read().clone(),
        );
        all.extend(c1);
        all.extend(film);
        all.extend(imported);
        dedupe_styles(all)
    }

    fn film_curves(&self) -> Vec<String> {
        crate::filmcurve::names()
    }

    fn import_style(&self, path: &Path) -> anyhow::Result<StyleInfo> {
        let style = crate::c1_styles::load_style(path)?;
        let info = StyleInfo {
            name: style.name,
            group: "Imported".into(),
            adjustments: style.adjustments,
            source: Some(path.to_path_buf()),
            has_look: style.look.is_some(),
        };
        let mut imported = self.imported_styles.write();
        imported.retain(|s| !(s.group == info.group && s.name == info.name));
        imported.push(info.clone());
        Ok(info)
    }

    fn style_for_image(
        &self,
        image: ImageId,
        style: &StyleInfo,
        target: &AdjustmentSet,
    ) -> anyhow::Result<AdjustmentSet> {
        self.style_for_image_with(image, style, target, crate::matchlook::predict_look_reference)
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
        let (model, iso, path) = {
            let st = self.state.lock();
            let Some(entry) = st.entries.iter().find(|e| e.info.id == image) else {
                return crate::ParamOverrides::default();
            };
            (entry.info.exif.camera_model.clone(), entry.info.exif.iso, entry.path.clone())
        };
        // The Nikon block sits in the packed RAW, not in the EIP container.
        let path = if is_eip(&path) {
            match materialize_eip(&self.root, &path) {
                Ok(assets) => assets.raw,
                Err(_) => path,
            }
        } else {
            path
        };
        self.frame_defaults(model.as_deref(), iso, &path)
    }

    fn lens_profiles(&self) -> Vec<String> {
        Vec::new()
    }

    fn start_export(&self, job: ExportJob) -> anyhow::Result<()> {
        self.exports().start_export(job)
    }

    fn job_status(&self, job_id: u64) -> JobStatus {
        self.exports().job_status(job_id)
    }

    fn cancel_job(&self, job_id: u64) -> bool {
        self.exports().cancel_job(job_id)
    }

    fn source_status(&self, image: ImageId) -> Option<String> {
        self.catalog().source_status(image)
    }

    fn tether_status(&self) -> TetherStatus {
        self.captures().tether_status()
    }
    fn capture_devices(&self) -> Vec<crate::CaptureDevice> {
        self.captures().capture_devices()
    }
    fn portrait_capabilities(&self) -> Vec<crate::PortraitCapability> {
        crate::panel_commands::portrait_capabilities()
    }
    fn layer_summaries(&self, image: ImageId) -> Vec<crate::engine::LayerSummary> {
        self.catalog().layer_summaries(image)
    }
    fn retouch_masks(&self, image: ImageId) -> Vec<crate::c1_retouch::RetouchMaskSummary> {
        self.catalog().retouch_masks(image)
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
            blemish_gainmap: true,
            blemish_gainmap_model: crate::blemish_gainmap::model_available(),
            export_formats: crate::output::ExportFormat::ALL
                .into_iter()
                .filter(|format| format.is_supported())
                .collect(),
        }
    }
    fn camera_status(&self) -> CameraStatus {
        self.captures().camera_status()
    }
    fn camera_settings(&self) -> Option<CameraSettings> {
        self.captures().camera_settings()
    }
    fn capture(&self) -> anyhow::Result<ImageInfo> {
        self.captures().capture()
    }
    fn live_view_frame(&self, width: u32) -> Option<RgbaImage> {
        self.captures().live_view_frame(width)
    }
    /// See [`services::CaptureService::live_view_focus`]: focus is a no-op
    /// because this binding does not read the camera's config tree.
    fn live_view_focus(&self, steps: i32) {
        self.captures().live_view_focus(steps)
    }
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
            Some(StyleInfo {
                name: name.clone(),
                group: "Film Curve".into(),
                adjustments,
                source: None,
                has_look: false,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_provenance_is_the_only_c1_z6iii_linear_switch() {
        use crate::raw::{DecodeBackend, DecodeProvenance, OpenClStatus, PixelSpace};

        let opencl =
            DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiOpenCl);
        assert!(opencl.unlocks_c1_z6iii_linear());
        assert_eq!(opencl.pixel_space, PixelSpace::CameraLinear16);

        // OpenCL failing into the CPU C1 chain still yields camera-linear Z6III
        // samples, but the failure state stays visible.
        let cpu_failed = DecodeProvenance::aurora(
            crate::aurora_raw::AuroraRoute::Z6iiiCpuOpenClFailed,
        );
        assert!(cpu_failed.unlocks_c1_z6iii_linear());
        assert_eq!(cpu_failed.opencl, OpenClStatus::Failed);

        // LibRaw and preview fallbacks must never unlock the pre-matrix.
        let libraw = DecodeProvenance::libraw16();
        assert_eq!(libraw.backend, DecodeBackend::LibRaw);
        assert_eq!(libraw.pixel_space, PixelSpace::LibRawLinear16);
        assert!(!libraw.unlocks_c1_z6iii_linear());
        let preview = DecodeProvenance::camera_preview();
        assert_eq!(preview.pixel_space, PixelSpace::EmbeddedPreview);
        assert!(!preview.unlocks_c1_z6iii_linear());

        // Generic Bayer is camera-linear but not Z6III.
        let generic = DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Cpu);
        assert_eq!(generic.pixel_space, PixelSpace::CameraLinear16);
        assert!(!generic.unlocks_c1_z6iii_linear());
    }

    #[test]
    fn native_and_proxy_raw_sources_share_banding_identity() {
        // Synthetic capture metadata: no NEF or C1 install involved. The proxy
        // source() reads the file EXIF; native_raw_source() reads entry.info.exif.
        // Both funnel the same ISO/model fields through raw_banding_identity().
        let exif = crate::Exif {
            camera_model: Some("NIKON Z 6III".to_owned()),
            iso: Some(320),
            ..Default::default()
        };
        let proxy_identity = raw_banding_identity(&exif);
        let native_identity = raw_banding_identity(&exif);
        assert_eq!(native_identity, proxy_identity);
        assert_eq!(proxy_identity, (Some(320), Some("NIKON Z 6III".to_owned())));

        // The identity is resolution-independent: the full-size native shape and
        // the 4096 proxy shape are built through the same metadata helper.
        let decorate = |source: Source, identity: &(Option<u32>, Option<String>)| {
            source.with_banding_identity(identity.0, identity.1.as_deref())
        };
        let native_shape = decorate(Source::from_rgba16(8, 6, &[0u16; 8 * 6 * 4]), &native_identity);
        let proxy_shape = decorate(Source::from_rgba16(4, 3, &[0u16; 4 * 3 * 4]), &proxy_identity);
        std::hint::black_box((native_shape, proxy_shape));

        // Either field missing keeps BandingSuppression off for both paths.
        let missing_model = crate::Exif { iso: Some(320), ..Default::default() };
        assert_eq!(raw_banding_identity(&missing_model), (Some(320), None));
        let missing_iso = crate::Exif { camera_model: Some("NIKON Z 6III".to_owned()), ..Default::default() };
        assert_eq!(raw_banding_identity(&missing_iso), (None, Some("NIKON Z 6III".to_owned())));
    }

    #[test]
    fn aicgr_target_style_changes_exposure_and_white_balance_against_the_old_path() {
        use base64::Engine as _;
        use crate::matchlook::{LookPrediction, LOOK_BW_VECTOR, LOOK_COLOR_VECTOR};

        let dir = std::env::temp_dir().join(format!("aurora-aicgr-target-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("Target Look.costyle");
        let mut tiff = Vec::new();
        tiff::encoder::TiffEncoder::new(std::io::Cursor::new(&mut tiff))
            .unwrap()
            .write_image::<tiff::encoder::colortype::RGB8>(1, 1, &[10, 20, 30])
            .unwrap();
        let bitmaps = base64::engine::general_purpose::STANDARD.encode(tiff);
        std::fs::write(
            &path,
            format!(
                r#"<SL><E K="Name" V="Target Look"/><AICGR><E K="Bitmaps" V="{bitmaps}"/><E K="Impact" V="50"/></AICGR></SL>"#
            ),
        )
        .unwrap();

        let plain = crate::c1_styles::load_style(&path).unwrap();
        let style = StyleInfo {
            name: plain.name.clone(),
            group: "Imported".into(),
            adjustments: plain.adjustments.clone(),
            source: Some(path.clone()),
            has_look: plain.look.is_some(),
        };
        let engine = FileEngine::open(&dir).unwrap();
        let mut target = AdjustmentSet::default();
        target.exposure.exposure = 1.0;
        target.white_balance.gains = Some([1.625, 1.0, 1.412]);
        target.white_balance.kelvin = 4779.55;
        target.white_balance.tint = -4.4875;

        let adjusted = engine
            .style_for_image_with(ImageId(1), &style, &target, |reference| {
                assert_eq!(reference.thumbnails[0].rgba, [10, 20, 30, 255]);
                let mut color = [0.5; LOOK_COLOR_VECTOR];
                color[0] = 0.75;
                color[1] = 0.75;
                color[2] = 0.6;
                Some(LookPrediction { color, bw: [0.5; LOOK_BW_VECTOR] })
            })
            .unwrap();

        assert!((adjusted.exposure.exposure - 1.9).abs() < 1e-5, "{adjusted:?}");
        assert_ne!(adjusted.exposure.exposure, plain.adjustments.exposure.exposure);
        assert_ne!(
            adjusted.white_balance.gains,
            plain.adjustments.white_balance.gains,
            "target-aware WB must differ from the old default-gains path"
        );
        assert_ne!(adjusted.white_balance.gains, target.white_balance.gains);
        assert_eq!(target.exposure.exposure, 1.0);

        let fallback = engine
            .style_for_image_with(ImageId(1), &style, &target, |_| None)
            .unwrap();
        assert_eq!(fallback, style.adjustments, "no model output falls back to the plain style");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn applies_cos_global_and_masked_layers_from_the_settings_folder() {
        let dir = std::env::temp_dir().join(format!("aurora-cos-{}", std::process::id()));
        let capture = dir.join("Capture");
        let settings = dir.join("CaptureOne").join("Settings1680");
        std::fs::create_dir_all(&settings).unwrap();
        std::fs::create_dir_all(&capture).unwrap();
        let raw = capture.join("IMG_0001.IIQ");
        std::fs::write(&raw, b"not a raw").unwrap();
        std::fs::write(
            settings.join("IMG_0001.IIQ.cos"),
            br#"<IMG><E K="Version" V="1530"/><VAR Engine="600"><DL><E K="Exposure" V="1.25"/></DL><AL><E K="MaskUUID" V="MASK-1"/><E K="Exposure" V="-0.75"/></AL></VAR></IMG>"#,
        )
        .unwrap();

        let adjustments = cos_adjustments(&raw).expect("`.cos` applies");
        assert!((adjustments.exposure.exposure - 1.25).abs() < 1e-6);
        assert_eq!(adjustments.layers.len(), 1);
        assert!((adjustments.layers[0].adjustments.exposure.exposure + 0.75).abs() < 1e-6);
        std::fs::remove_dir_all(dir).ok();
    }

    fn lcc_fixture(name: &str) -> (PathBuf, PathBuf, crate::dust_lcc::LightFalloff) {
        let dir = std::env::temp_dir().join(format!("aurora-lcc-command-{}-{name}", std::process::id()));
        let settings = dir.join("CaptureOne").join("Settings1680");
        std::fs::create_dir_all(&settings).unwrap();
        let raw = dir.join("a.png");
        image::RgbaImage::from_pixel(4, 3, image::Rgba([20, 40, 60, 255]))
            .save(&raw)
            .unwrap();
        std::fs::write(
            settings.join("a.png.cos"),
            br#"<IMG><VAR Engine="600"><DL><E K="Exposure" V="0"/></DL></VAR></IMG>"#,
        )
        .unwrap();
        let sample = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/lcc/vignette-1ev-1500x1000.lcc");
        let grid = crate::dust_lcc::load_light_falloff(&sample).unwrap();
        (dir, settings, grid)
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn lcc_calibration_round_trips_through_cos_lccid_and_sidecar() {
        let (dir, settings, grid) = lcc_fixture("round-trip");
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        engine.set_lcc_calibration(id, Some(grid.clone())).unwrap();
        assert_eq!(engine.lcc_calibration(id), Some(grid.clone()));

        let document = crate::c1_cos::CosDocument::parse(&std::fs::read(settings.join("a.png.cos")).unwrap())
            .unwrap();
        assert!(document.variant().unwrap().lcc_id.is_some());
        let sidecars: Vec<_> = std::fs::read_dir(&settings)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "lcc"))
            .collect();
        assert_eq!(sidecars.len(), 1);

        drop(engine);
        let reopened = FileEngine::open(&dir).unwrap();
        let id = reopened.images()[0].id;
        assert_eq!(reopened.lcc_calibration(id), Some(grid));
        assert!(reopened.load_adjustments(id).lens.uniform_light_grid.is_some());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn clearing_lcc_calibration_removes_the_cos_pointer() {
        let (dir, settings, grid) = lcc_fixture("clear");
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        engine.set_lcc_calibration(id, Some(grid)).unwrap();
        engine.set_lcc_calibration(id, None).unwrap();
        assert_eq!(engine.lcc_calibration(id), None);
        let document = crate::c1_cos::CosDocument::parse(&std::fs::read(settings.join("a.png.cos")).unwrap())
            .unwrap();
        assert_eq!(document.variant().unwrap().lcc_id, None);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn setting_lcc_on_an_unknown_image_is_an_error() {
        let (dir, _settings, grid) = lcc_fixture("unknown");
        let engine = FileEngine::open(&dir).unwrap();
        assert!(engine.set_lcc_calibration(ImageId(999), Some(grid)).is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    #[ignore = "profiles folder rescan metadata extraction"]
    fn bench_folder_rescan_metadata() {
        use std::hash::{Hash, Hasher};
        use std::time::Instant;

        let dir = std::env::var_os("AURORA_RESCAN_BENCH_DIR")
            .expect("set AURORA_RESCAN_BENCH_DIR");
        let engine = FileEngine::open(&dir).expect("open benchmark folder");
        let mut times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            engine.rescan().expect("rescan");
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        for image in engine.images() {
            format!("{image:?}").hash(&mut hash);
        }
        println!(
            "RESCAN_METADATA count={} p50={:.3} ms p95={:.3} ms hash={:016x}",
            engine.images().len(),
            times[2],
            times[4],
            hash.finish()
        );
    }

    #[test]
    #[ignore = "profiles bulk import metadata extraction"]
    fn bench_bulk_import_metadata() {
        use std::hash::{Hash, Hasher};
        use std::time::Instant;

        let source = std::env::var_os("AURORA_IMPORT_BENCH_SOURCE")
            .expect("set AURORA_IMPORT_BENCH_SOURCE");
        let root = std::env::var_os("AURORA_IMPORT_BENCH_ROOT")
            .expect("set AURORA_IMPORT_BENCH_ROOT");
        let engine = FileEngine::open(&root).expect("open empty import root");
        let mut paths: Vec<PathBuf> = std::fs::read_dir(source)
            .expect("read import source")
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .collect();
        paths.sort();
        let start = Instant::now();
        let added = engine.import(&paths).expect("bulk import");
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        for image in &added {
            image.file_name.hash(&mut hash);
            image.width.hash(&mut hash);
            image.height.hash(&mut hash);
            image.file_size.hash(&mut hash);
            format!("{:?}", image.exif).hash(&mut hash);
            format!("{:?}", image.iptc).hash(&mut hash);
        }
        println!(
            "IMPORT_METADATA count={} ms={elapsed:.3} hash={:016x}",
            added.len(),
            hash.finish()
        );
    }

    #[test]
    #[ignore = "profiles startup adjustment loading"]
    fn bench_load_adjustments_serial_parallel() {
        use std::hash::{Hash, Hasher};
        use std::time::Instant;

        let dir = std::env::var_os("AURORA_RESCAN_BENCH_DIR")
            .expect("set AURORA_RESCAN_BENCH_DIR");
        let engine = FileEngine::open(&dir).expect("open benchmark folder");
        let ids: Vec<_> = engine.images().into_iter().map(|image| image.id).collect();

        let start = Instant::now();
        let serial: Vec<_> = ids
            .iter()
            .map(|id| engine.load_adjustments(*id))
            .collect();
        let serial_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let parallel: Vec<_> = ids
            .par_iter()
            .map(|id| engine.load_adjustments(*id))
            .collect();
        let parallel_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let batch: Vec<_> = engine
            .load_adjustments_batch(&ids)
            .into_iter()
            .map(|(_, adjustments)| adjustments)
            .collect();
        let batch_ms = start.elapsed().as_secs_f64() * 1000.0;

        let hash = |values: &[AdjustmentSet]| {
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            for value in values {
                format!("{value:?}").hash(&mut hash);
            }
            hash.finish()
        };
        let serial_hash = hash(&serial);
        let parallel_hash = hash(&parallel);
        let batch_hash = hash(&batch);
        assert_eq!(serial_hash, parallel_hash);
        assert_eq!(serial_hash, batch_hash);
        println!(
            "LOAD_ADJUSTMENTS count={} serial={serial_ms:.3} ms parallel={parallel_ms:.3} ms \
             batch={batch_ms:.3} ms hash={serial_hash:016x}",
            ids.len()
        );
    }

    #[test]
    fn load_adjustments_batch_matches_serial_and_keeps_variants() {
        let dir = std::env::temp_dir()
            .join(format!("aurora-adjust-batch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, rgb) in [
            ("first.jpg", [12u8, 24, 36]),
            ("second.jpg", [48, 60, 72]),
            ("third.jpg", [84, 96, 108]),
        ] {
            image::RgbImage::from_pixel(6, 4, image::Rgb(rgb))
                .save(dir.join(name))
                .unwrap();
        }
        let engine = FileEngine::open(&dir).unwrap();
        let ids: Vec<ImageId> = engine.images().into_iter().map(|image| image.id).collect();
        assert_eq!(ids.len(), 3);

        let mut edited = engine.load_adjustments(ids[0]);
        edited.exposure.exposure = 1.75;
        engine.save_adjustments(ids[0], &edited).unwrap();

        let mut variant_seed = engine.load_adjustments(ids[1]);
        variant_seed.exposure.exposure = -2.0;
        let variant = engine.new_variant(ids[1], &variant_seed).unwrap();

        let all: Vec<ImageId> = engine.images().into_iter().map(|image| image.id).collect();
        assert!(all.contains(&variant.id), "the variant joins the startup list");
        let serial: Vec<(ImageId, AdjustmentSet)> = all
            .iter()
            .map(|&image| (image, engine.load_adjustments(image)))
            .collect();
        let batch = engine.load_adjustments_batch(&all);
        assert_eq!(batch, serial, "batch must resolve every variant identically");

        let subset = vec![ids[0], variant.id, ids[2]];
        let serial_subset: Vec<(ImageId, AdjustmentSet)> = subset
            .iter()
            .map(|&image| (image, engine.load_adjustments(image)))
            .collect();
        assert_eq!(engine.load_adjustments_batch(&subset), serial_subset);

        let unknown = ImageId(9_999_999);
        assert_eq!(
            engine.load_adjustments_batch(&[unknown]),
            vec![(unknown, AdjustmentSet::default())]
        );

        let repeated = vec![ids[0], ids[0]];
        let repeated_serial: Vec<(ImageId, AdjustmentSet)> = repeated
            .iter()
            .map(|&image| (image, engine.load_adjustments(image)))
            .collect();
        assert_eq!(engine.load_adjustments_batch(&repeated), repeated_serial);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_adjustments_batch_resolves_synthetic_raw_entries_like_single_loads() {
        let dir = std::env::temp_dir()
            .join(format!("aurora-adjust-batch-raw-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();

        let raw_info = |id: u64, variant: u32| ImageInfo {
            id: ImageId(id),
            path: PathBuf::from(format!(r"C:\synthetic\IMG_{id}.NEF")),
            file_name: format!("IMG_{id}.NEF"),
            kind: FileKind::Raw,
            width: 6048,
            height: 4024,
            file_size: 25_000_000,
            exif: Exif {
                camera_model: Some("NIKON Z 6III".to_owned()),
                lens: Some("NIKKOR Z 24-70mm f/2.8 S".to_owned()),
                focal_length_mm: Some(50.0),
                iso: Some(3200),
                ..Default::default()
            },
            iptc: Iptc::default(),
            keywords: Vec::new(),
            rating: 0,
            color_tag: ColorTag::default(),
            variant,
            has_adjustments: false,
            trashed: false,
            heif: None,
        };
        let mut seeded = AdjustmentSet::default();
        seeded.exposure.exposure = 2.5;
        seeded.noise_reduction.luminance = 7.0;
        {
            let mut st = engine.state.lock();
            st.entries.push(FileEntry {
                info: raw_info(9001, 1),
                path: PathBuf::from(r"C:\synthetic\IMG_9001.NEF"),
                rel_path: "IMG_9001.NEF".to_owned(),
                adjustments: seeded.clone(),
            });
            st.entries.push(FileEntry {
                info: raw_info(9002, 2),
                path: PathBuf::from(r"C:\synthetic\IMG_9001.NEF"),
                rel_path: "IMG_9001.NEF".to_owned(),
                adjustments: AdjustmentSet::default(),
            });
        }

        let ids = [ImageId(9001), ImageId(9002)];
        let serial: Vec<(ImageId, AdjustmentSet)> = ids
            .iter()
            .map(|&image| (image, engine.load_adjustments(image)))
            .collect();
        let batch = engine.load_adjustments_batch(&ids);
        assert_eq!(batch, serial, "the RAW/ICC/lens defaults are shared verbatim");
        assert_eq!(
            batch[0].1.exposure.exposure, 2.5,
            "a stored edit survives the import defaults"
        );
        assert_eq!(
            batch[0].1.noise_reduction.luminance, 7.0,
            "a non-default value is not replaced"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "profiles the complete engine style index"]
    fn bench_engine_style_index() {
        use std::time::Instant;

        let dir = std::env::var_os("AURORA_RESCAN_BENCH_DIR")
            .expect("set AURORA_RESCAN_BENCH_DIR");
        let engine = FileEngine::open(&dir).expect("open benchmark folder");
        let start = Instant::now();
        let styles = engine.styles();
        println!(
            "ENGINE_STYLE_INDEX count={} ms={:.3}",
            styles.len(),
            start.elapsed().as_secs_f64() * 1000.0
        );
    }

    #[test]
    #[ignore = "profiles real NEF decode through Source construction; set AURORA_TEST_RAW"]
    fn bench_real_nef_source_pipeline() {
        use std::time::Instant;

        let path = std::env::var_os("AURORA_TEST_RAW")
            .map(PathBuf::from)
            .expect("set AURORA_TEST_RAW to a real NEF file");
        for round in 0..3 {
            let started = Instant::now();
            let (w, h, px) = crate::raw::decode16(&path).expect("decode16");
            let decode_ms = started.elapsed().as_secs_f64() * 1000.0;

            let started = Instant::now();
            let (tw, th) = fit_long_side((w, h));
            let scaled = crate::recipe::resize_rgba16(&px, w, h, tw, th);
            let resize_ms = started.elapsed().as_secs_f64() * 1000.0;
            drop(px);

            let started = Instant::now();
            let source = Source::from_rgba16(tw, th, &scaled);
            let source_ms = started.elapsed().as_secs_f64() * 1000.0;
            println!(
                "NEF_SOURCE round={round} decode={decode_ms:.1} ms resize={resize_ms:.1} ms \
                 source={source_ms:.1} ms size={tw}x{th}"
            );
            std::hint::black_box(source);
        }

        let started = Instant::now();
        let cold = raw16_source(&path).expect("cold cached source").0;
        let cold_ms = started.elapsed().as_secs_f64() * 1000.0;
        drop(cold);
        let started = Instant::now();
        let warm = raw16_source(&path).expect("warm cached source").0;
        let warm_ms = started.elapsed().as_secs_f64() * 1000.0;
        println!(
            "NEF_FILE_ENGINE_SOURCE cold={cold_ms:.1} ms warm={warm_ms:.1} ms speedup={:.2}x",
            cold_ms / warm_ms
        );
        std::hint::black_box(warm);
    }

    #[test]
    fn midpoint_warning_only_touches_non_default_gradients() {
        warn_unverified_midpoint("default", [0.0, 0.0, 10.0, 0.0, 0.5]);
        warn_unverified_midpoint("skewed", [0.0, 0.0, 10.0, 0.0, 0.25]);
        warn_unverified_midpoint("skewed", [0.0, 0.0, 10.0, 0.0, 0.75]);
        warn_unverified_midpoint("broken", [0.0, 0.0, 10.0, 0.0, f32::NAN]);
    }

    /// A missing camera must fail before anything is written or catalogued; the
    /// mock engine is never consulted from this path.
    #[test]
    fn capture_without_a_camera_leaves_the_catalog_unchanged() {
        if !matches!(tether::probe(), TetherStatus::NoCamera) {
            return;
        }
        let dir = std::env::temp_dir().join(format!("portrait-nocam-{}", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!("portrait-caps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let caps = FileEngine::open(&dir).unwrap().capabilities();
        assert!(caps.eye_report && caps.auto_rotate && caps.match_look);
        assert!(caps.blemish_gainmap);
        assert_eq!(
            caps.blemish_gainmap_model,
            crate::blemish_gainmap::model_available()
        );
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
    fn heif_sources_decode_through_libheif() {
        if !crate::heif_write::available() {
            eprintln!("libheif unavailable; skipped the HEIF import round-trip");
            return;
        }
        let dir = std::env::temp_dir().join(format!("portrait-heif-source-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shot.heic");
        crate::heif_write::encode(&path, &vec![80u8; 32 * 24 * 4], 32, 24, 80, None, None, None)
            .unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let info = engine.images()[0].clone();
        assert_eq!(info.kind, FileKind::Heif);
        assert_eq!((info.width, info.height), (32, 24));
        let frame = engine.render_preview(&PreviewRequest {
            image: info.id,
            adjustments: AdjustmentSet::default(),
            region: NormRect::FULL,
            max_size: 32,
            quality: crate::RenderQuality::Final,
            bypass: false,
            mask_overlay: None,
            generation: 1,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
            rendering_intent: None,
            black_point_compensation: false,
        }).unwrap();
        assert_eq!((frame.image.width, frame.image.height), (32, 24));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn importing_an_unsupported_file_adds_nothing() {
        let dir = std::env::temp_dir().join(format!("portrait-import-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let note = dir.join("note.txt");
        std::fs::write(&note, b"not an image").unwrap();
        let added = engine.import(&[note]).unwrap();
        assert!(added.is_empty());
        assert_eq!(engine.images().len(), 0);
    }

    #[test]
    fn synthetic_import_returns_infos_one_succeeded_job_and_is_idempotent() {
        let root = s2_test_dir("import-service-root");
        let source = s2_test_dir("import-service-source");
        // Opening an empty library uses the unrecorded opening scan, so the
        // snapshot starts empty and the record below belongs to `import`.
        let engine = FileEngine::open(&root).unwrap();
        assert!(engine.import_job_snapshot().scope(JobScope::Import).is_none());

        let first = write_test_png(&source, "first.png", 32, 24, 11);
        let second = write_test_png(&source, "second.png", 40, 30, 22);
        let added = engine.import(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(added.len(), 2);
        assert_eq!(engine.images().len(), 2);
        let mut names: Vec<String> = added.iter().map(|info| info.file_name.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["first.png".to_string(), "second.png".to_string()]);
        assert!(added.iter().all(|info| info.width > 0 && info.height > 0 && info.file_size > 0));

        let snapshot = engine.import_job_snapshot();
        let import = snapshot.scope(JobScope::Import).expect("one import scope");
        assert_eq!(import.succeeded, 1, "one import call is one Succeeded record");
        assert_eq!(import.live(), 0);

        // Re-importing the same paths adds no entries; it still records the
        // call as Succeeded.
        let again = engine.import(&[first, second]).unwrap();
        assert!(again.is_empty());
        assert_eq!(engine.images().len(), 2);
        let snapshot = engine.import_job_snapshot();
        let import = snapshot.scope(JobScope::Import).expect("import scope");
        assert_eq!(import.succeeded, 2);
        assert_eq!(import.live(), 0);

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(&source).ok();
    }

    #[test]
    fn synthetic_rescan_records_one_succeeded_job_per_call() {
        let dir = s2_test_dir("rescan-service");
        write_test_png(&dir, "one.png", 32, 24, 3);
        let engine = FileEngine::open(&dir).unwrap();
        assert!(engine.import_job_snapshot().scope(JobScope::Import).is_none());

        engine.rescan().unwrap();
        engine.rescan().unwrap();
        let snapshot = engine.import_job_snapshot();
        let import = snapshot.scope(JobScope::Import).expect("rescan scope");
        assert_eq!(import.succeeded, 2, "one Succeeded record per rescan");
        assert_eq!(import.live(), 0);
        assert_eq!(engine.images().len(), 1);

        std::fs::remove_dir_all(&dir).ok();
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
        let dir = std::env::temp_dir().join(format!("portrait-cancel-job-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let jid = engine.export_jobs.submit(JobScope::Export, CancelToken::new());
        engine.export_job_ids.lock().insert(77, jid);
        assert!(engine.cancel_job(77));
        assert_eq!(engine.job_status(77), JobStatus::Failed { message: "cancelled".into() });
        assert!(!engine.cancel_job(77));
        assert_eq!(engine.source_status(ImageId(1)), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_imported_style_appears_once_in_styles() {
        let dir = std::env::temp_dir().join(format!("portrait-import-style-{}", std::process::id()));
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
    #[ignore = "requires an isolated directory with a real RAW above 4096 px"]
    fn native_raw_preview_real_fixture() {
        let dir = PathBuf::from(std::env::var_os("AURORA_NATIVE_RAW_TEST_DIR").expect("fixture directory"));
        let engine = FileEngine::open(&dir).unwrap();
        let info = engine.images().into_iter().find(|info| crate::raw::is_raw(&info.path)).expect("RAW fixture");
        let adj = AdjustmentSet::default();
        let roi = NormRect { x: 0.25, y: 0.25, width: 0.125, height: 0.125 };
        assert!(engine.native_raw_source(info.id, Some((&adj.geometry, NormRect::FULL, 1024))).is_none());
        let native = engine.native_raw_source(info.id, Some((&adj.geometry, roi, 1024))).expect("native RAW decode");
        assert!(native.size().0.max(native.size().1) > SOURCE_LONG_SIDE);
        #[cfg(all(windows, feature = "z6iii-raw-export"))]
        if crate::z6iii::active_size(&info.path).is_some() {
            assert_eq!((info.width, info.height), native.size(), "viewer and decoded RAW dimensions differ");
        }
        let proxy = engine.source(info.id).expect("proxy RAW decode");
        assert_eq!(proxy.size().0.max(proxy.size().1), SOURCE_LONG_SIDE);
        assert_eq!(engine.native_raw_source(info.id, Some((&adj.geometry, roi, 1536))).unwrap().snapshot_id(), native.snapshot_id());
        let render = |src: &Source| process::render(src, &adj, roi, 1024, true, None);
        let expected = render(&native);
        assert_ne!(expected.rgba, render(&proxy).rgba, "fixture must distinguish native from proxy");
        let preview = engine.preview_frame(&PreviewRequest {
            image: info.id, adjustments: adj, region: roi, max_size: 1024, bypass: true,
            quality: crate::RenderQuality::Final, mask_overlay: None, generation: 1,
            display_profile: None, proof_profile: None, proof_paper_white: 1.0,
            proof_black_point: 0.0, rendering_intent: None, black_point_compensation: false,
        }, false).unwrap();
        assert_eq!((preview.image.width, preview.image.height), (expected.width, expected.height));
        assert!(preview.image.pixels.as_slice() == expected.rgba.as_slice(), "final viewport did not use the native RAW");
        eprintln!("native {:?}, proxy {:?}, viewport {}x{}", native.size(), proxy.size(), expected.width, expected.height);
    }

    #[test]
    #[ignore = "requires an isolated directory with a real RAW above 4096 px"]
    fn adjusted_raw_export_scales_the_native_source_once() {
        let dir = PathBuf::from(std::env::var_os("AURORA_NATIVE_RAW_TEST_DIR").expect("fixture directory"));
        let engine = FileEngine::open(&dir).unwrap();
        let info = engine.images().into_iter().find(|info| crate::raw::is_raw(&info.path)).expect("RAW fixture");
        let native = engine.native_raw_source(info.id, None).expect("native RAW decode");
        let size = native.size();
        assert!(size.0.max(size.1) > SOURCE_LONG_SIDE);
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = 0.25;
        engine.save_adjustments(info.id, &adj).unwrap();
        let destination = dir.join("adjusted-export-test");
        let recipe = crate::ProcessRecipe {
            format: crate::ExportFormat::Tiff,
            bit_depth: 16,
            scale: crate::ScaleMode::Percent,
            scale_value: (50, 0),
            destination: Some(destination),
            include_metadata: crate::MetadataPolicy::StripAll,
            ..crate::ProcessRecipe::default()
        };
        let expected = crate::recipe::output_size(&recipe, size);
        engine.start_export(ExportJob { job_id: 19, image: info.id, recipe }).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let output = loop {
            match engine.job_status(19) {
                JobStatus::Done { output } => break output,
                JobStatus::Failed { message } => panic!("{message}"),
                _ => {
                    assert!(std::time::Instant::now() < deadline, "adjusted RAW export timed out");
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
        };
        assert_eq!(image::image_dimensions(&output).unwrap(), expected);
        eprintln!("adjusted TIFF: {}x{} from native {:?}", expected.0, expected.1, size);
    }

    #[test]
    fn native_raw_preview_switches_using_visible_region_and_sensor_resolution() {
        let g = crate::Geometry::default();
        let sensor = (6064, 4040);
        let roi = NormRect { x: 0.25, y: 0.25, width: 0.25, height: 0.25 };
        assert!(!wants_native_raw(sensor, &g, NormRect::FULL, 1516));
        assert!(!wants_native_raw(sensor, &g, roi, 758));
        assert!(wants_native_raw(sensor, &g, roi, 1516));
        assert!(wants_native_raw(sensor, &g, roi, 3032));
        assert!(!wants_native_raw((4096, 2730), &g, roi, 3032));
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
        for ext in ["jpg", "png", "tif", "webp", "heic", "heif"] {
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
    fn opening_an_empty_folder_yields_no_images() {
        let dir = std::env::temp_dir().join(format!("portrait-empty-{}", std::process::id()));
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

        let dir = std::env::temp_dir().join(format!("portrait-maskcat-{}", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!("portrait-mixed-{}", std::process::id()));
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

        let dir = std::env::temp_dir().join(format!("portrait-mask-reopen-{}", std::process::id()));
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
        let plane = crate::aimask::lookup(&key).expect("preloaded from .portrait/masks");
        assert_eq!((plane.w, plane.h), (4, 3));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn trash_survives_reopening_the_folder() {
        let dir = std::env::temp_dir().join(format!("portrait-trash-test-{}", std::process::id()));
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
        assert!(reopened.images()[0].trashed, "trash is persisted in .portrait/catalog.json");
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

    #[test]
    fn batch_metadata_and_adjustments_commit_every_image() {
        use crate::ImageEngine;

        let dir = std::env::temp_dir().join(format!("portrait-batch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..3 {
            image::RgbaImage::from_pixel(8, 6, image::Rgba([i as u8 * 20, 40, 60, 255]))
                .save(dir.join(format!("frame-{i}.png")))
                .unwrap();
        }
        let engine = FileEngine::open(&dir).unwrap();
        let mut infos = engine.images();
        let mut invalid = infos.clone();
        invalid[0].rating = 2;
        invalid[1].id = ImageId(999_999);
        assert!(engine.update_info_batch(&invalid).is_err());
        assert_eq!(engine.images()[0].rating, infos[0].rating, "invalid batch wrote partially");

        for (i, info) in infos.iter_mut().enumerate() {
            info.rating = i as i8 + 2;
            info.color_tag = ColorTag::ALL[i + 1];
        }
        engine.update_info_batch(&infos).unwrap();
        let items: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                let mut adjustments = AdjustmentSet::default();
                adjustments.exposure.exposure = i as f32 + 0.25;
                (info.id, adjustments)
            })
            .collect();
        engine.save_adjustments_batch(&items).unwrap();

        let reopened = FileEngine::open(&dir).unwrap();
        for (i, info) in reopened.images().iter().enumerate() {
            assert_eq!(info.rating, i as i8 + 2);
            assert_eq!(info.color_tag, ColorTag::ALL[i + 1]);
            assert!(info.has_adjustments);
            assert!((reopened.load_adjustments(info.id).exposure.exposure - (i as f32 + 0.25)).abs() < 1e-6);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn save_adjustments_batch_keeps_last_duplicate_and_writes_nothing_on_unknown() {
        let dir = std::env::temp_dir().join(format!("portrait-batch-dup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..3 {
            image::RgbaImage::from_pixel(8, 6, image::Rgba([i as u8 * 20, 40, 60, 255]))
                .save(dir.join(format!("frame-{i}.png")))
                .unwrap();
        }
        let engine = FileEngine::open(&dir).unwrap();
        let infos = engine.images();

        let id = infos[0].id;
        let mut first = AdjustmentSet::default();
        first.exposure.exposure = 1.0;
        let mut last = AdjustmentSet::default();
        last.exposure.exposure = 3.0;
        engine.save_adjustments_batch(&[(id, first), (id, last)]).unwrap();
        assert_eq!(
            engine.load_adjustments(id).exposure.exposure,
            3.0,
            "the last duplicate in the batch wins"
        );

        let mut before = engine.load_adjustments(infos[1].id);
        before.exposure.exposure = 2.0;
        let poison = AdjustmentSet::default();
        let error = engine
            .save_adjustments_batch(&[(infos[1].id, before), (ImageId(99_999_999), poison)])
            .unwrap_err();
        assert_eq!(error.to_string(), "unknown image");
        assert_eq!(
            engine.load_adjustments(infos[1].id).exposure.exposure,
            0.0,
            "an invalid id rejects the whole batch before any entry is written"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[ignore = "profiles batch adjustment id/catalog lookup; run with --ignored --nocapture"]
    fn bench_batch_edit_index_equivalence() {
        use std::time::Instant;

        fn linear_positions(entries: &[FileEntry], ids: &[ImageId]) -> Vec<usize> {
            ids.iter()
                .map(|id| {
                    entries
                        .iter()
                        .position(|entry| entry.info.id == *id)
                        .expect("known id")
                })
                .collect()
        }

        fn p50(mut run: impl FnMut()) -> f64 {
            let mut samples: Vec<f64> = (0..5)
                .map(|_| {
                    let started = Instant::now();
                    run();
                    started.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        }

        fn synthetic_info(id: u64) -> ImageInfo {
            ImageInfo {
                id: ImageId(id),
                path: PathBuf::from(format!("IMG_{id}.jpg")),
                file_name: format!("IMG_{id}.jpg"),
                kind: FileKind::Jpeg,
                width: 16,
                height: 12,
                file_size: 1,
                exif: Exif::default(),
                iptc: Iptc::default(),
                keywords: Vec::new(),
                rating: 0,
                color_tag: ColorTag::default(),
                variant: 1,
                has_adjustments: false,
                trashed: false,
                heif: None,
            }
        }

        fn catalog_entry(path: String, rating: i8) -> Entry {
            Entry {
                rel_path: path,
                rating,
                color_tag: 0,
                keywords: Vec::new(),
                variant: 1,
                updated_at: 0,
                trashed: false,
            }
        }

        let entries: Vec<FileEntry> = (0..20_000)
            .map(|id| FileEntry {
                info: synthetic_info(id),
                path: PathBuf::from(format!("IMG_{id}.jpg")),
                rel_path: format!("IMG_{id}.jpg"),
                adjustments: AdjustmentSet::default(),
            })
            .collect();
        let mut catalog_seed: Vec<Entry> = (0..20_000)
            .map(|slot| catalog_entry(format!("IMG_{slot}.jpg"), (slot % 7) as i8))
            .collect();
        // A duplicated path keeps its first slot, the case the batch index
        // must reproduce from `Catalog::upsert`.
        catalog_seed[1] = catalog_entry("IMG_0.jpg".to_owned(), 6);

        for count in [1usize, 8, 100, 10_000] {
            // Scattered ids: a selection lands anywhere in the library, unlike
            // a prefix that would flatter the linear scan.
            let ids: Vec<ImageId> = (0..count as u64)
                .map(|i| ImageId((i * 7_919) % 20_000))
                .collect();
            let expected = linear_positions(&entries, &ids);
            assert_eq!(image_positions(&entries, &ids).unwrap(), expected);
            let lookup_old = p50(|| {
                std::hint::black_box(linear_positions(&entries, &ids));
            });
            let lookup_new = p50(|| {
                std::hint::black_box(image_positions(&entries, &ids).unwrap());
            });

            let updates: Vec<Entry> = (0..count as u64)
                .map(|i| match i % 3 {
                    0 => catalog_entry(format!("IMG_{i}.jpg"), -1),
                    1 => catalog_entry(format!("IMG_{}.jpg", 20_000 + i), -1),
                    _ => catalog_entry("IMG_2.jpg".to_owned(), -1),
                })
                .collect();
            let mut old_catalog = Catalog::default();
            old_catalog.images = catalog_seed.clone();
            for entry in updates.iter().cloned() {
                old_catalog.upsert(entry);
            }
            let mut new_catalog = Catalog::default();
            new_catalog.images = catalog_seed.clone();
            upsert_entries(&mut new_catalog, updates.clone());
            assert_eq!(new_catalog, old_catalog, "batch catalog write changed the result");

            let catalog_old = p50(|| {
                let mut catalog = Catalog::default();
                catalog.images = catalog_seed.clone();
                for entry in updates.iter().cloned() {
                    catalog.upsert(entry);
                }
                std::hint::black_box(catalog);
            });
            let catalog_new = p50(|| {
                let mut catalog = Catalog::default();
                catalog.images = catalog_seed.clone();
                upsert_entries(&mut catalog, updates.clone());
                std::hint::black_box(catalog);
            });
            println!(
                "BATCH_EDIT_INDEX count={count} lookup_old_p50={lookup_old:.3}ms lookup_new_p50={lookup_new:.3}ms \
                 catalog_old_p50={catalog_old:.3}ms catalog_new_p50={catalog_new:.3}ms"
            );
        }

        assert!(image_positions(&entries, &[ImageId(99_999_999)]).is_err());
        assert_eq!(
            image_positions(&entries, &[ImageId(3), ImageId(3)]).unwrap(),
            vec![3, 3]
        );
        let mut duplicate_entries = vec![entries[5].clone(), entries[5].clone()];
        duplicate_entries[1].rel_path = "IMG_5_dup.jpg".to_owned();
        assert_eq!(
            image_positions(&duplicate_entries, &[ImageId(5)]).unwrap(),
            vec![0],
            "a duplicated id resolves to its first entry"
        );
    }

    #[test]
    fn ai_input_cache_reuses_source_and_invalidates_on_geometry() {
        let dir = std::env::temp_dir().join(format!("portrait-ai-input-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::from_pixel(32, 24, image::Rgba([20, 40, 60, 255]))
            .save(dir.join("frame.png"))
            .unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        let first = engine.ai_input(id).unwrap();
        let second = engine.ai_input(id).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!((first.width, first.height), (1024, 768));

        let mut changed = engine.load_adjustments(id);
        changed.geometry.orientation = 1;
        engine.save_adjustments(id, &changed).unwrap();
        let rotated = engine.ai_input(id).unwrap();
        assert!(!Arc::ptr_eq(&first, &rotated));
        assert_eq!((rotated.width, rotated.height), (768, 1024));
        assert!(Arc::ptr_eq(&rotated, &engine.ai_input(id).unwrap()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Copies the C1 fixture into a temp folder so the store lands there too.
    fn c1_copy(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("portrait-c1-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-sample.cocatalogdb");
        std::fs::copy(source, dir.join("Capture One.cocatalogdb")).unwrap();
        dir
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn c1_catalog_imports_every_variant_once() {
        let dir = c1_copy("import");
        let engine = FileEngine::open_c1_catalog(&dir).unwrap();
        let images = engine.images();
        assert_eq!(images.len(), 4, "3 images, 4 variants");
        assert_eq!(images.iter().filter(|i| i.file_name == "_DSC5673.NEF").count(), 2);

        let again = FileEngine::open_c1_catalog(&dir).unwrap();
        assert_eq!(again.images().len(), 4, "reopening must not duplicate entries");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn c1_import_keeps_adjustments_and_metadata() {
        let dir = c1_copy("meta");
        let engine = FileEngine::open_c1_catalog(&dir).unwrap();
        let first = &engine.images()[0];
        assert!(first.has_adjustments);
        assert_eq!(first.rating, 5);
        let adj = engine.load_adjustments(first.id);
        // C1's own IC_RGBToKelvinTint reports 4779.55 K for these Z6III gains.
        assert!((adj.white_balance.kelvin - 4779.55).abs() < 20.0);
        assert!(adj.geometry.crop.width > 0.0 && adj.geometry.crop.width <= 1.0);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Opening a C1 catalog must publish C1's own albums, not the folder tree.
    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn c1_collections_replace_the_folder_tree() {
        let dir = c1_copy("collections");
        let engine = FileEngine::open_c1_catalog(&dir).unwrap();
        let collections = engine.collections();
        assert!(!collections.is_empty(), "the fixture stores collections");
        assert!(
            collections.iter().all(|c| c.id >= crate::c1import::C1_COLLECTION_ID_BASE),
            "ids must come from C1, got {:?}",
            collections.iter().map(|c| c.id).collect::<Vec<_>>()
        );
        fn walk(list: &[Collection], out: &mut Vec<u64>) {
            for c in list {
                out.push(c.id);
                walk(&c.children, out);
            }
        }
        let mut ids = Vec::new();
        walk(&collections, &mut ids);
        for id in ids {
            assert!(
                engine.collection_images(id).is_some(),
                "C1 collection {id} must resolve to a member list"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// C1 local layers are attached only when their `.comask` mask can be decoded;
    /// the fixture has no sidecar, so it must stay unattached and safe.
    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn c1_layers_without_a_mask_stay_out_of_the_render() {
        let dir = c1_copy("layers");
        let engine = FileEngine::open_c1_catalog(&dir).unwrap();
        let with_layers = engine
            .images()
            .into_iter()
            .find(|info| !engine.layer_summaries(info.id).is_empty())
            .expect("the fixture keeps local layers on one variant");
        let layers = engine.layer_summaries(with_layers.id);
        assert!(!layers.is_empty());
        assert!(layers.iter().all(|layer| layer.mask_uuid.is_some()));
        assert!(
            engine.load_adjustments(with_layers.id).layers.is_empty(),
            "no layer may render without its mask"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn c1_import_saves_to_portrait_not_the_catalog() {
        let dir = c1_copy("store");
        let catalog_file = dir.join("Capture One.cocatalogdb");
        let before = std::fs::metadata(&catalog_file).unwrap().len();
        let engine = FileEngine::open_c1_catalog(&dir).unwrap();
        assert_eq!(std::fs::metadata(&catalog_file).unwrap().len(), before);
        assert!(!dir.join(".portrait/catalog.json").exists(), "opening is not an edit");
        assert_eq!(engine.images().len(), 4, "the catalog is materialized in memory");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A real C1 catalog is a package folder; our store must stay outside it.
    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn c1_bundle_leaves_no_portrait_inside_the_package() {
        let dir = std::env::temp_dir().join(format!("portrait-c1-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let package = dir.join("Capture One.cocatalogdb");
        let inner = package.join("Capture One.cocatalogdb");
        std::fs::create_dir_all(&package).unwrap();
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-sample.cocatalogdb");
        std::fs::copy(fixture, &inner).unwrap();
        let engine = FileEngine::open_c1_catalog(&inner).unwrap();
        assert_eq!(engine.images().len(), 4);
        assert!(!package.join(".portrait").exists(), "no sidecar inside the C1 package");
        assert!(!dir.join(".portrait/catalog.json").exists(), "opening does not rewrite the index");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn c1_import_keeps_portrait_edits_on_reopen() {
        let dir = c1_copy("reopen");
        {
            let engine = FileEngine::open_c1_catalog(&dir).unwrap();
            let id = engine.images()[0].id;
            let mut adj = engine.load_adjustments(id);
            adj.exposure.exposure = 1.5;
            engine.save_adjustments(id, &adj).unwrap();
        }
        let reopened = FileEngine::open_c1_catalog(&dir).unwrap();
        let same = reopened.images().into_iter().find(|i| i.variant == 1).unwrap();
        assert!((reopened.load_adjustments(same.id).exposure.exposure - 1.5).abs() < 1e-6);
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
        let dir = std::env::temp_dir().join(format!("portrait-icc-{}-{name}", std::process::id()));
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

    #[test]
    fn exif_lens_name_enables_only_a_database_match() {
        let lenses = crate::lensdb::parse_lenses(
            r#"<lenses>
  <lens ID="Z2470" type="Zoom">
    <fullName>Nikon NIKKOR Z 24-70mm f/4 S</fullName>
    <shortName>NIKKOR Z 24-70mm f/4 S</shortName>
    <brand>Nikon</brand>
    <focalLength dependent="true" min="24.0" max="70.0"/>
  </lens>
</lenses>"#,
        );

        let mut known = crate::LensCorrection::default();
        let exif = Exif {
            lens: Some("NIKKOR Z 24-70mm f/4 S".into()),
            focal_length_mm: Some(70.0),
            ..Default::default()
        };
        apply_auto_lens(&mut known, &lenses, &exif);
        assert_eq!(known.profile.as_deref(), Some("Nikon NIKKOR Z 24-70mm f/4 S"));
        assert!(known.use_profile, "a database hit turns the calibrated warp on");

        let mut unknown = crate::LensCorrection::default();
        let exif = Exif {
            lens: Some("NIKKOR Z 24-120mm f/4 S".into()),
            focal_length_mm: Some(120.0),
            ..Default::default()
        };
        apply_auto_lens(&mut unknown, &lenses, &exif);
        assert_eq!(unknown.profile, None);
        assert!(!unknown.use_profile, "an unknown lens must stay off");

        let mut missing = crate::LensCorrection::default();
        apply_auto_lens(&mut missing, &lenses, &Exif::default());
        assert_eq!(missing.profile, None);
        assert!(!missing.use_profile);

        let mut manual = crate::LensCorrection { profile: Some("Manual".into()), ..Default::default() };
        apply_auto_lens(&mut manual, &lenses, &Exif { lens: Some("NIKKOR Z 24-70mm f/4 S".into()), focal_length_mm: Some(50.0), ..Default::default() });
        assert_eq!(manual.profile.as_deref(), Some("Manual"));
        assert!(!manual.use_profile, "a user-picked profile is not replaced");
    }

    #[test]
    fn merge_default_options_match_the_old_methods_byte_for_byte() {
        let dir = std::env::temp_dir().join(format!("portrait-mergeopts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame.tif");
        let pixels: image::ImageBuffer<image::Rgb<u16>, Vec<u16>> = image::ImageBuffer::from_fn(
            16,
            8,
            |x, y| image::Rgb([(x * 1000) as u16, (y * 2000) as u16, 30_000]),
        );
        image::DynamicImage::ImageRgb16(pixels)
            .save_with_format(&path, image::ImageFormat::Tiff)
            .unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine
            .images()
            .into_iter()
            .find(|image| image.file_name == "frame.tif")
            .unwrap()
            .id;

        let hdr_old = FileEngine::merge_hdr_frames(&engine, &[id]).unwrap();
        let hdr_new = engine.merge_hdr_with_options(&[id], &HdrOptions::default()).unwrap();
        assert_ne!(hdr_old.path, hdr_new.path);
        assert_eq!(
            std::fs::read(&hdr_old.path).unwrap(),
            std::fs::read(&hdr_new.path).unwrap(),
            "default HDR options changed the merged TIFF"
        );

        let pano_old = FileEngine::merge_panorama_frames(&engine, &[id]).unwrap();
        let pano_new = engine
            .merge_panorama_with_options(&[id], &PanoramaOptions::default())
            .unwrap();
        assert_ne!(pano_old.path, pano_new.path);
        assert_eq!(
            std::fs::read(&pano_old.path).unwrap(),
            std::fs::read(&pano_new.path).unwrap(),
            "default panorama options changed the merged TIFF"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eip_packages_scan_and_materialize_their_raw_and_comask() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../../../04_reference/legacy-root/逆向工程/Capture One/resources/eip_samples/FULL.EIP",
        );
        if !fixture.is_file() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("portrait-eip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(&fixture, dir.join("FULL.EIP")).unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let image = engine
            .images()
            .into_iter()
            .find(|image| image.file_name == "FULL.EIP")
            .expect("an .eip package is a catalog image");
        assert_eq!(image.kind, FileKind::Raw);
        let assets = engine.eip_assets(image.id).expect("packed members materialize");
        assert_eq!(std::fs::metadata(&assets.raw).unwrap().len(), 128, "the RAW member, not the ZIP");
        assert_eq!(assets.raw.extension().and_then(|e| e.to_str()), Some("IIQ"));
        assert_eq!(std::fs::metadata(assets.settings.as_ref().unwrap()).unwrap().len(), 6);
        let comask = assets.comask.expect("FULL.EIP declares CoMaskPath");
        assert_eq!(std::fs::read(comask).unwrap().len(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn fixture_eip(members: &[(&str, &[u8])], deflated: bool) -> Vec<u8> {
        use std::io::Write;
        let mut archive = Vec::new();
        let mut directory = Vec::new();
        for (name, bytes) in members {
            let compressed;
            let payload = if deflated {
                let mut encoder = flate2::write::DeflateEncoder::new(
                    Vec::new(),
                    flate2::Compression::fast(),
                );
                encoder.write_all(bytes).unwrap();
                compressed = encoder.finish().unwrap();
                compressed.as_slice()
            } else {
                bytes
            };
            let offset = (archive.len() as u32).to_le_bytes();
            let size = (bytes.len() as u32).to_le_bytes();
            let name_len = (name.len() as u16).to_le_bytes();
            let crc = crc32fast::hash(bytes).to_le_bytes();
            let mut local = [0u8; 30];
            local[..4].copy_from_slice(b"PK\x03\x04");
            local[4] = 20;
            local[8] = if deflated { 8 } else { 0 };
            local[14..18].copy_from_slice(&crc);
            local[18..22].copy_from_slice(&(payload.len() as u32).to_le_bytes());
            local[22..26].copy_from_slice(&size);
            local[26..28].copy_from_slice(&name_len);
            archive.extend_from_slice(&local);
            archive.extend_from_slice(name.as_bytes());
            archive.extend_from_slice(payload);
            let mut central = [0u8; 46];
            central[..4].copy_from_slice(b"PK\x01\x02");
            central[4] = 20;
            central[6..8].copy_from_slice(&local[4..6]);
            central[10..12].copy_from_slice(&local[8..10]);
            central[16..30].copy_from_slice(&local[14..28]);
            central[42..46].copy_from_slice(&offset);
            directory.extend_from_slice(&central);
            directory.extend_from_slice(name.as_bytes());
        }
        let mut footer = [0u8; 22];
        let count = (members.len() as u16).to_le_bytes();
        footer[..4].copy_from_slice(b"PK\x05\x06");
        footer[8..10].copy_from_slice(&count);
        footer[10..12].copy_from_slice(&count);
        footer[12..16].copy_from_slice(&(directory.len() as u32).to_le_bytes());
        footer[16..20].copy_from_slice(&(archive.len() as u32).to_le_bytes());
        archive.extend_from_slice(&directory);
        archive.extend_from_slice(&footer);
        archive
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn eip_restores_packed_edits_on_import_and_reopen() {
        assert_packed_edits_on_import_and_reopen(false);
    }

    #[test]
    fn eip_restores_deflated_edits_on_import_and_reopen() {
        assert_packed_edits_on_import_and_reopen(true);
    }

    fn assert_packed_edits_on_import_and_reopen(deflated: bool) {
        let dir = std::env::temp_dir().join(format!(
            "aurora-eip-edits-{}-{deflated}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let path = dir.join("edited.eip");
        let manifest = br#"<Manifest><File><RawPath>photo.png</RawPath><SettingsPath>settings/inner.cos</SettingsPath><CoMaskPath>masks/another-name.comask</CoMaskPath></File></Manifest>"#;
        let settings = br#"<IMG><VAR><DL><E K="Exposure" V="1.25"/></DL><AL><E K="MaskUUID" V="24391591-4129-4c9f-a9f5-bb4dd4c41289"/><E K="Exposure" V="-0.75"/></AL></VAR><VAR><DL><E K="Exposure" V="-2.0"/></DL><AL><E K="MaskUUID" V="be040daa-3357-4998-b5d5-c3c2cae23759"/><E K="Exposure" V="0.5"/></AL></VAR></IMG>"#;
        let mask = crate::comask::gradient_container_with(&[
            ("24391591-4129-4c9f-a9f5-bb4dd4c41289", [0.0, 0.0, 7.0, 0.0, 0.5]),
            ("be040daa-3357-4998-b5d5-c3c2cae23759", [1.0, 0.0, 4.0, 0.0, 0.5]),
        ]);
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(8, 1).write_to(&mut png, image::ImageFormat::Png).unwrap();
        let bytes = fixture_eip(&[
            ("manifest.xml", manifest),
            ("photo.png", png.get_ref()),
            ("settings/inner.cos", settings),
            ("masks/another-name.comask", &mask),
        ], deflated);
        std::fs::write(&path, &bytes).unwrap();

        let assert_edits = |engine: &FileEngine, info: &ImageInfo| {
            assert!(info.has_adjustments);
            assert_eq!((info.width, info.height), (8, 1));
            let adjustments = engine.load_adjustments(info.id);
            assert_eq!(adjustments.exposure.exposure, 1.25);
            assert_eq!(adjustments.layers.len(), 1);
            let layer = &adjustments.layers[0];
            assert_eq!(layer.adjustments.exposure.exposure, -0.75);
            let [MaskShape::Linear(gradient)] = layer.mask.shapes.as_slice() else {
                panic!("packed layer did not resolve its mask UUID");
            };
            assert_eq!(gradient.start, [7.0 / 8.0, 0.0]);
            assert_eq!(gradient.end, [0.0, 0.0]);
        };
        let imported = engine.import(std::slice::from_ref(&path)).unwrap();
        assert_eq!(imported.len(), 2);
        assert_edits(&engine, &imported[0]);
        assert_eq!(imported[0].variant, 1);
        assert_eq!(imported[1].variant, 2);
        assert_ne!(imported[0].id, imported[1].id);
        assert_eq!(imported[0].path, imported[1].path);
        let assert_second = |engine: &FileEngine, info: &ImageInfo, exposure| {
            let adjustments = engine.load_adjustments(info.id);
            assert_eq!(adjustments.exposure.exposure, exposure);
            let [layer] = adjustments.layers.as_slice() else {
                panic!("expected exactly one layer on the second variant");
            };
            assert_eq!(layer.adjustments.exposure.exposure, 0.5);
            let [MaskShape::Linear(gradient)] = layer.mask.shapes.as_slice() else {
                panic!("second variant did not resolve its own mask UUID");
            };
            assert_eq!(gradient.start, [4.0 / 8.0, 0.0]);
            assert_eq!(gradient.end, [1.0 / 8.0, 0.0]);
        };
        assert_second(&engine, &imported[1], -2.0);
        let cache = dir.join(".portrait").join("eip-cache");
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 3);
        assert!(engine.import(std::slice::from_ref(&path)).unwrap().is_empty());
        let mut second_edit = engine.load_adjustments(imported[1].id);
        second_edit.exposure.exposure = -3.0;
        engine.save_adjustments(imported[1].id, &second_edit).unwrap();
        let mut rated = imported[0].clone();
        rated.rating = 3;
        engine.update_info(&rated).unwrap();
        drop(engine);

        let reopened = FileEngine::open(&dir).unwrap();
        let images = reopened.images();
        assert_eq!(images.len(), 2);
        let info = images[0].clone();
        assert_edits(&reopened, &info);
        assert_eq!(images[1].variant, 2);
        assert_second(&reopened, &images[1], -3.0);
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 3);
        assert_eq!(images[1].rating, 0);
        assert_eq!(info.rating, 3);
        let mut ours = reopened.load_adjustments(info.id);
        ours.exposure.exposure = 2.5;
        reopened.save_adjustments(info.id, &ours).unwrap();
        drop(reopened);
        let reopened = FileEngine::open(&dir).unwrap();
        assert_eq!(reopened.load_adjustments(reopened.images()[0].id).exposure.exposure, 2.5);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        drop(reopened);
        if !deflated {
            assert_eip_catalog_session_round_trip(&dir, &path, &bytes);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn assert_eip_catalog_session_round_trip(dir: &Path, package: &Path, bytes: &[u8]) {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-sample.cocatalogdb");
        let mut source = crate::c1import::C1Import::open(&fixture).unwrap();
        let image_pk = source.images().iter().find(|image| {
            source.variants_of(image.pk).len() == 2
        }).unwrap().pk;
        let source_image = source.catalog.images.iter_mut().find(|image| image.pk == image_pk).unwrap();
        source_image.file_name = "edited.eip".into();
        source_image.abs_path = package.to_string_lossy().into_owned();
        let source_variants: Vec<_> = source.variants_of(image_pk)
            .iter().map(|variant| (variant.pk, variant.index)).collect();

        let target = dir.join("Template.cosessiondb");
        let export = crate::c1session_export::export_session(&source, &target).unwrap();
        let saved_package = target.join("Capture/edited.eip");
        assert_eq!(std::fs::read(&saved_package).unwrap(), bytes);
        let reopened = crate::c1import::C1Import::open(&target).unwrap();
        let exported_image = reopened.images().iter().find(|image| image.pk == image_pk).unwrap();
        assert_eq!(exported_image.file_name, "edited.eip");
        assert!(exported_image.abs_path.ends_with("Capture/edited.eip")
            || exported_image.abs_path.ends_with("Capture\\edited.eip"));
        let exported_variants: Vec<_> = reopened.variants_of(image_pk)
            .iter().map(|variant| (variant.pk, variant.index)).collect();
        assert_eq!(exported_variants, source_variants);
        let database = rusqlite::Connection::open(&export.database).unwrap();
        let memberships: i64 = database.query_row(
            "SELECT COUNT(*) FROM ZIMAGEINCOLLECTION m
             JOIN ZCOLLECTION c ON c.Z_PK=m.ZCOLLECTION
             WHERE m.ZIMAGE=?1 AND c.Z_ENT=5 AND c.ZCOLLECTIONINDEX=5",
            [image_pk],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(memberships, 1);

        let capture = FileEngine::open(&target.join("Capture")).unwrap();
        let packed: Vec<_> = capture.images().into_iter()
            .filter(|image| image.file_name == "edited.eip")
            .collect();
        assert_eq!(packed.len(), 2);
        assert_ne!(packed[0].id, packed[1].id);
        assert_eq!(packed[0].path, packed[1].path);
        let first = capture.load_adjustments(packed[0].id);
        assert_eq!(first.exposure.exposure, 1.25);
        assert!(matches!(first.layers[0].mask.shapes.as_slice(), [MaskShape::Linear(_)]));
        assert_eq!(capture.load_adjustments(packed[1].id).exposure.exposure, -2.0);
    }

    // ---- BO-4: export writes atomically through a staging folder ------------

    struct ExportFixture {
        dir: PathBuf,
        engine: FileEngine,
        image: ImageId,
    }

    impl Drop for ExportFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn export_fixture(tag: &str) -> ExportFixture {
        let dir = std::env::temp_dir()
            .join(format!("aurora-export-bo4-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbImage::from_pixel(8, 6, image::Rgb([30, 60, 90]))
            .save(dir.join("shot.jpg"))
            .unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let image = engine.images()[0].id;
        ExportFixture { dir, engine, image }
    }

    fn wait_export(engine: &FileEngine, job_id: u64) -> JobStatus {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let status = engine.job_status(job_id);
            let is_terminal = engine
                .export_job_ids
                .lock()
                .get(&job_id)
                .copied()
                .and_then(|jid| engine.export_jobs.job(jid))
                .map(|record| record.state().is_terminal())
                .unwrap_or(true);
            match status {
                JobStatus::Queued | JobStatus::Running { .. } => {
                    assert!(std::time::Instant::now() < deadline, "export job timed out");
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                _ if !is_terminal => {
                    assert!(std::time::Instant::now() < deadline, "export job timed out waiting for terminal state");
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                status => return status,
            }
        }
    }

    fn jpeg_recipe(destination: PathBuf) -> crate::ProcessRecipe {
        crate::ProcessRecipe {
            destination: Some(destination),
            include_metadata: crate::MetadataPolicy::StripAll,
            ..crate::ProcessRecipe::default()
        }
    }

    fn staging_leftovers(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|path| crate::output::is_staging_path(path))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn export_publishes_through_a_staging_folder_and_leaves_no_temp() {
        let fixture = export_fixture("publish");
        let out = fixture.dir.join("Output");
        let recipe = jpeg_recipe(out.clone());
        fixture.engine.start_export(ExportJob { job_id: 41, image: fixture.image, recipe }).unwrap();
        let JobStatus::Done { output } = wait_export(&fixture.engine, 41) else {
            panic!("export did not finish: {:?}", fixture.engine.job_status(41));
        };
        assert!(output.is_file());
        assert_eq!(output.file_name().and_then(|n| n.to_str()), Some("shot.jpg"));
        assert!(staging_leftovers(&out).is_empty(), "the staging folder must be removed");
        assert!(image::image_dimensions(&output).is_ok(), "a complete JPEG was published");
    }

    #[test]
    fn standard_percent_export_scales_the_edited_frame_once() {
        let fixture = export_fixture("percent");
        let mut adjustments = fixture.engine.load_adjustments(fixture.image);
        for (job_id, crop, expected) in [
            (47, NormRect::FULL, (4, 3)),
            (48, NormRect { x: 0.125, y: 0.0, width: 0.75, height: 2.0 / 3.0 }, (3, 2)),
        ] {
            adjustments.geometry.crop = crop;
            fixture.engine.save_adjustments(fixture.image, &adjustments).unwrap();
            let recipe = crate::ProcessRecipe {
                scale: crate::ScaleMode::Percent,
                scale_value: (50, 0),
                ..jpeg_recipe(fixture.dir.join(format!("Output-{job_id}")))
            };
            fixture.engine.start_export(ExportJob { job_id, image: fixture.image, recipe }).unwrap();
            let status = wait_export(&fixture.engine, job_id);
            let JobStatus::Done { output } = status else { panic!("export failed: {status:?}") };
            assert_eq!(image::image_dimensions(output).unwrap(), expected);
        }
    }

    #[test]
    fn export_never_overwrites_an_existing_output() {
        let fixture = export_fixture("collision");
        let out = fixture.dir.join("Output");
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("shot.jpg"), b"previous output").unwrap();

        let recipe = jpeg_recipe(out.clone());
        fixture.engine.start_export(ExportJob { job_id: 42, image: fixture.image, recipe }).unwrap();
        let JobStatus::Done { output } = wait_export(&fixture.engine, 42) else {
            panic!("export did not finish: {:?}", fixture.engine.job_status(42));
        };

        assert_eq!(output.file_name().and_then(|n| n.to_str()), Some("shot-1.jpg"));
        assert_eq!(std::fs::read(out.join("shot.jpg")).unwrap(), b"previous output");
        assert!(output.is_file());
    }

    #[test]
    fn cancelling_at_the_publish_point_keeps_the_existing_output_bytes() {
        let fixture = export_fixture("cancel");
        let out = fixture.dir.join("Output");
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("shot.jpg"), b"keep these bytes").unwrap();

        let recipe = jpeg_recipe(out.clone());
        fixture.engine.start_export(ExportJob { job_id: 43, image: fixture.image, recipe }).unwrap();
        let cancelled = fixture.engine.cancel_job(43);

        // Whether the cancel won the race or the tiny render finished first,
        // the pre-existing target keeps its bytes.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !staging_leftovers(&out).is_empty() {
            assert!(std::time::Instant::now() < deadline, "cancelled export kept its staging folder");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(std::fs::read(out.join("shot.jpg")).unwrap(), b"keep these bytes");
        if cancelled {
            assert!(matches!(fixture.engine.job_status(43), JobStatus::Failed { .. }));
            assert!(
                std::fs::read_dir(&out).unwrap().flatten().count() == 1,
                "a cancelled export publishes nothing"
            );
        }
    }

    #[test]
    fn a_cancel_that_reports_true_never_leaves_a_published_file() {
        // The publish gate makes this a two-outcome guarantee: either
        // `cancel_job` wins and no file appears, or the publish wins and the
        // job is already `Done` when `cancel_job` runs (it returns false).
        for attempt in 0..8 {
            let fixture = export_fixture(&format!("cancel-gate-{attempt}"));
            let out = fixture.dir.join("Output");
            std::fs::create_dir_all(&out).unwrap();
            fixture
                .engine
                .start_export(ExportJob { job_id: 1, image: fixture.image, recipe: jpeg_recipe(out.clone()) })
                .unwrap();
            std::thread::sleep(std::time::Duration::from_micros(attempt as u64 * 400));
            let cancelled = fixture.engine.cancel_job(1);
            let status = wait_export(&fixture.engine, 1);
            let published = std::fs::read_dir(&out)
                .unwrap()
                .flatten()
                .filter(|entry| !crate::output::is_staging_path(&entry.path()))
                .count();

            if cancelled {
                assert!(matches!(status, JobStatus::Failed { .. }), "cancelled job must not be Done");
                assert_eq!(published, 0, "cancel_job == true must publish nothing (attempt {attempt})");
            } else {
                assert!(matches!(status, JobStatus::Done { .. }), "uncancelled job must finish");
                assert_eq!(published, 1, "a finished export publishes exactly one file");
            }
        }
    }

    #[test]
    fn export_job_snapshot_counts_the_export_scope() {
        let fixture = export_fixture("snapshot");
        let out = fixture.dir.join("Output");
        fixture
            .engine
            .start_export(ExportJob { job_id: 51, image: fixture.image, recipe: jpeg_recipe(out.clone()) })
            .unwrap();
        let JobStatus::Done { .. } = wait_export(&fixture.engine, 51) else {
            panic!("export did not finish: {:?}", fixture.engine.job_status(51));
        };
        let snapshot = fixture.engine.export_job_snapshot();
        let export = snapshot.scope(JobScope::Export).expect("export scope counts");
        assert_eq!(export.total, 1);
        assert_eq!(export.succeeded, 1);
        assert_eq!(export.live(), 0);
        assert_eq!(snapshot.terminal(), 1);
        assert_eq!(snapshot.live(), 0);
    }

    #[test]
    fn unknown_export_job_is_failed_not_queued() {
        let fixture = export_fixture("unknown");
        assert_eq!(
            fixture.engine.job_status(987_654),
            JobStatus::Failed { message: "unknown job".into() }
        );
        assert!(!fixture.engine.cancel_job(987_654));
        assert_eq!(fixture.engine.export_job_snapshot().total, 0);
    }

    #[test]
    fn cancelling_before_the_worker_starts_publishes_nothing() {
        let fixture = export_fixture("cancel-early");
        let out = fixture.dir.join("Output");
        std::fs::create_dir_all(&out).unwrap();

        // Hold the whole export budget so the worker cannot leave the
        // scheduler wait before `cancel_job` runs. The guard is dropped before
        // we wait on the job, so this cannot stall another export test.
        let hold = export_scheduler()
            .acquire_cancellable(usize::MAX, || false)
            .expect("the test acquires the idle export budget");
        fixture
            .engine
            .start_export(ExportJob { job_id: 52, image: fixture.image, recipe: jpeg_recipe(out.clone()) })
            .unwrap();
        assert!(fixture.engine.cancel_job(52));
        // `cancel_job` moves the registry to `Cancelling` synchronously, so the
        // old immediate-cancel contract holds before the worker runs.
        assert_eq!(
            fixture.engine.job_status(52),
            JobStatus::Failed { message: "cancelled".into() }
        );
        drop(hold);

        assert!(matches!(wait_export(&fixture.engine, 52), JobStatus::Failed { .. }));
        assert!(staging_leftovers(&out).is_empty(), "no staging folder survives");
        assert_eq!(
            std::fs::read_dir(&out).unwrap().flatten().count(),
            0,
            "a cancelled export publishes nothing"
        );
        // Exactly one terminal state; a second cancel is a no-op.
        assert!(!fixture.engine.cancel_job(52));
        let snapshot = fixture.engine.export_job_snapshot();
        assert_eq!(snapshot.terminal(), 1);
        assert_eq!(snapshot.scope(JobScope::Export).unwrap().cancelled, 1);
    }

    #[test]
    fn a_late_success_cannot_overwrite_a_cancelled_export() {
        let fixture = export_fixture("stale-success");
        let jid = fixture.engine.export_jobs.submit(JobScope::Export, CancelToken::new());
        fixture.engine.export_job_ids.lock().insert(53, jid);

        assert!(fixture.engine.cancel_job(53));
        // The worker observes the token at the publish point and settles the
        // job as `Cancelled`; a late `Succeeded` is then rejected by the
        // terminal registry state and the output map never grows.
        fixture.engine.export_jobs.transition(jid, JobState::Cancelled).unwrap();
        assert!(fixture.engine.export_jobs.transition(jid, JobState::Succeeded).is_err());
        assert_eq!(
            fixture.engine.job_status(53),
            JobStatus::Failed { message: "cancelled".into() },
            "a cancelled job is never reported as Done"
        );
        assert!(!fixture.engine.export_outputs.lock().contains_key(&53));
        let snapshot = fixture.engine.export_job_snapshot();
        assert_eq!(snapshot.terminal(), 1, "exactly one terminal state");
        assert_eq!(snapshot.scope(JobScope::Export).unwrap().cancelled, 1);
    }

    #[test]
    fn export_history_is_bounded_and_keeps_live_and_newest_outputs() {
        let fixture = export_fixture("bounded-history");
        let engine = &fixture.engine;
        let cap = 16usize;

        // A live job that pruning must never touch.
        let live_jid = engine.export_jobs.submit(JobScope::Export, CancelToken::new());
        engine.export_jobs.transition(live_jid, JobState::running(0.5)).unwrap();
        engine.export_job_ids.lock().insert(9_000, live_jid);

        // `cap` terminal jobs, oldest first, each with an output entry.
        let mut terminal_jids = Vec::new();
        for job_id in 0..cap as u64 {
            let jid = engine.export_jobs.submit(JobScope::Export, CancelToken::new());
            engine.export_jobs.transition(jid, JobState::running(1.0)).unwrap();
            engine.export_jobs.transition(jid, JobState::Succeeded).unwrap();
            engine.export_job_ids.lock().insert(job_id, jid);
            engine.export_outputs.lock().insert(job_id, PathBuf::from(format!("/out/{job_id}.jpg")));
            terminal_jids.push(jid);
        }
        // One newer terminal job: with the `cap` above, this overflows by one.
        let newest_jid = engine.export_jobs.submit(JobScope::Export, CancelToken::new());
        engine.export_jobs.transition(newest_jid, JobState::running(1.0)).unwrap();
        engine.export_jobs.transition(newest_jid, JobState::Succeeded).unwrap();
        engine.export_job_ids.lock().insert(9_999, newest_jid);
        engine.export_outputs.lock().insert(9_999, PathBuf::from("/out/newest.jpg"));

        assert_eq!(engine.exports().prune_export_history_for_test(cap), 1);

        // The live job and its status survive.
        assert_eq!(engine.job_status(9_000), JobStatus::Running { progress: 0.5 });
        assert!(engine.export_job_ids.lock().contains_key(&9_000));

        // The newest terminal output survives; the oldest one is dropped.
        assert_eq!(
            engine.job_status(9_999),
            JobStatus::Done { output: PathBuf::from("/out/newest.jpg") }
        );
        assert!(engine.export_outputs.lock().contains_key(&9_999));
        assert!(!engine.export_outputs.lock().contains_key(&0));
        assert_eq!(
            engine.job_status(0),
            JobStatus::Failed { message: "unknown job".into() },
            "a pruned id reports the same unknown-job answer as a never-submitted one"
        );
        assert!(!engine.export_job_ids.lock().contains_key(&0));

        // The rest of the retained window still reports Done.
        assert!(matches!(engine.job_status(1), JobStatus::Done { .. }));
        assert!(matches!(engine.job_status(cap as u64 - 1), JobStatus::Done { .. }));

        // Registry terminal records are bounded too, and the live one is kept.
        let snapshot = engine.export_job_snapshot();
        assert_eq!(snapshot.terminal(), cap, "terminal records bounded to the cap");
        assert_eq!(snapshot.live(), 1, "the live export is preserved");
        assert!(!engine.export_jobs.contains(terminal_jids[0]));
    }

    #[test]
    fn a_destination_that_is_a_file_is_refused_before_rendering() {
        let fixture = export_fixture("bad-destination");
        let not_a_folder = fixture.dir.join("not-a-folder.jpg");
        std::fs::write(&not_a_folder, b"x").unwrap();

        let recipe = jpeg_recipe(not_a_folder.clone());
        let error = fixture
            .engine
            .start_export(ExportJob { job_id: 44, image: fixture.image, recipe })
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<crate::output::OutputPathRefusal>(),
            Some(&crate::output::OutputPathRefusal::NotADirectory)
        );
        assert_eq!(std::fs::read(&not_a_folder).unwrap(), b"x");
    }

    #[test]
    fn export_log_lines_do_not_carry_the_photo_folder() {
        // The message the exporter logs goes through the sanitizer; the file
        // name stays, the folder never does.
        let message = format!(
            "export job 9 for shot.jpg failed: {}",
            crate::output::sanitize_log_text(
                r"copying C:\Users\alice\Pictures\Shoot\shot.jpg to O:\out\shot.jpg"
            )
        );
        assert_eq!(message, "export job 9 for shot.jpg failed: copying <path> to <path>");
        assert!(!message.contains("alice"));
        assert!(!message.contains(r"C:\"));
    }

    // ---- export source acquisition above SOURCE_LONG_SIDE ----
    //
    // A 4096-capped proxy upscaled to a larger output violates the repository
    // rule against upsampling. These tests pin the replacement: at or below the
    // cap the cached `source()` path is unchanged; above it the original file is
    // decoded once at full resolution (or the job fails) and the cache is left
    // alone.

    /// [`FULL_RES_EXPORT_DECODES`] is process-global; serialize the tests that
    /// read it so an exact before/after delta also holds when the suite runs
    /// with more than one test thread.
    static FULL_RES_EXPORT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn png_recipe(destination: PathBuf) -> crate::ProcessRecipe {
        crate::ProcessRecipe {
            format: crate::ExportFormat::Png,
            destination: Some(destination),
            include_metadata: crate::MetadataPolicy::StripAll,
            ..crate::ProcessRecipe::default()
        }
    }

    /// Flat RGBA background with a 1-pixel black/white checker patch. The flat
    /// field keeps a 13.5 MP PNG encode fast and small; the patch is the
    /// high-frequency feature an upscaled proxy cannot preserve.
    fn write_checker_png(path: &Path, width: u32, height: u32, patch: (u32, u32, u32)) {
        use image::ImageEncoder as _;
        let mut pixels =
            image::RgbaImage::from_pixel(width, height, image::Rgba([128, 128, 128, 255]));
        let (px, py, size) = patch;
        for y in py..(py + size).min(height) {
            for x in px..(px + size).min(width) {
                let value = if (x - px + y - py) % 2 == 0 { 0 } else { 255 };
                pixels.put_pixel(x, y, image::Rgba([value, value, value, 255]));
            }
        }
        let file = std::fs::File::create(path).unwrap();
        image::codecs::png::PngEncoder::new_with_quality(
            std::io::BufWriter::new(file),
            image::codecs::png::CompressionType::Fast,
            image::codecs::png::FilterType::NoFilter,
        )
        .write_image(pixels.as_raw(), width, height, image::ExtendedColorType::Rgba8)
        .unwrap();
    }

    fn export_png_fixture(tag: &str, width: u32, height: u32, patch: (u32, u32, u32)) -> ExportFixture {
        let dir = std::env::temp_dir()
            .join(format!("aurora-export-raster-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_checker_png(&dir.join("shot.png"), width, height, patch);
        let engine = FileEngine::open(&dir).unwrap();
        let image = engine
            .images()
            .into_iter()
            .find(|info| info.file_name == "shot.png")
            .expect("the PNG was scanned")
            .id;
        ExportFixture { dir, engine, image }
    }

    /// A real 1x1 PNG whose IHDR is rewritten to `width`x`height` (CRC fixed):
    /// scanning reads the advertised size, but the pixel data can only decode
    /// as 1x1, so the full-resolution decode fails.
    fn png_with_patched_dimensions(width: u32, height: u32) -> Vec<u8> {
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgba8(1, 1)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let mut bytes = encoded.into_inner();
        // PNG layout: 8-byte signature, then IHDR length(4) type(4) data(13)
        // CRC(4). Width/height are the first 8 bytes of the IHDR data.
        bytes[16..20].copy_from_slice(&width.to_be_bytes());
        bytes[20..24].copy_from_slice(&height.to_be_bytes());
        let crc = crc32fast::hash(&bytes[12..29]);
        bytes[29..33].copy_from_slice(&crc.to_be_bytes());
        bytes
    }

    #[test]
    fn export_above_the_proxy_decodes_the_original_at_full_resolution() {
    let _serial = full_res_export_test_lock();
        let _guard = FULL_RES_EXPORT_TEST_LOCK.lock();
        let fixture = export_png_fixture("full", 4500, 3000, (100, 100, 64));
        let before_misses = fixture.engine.source_cache_stats().misses;
        let before_decodes = full_res_export_decode_count();
        let out = fixture.dir.join("Output");
        fixture
            .engine
            .start_export(ExportJob { job_id: 71, image: fixture.image, recipe: png_recipe(out.clone()) })
            .unwrap();
        let JobStatus::Done { output } = wait_export(&fixture.engine, 71) else {
            panic!("export did not finish: {:?}", fixture.engine.job_status(71));
        };
        // The 4096 proxy would render 4500x2999 (4500 * 2730 / 4096); the full
        // original renders exactly 4500x3000.
        assert_eq!(image::image_dimensions(&output).unwrap(), (4500, 3000));
        assert_eq!(
            full_res_export_decode_count() - before_decodes,
            1,
            "exactly one full-resolution export decode"
        );
        assert_eq!(
            fixture.engine.source_cache_stats().misses,
            before_misses,
            "the full-resolution export decode must not touch the preview source cache"
        );

        let exported = image::open(&output).unwrap().to_rgba8();
        assert_eq!(exported.dimensions(), (4500, 3000));
        let row: Vec<u8> = (0..64).map(|i| exported.get_pixel(100 + i, 110).0[0]).collect();
        let dark = row.iter().filter(|value| **value < 60).count();
        let bright = row.iter().filter(|value| **value > 195).count();
        assert!(dark >= 20 && bright >= 20, "checker polarity lost: {row:?}");
        let max_step =
            row.windows(2).map(|pair| pair[0].abs_diff(pair[1])).max().unwrap_or(0);
        assert!(
            max_step > 120,
            "an upscaled proxy would blur the 1-pixel checker (max adjacent step {max_step})"
        );
    }

    #[test]
    fn export_at_or_below_the_proxy_keeps_the_cached_source_path() {
    let _serial = full_res_export_test_lock();
        let _guard = FULL_RES_EXPORT_TEST_LOCK.lock();
        let fixture = export_png_fixture("small", 800, 600, (50, 50, 32));
        let before_misses = fixture.engine.source_cache_stats().misses;
        let before_decodes = full_res_export_decode_count();
        let out = fixture.dir.join("Output");
        fixture
            .engine
            .start_export(ExportJob { job_id: 72, image: fixture.image, recipe: png_recipe(out.clone()) })
            .unwrap();
        let JobStatus::Done { output } = wait_export(&fixture.engine, 72) else {
            panic!("export did not finish: {:?}", fixture.engine.job_status(72));
        };
        assert_eq!(image::image_dimensions(&output).unwrap(), (800, 600));
        assert_eq!(
            full_res_export_decode_count(),
            before_decodes,
            "a source at or below SOURCE_LONG_SIDE must keep the existing preview decode"
        );
        assert_eq!(
            fixture.engine.source_cache_stats().misses,
            before_misses + 1,
            "the existing source() path decoded the preview exactly once"
        );
    }

    #[test]
    fn full_resolution_decode_failure_fails_the_job_and_publishes_nothing() {
    let _serial = full_res_export_test_lock();
        // This also increments the process-global full-res counter (the decode
        // is attempted), so share the lock with the delta assertions above.
        let _guard = FULL_RES_EXPORT_TEST_LOCK.lock();
        let dir = std::env::temp_dir()
            .join(format!("aurora-export-raster-broken-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shot.png"), png_with_patched_dimensions(4500, 3000)).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let info = engine
            .images()
            .into_iter()
            .find(|info| info.file_name == "shot.png")
            .expect("the header-only PNG was scanned");
        assert_eq!((info.width, info.height), (4500, 3000));

        let out = dir.join("Output");
        // The full-resolution decode is deferred to the export worker, so
        // `start_export` only queues the job; the typed failure arrives through
        // `job_status`.
        engine
            .start_export(ExportJob { job_id: 73, image: info.id, recipe: png_recipe(out.clone()) })
            .expect("the job is queued");
        let JobStatus::Failed { message } = wait_export(&engine, 73) else {
            panic!("a failed full decode must fail the job: {:?}", engine.job_status(73));
        };
        assert!(message.contains("shot.png"), "the error names the file: {message}");
        assert!(message.contains("full resolution"), "the error says what failed: {message}");
        let published: Vec<PathBuf> = std::fs::read_dir(&out)
            .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
            .unwrap_or_default();
        assert!(published.is_empty(), "a failed full decode publishes nothing: {published:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn full_res_export_memory_guard_refuses_only_above_the_cap() {
        let cap = 2 * 1024 * 1024 * 1024u64;
        // 4500x3000 (~13.5 MP) is a normal export and stays well below 2 GiB.
        assert!(full_res_export_decode_fits(4500, 3000, cap).is_ok());
        // 45 MP too.
        assert!(full_res_export_decode_fits(8000, 6000, cap).is_ok());
        // 100 MP exceeds the unknown-memory fallback cap and is refused with the
        // estimate so the caller can report it.
        let refused = full_res_export_decode_fits(10000, 10000, cap).unwrap_err();
        assert_eq!(refused, full_res_export_estimate_bytes(10000, 10000));
        assert!(refused > cap);
        // The derived cap never undercuts the estimate formula for a small cap.
        assert_eq!(full_res_export_estimate_bytes(100, 50), 100 * 50 * FULL_RES_EXPORT_BYTES_PER_PX);
        assert!(full_res_export_decode_fits(0, 0, 0).is_ok(), "an empty frame is not refused");
    }

    // ---- S2: decoded-source byte budget + per-image single-flight ----

    fn s2_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aurora-s2-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_test_png(dir: &Path, name: &str, width: u32, height: u32, seed: u8) -> PathBuf {
        let path = dir.join(name);
        let image = image::RgbaImage::from_fn(width, height, |x, y| {
            image::Rgba([(x as u8).wrapping_add(seed), (y as u8).wrapping_mul(3), seed, 255])
        });
        image.save(&path).unwrap();
        path
    }

    fn pixel_hash(rgba: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in rgba {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    #[test]
    fn source_cache_bytes_count_every_mip_level() {
        let rgba = vec![128u8; 1024 * 768 * 4];
        let source = Source::from_rgba(1024, 768, &rgba);
        let expected: usize = source
            .levels
            .iter()
            .map(|level| level.data.capacity() * std::mem::size_of::<[f32; 3]>())
            .sum();
        assert_eq!(source_cache_bytes(&source), expected);
        assert!(source.levels.len() > 1, "mip pyramid missing");
        assert!(source_cache_bytes(&source) >= 1024 * 768 * std::mem::size_of::<[f32; 3]>());
    }

    #[test]
    fn native_lazy_source_publishes_the_reserved_mip_tail() {
        let rgba = vec![32000u16; 1024 * 768 * 4];
        let source = Source::from_rgba16_native(1024, 768, &rgba);
        let level0 = source.levels[0].data.capacity() * std::mem::size_of::<[f32; 3]>();
        let tail: usize = source.levels[1..]
            .iter()
            .map(|level| {
                level.width as usize * level.height as usize * std::mem::size_of::<[f32; 3]>()
            })
            .sum();
        assert_eq!(
            source_cache_bytes(&source),
            level0 + tail,
            "a native slot must reserve the full tail at publication"
        );
        // Sampling level 0 only must not materialise the tail.
        let _ = crate::render(
            &source,
            &AdjustmentSet::default(),
            NormRect::FULL,
            1024,
            false,
            None,
        );
        assert!(
            source.levels[1..].iter().all(|level| level.data.is_empty()),
            "a 1:1 render must not build the deferred tail"
        );
        assert_eq!(source_cache_bytes(&source), level0 + tail);
        // Zoom out: the tail materialises and accounting follows its capacity.
        let _ = crate::render(
            &source,
            &AdjustmentSet::default(),
            NormRect::FULL,
            256,
            false,
            None,
        );
        let materialized: usize = source
            .levels
            .iter()
            .map(|level| level.allocated_bytes())
            .sum();
        assert_eq!(source_cache_bytes(&source), materialized);
        assert!(source.levels[1..].iter().all(|level| level.is_materialized()));
    }

    #[test]
    fn source_cache_budget_follows_installed_memory() {
        let mib = 1024 * 1024usize;
        let small = source_cache_budget_from_memory(16 * 1024 * mib as u64, 8 * 1024 * mib as u64);
        assert!((1024 * mib..=1536 * mib).contains(&small), "16 GiB → {small}");
        let large = source_cache_budget_from_memory(32 * 1024 * mib as u64, 20 * 1024 * mib as u64);
        assert!((2048 * mib..=3072 * mib).contains(&large), "32 GiB → {large}");
        assert_eq!(
            source_cache_budget_from_memory(2 * 1024 * mib as u64, 1024 * mib as u64),
            SOURCE_CACHE_MIN_BYTES
        );
        assert_eq!(
            source_cache_budget_from_memory(128 * 1024 * mib as u64, 100 * 1024 * mib as u64),
            SOURCE_CACHE_MAX_BYTES
        );
        if std::env::var_os("AURORA_SOURCE_CACHE_BYTES").is_none()
            && std::env::var_os("AURORA_SOURCE_CACHE_MB").is_none()
        {
            let default = default_source_cache_bytes();
            assert!((SOURCE_CACHE_MIN_BYTES..=SOURCE_CACHE_MAX_BYTES).contains(&default), "{default}");
        }
    }

    fn export_budget_mib(total_mib: u64, available_mib: u64) -> usize {
        export_memory_budget_from_memory(total_mib * 1024 * 1024, available_mib * 1024 * 1024) / (1024 * 1024)
    }

    /// Admits `charge` on a worker thread and reports whether it started within
    /// 150 ms; a refusal is cancelled so the probe cannot hang the test.
    fn admitted_while_busy(scheduler: &Arc<ExportScheduler>, charge: usize) -> bool {
        use std::sync::atomic::AtomicBool;
        let admitted = Arc::new(AtomicBool::new(false));
        let cancel = Arc::new(AtomicBool::new(false));
        let worker = {
            let scheduler = Arc::clone(scheduler);
            let admitted = Arc::clone(&admitted);
            let cancel = Arc::clone(&cancel);
            std::thread::spawn(move || {
                if scheduler
                    .acquire_cancellable(charge, || cancel.load(Ordering::SeqCst))
                    .is_some()
                {
                    admitted.store(true, Ordering::SeqCst);
                }
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(150));
        let started = admitted.load(Ordering::SeqCst);
        cancel.store(true, Ordering::SeqCst);
        scheduler.notify_all();
        worker.join().unwrap();
        started
    }

    #[test]
    fn export_budget_scales_from_installed_memory_and_stays_in_bounds() {
        let mib = 1024 * 1024usize;
        let ceiling = export_budget_ceiling();
        assert_eq!(ceiling, 3_686_400_000, "2.4 standard-job charges: {ceiling}");
        assert_eq!(
            export_memory_budget_from_memory(32 * 1024 * mib as u64, 24 * 1024 * mib as u64),
            ceiling,
            "32 GiB must reach the derived ceiling"
        );
        assert_eq!(export_budget_mib(16 * 1024, 16 * 1024), 2048, "16 GiB holds one standard job, not two");
        assert_eq!(export_budget_mib(8 * 1024, 8 * 1024), 1024, "an 8 GiB machine keeps one job in flight");
        assert_eq!(
            export_memory_budget_from_memory(128 * 1024 * mib as u64, 120 * 1024 * mib as u64),
            ceiling,
            "larger machines keep the derived ceiling"
        );
        assert_eq!(export_budget_mib(32 * 1024, 5 * 1024), 512, "pressure collapses the budget to the floor");
        assert_eq!(export_budget_mib(32 * 1024, 6 * 1024), 512, "the reserve exactly offsets a 6 GiB free reading");
        assert_eq!(export_budget_mib(32 * 1024, 7 * 1024), 1024, "just above the reserve leaves 1 GiB");
    }

    #[test]
    fn export_budget_admits_two_24mp_jobs_and_not_three() {
        use std::sync::atomic::AtomicBool;
        let mib = 1024 * 1024usize;
        let budget = export_memory_budget_from_memory(32 * 1024 * mib as u64, 24 * 1024 * mib as u64);
        let scheduler = Arc::new(ExportScheduler::with_limits(4, budget));
        let full = (6048u32, 4032u32);
        let charge = estimated_decode_bytes(full.0, full.1, DecodeFormat::RawNative)
            + estimated_export_bytes(full.0, full.1, 16);
        let first = scheduler.acquire_cancellable(charge, || false);
        assert!(first.is_some(), "the first standard job must always be admitted");
        let hold = Arc::new(AtomicBool::new(true));
        let worker = {
            let scheduler = Arc::clone(&scheduler);
            let hold = Arc::clone(&hold);
            std::thread::spawn(move || {
                let second = scheduler.acquire_cancellable(charge, || false);
                assert!(second.is_some(), "the second standard job must be admitted");
                while hold.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert_eq!(scheduler.stats().0, 2, "exactly two 24 MP jobs are active");
        assert!(!admitted_while_busy(&scheduler, charge), "a third standard job must wait: {charge} * 3 > {budget}");
        hold.store(false, Ordering::SeqCst);
        worker.join().unwrap();
        drop(first);
        scheduler.notify_all();
        let (active, used, _) = scheduler.stats();
        assert_eq!((active, used), (0, 0), "both permits were released");
    }

    #[test]
    fn export_budget_binds_two_45mp_jobs() {
        let mib = 1024 * 1024usize;
        let budget = export_memory_budget_from_memory(32 * 1024 * mib as u64, 24 * 1024 * mib as u64);
        let scheduler = Arc::new(ExportScheduler::with_limits(4, budget));
        let full = (8256u32, 5504u32);
        let charge = estimated_decode_bytes(full.0, full.1, DecodeFormat::RawNative)
            + estimated_export_bytes(full.0, full.1, 16);
        let first = scheduler.acquire_cancellable(charge, || false);
        assert!(first.is_some(), "a 45 MP job alone must still be admitted");
        assert!(!admitted_while_busy(&scheduler, charge), "two 45 MP jobs must wait: {charge} * 2 > {budget}");
    }

    #[test]
    fn export_budget_never_refuses_a_single_over_budget_job() {
        let mib = 1024 * 1024usize;
        let budget = export_memory_budget_from_memory(32 * mib as u64, 5 * mib as u64);
        assert_eq!(budget, 512 * mib, "the pressure floor is the budget here");
        let scheduler = Arc::new(ExportScheduler::with_limits(4, budget));
        let permit = scheduler
            .acquire_cancellable(4096 * mib, || false)
            .expect("an over-budget job runs alone on an idle scheduler");
        assert_eq!(scheduler.stats().0, 1);
        let (second, third) = (
            admitted_while_busy(&scheduler, mib),
            admitted_while_busy(&scheduler, mib),
        );
        assert!(!second, "the floor budget must not pair anything with the big job");
        assert!(!third);
        drop(permit);
        assert_eq!(scheduler.stats().0, 0);
        let _ = scheduler.acquire_cancellable(4096 * mib, || false).expect("the door reopens after release");
    }

    // ---- S2: decoded-source byte budget + per-image single-flight ----
    #[test]
    fn decode_gate_admits_preview_before_thumbnail() {
        let gate = Arc::new(DecodeGate::new(1));
        let first = gate.acquire(DecodePriority::Interactive, 1);
        let order = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(std::sync::Barrier::new(3));
        std::thread::scope(|scope| {
            for priority in [DecodePriority::Thumbnail, DecodePriority::Interactive] {
                let gate = Arc::clone(&gate);
                let order = Arc::clone(&order);
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    let permit = gate.acquire(priority, 1);
                    order.lock().push(priority);
                    drop(permit);
                });
            }
            barrier.wait();
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(first);
        });
        assert_eq!(*order.lock(), vec![DecodePriority::Interactive, DecodePriority::Thumbnail]);
    }

    #[test]
    fn decode_units_asks_are_clamped_by_the_gate() {
        assert_eq!(decode_units(1), 1);
        assert_eq!(decode_units(DECODE_UNIT_BYTES * 3), 3);
        // The caller's ask is not pre-capped: a 99-unit frame on a 2-unit gate
        // still asks for 99 and the gate charges only what it has.
        assert_eq!(decode_units(DECODE_UNIT_BYTES * 99), 99);
        let gate = DecodeGate::new(2);
        let permit = gate.acquire(DecodePriority::Interactive, decode_units(DECODE_UNIT_BYTES * 99));
        assert_eq!(permit.units, 2);
        drop(permit);
        assert_eq!(gate_state(&gate).occupied, 0);
        // A 45MP RAW proxy and a 45MP native view reserve different shares.
        assert!(
            estimated_decode_bytes(8256, 5504, DecodeFormat::RawNative)
                > estimated_decode_bytes(8256, 5504, DecodeFormat::RawProxy)
        );
    }

    #[test]
    fn decode_estimate_uses_source_resolution_for_transient_work() {
        // 45MP RAW proxy: the retained proxy is 4096-long-side, but the decoder
        // still touches every source pixel, so the estimate must not collapse
        // to the ~11MP proxy size (the old bug underestimated ~3x).
        let (w, h) = (8256u32, 5504u32);
        let source_px = w as usize * h as usize;
        let proxy = estimated_decode_bytes(w, h, DecodeFormat::RawProxy);
        let native = estimated_decode_bytes(w, h, DecodeFormat::RawNative);
        let raster = estimated_decode_bytes(w, h, DecodeFormat::Raster);
        assert!(proxy >= source_px * RAW_DECODE_TRANSIENT_BYTES_PER_PX, "{proxy}");
        assert!(native > proxy, "native {native} should exceed proxy {proxy}");
        assert!(raster > proxy, "raster {raster} should exceed proxy {proxy}");
        // The retained mip is sized from the 4096 proxy, not the source.
        let kept = fit_long_side((w, h));
        let kept_px = kept.0 as usize * kept.1 as usize;
        assert!(kept_px < source_px / 3, "proxy should retain far fewer pixels");
        assert_eq!(proxy, kept_px * RETAINED_BYTES_PER_PX * 4 / 3 + source_px * RAW_DECODE_TRANSIENT_BYTES_PER_PX);
        assert_eq!(native, source_px * RETAINED_BYTES_PER_PX * 4 / 3 + source_px * RAW_DECODE_TRANSIENT_BYTES_PER_PX);
        // 100MP raster: full DynamicImage + RGBA copy + downscale frame.
        let (w, h) = (12288u32, 8192u32);
        let source_px = w as usize * h as usize;
        let raster = estimated_decode_bytes(w, h, DecodeFormat::Raster);
        assert!(raster >= source_px * 14, "{raster} vs {}", source_px * 14);
    }

    #[test]
    fn observed_decode_estimate_counts_file_bytes_and_backend_buffers() {
        let (w, h) = (6048u32, 4032u32);
        let px = w as usize * h as usize;
        let file_bytes = 50_000_000usize;
        let route = |c1_raw_enabled: bool, z6iii_nef: bool| crate::raw::DecodeRoute {
            pipeline_fingerprint: 0,
            c1_raw_enabled,
            z6iii_nef,
            z6iii_feature: true,
            libraw_available: true,
            opencl: crate::raw::OpenClRouteState::NotCompiled,
            demosaic: 3,
        };
        let libraw = route(false, false);
        let generic = route(true, false);
        let z6iii = route(true, true);
        let kept = fit_long_side((w, h));
        let retained = kept.0 as usize * kept.1 as usize * RETAINED_BYTES_PER_PX * 4 / 3;

        let libraw_bytes =
            estimated_decode_bytes_observed(w, h, DecodeFormat::RawProxy, file_bytes, &libraw);
        assert_eq!(libraw_bytes, file_bytes + retained + px * 20);
        let generic_bytes =
            estimated_decode_bytes_observed(w, h, DecodeFormat::RawProxy, file_bytes, &generic);
        assert_eq!(generic_bytes, file_bytes + retained + px * 46);
        let z6iii_bytes =
            estimated_decode_bytes_observed(w, h, DecodeFormat::RawProxy, file_bytes, &z6iii);
        assert_eq!(z6iii_bytes, file_bytes + retained + px * 40);

        // The detailed estimate exceeds the unchanged admission value: the gate
        // still undercounts until real peak-RSS data justifies raising it.
        assert!(libraw_bytes > estimated_decode_bytes(w, h, DecodeFormat::RawProxy));
        assert!(generic_bytes > z6iii_bytes && z6iii_bytes > libraw_bytes);

        // Native keeps the full source resolution in the retained mip.
        let native =
            estimated_decode_bytes_observed(w, h, DecodeFormat::RawNative, file_bytes, &z6iii);
        assert_eq!(native, file_bytes + px * RETAINED_BYTES_PER_PX * 4 / 3 + px * 40);

        // Saturation, not panic, on absurd inputs.
        assert_eq!(
            estimated_decode_bytes_observed(
                u32::MAX,
                u32::MAX,
                DecodeFormat::RawNative,
                usize::MAX,
                &generic,
            ),
            usize::MAX
        );
    }

    #[test]
    fn decode_gate_units_follow_byte_budget() {
        let mib = 1024 * 1024usize;
        assert_eq!(decode_gate_units(512 * mib), 2);
        assert_eq!(decode_gate_units(4 * 1024 * mib), 16);
        let gates = DecodeGates::new(512 * mib);
        assert_eq!(gates.heavy.total_units(), 2);
        assert_eq!(gates.light.total_units(), 2);
    }

    struct GateState {
        total: usize,
        occupied: usize,
        free: usize,
        active: usize,
        /// `(charged units, granted)` per registered request, in ticket order.
        waiters: Vec<(usize, bool)>,
    }

    fn gate_state(gate: &DecodeGate) -> GateState {
        let inner = gate.inner.lock();
        GateState {
            total: inner.total_units,
            occupied: inner.occupied_units,
            free: inner.free_units(),
            active: inner.active,
            waiters: inner.waiters.iter().map(|waiter| (waiter.units, waiter.granted)).collect(),
        }
    }

    #[test]
    fn decode_gate_shrink_below_two_active_permits_admits_nothing() {
        // Two 2-unit decodes are in flight on a 4-unit gate when the budget
        // shrinks the gate to 2 — below what is already running.
        let gate = DecodeGate::new(4);
        let first = gate.acquire(DecodePriority::Interactive, 2);
        let second = gate.acquire(DecodePriority::Interactive, 2);
        let full = gate_state(&gate);
        assert_eq!((full.occupied, full.free), (4, 0));

        gate.set_total_units(2);
        let shrunk = gate_state(&gate);
        assert_eq!(shrunk.total, 2);
        assert_eq!(shrunk.occupied, 4, "in-flight decodes stay charged above the new budget");
        assert_eq!(shrunk.free, 0, "a budget below occupancy has no free units");

        // A weakly-charged thumbnail queues behind the over-budget decodes.
        let queued = {
            let mut inner = gate.inner.lock();
            DecodeGate::enqueue(&mut inner, DecodePriority::Thumbnail, 1)
        };
        drop(first);
        let after_one = gate_state(&gate);
        assert_eq!(after_one.waiters, vec![(1, false)], "the queued 1-unit decode must not be admitted");
        assert_eq!(after_one.occupied, 2);
        assert_eq!(after_one.free, 0, "one release still leaves the gate fully occupied");

        drop(second);
        let after_all = gate_state(&gate);
        assert_eq!((after_all.occupied, after_all.free), (1, 1));
        assert_eq!(after_all.waiters, vec![(1, true)], "the queued decode runs once the gate drains");

        let permit = GatePermit { gate: &gate, units: DecodeGate::claim(&mut gate.inner.lock(), queued) };
        assert_eq!(permit.units, 1);
        drop(permit);
        let drained = gate_state(&gate);
        assert_eq!((drained.occupied, drained.free, drained.active), (0, 2, 0));
        assert!(drained.waiters.is_empty());
    }

    #[test]
    fn decode_gate_partial_release_then_growth_recredits_only_real_headroom() {
        let gate = DecodeGate::new(4);
        let first = gate.acquire(DecodePriority::Interactive, 2);
        let second = gate.acquire(DecodePriority::Interactive, 2);
        gate.set_total_units(2);
        drop(second);
        let held = gate_state(&gate);
        assert_eq!(
            (held.occupied, held.free),
            (2, 0),
            "the release below the new budget must not free units that are still held"
        );

        let (big, clamped) = {
            let mut inner = gate.inner.lock();
            (
                DecodeGate::enqueue(&mut inner, DecodePriority::Interactive, 5),
                DecodeGate::enqueue(&mut inner, DecodePriority::Thumbnail, 6),
            )
        };
        assert_eq!(gate_state(&gate).waiters, vec![(2, false), (2, false)], "queued asks are clamped to the small gate");

        // Growth re-opens 6 - 2 = 4 units, not the 6 the stale free counter
        // would have invented, and queued asks return to their original size.
        gate.set_total_units(6);
        let grown = gate_state(&gate);
        assert_eq!((grown.total, grown.occupied, grown.free), (6, 2, 4));
        assert_eq!(grown.waiters, vec![(5, false), (6, false)]);

        drop(first);
        let after_release = gate_state(&gate);
        assert_eq!((after_release.occupied, after_release.free), (5, 1));
        assert_eq!(after_release.waiters, vec![(5, true), (6, false)], "only the 5-unit decode fits the reclaimed headroom");

        let big_permit = GatePermit { gate: &gate, units: DecodeGate::claim(&mut gate.inner.lock(), big) };
        assert_eq!(big_permit.units, 5);
        drop(big_permit);
        let unblocked = gate_state(&gate);
        assert_eq!((unblocked.occupied, unblocked.free), (6, 0));
        assert_eq!(unblocked.waiters, vec![(6, true)]);

        let clamped_permit = GatePermit { gate: &gate, units: DecodeGate::claim(&mut gate.inner.lock(), clamped) };
        assert_eq!(clamped_permit.units, 6);
        drop(clamped_permit);
        assert_eq!(gate_state(&gate).occupied, 0);
    }

    #[test]
    fn decode_gate_grant_retune_before_claim_releases_charged_units() {
        let gate = DecodeGate::new(4);
        let held = gate.acquire(DecodePriority::Interactive, 2);
        let queued = {
            let mut inner = gate.inner.lock();
            DecodeGate::enqueue(&mut inner, DecodePriority::Thumbnail, 4)
        };
        assert_eq!(gate_state(&gate).free, 2);

        gate.set_total_units(6);
        let granted = gate_state(&gate);
        assert_eq!((granted.occupied, granted.free), (6, 0));
        assert_eq!(granted.waiters, vec![(4, true)]);

        // The budget shrinks while the grant is still unclaimed: the granted
        // charge is already in flight and must not be retuned.
        gate.set_total_units(2);
        let retuned = gate_state(&gate);
        assert_eq!(retuned.waiters, vec![(4, true)], "granted charge must not be clamped");
        assert_eq!((retuned.total, retuned.occupied, retuned.free), (2, 6, 0));

        let charged = DecodeGate::claim(&mut gate.inner.lock(), queued);
        assert_eq!(charged, 4, "the permit returns the units actually charged, not the caller's ask");
        let permit = GatePermit { gate: &gate, units: charged };
        drop(permit);
        let after = gate_state(&gate);
        assert_eq!(after.occupied, 2, "exactly the charged 4 units are released");
        assert_eq!(after.free, 0, "the surviving 2-unit decode still fills the 2-unit gate");
        assert_eq!(after.active, 1);

        drop(held);
        let empty = gate_state(&gate);
        assert_eq!((empty.occupied, empty.free, empty.active), (0, 2, 0));
    }

    #[test]
    fn large_raster_reserves_most_of_a_full_gate() {
        // A ~64MP raster cannot be admitted several times at once.
        let bytes = estimated_decode_bytes(8000, 8000, DecodeFormat::Raster);
        assert!(decode_units(bytes) >= 5, "{bytes}");
    }

    #[test]
    fn source_cache_evicts_by_bytes_and_keeps_recent_pins() {
        let dir = s2_test_dir("evict");
        for index in 0..6 {
            write_test_png(&dir, &format!("evict-{index}.png"), 400, 300, index as u8);
        }
        let engine = FileEngine::open(&dir).unwrap();
        let ids: Vec<ImageId> = engine.images().iter().map(|image| image.id).collect();
        assert_eq!(ids.len(), 6);
        let adjustments = AdjustmentSet::default();
        engine.thumbnail(ids[0], &adjustments, 64).unwrap();
        let one = engine.source_cache_stats().current_bytes;
        assert!(one > 0);
        engine.set_source_cache_budget_bytes(one * 2);
        for id in &ids {
            engine.thumbnail(*id, &adjustments, 64).unwrap();
        }
        let stats = engine.source_cache_stats();
        eprintln!("eviction stats: {stats:?}");
        assert!(stats.evictions >= 1, "{stats:?}");
        assert!(stats.cached_sources <= 4, "{stats:?}");
        assert!(stats.pinned_sources >= 2, "{stats:?}");
        assert_eq!(stats.max_bytes, one * 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn same_image_concurrent_requests_decode_once() {
        let dir = s2_test_dir("flight");
        write_test_png(&dir, "same.png", 1600, 1200, 7);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        let adjustments = AdjustmentSet::default();
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    engine.thumbnail(id, &adjustments, 64).unwrap();
                });
            }
        });
        let stats = engine.source_cache_stats();
        eprintln!("single-flight stats: {stats:?}");
        assert_eq!(stats.misses, 1, "the same image decoded more than once: {stats:?}");
        assert_eq!(stats.hits + stats.single_flight_waits, 7, "{stats:?}");
        assert_eq!(stats.cached_sources, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn different_images_decode_in_parallel() {
        let dir = s2_test_dir("parallel");
        for index in 0..3 {
            write_test_png(&dir, &format!("parallel-{index}.png"), 2400, 1600, index as u8);
        }
        let engine = FileEngine::open(&dir).unwrap();
        let ids: Vec<ImageId> = engine.images().iter().map(|image| image.id).collect();
        let adjustments = AdjustmentSet::default();
        let barrier = std::sync::Barrier::new(ids.len());
        std::thread::scope(|scope| {
            for id in &ids {
                scope.spawn(|| {
                    barrier.wait();
                    engine.thumbnail(*id, &adjustments, 64).unwrap();
                });
            }
        });
        let stats = engine.source_cache_stats();
        eprintln!("parallel decode stats: {stats:?}");
        assert_eq!(stats.misses, 3, "{stats:?}");
        assert!(stats.decode_peak_concurrency >= 2, "decodes never overlapped: {stats:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn source_cache_eviction_does_not_change_pixels() {
        let dir = s2_test_dir("pixels");
        write_test_png(&dir, "pixels.png", 800, 600, 21);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        let adjustments = AdjustmentSet::default();
        let before = pixel_hash(&engine.thumbnail(id, &adjustments, 128).unwrap().pixels);
        engine.clear_source_cache();
        assert_eq!(engine.source_cache_stats().current_bytes, 0);
        let after = pixel_hash(&engine.thumbnail(id, &adjustments, 128).unwrap().pixels);
        assert_eq!(before, after, "cache eviction changed the rendered pixels");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[ignore = "requires AURORA_NATIVE_RAW_TEST_DIR with a real RAW"]
    fn native_raw_is_charged_to_the_same_budget() {
        let dir = PathBuf::from(std::env::var_os("AURORA_NATIVE_RAW_TEST_DIR").expect("fixture directory"));
        let engine = FileEngine::open(&dir).unwrap();
        let info = engine.images().into_iter().find(|info| crate::raw::is_raw(&info.path)).expect("RAW fixture");
        let adjustments = AdjustmentSet::default();
        let roi = NormRect { x: 0.25, y: 0.25, width: 0.125, height: 0.125 };
        let native = engine
            .native_raw_source(info.id, Some((&adjustments.geometry, roi, 1536)))
            .expect("native RAW decode");
        let expected = source_cache_bytes(&native);
        let stats = engine.source_cache_stats();
        eprintln!("native budget stats: {stats:?}, native {expected} bytes");
        assert_eq!(stats.native_bytes, expected);
        assert!(stats.current_bytes >= stats.native_bytes);
        assert_eq!(engine.native_raw_source(info.id, Some((&adjustments.geometry, roi, 1536))).unwrap().snapshot_id(), native.snapshot_id());
        assert_eq!(engine.source_cache_stats().native_hits, 1);
    }

    #[test]
    fn source_cache_bytes_count_lazy_portrait_derivatives() {
        let rgba = vec![96u8; 320 * 240 * 4];
        let source = Source::from_rgba(320, 240, &rgba);
        let before = source_cache_bytes(&source);
        let _face = source.face_input_image();
        let after = source_cache_bytes(&source);
        assert!(
            after >= before + 320 * 240 * 4,
            "face input not charged: {before} -> {after}"
        );
        assert!(source.cached_derived_bytes() >= 320 * 240 * 4);
    }

    #[test]
    fn ai_input_is_charged_to_the_source_cache() {
        let dir = s2_test_dir("ai-bytes");
        write_test_png(&dir, "ai.png", 64, 48, 5);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        let input = engine.ai_input(id).unwrap();
        let pixels = (input.width as usize) * (input.height as usize) * 4;
        let stats = engine.source_cache_stats();
        eprintln!("ai input stats: {stats:?}");
        assert!(stats.ai_input_bytes >= pixels, "{stats:?}");
        assert!(stats.current_bytes >= stats.ai_input_bytes, "{stats:?}");
        assert_eq!(stats.ai_input_bytes, ai_input_bytes(&input));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rescan_keeps_unchanged_ids_and_drops_removed_sources() {
        let dir = s2_test_dir("rescan");
        write_test_png(&dir, "keep.png", 96, 64, 1);
        write_test_png(&dir, "drop.png", 96, 64, 2);
        let engine = FileEngine::open(&dir).unwrap();
        let adjustments = AdjustmentSet::default();
        let before: Vec<(String, ImageId)> = engine
            .images()
            .into_iter()
            .map(|info| (info.file_name.clone(), info.id))
            .collect();
        for (_, id) in &before {
            engine.thumbnail(*id, &adjustments, 64).unwrap();
        }
        let bytes = engine.source_cache_stats().current_bytes;
        assert!(bytes > 0);

        engine.rescan().unwrap();
        let after: Vec<(String, ImageId)> = engine
            .images()
            .into_iter()
            .map(|info| (info.file_name.clone(), info.id))
            .collect();
        assert_eq!(before, after, "rescan reassigned an unchanged image id");
        assert_eq!(engine.source_cache_stats().current_bytes, bytes, "rescan lost cached bytes");

        // Fill the preview cache so the removal path is observable.
        engine.preview_frame(&PreviewRequest {
            image: before[1].1,
            adjustments: adjustments.clone(),
            region: NormRect::FULL,
            max_size: 64,
            quality: RenderQuality::Final,
            bypass: false,
            mask_overlay: None,
            generation: 1,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
            rendering_intent: None,
            black_point_compensation: false,
        }, false).unwrap();
        assert!(engine.state.lock().preview.len() > 0);

        std::fs::remove_file(dir.join("drop.png")).unwrap();
        engine.rescan().unwrap();
        let stats = engine.source_cache_stats();
        eprintln!("rescan removal stats: {stats:?}");
        assert_eq!(stats.cached_sources, 1, "{stats:?}");
        assert!(stats.current_bytes < bytes, "{stats:?}");
        assert_eq!(engine.state.lock().preview.len(), 0, "rescan did not clear previews");
        let kept: Vec<ImageId> = engine.images().iter().map(|info| info.id).collect();
        assert_eq!(kept, vec![before[1].1], "unchanged file lost its id");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn source_identity_recheck_restarts_the_budget_for_proxy_and_native() {
        let dir = s2_test_dir("identity-recheck");
        let path = write_test_png(&dir, "shot.png", 48, 32, 1);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        engine.source(id).unwrap();

        let mut st = engine.state.lock();
        let identity = st.sources[&id].identity;
        st.native_raw = Some(NativeRawSlot {
            image: id,
            source: None,
            bytes: 0,
            last_used: 0,
            identity,
            last_verified: Instant::now(),
            hits_since_verify: 0,
        });
        for native in [false, true] {
            for by_time in [false, true] {
                let last_verified = if by_time { Instant::now() - SOURCE_IDENTITY_RECHECK } else { Instant::now() };
                let hits = if by_time { 0 } else { SOURCE_IDENTITY_RECHECK_HITS };
                if native {
                    let slot = st.native_raw.as_mut().unwrap();
                    slot.last_verified = last_verified;
                    slot.hits_since_verify = hits;
                } else {
                    let entry = st.sources.get_mut(&id).unwrap();
                    entry.last_verified = last_verified;
                    entry.hits_since_verify = hits;
                }

                for expected_hits in [0, 1] {
                    assert!(FileEngine::refresh_source_identity(&mut st, id, &path, native));
                    let (verified_at, hits_since_verify) = if native {
                        let slot = st.native_raw.as_ref().unwrap();
                        (slot.last_verified, slot.hits_since_verify)
                    } else {
                        let entry = &st.sources[&id];
                        (entry.last_verified, entry.hits_since_verify)
                    };
                    assert_eq!(hits_since_verify, expected_hits);
                    assert!(verified_at >= last_verified);
                }
            }
        }
        drop(st);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn replaced_source_file_invalidates_the_cache() {
        let dir = s2_test_dir("identity");
        let path = write_test_png(&dir, "shot.png", 48, 32, 1);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        let adjustments = AdjustmentSet::default();
        engine.thumbnail(id, &adjustments, 64).unwrap();
        let first = engine.source(id).unwrap().snapshot_id();
        assert_eq!(engine.source_cache_stats().identity_invalidations, 0);

        // Same dimensions, different pixels. The trailing byte forces a length
        // change as well, so the identity check cannot depend on mtime alone.
        let rewritten = image::RgbaImage::from_fn(48, 32, |x, y| {
            image::Rgba([(255 - x) as u8, (y * 7) as u8, 200, 255])
        });
        rewritten.save(&path).unwrap();
        {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            file.write_all(b"x").unwrap();
        }
        let second = engine.source(id).unwrap().snapshot_id();
        assert_ne!(first, second, "rewritten file served the old cached source");
        assert!(engine.source_cache_stats().identity_invalidations >= 1);

        // Repeated hits on an unchanged file must not invalidate it.
        for _ in 0..10 {
            engine.source(id).unwrap();
        }
        assert_eq!(engine.source_cache_stats().identity_invalidations, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decode_jobs_record_success_and_failure() {
        let dir = s2_test_dir("decode-lifecycle");
        write_test_png(&dir, "good.png", 16, 12, 1);
        std::fs::write(dir.join("broken.png"), b"\x89PNG\r\n\x1a\nnot a real png").unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        assert_eq!(engine.decode_job_snapshot().total, 0, "opening the library must not decode");

        let (good, broken) = {
            let images = engine.images();
            let good = images.iter().find(|image| image.file_name == "good.png").unwrap().id;
            let broken = images.iter().find(|image| image.file_name == "broken.png").unwrap().id;
            (good, broken)
        };

        engine.source(good).unwrap();
        let snapshot = engine.decode_job_snapshot();
        assert_eq!(snapshot.total, 1);
        let decode = snapshot.scope(JobScope::Decode).expect("decode scope counts");
        assert_eq!(decode.succeeded, 1);
        assert_eq!(decode.failed, 0);
        assert_eq!(decode.live(), 0);

        assert!(engine.source(broken).is_err(), "corrupt PNG must fail to decode");
        let snapshot = engine.decode_job_snapshot();
        assert_eq!(snapshot.total, 2);
        let decode = snapshot.scope(JobScope::Decode).expect("decode scope counts");
        assert_eq!(decode.succeeded, 1);
        assert_eq!(decode.failed, 1);
        assert_eq!(decode.live(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn decode_jobs_stay_bounded_across_many_flights() {
        let dir = s2_test_dir("decode-lifecycle-bound");
        // More distinct images than the decode record cap, so pruning runs.
        let count = 80u32;
        for index in 0..count {
            write_test_png(&dir, &format!("shot-{index:03}.png"), 8, 8, (index % 251) as u8);
        }
        let engine = FileEngine::open(&dir).unwrap();
        let images: Vec<_> = engine.images().into_iter().map(|info| info.id).collect();
        assert_eq!(images.len(), count as usize);

        // Each distinct image is a new leader flight and therefore a new record.
        for image in images {
            engine.source(image).unwrap();
        }
        let snapshot = engine.decode_job_snapshot();
        assert!(snapshot.total <= 64, "decode records grew to {}", snapshot.total);
        let decode = snapshot.scope(JobScope::Decode).expect("decode scope counts");
        assert_eq!(decode.live(), 0, "all flights settled, no live record leaked");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn preview_jobs_record_terminal_calls_and_stay_bounded() {
        let dir = s2_test_dir("preview-lifecycle");
        write_test_png(&dir, "shot.png", 32, 24, 2);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        let request = |generation: u64| PreviewRequest {
            image: id,
            adjustments: AdjustmentSet::default(),
            region: NormRect::FULL,
            max_size: 32,
            quality: crate::RenderQuality::Final,
            bypass: false,
            mask_overlay: None,
            generation,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
            rendering_intent: None,
            black_point_compensation: false,
        };

        engine.render_preview(&request(1)).unwrap();
        engine.render_preview_fast(&request(2), true).unwrap();
        let snapshot = engine.preview_job_snapshot();
        assert_eq!(snapshot.total, 2);
        let preview = snapshot.scope(JobScope::Preview).expect("preview scope counts");
        assert_eq!(preview.succeeded, 2);
        assert_eq!(preview.live(), 0);

        // Repeated drags must not grow the record map without bound. The cache
        // makes these cheap, but each call still leaves a terminal record.
        for generation in 0..200 {
            engine.render_preview_fast(&request(generation), true).unwrap();
        }
        let snapshot = engine.preview_job_snapshot();
        assert!(snapshot.total <= 64, "preview records grew to {}", snapshot.total);
        assert_eq!(
            snapshot.scope(JobScope::Preview).expect("preview scope counts").live(),
            0
        );

        // A failed render still settles its record as Failed.
        let missing = PreviewRequest { image: ImageId(999_999), ..request(0) };
        assert!(engine.render_preview(&missing).is_err());
        let snapshot = engine.preview_job_snapshot();
        assert_eq!(
            snapshot.scope(JobScope::Preview).expect("preview scope counts").failed,
            1
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn thumbnail_jobs_record_success_and_failure_and_stay_bounded() {
        let dir = s2_test_dir("thumbnail-lifecycle");
        write_test_png(&dir, "thumb.png", 32, 24, 4);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;
        let adjustments = AdjustmentSet::default();

        engine.thumbnail(id, &adjustments, 16).unwrap();
        let snapshot = engine.thumbnail_job_snapshot();
        assert_eq!(snapshot.total, 1);
        let thumbnail = snapshot.scope(JobScope::Thumbnail).expect("thumbnail scope counts");
        assert_eq!(thumbnail.succeeded, 1);
        assert_eq!(thumbnail.failed, 0);
        assert_eq!(thumbnail.live(), 0);

        // An unknown image fails its record and returns the usual error.
        assert!(engine.thumbnail(ImageId(999_999), &adjustments, 16).is_err());
        let snapshot = engine.thumbnail_job_snapshot();
        let thumbnail = snapshot.scope(JobScope::Thumbnail).expect("thumbnail scope counts");
        assert_eq!(thumbnail.failed, 1);
        assert_eq!(thumbnail.live(), 0);

        // Repeated calls prune terminal records instead of growing without bound.
        for _ in 0..200 {
            engine.thumbnail(id, &adjustments, 16).unwrap();
        }
        let snapshot = engine.thumbnail_job_snapshot();
        assert!(snapshot.total <= 64, "thumbnail records grew to {}", snapshot.total);
        assert_eq!(
            snapshot.scope(JobScope::Thumbnail).expect("thumbnail scope counts").live(),
            0
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compute_ai_mask_failure_is_a_failed_inference_record() {
        let dir = s2_test_dir("inference-mask");
        write_test_png(&dir, "inference-mask.png", 32, 24, 5);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;

        // A category the people parser cannot produce fails without any model.
        let err = engine.compute_ai_mask(id, AiMaskKind::Eyes).unwrap_err();
        assert!(err.to_string().contains("no Eyes class"), "{err}");
        let snapshot = engine.inference_job_snapshot();
        let inference = snapshot.scope(JobScope::Inference).expect("inference scope counts");
        assert_eq!(inference.total, 1);
        assert_eq!(inference.failed, 1);
        assert_eq!(inference.succeeded, 0);
        assert_eq!(inference.live(), 0);

        // With no subject model installed, the typed "no model/runtime" error is
        // recorded as well. A machine that ships the model keeps this branch out.
        if !crate::aimask::model_available("subjectMasking.onnx") {
            let err = engine.compute_ai_mask(id, AiMaskKind::Subject).unwrap_err();
            let text = err.to_string();
            assert!(
                text.contains("AI model subjectMasking.onnx not found")
                    || text.contains("ONNX Runtime is not available"),
                "{text}"
            );
            let snapshot = engine.inference_job_snapshot();
            let inference = snapshot.scope(JobScope::Inference).expect("inference scope counts");
            assert_eq!(inference.total, 2);
            assert_eq!(inference.failed, 2);
            assert_eq!(inference.live(), 0);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn eye_report_records_both_outcomes_without_external_fixtures() {
        let dir = s2_test_dir("inference-eyes");
        write_test_png(&dir, "inference-eyes.png", 32, 24, 6);
        let engine = FileEngine::open(&dir).unwrap();
        let id = engine.images()[0].id;

        // The eye models are optional local assets. Each call must still leave
        // one terminal record whether the check runs or reports its typed error.
        let first = engine.eye_report(id);
        let second = engine.eye_report(id);
        let snapshot = engine.inference_job_snapshot();
        let inference = snapshot.scope(JobScope::Inference).expect("inference scope counts");
        assert_eq!(inference.total, 2);
        assert_eq!(inference.live(), 0);
        match (&first, &second) {
            (Ok(_), Ok(_)) => assert_eq!(inference.succeeded, 2),
            (Err(_), Err(_)) => assert_eq!(inference.failed, 2),
            other => panic!("eye report outcome changed between calls: {other:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn capture_methods_without_a_camera_keep_their_contract() {
        let dir = s2_test_dir("capture-lifecycle");
        write_test_png(&dir, "capture.png", 8, 6, 7);
        let engine = FileEngine::open(&dir).unwrap();
        if matches!(engine.tether_status(), TetherStatus::Available { .. }) {
            // A real camera makes the no-camera assertions inapplicable.
            std::fs::remove_dir_all(&dir).ok();
            return;
        }

        if cfg!(not(windows)) {
            assert_eq!(engine.camera_status(), CameraStatus::Disconnected);
            assert_eq!(engine.camera_settings(), None);
        }
        assert!(engine.live_view_frame(640).is_none());
        engine.live_view_focus(2);

        let err = engine.capture().expect_err("capture must fail without a camera");
        assert!(err.to_string().contains("NoCamera"), "{err}");
        assert!(
            !dir.join("Capture").exists(),
            "a failed capture must not create the output folder"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn export_scheduler_bounds_concurrent_slots_and_releases_on_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let scheduler = Arc::new(ExportScheduler::with_limits(2, 1024 * 1024 * 1024));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..6 {
            let scheduler = Arc::clone(&scheduler);
            let active = Arc::clone(&active);
            let peak = Arc::clone(&peak);
            handles.push(std::thread::spawn(move || {
                let _permit = scheduler
                    .acquire_cancellable(16 * 1024 * 1024, || false)
                    .expect("permit should be granted");
                let cur = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(cur, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(15));
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!(peak.load(Ordering::SeqCst) <= 2, "concurrent exports exceeded max_slots=2");
        let (left_active, left_bytes, recorded_peak) = scheduler.stats();
        assert_eq!(left_active, 0);
        assert_eq!(left_bytes, 0);
        assert!(recorded_peak <= 2);
    }

    #[test]
    fn export_scheduler_serializes_large_frames_by_memory_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // 4 slots available, but 1 GiB budget forces 24MP/45MP 16-bit exports to serialize.
        let scheduler = Arc::new(ExportScheduler::with_limits(4, 1024 * 1024 * 1024));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let bytes_24mp = estimated_export_bytes(6000, 4000, 16);
        let bytes_45mp = estimated_export_bytes(8256, 5504, 16);

        let mut handles = Vec::new();
        for bytes in [bytes_24mp, bytes_45mp, bytes_24mp] {
            let scheduler = Arc::clone(&scheduler);
            let active = Arc::clone(&active);
            let peak = Arc::clone(&peak);
            handles.push(std::thread::spawn(move || {
                let _permit = scheduler
                    .acquire_cancellable(bytes, || false)
                    .expect("large frame should still acquire when idle");
                let cur = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(cur, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(15));
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 1, "24MP/45MP exports must serialize under 1 GiB budget");
        let (left_active, left_bytes, _) = scheduler.stats();
        assert_eq!(left_active, 0);
        assert_eq!(left_bytes, 0);
    }

    #[test]
    fn export_scheduler_aborts_wait_when_cancelled() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let scheduler = Arc::new(ExportScheduler::with_limits(1, 512 * 1024 * 1024));
        let held = scheduler.acquire_cancellable(1024, || false).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let waiting = Arc::new(AtomicBool::new(false));

        let s2 = Arc::clone(&scheduler);
        let c2 = Arc::clone(&cancelled);
        let w2 = Arc::clone(&waiting);
        let waiter = std::thread::spawn(move || {
            w2.store(true, Ordering::SeqCst);
            s2.acquire_cancellable(1024, || c2.load(Ordering::SeqCst)).is_some()
        });

        while !waiting.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        cancelled.store(true, Ordering::SeqCst);
        scheduler.notify_all();

        let acquired = waiter.join().unwrap();
        assert!(!acquired, "cancelled job must abort waiting without acquiring permit");
        assert_eq!(scheduler.stats().0, 1);

        drop(held);
        let (left_active, left_bytes, _) = scheduler.stats();
        assert_eq!(left_active, 0);
        assert_eq!(left_bytes, 0);
    }

    #[test]
    fn export_scheduler_releases_slot_on_panic_unwind() {
        let scheduler = ExportScheduler::with_limits(1, 512 * 1024 * 1024);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _permit = scheduler.acquire_cancellable(4096, || false).unwrap();
            panic!("simulated export panic");
        }));
        assert!(res.is_err());
        let (left_active, left_bytes, _) = scheduler.stats();
        assert_eq!(left_active, 0);
        assert_eq!(left_bytes, 0);
        assert!(scheduler.acquire_cancellable(4096, || false).is_some());
    }
}

#[cfg(test)]
mod save_report_tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aurora-save-report-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_png(dir: &Path, name: &str, seed: u8) {
        let image = image::RgbaImage::from_fn(8, 8, |x, y| {
            image::Rgba([(x as u8).wrapping_add(seed), (y as u8).wrapping_mul(3), seed, 255])
        });
        image.save(dir.join(name)).unwrap();
    }

    #[test]
    fn batch_report_separates_saved_and_failed_documents() {
        let dir = temp_dir("partial");
        write_png(&dir, "one.png", 1);
        write_png(&dir, "two.png", 2);
        let engine = FileEngine::open(&dir).unwrap();
        assert_eq!(engine.images().len(), 2);
        let (id_a, rel_a, id_b, rel_b) = {
            let st = engine.state.lock();
            let a = &st.entries[0];
            let b = &st.entries[1];
            (a.info.id, a.rel_path.clone(), b.info.id, b.rel_path.clone())
        };

        // Write `b` once, then make its target a directory so the next atomic
        // rename over it fails while `a` keeps writing normally.
        let store = engine.store.as_ref().unwrap();
        store.save_adjustments(&rel_b, &AdjustmentSet::default()).unwrap();
        let target = std::fs::read_dir(store.root().join("adjust"))
            .unwrap()
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .find(|path| path.extension().and_then(|e| e.to_str()) == Some("json"))
            .expect("b's adjustment file");
        std::fs::remove_file(&target).unwrap();
        std::fs::create_dir(&target).unwrap();

        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = 0.5;
        let report = engine.save_adjustments_batch_report(&[(id_a, adj.clone()), (id_b, adj.clone())]);
        assert_eq!(report.documents.len(), 2);
        assert_eq!(report.saved(), 1);
        assert_eq!(report.failed(), 1);
        assert_eq!(report.superseded(), 0);
        assert!(report.index_error.is_none());
        assert_eq!(report.documents.iter().find(|d| d.is_saved()).unwrap().path, rel_a);
        let failed = report.first_failure().expect("one failed document");
        assert_eq!(failed.image, id_b);
        assert_eq!(failed.path, rel_b);
        assert!(!report.is_complete());
        assert!(engine.save_adjustments_batch(&[(id_a, adj.clone()), (id_b, adj)]).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_stale_report_does_not_restore_older_in_memory_edits() {
        let dir = temp_dir("memory-order");
        write_png(&dir, "shot.png", 4);
        let engine = FileEngine::open(&dir).unwrap();
        let (id, rel_path) = {
            let st = engine.state.lock();
            (st.entries[0].info.id, st.entries[0].rel_path.clone())
        };

        let mut newer = AdjustmentSet::default();
        newer.exposure.exposure = 2.0;
        assert_eq!(
            engine.save_adjustments_batch_report_at(5, &[(id, newer)]).saved(),
            1
        );

        // An older call that reaches the engine late must not restore its
        // stale edits in memory, even though its disk write is skipped.
        let mut older = AdjustmentSet::default();
        older.exposure.exposure = -1.0;
        let stale = engine.save_adjustments_batch_report_at(4, &[(id, older)]);
        assert_eq!(stale.superseded(), 1);
        assert_eq!(ImageEngine::load_adjustments(&engine, id).exposure.exposure, 2.0);
        let store = engine.store.as_ref().unwrap();
        assert_eq!(store.load_adjustments(&rel_path).unwrap().exposure.exposure, 2.0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_stale_sequence_does_not_overwrite_newer_bytes() {
        let dir = temp_dir("order");
        write_png(&dir, "shot.png", 3);
        let engine = FileEngine::open(&dir).unwrap();
        let (id, rel_path, info) = {
            let st = engine.state.lock();
            let entry = &st.entries[0];
            (entry.info.id, entry.rel_path.clone(), entry.info.clone())
        };

        let mut newer = AdjustmentSet::default();
        newer.exposure.exposure = 2.0;
        let first = engine.persist_batch(&[(rel_path.clone(), newer.clone(), info.clone())]);
        assert_eq!(first.saved(), 1);

        // Sequence 0 is older than every allocated batch sequence.
        let mut older = AdjustmentSet::default();
        older.exposure.exposure = -2.0;
        let stale = engine.persist_batch_at(0, &[(rel_path.clone(), older, info)]);
        assert_eq!(stale.superseded(), 1);
        assert_eq!(stale.saved(), 0);
        assert!(stale.into_error().is_none(), "superseded is not a failure");

        let loaded = engine.store.as_ref().unwrap().load_adjustments(&rel_path).unwrap();
        assert_eq!(loaded.exposure.exposure, 2.0, "older write replaced newer bytes");

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod queued_save_tests {
    use super::*;
    use crate::save_queue::PathState;

    fn queued_temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "aurora-queued-save-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn queued_write_png(dir: &Path, name: &str, seed: u8) {
        let image = image::RgbaImage::from_fn(8, 8, |x, y| {
            image::Rgba([(x as u8).wrapping_add(seed), (y as u8).wrapping_mul(3), seed, 255])
        });
        image.save(dir.join(name)).unwrap();
    }

    fn id_by_path(engine: &FileEngine, rel_path: &str) -> ImageId {
        engine
            .state
            .lock()
            .entries
            .iter()
            .find(|entry| entry.rel_path == rel_path)
            .map(|entry| entry.info.id)
            .unwrap_or_else(|| panic!("{rel_path} is not in the engine"))
    }

    fn rel_of(engine: &FileEngine, image: ImageId) -> String {
        engine
            .state
            .lock()
            .entries
            .iter()
            .find(|entry| entry.info.id == image)
            .map(|entry| entry.rel_path.clone())
            .unwrap()
    }

    fn exposure(value: f32) -> AdjustmentSet {
        let mut adjustments = AdjustmentSet::default();
        adjustments.exposure.exposure = value;
        adjustments
    }

    /// Blocks the first `write` until the test releases it; later writes pass
    /// straight through (the release sender is dropped after the first send, so
    /// `recv` returns immediately once disconnected).
    struct WriteGate {
        entered: std::sync::mpsc::Sender<()>,
        release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl WriteGate {
        fn enter_and_wait(&self) {
            let _ = self.entered.send(());
            let receiver = self.release.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let _ = receiver.recv();
        }
    }

    /// A `SaveWriter` fake that records what the queue would have written.
    struct RecordingWriter {
        writes: std::sync::Mutex<Vec<(String, Option<String>)>>,
        gate: Option<Arc<WriteGate>>,
    }

    impl RecordingWriter {
        fn plain() -> Arc<Self> {
            Arc::new(RecordingWriter { writes: std::sync::Mutex::new(Vec::new()), gate: None })
        }

        fn gated(gate: Arc<WriteGate>) -> Arc<Self> {
            Arc::new(RecordingWriter { writes: std::sync::Mutex::new(Vec::new()), gate: Some(gate) })
        }

        fn writes_for(&self, rel_path: &str) -> Vec<Option<String>> {
            self.writes
                .lock()
                .unwrap()
                .iter()
                .filter(|(path, _)| path == rel_path)
                .map(|(_, marker)| marker.clone())
                .collect()
        }
    }

    impl SaveWriter for RecordingWriter {
        fn write(&self, rel_path: &str, adjustments: &AdjustmentSet) -> anyhow::Result<()> {
            if let Some(gate) = &self.gate {
                gate.enter_and_wait();
            }
            self.writes
                .lock()
                .unwrap()
                .push((rel_path.to_string(), adjustments.style.clone()));
            Ok(())
        }

        fn commit_index(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn queued_save_writes_documents_and_index_then_reloads() {
        let dir = queued_temp_dir("round-trip");
        queued_write_png(&dir, "one.png", 1);
        queued_write_png(&dir, "two.png", 2);
        let engine = FileEngine::open(&dir).unwrap();

        let rel_a = "one.png".to_string();
        let rel_b = "two.png".to_string();
        let id_a = id_by_path(&engine, &rel_a);
        let id_b = id_by_path(&engine, &rel_b);
        let adj_a = exposure(0.75);
        let adj_b = exposure(-0.5);

        let summary =
            engine.save_adjustments_queued(&[(id_a, adj_a.clone()), (id_b, adj_b.clone())]);
        assert_eq!(summary.accepted(), 2, "{summary:?}");
        assert_eq!(summary.failed(), 0, "{summary:?}");
        assert!(matches!(
            summary.outcome_of(id_a),
            Some(QueuedSaveOutcome::Accepted { revision }) if revision.get() >= 1
        ));

        // The edit is visible immediately, before the writer has necessarily run.
        assert_eq!(ImageEngine::load_adjustments(&engine, id_a), adj_a);

        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert!(!report.timed_out, "{report:?}");
        assert_eq!(report.saved(), 2, "{report:?}");
        assert_eq!(report.failed(), 0, "{report:?}");
        assert!(report.index_error.is_none(), "{report:?}");
        assert_eq!(report.state_of(&rel_a), Some(&PathState::Saved), "{report:?}");
        assert_eq!(report.state_of(&rel_b), Some(&PathState::Saved), "{report:?}");

        let store = engine.store.as_ref().unwrap();
        assert_eq!(store.load_adjustments(&rel_a).unwrap(), adj_a);
        assert_eq!(store.load_adjustments(&rel_b).unwrap(), adj_b);
        let index = store.load_index().unwrap();
        assert_eq!(index.entry(&rel_a).map(|entry| entry.rel_path.as_str()), Some(rel_a.as_str()));
        assert_eq!(index.entry(&rel_b).map(|entry| entry.rel_path.as_str()), Some(rel_b.as_str()));

        drop(engine);
        let reopened = FileEngine::open(&dir).unwrap();
        let reopened_a = id_by_path(&reopened, &rel_a);
        let reopened_b = id_by_path(&reopened, &rel_b);
        assert_eq!(ImageEngine::load_adjustments(&reopened, reopened_a), adj_a);
        assert_eq!(ImageEngine::load_adjustments(&reopened, reopened_b), adj_b);

        drop(reopened);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn queued_save_failure_is_per_document_and_the_writer_survives() {
        let dir = queued_temp_dir("partial");
        queued_write_png(&dir, "one.png", 1);
        queued_write_png(&dir, "two.png", 2);
        let engine = FileEngine::open(&dir).unwrap();

        let rel_a = "one.png".to_string();
        let rel_b = "two.png".to_string();
        let id_a = id_by_path(&engine, &rel_a);
        let id_b = id_by_path(&engine, &rel_b);

        // Turn `b`'s adjustment target into a directory so the atomic rename
        // fails while `a` keeps working. Writing b first names the only file.
        let store = engine.store.as_ref().unwrap();
        store.save_adjustments(&rel_b, &AdjustmentSet::default()).unwrap();
        let target = std::fs::read_dir(store.root().join("adjust"))
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .find(|path| path.extension().and_then(|extension| extension.to_str()) == Some("json"))
            .expect("b's adjustment file");
        std::fs::remove_file(&target).unwrap();
        std::fs::create_dir(&target).unwrap();

        let adj_a = exposure(0.5);
        let adj_b = exposure(0.25);
        let summary =
            engine.save_adjustments_queued(&[(id_a, adj_a.clone()), (id_b, adj_b.clone())]);
        assert_eq!(summary.accepted(), 2, "{summary:?}");

        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert!(!report.timed_out, "{report:?}");
        assert_eq!(report.saved(), 1, "{report:?}");
        assert_eq!(report.failed(), 1, "{report:?}");
        assert_eq!(report.state_of(&rel_a), Some(&PathState::Saved), "{report:?}");
        assert!(
            matches!(report.state_of(&rel_b), Some(PathState::Failed { .. })),
            "{report:?}"
        );
        // The failed edit was never dropped from memory.
        assert_eq!(ImageEngine::load_adjustments(&engine, id_b), adj_b);

        // Repair the path; the writer is still alive and takes new work.
        std::fs::remove_dir_all(&target).unwrap();
        let retry = engine.save_adjustments_queued(&[(id_b, adj_b.clone())]);
        assert_eq!(retry.accepted(), 1, "{retry:?}");
        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert!(!report.timed_out, "{report:?}");
        assert_eq!(report.state_of(&rel_b), Some(&PathState::Saved), "{report:?}");
        let store = engine.store.as_ref().unwrap();
        assert_eq!(store.load_adjustments(&rel_b).unwrap(), adj_b);

        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn queue_full_keeps_the_in_memory_edit_and_reports_capacity() {
        let dir = queued_temp_dir("full");
        queued_write_png(&dir, "one.png", 1);
        queued_write_png(&dir, "two.png", 2);
        let mut engine = FileEngine::open(&dir).unwrap();

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(WriteGate {
            entered: entered_tx,
            release: std::sync::Mutex::new(release_rx),
        });
        let writer = RecordingWriter::gated(gate);
        engine.install_test_save_queue(writer.clone(), 1);

        let rel_a = "one.png".to_string();
        let rel_b = "two.png".to_string();
        let id_a = id_by_path(&engine, &rel_a);
        let id_b = id_by_path(&engine, &rel_b);

        let first = exposure(0.5);
        let second = exposure(0.25);
        assert_eq!(
            engine.save_adjustments_queued(&[(id_a, first)]).accepted(),
            1,
            "the first path uses the single outstanding slot"
        );
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the writer never entered write()");

        let summary = engine.save_adjustments_queued(&[(id_b, second.clone())]);
        assert_eq!(summary.full(), 1, "{summary:?}");
        assert!(matches!(
            summary.outcome_of(id_b),
            Some(QueuedSaveOutcome::Full { capacity }) if *capacity == 1
        ));
        // A refused submission still applies the edit the user just made.
        assert_eq!(ImageEngine::load_adjustments(&engine, id_b), second);

        let _ = release_tx.send(());
        drop(release_tx);
        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert_eq!(report.state_of(&rel_a), Some(&PathState::Saved), "{report:?}");
        assert!(writer.writes_for(&rel_b).is_empty(), "a Full submission must not be written");

        // Resubmitting after the queue drains is the documented retry.
        let retry = engine.save_adjustments_queued(&[(id_b, second)]);
        assert_eq!(retry.accepted(), 1, "{retry:?}");
        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert_eq!(report.state_of(&rel_b), Some(&PathState::Saved), "{report:?}");

        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn queued_revisions_keep_the_newest_write_for_a_path() {
        let dir = queued_temp_dir("revisions");
        queued_write_png(&dir, "one.png", 1);
        let mut engine = FileEngine::open(&dir).unwrap();

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(WriteGate {
            entered: entered_tx,
            release: std::sync::Mutex::new(release_rx),
        });
        let writer = RecordingWriter::gated(gate);
        engine.install_test_save_queue(writer.clone(), 8);

        let rel = "one.png".to_string();
        let image = id_by_path(&engine, &rel);

        let mut r1 = AdjustmentSet::default();
        r1.style = Some("r1".to_string());
        let first = engine.save_adjustments_queued(&[(image, r1)]);
        let revision1 = match first.outcome_of(image) {
            Some(QueuedSaveOutcome::Accepted { revision }) => *revision,
            other => panic!("expected the first revision to be accepted, got {other:?}"),
        };
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the writer never entered write()");
        assert_eq!(first.accepted(), 1);

        let mut r2 = AdjustmentSet::default();
        r2.style = Some("r2".to_string());
        let second = engine.save_adjustments_queued(&[(image, r2)]);
        let revision2 = match second.outcome_of(image) {
            Some(QueuedSaveOutcome::Accepted { revision }) => *revision,
            other => panic!("expected the newer revision to be accepted, got {other:?}"),
        };
        assert!(revision2 > revision1, "revisions must strictly increase: {revision1:?} {revision2:?}");

        let _ = release_tx.send(());
        drop(release_tx);
        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert!(!report.timed_out, "{report:?}");
        assert_eq!(report.state_of(&rel), Some(&PathState::Saved), "{report:?}");
        let status = report
            .documents
            .iter()
            .find(|document| document.rel_path == rel)
            .expect("the path in the report");
        assert_eq!(
            status.revision, revision2,
            "the report must name the revision that owns the path"
        );
        assert_eq!(
            writer.writes_for(&rel),
            [Some("r1".to_string()), Some("r2".to_string())],
            "the in-flight r1 may finish, but r2 must land last"
        );

        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_infos_stay_bounded_keep_in_flight_and_reinsert_on_retry() {
        let dir = queued_temp_dir("save-info-bound");
        queued_write_png(&dir, "one.png", 1);
        let mut engine = FileEngine::open(&dir).unwrap();
        let image = id_by_path(&engine, "one.png");
        let rel = "one.png".to_string();

        // Gate the writer so the path is genuinely in flight during pruning.
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(WriteGate {
            entered: entered_tx,
            release: std::sync::Mutex::new(release_rx),
        });
        engine.install_test_save_queue(RecordingWriter::gated(gate), 8);

        assert_eq!(engine.save_adjustments_queued(&[(image, exposure(0.5))]).accepted(), 1);
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the writer never entered write()");
        assert!(engine.save_infos.get(&rel).is_some(), "info recorded before the write");

        // Fill past a small cap with synthetic terminal entries, then prune.
        let cap = 4usize;
        let info = engine.images()[0].clone();
        for index in 0..(cap as u64 + 4) {
            engine.save_infos.insert(format!("synthetic/{index}.png"), info.clone());
        }
        let removed = engine.save_infos.prune(cap, engine.save_queue.as_deref());
        assert_eq!(removed, 4, "only the oldest terminal entries are dropped");
        assert_eq!(
            engine.save_infos.len(),
            cap + 1,
            "the in-flight path plus the newest cap terminal entries survive"
        );
        assert!(engine.save_infos.get(&rel).is_some(), "an in-flight path is never pruned");
        assert!(engine.save_infos.get("synthetic/0.png").is_none(), "oldest dropped");
        assert!(
            engine.save_infos.get(&format!("synthetic/{}.png", cap + 3)).is_some(),
            "newest retained"
        );

        let _ = release_tx.send(());
        drop(release_tx);
        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert_eq!(report.state_of(&rel), Some(&PathState::Saved), "{report:?}");
        // Within the cap the terminal entry is retained; a zero cap removes it.
        assert!(engine.save_infos.get(&rel).is_some());
        assert_eq!(engine.save_infos.prune(0, engine.save_queue.as_deref()), cap + 1);
        assert_eq!(engine.save_infos.len(), 0);

        // A retry re-inserts the metadata before its new revision is written,
        // so the writer still finds it after a prune.
        let retry = engine.save_adjustments_queued(&[(image, exposure(0.25))]);
        assert_eq!(retry.accepted(), 1, "{retry:?}");
        assert!(engine.save_infos.get(&rel).is_some(), "the retry re-recorded the info");
        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert_eq!(report.state_of(&rel), Some(&PathState::Saved), "{report:?}");

        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn queued_unknown_image_is_reported_and_does_not_panic() {
        let dir = queued_temp_dir("unknown");
        queued_write_png(&dir, "one.png", 1);
        let engine = FileEngine::open(&dir).unwrap();

        let summary = engine.save_adjustments_queued(&[(ImageId(9_999), exposure(1.0))]);
        assert_eq!(summary.failed(), 1, "{summary:?}");
        assert_eq!(summary.accepted(), 0, "{summary:?}");
        assert!(summary.first_failure().is_some_and(|failure| failure.path.is_empty()));

        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synchronous_batch_report_stays_off_the_queue() {
        let dir = queued_temp_dir("sync");
        queued_write_png(&dir, "one.png", 1);
        let engine = FileEngine::open(&dir).unwrap();

        let rel = "one.png".to_string();
        let image = id_by_path(&engine, &rel);
        let adj = exposure(1.0);
        let report = engine.save_adjustments_batch_report(&[(image, adj.clone())]);
        assert_eq!(report.saved(), 1, "{report:?}");

        // The synchronous path wrote through immediately and never touched the
        // bounded queue.
        let store = engine.store.as_ref().unwrap();
        assert_eq!(store.load_adjustments(&rel).unwrap(), adj);
        let status = engine.save_queue_status().expect("queue");
        assert_eq!(status.pending, 0, "{status:?}");
        assert_eq!(status.in_flight, 0, "{status:?}");
        assert!(status.is_idle(), "{status:?}");

        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn queue_status_and_flush_stay_responsive_while_the_writer_blocks() {
        let dir = queued_temp_dir("responsive");
        queued_write_png(&dir, "one.png", 1);
        let mut engine = FileEngine::open(&dir).unwrap();

        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(WriteGate {
            entered: entered_tx,
            release: std::sync::Mutex::new(release_rx),
        });
        let writer = RecordingWriter::gated(gate);
        engine.install_test_save_queue(writer, 8);

        let rel = "one.png".to_string();
        let image = id_by_path(&engine, &rel);
        assert_eq!(engine.save_adjustments_queued(&[(image, exposure(0.5))]).accepted(), 1);
        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the writer never entered write()");

        // If the engine held the queue lock across I/O, these calls would never
        // return; the watchdog makes that a clear failure rather than a hang.
        let (probe_tx, probe_rx) = std::sync::mpsc::channel();
        let rel_probe = rel.clone();
        let observed = std::thread::scope(|scope| {
            let engine_ref = &engine;
            scope.spawn(move || {
                let status = engine_ref.save_queue_status().expect("queue");
                let report = engine_ref.flush_saves(Duration::from_millis(20)).expect("queue");
                let _ = probe_tx
                    .send((status.in_flight, report.timed_out, report.state_of(&rel_probe).cloned()));
            });
            probe_rx.recv_timeout(Duration::from_secs(5))
        });

        // Always release before asserting so a failure cannot leave the writer
        // blocked.
        let _ = release_tx.send(());
        drop(release_tx);
        let (in_flight, timed_out, state) = observed.expect(
            "save_queue_status()/flush_saves() blocked while the writer was inside write()",
        );
        assert_eq!(in_flight, 1);
        assert!(timed_out, "flush must return at its deadline, not wait for I/O");
        assert_eq!(state, Some(PathState::NotWritten));

        let report = engine.flush_saves(Duration::from_secs(10)).expect("queue");
        assert_eq!(report.state_of(&rel), Some(&PathState::Saved), "{report:?}");

        drop(engine);
        std::fs::remove_dir_all(&dir).ok();
    }
}
