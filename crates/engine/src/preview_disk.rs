//! Disk-backed preview placeholder for re-opened images and app restarts.
//!
//! The last settled (`RenderQuality::Final`, full-frame) preview of an image is
//! written to a bounded on-disk cache ([`crate::disk_preview_cache`]). When the
//! same image is selected again with the same adjustments, output size and
//! engine build, the stored pixels are shown at once while the real render
//! runs.
//!
//! **A disk hit is only ever a placeholder.** The key covers the source file
//! (path, length, mtime), the full adjustment fingerprint, region, output size,
//! bypass flag and the engine build/algorithm version, but it cannot cover
//! producer/runtime inputs, AI mask caches, or whether the GPU or the CPU
//! numeric path rendered the stored frame. So a hit is delivered through the
//! progressive first-frame hook (`quick_cold_raw_preview`), which the app
//! installs only while nothing newer is displayed, and the real render always
//! follows and replaces it. A hit is never cached in RAM, never marked final.
//!
//! Layout and policy:
//! * directory: per-platform cache dir next to the RAW cache
//!   (`%LOCALAPPDATA%\Aurora Studio\Cache\preview`, `~/Library/Caches/Aurora
//!   Studio/preview`, `$XDG_CACHE_HOME|~/.cache/aurora-studio/preview`);
//!   `AURORA_PREVIEW_CACHE_DIR` overrides it, `AURORA_DISABLE_PREVIEW_DISK_CACHE`
//!   turns the cache off;
//! * budget: `AURORA_PREVIEW_CACHE_MB` (default 512, `0` disables), LRU eviction
//!   by the SQLite index in [`crate::disk_preview_cache`];
//! * writes are staged in a temp file and renamed into place; every blob carries
//!   a header and a checksum, and a blob that fails either is a miss.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::disk_preview_cache::{
    DiskPreviewCache, DiskPreviewCacheConfig, PyramidLevelKind, PyramidLevelWrite,
};
use crate::{AdjustmentSet, NormRect};

const MAGIC: [u8; 4] = *b"APV1";
const HEADER_LEN: usize = 4 + 4 + 4 + 8;
const DEFAULT_BUDGET_MB: u64 = 512;

/// Everything that names one stored frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiskKey {
    image_id: String,
    source_fingerprint: String,
    settings_digest: String,
    long_side: u32,
}

impl DiskKey {
    /// `None` when the request has no stable name (mask overlay) or the source
    /// file cannot be stat'ed.
    pub(crate) fn new(
        path: &Path,
        adj: &AdjustmentSet,
        region: NormRect,
        max_size: u32,
        bypass: bool,
        mask_overlay: Option<u64>,
        fast: bool,
    ) -> Option<Self> {
        let settings_digest =
            crate::preview_cache::persistent_digest(adj, region, max_size, bypass, mask_overlay, fast)?;
        let meta = std::fs::metadata(path).ok()?;
        let mtime_nanos = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
        Some(Self {
            image_id: path.to_string_lossy().into_owned(),
            source_fingerprint: format!("{}:{mtime_nanos}", meta.len()),
            settings_digest,
            long_side: max_size,
        })
    }
}

/// One verified frame read back from disk (RGBA8, sRGB, pre display transform).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DiskFrame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) rgba: Vec<u8>,
}

pub(crate) struct PreviewDisk {
    cache: DiskPreviewCache,
    max_bytes: u64,
}

impl PreviewDisk {
    pub(crate) fn open(dir: PathBuf, max_bytes: u64) -> Option<Self> {
        if max_bytes == 0 {
            return None;
        }
        match DiskPreviewCache::open(DiskPreviewCacheConfig::new(dir, max_bytes)) {
            Ok(cache) => Some(Self { cache, max_bytes }),
            Err(err) => {
                log::warn!("preview disk cache unavailable: {err:#}");
                None
            }
        }
    }

    /// Verified lookup. Any mismatch (size, header, checksum) is a miss.
    pub(crate) fn lookup(&self, key: &DiskKey) -> Option<DiskFrame> {
        let level = self
            .cache
            .get(
                &key.image_id,
                &key.settings_digest,
                &key.source_fingerprint,
                PyramidLevelKind::Standard,
            )
            .ok()??;
        if level.long_side != key.long_side {
            return None;
        }
        let blob = level.data;
        if blob.len() < HEADER_LEN || blob[..4] != MAGIC {
            return None;
        }
        let width = u32::from_le_bytes(blob[4..8].try_into().ok()?);
        let height = u32::from_le_bytes(blob[8..12].try_into().ok()?);
        let checksum = u64::from_le_bytes(blob[12..20].try_into().ok()?);
        let pixels = &blob[HEADER_LEN..];
        let expected = (width as usize).checked_mul(height as usize)?.checked_mul(4)?;
        if width == 0
            || height == 0
            || pixels.len() != expected
            || (width, height) != (level.width, level.height)
            || pixel_checksum(pixels) != checksum
        {
            return None;
        }
        Some(DiskFrame { width, height, rgba: pixels.to_vec() })
    }

    /// Stores one frame; false when it was skipped or the write failed.
    pub(crate) fn store(&self, key: &DiskKey, width: u32, height: u32, rgba: &[u8]) -> bool {
        let Some(expected) = (width as usize).checked_mul(height as usize).and_then(|n| n.checked_mul(4))
        else {
            return false;
        };
        // A frame that would take over most of the budget evicts everything
        // else for one entry; skip it.
        if width == 0 || height == 0 || rgba.len() != expected || (rgba.len() as u64) > self.max_bytes / 2 {
            return false;
        }
        let mut blob = Vec::with_capacity(HEADER_LEN + rgba.len());
        blob.extend_from_slice(&MAGIC);
        blob.extend_from_slice(&width.to_le_bytes());
        blob.extend_from_slice(&height.to_le_bytes());
        blob.extend_from_slice(&pixel_checksum(rgba).to_le_bytes());
        blob.extend_from_slice(rgba);
        let result = self.cache.put(PyramidLevelWrite {
            image_uuid: &key.image_id,
            settings_digest: &key.settings_digest,
            source_fingerprint: &key.source_fingerprint,
            level_kind: PyramidLevelKind::Standard,
            long_side: key.long_side,
            width,
            height,
            data: &blob,
        });
        match result {
            Ok(()) => true,
            Err(err) => {
                log::debug!("preview disk cache write failed: {err:#}");
                false
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn inner(&self) -> &DiskPreviewCache {
        &self.cache
    }
}

/// Word-at-a-time checksum; corruption detection, not cryptography.
fn pixel_checksum(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ bytes.len() as u64;
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        hash = (hash ^ word).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(29);
    }
    for &b in chunks.remainder() {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn default_dir() -> Option<PathBuf> {
    if std::env::var_os("AURORA_DISABLE_PREVIEW_DISK_CACHE").is_some() {
        return None;
    }
    if let Some(dir) = std::env::var_os("AURORA_PREVIEW_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .map(|dir| dir.join("Aurora Studio").join("Cache").join("preview"))
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME").map(PathBuf::from).map(|dir| {
            dir.join("Library").join("Caches").join("Aurora Studio").join("preview")
        })
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|dir| PathBuf::from(dir).join(".cache")))
            .map(|dir| dir.join("aurora-studio").join("preview"))
    }
}

fn budget_bytes() -> u64 {
    std::env::var("AURORA_PREVIEW_CACHE_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_BUDGET_MB)
        .saturating_mul(1024 * 1024)
}

/// The process-wide cache, opened on first use. Under `cfg(test)` it is off
/// unless `AURORA_PREVIEW_CACHE_DIR` is set before the first call, so engine
/// tests never write into the developer's real cache directory.
pub(crate) fn global() -> Option<&'static PreviewDisk> {
    static DISK: OnceLock<Option<PreviewDisk>> = OnceLock::new();
    DISK.get_or_init(|| {
        if cfg!(test) && std::env::var_os("AURORA_PREVIEW_CACHE_DIR").is_none() {
            return None;
        }
        PreviewDisk::open(default_dir()?, budget_bytes())
    })
    .as_ref()
}

/// Writes `rgba` on a background thread. At most one write is in flight; a
/// frame that arrives meanwhile is dropped (the next settled frame replaces it),
/// so a burst of settles never queues pixel copies.
pub(crate) fn store_async(disk: &'static PreviewDisk, key: DiskKey, width: u32, height: u32, rgba: Arc<Vec<u8>>) {
    static BUSY: AtomicBool = AtomicBool::new(false);
    if BUSY.swap(true, Ordering::AcqRel) {
        return;
    }
    let spawned = std::thread::Builder::new().name("preview-disk-write".into()).spawn(move || {
        disk.store(&key, width, height, &rgba);
        BUSY.store(false, Ordering::Release);
    });
    if spawned.is_err() {
        BUSY.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "aurora-preview-disk-{name}-{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn frame(w: u32, h: u32, seed: u8) -> Vec<u8> {
        (0..w * h * 4).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed)).collect()
    }

    fn key_for(path: &Path, adj: &AdjustmentSet, max: u32) -> DiskKey {
        DiskKey::new(path, adj, NormRect::FULL, max, false, None, false).expect("key")
    }

    fn source_file(dir: &Path, bytes: &[u8]) -> PathBuf {
        let path = dir.join("source.png");
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn roundtrip_is_byte_identical_and_survives_reopen() {
        let scratch = Scratch::new("roundtrip");
        let src = source_file(&scratch.0, b"source-v1");
        let adj = AdjustmentSet::default();
        let key = key_for(&src, &adj, 64);
        let pixels = frame(16, 8, 7);
        {
            let disk = PreviewDisk::open(scratch.0.join("cache"), 8 << 20).unwrap();
            assert!(disk.lookup(&key).is_none(), "cold cache is a miss");
            assert!(disk.store(&key, 16, 8, &pixels));
            let hit = disk.lookup(&key).expect("hit");
            assert_eq!((hit.width, hit.height), (16, 8));
            assert_eq!(hit.rgba, pixels);
        }
        // "App restart": a new instance over the same directory.
        let disk = PreviewDisk::open(scratch.0.join("cache"), 8 << 20).unwrap();
        assert_eq!(disk.lookup(&key).expect("hit after reopen").rgba, pixels);
    }

    #[test]
    fn any_identity_change_is_a_miss() {
        let scratch = Scratch::new("identity");
        let src = source_file(&scratch.0, b"source-v1");
        let adj = AdjustmentSet::default();
        let disk = PreviewDisk::open(scratch.0.join("cache"), 8 << 20).unwrap();
        let key = key_for(&src, &adj, 64);
        assert!(disk.store(&key, 8, 8, &frame(8, 8, 1)));

        // Different output size, adjustments or bypass: other digests.
        assert!(disk.lookup(&key_for(&src, &adj, 128)).is_none());
        let mut edited = AdjustmentSet::default();
        edited.exposure.exposure = 0.5;
        assert!(disk.lookup(&key_for(&src, &edited, 64)).is_none());
        let bypass = DiskKey::new(&src, &adj, NormRect::FULL, 64, true, None, false).unwrap();
        assert!(disk.lookup(&bypass).is_none());
        // A mask overlay has no stable name.
        assert!(DiskKey::new(&src, &adj, NormRect::FULL, 64, false, Some(9), false).is_none());

        // A replaced source file (different length) invalidates the entry.
        std::fs::write(&src, b"source-v2-longer").unwrap();
        assert!(disk.lookup(&key_for(&src, &adj, 64)).is_none());
        // The old key still names the old file state and the entry is gone too.
        assert!(disk.lookup(&key).is_none());
    }

    #[test]
    fn corrupt_or_truncated_blobs_are_ignored() {
        let scratch = Scratch::new("corrupt");
        let src = source_file(&scratch.0, b"source-v1");
        let adj = AdjustmentSet::default();
        let root = scratch.0.join("cache");
        let disk = PreviewDisk::open(root.clone(), 8 << 20).unwrap();
        let key = key_for(&src, &adj, 64);
        assert!(disk.store(&key, 8, 8, &frame(8, 8, 3)));
        let blobs: Vec<PathBuf> = walk(&root.join("blobs"));
        assert_eq!(blobs.len(), 1);

        // Flip one pixel byte: same length, wrong checksum.
        let mut bytes = std::fs::read(&blobs[0]).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&blobs[0], &bytes).unwrap();
        assert!(disk.lookup(&key).is_none());

        // Truncated blob.
        assert!(disk.store(&key, 8, 8, &frame(8, 8, 3)));
        let blob = walk(&root.join("blobs")).pop().unwrap();
        let bytes = std::fs::read(&blob).unwrap();
        std::fs::write(&blob, &bytes[..bytes.len() / 2]).unwrap();
        assert!(disk.lookup(&key).is_none());

        // Overwritten garbage of the right length.
        assert!(disk.store(&key, 8, 8, &frame(8, 8, 3)));
        let blob = walk(&root.join("blobs")).pop().unwrap();
        let len = std::fs::read(&blob).unwrap().len();
        std::fs::write(&blob, vec![0xabu8; len]).unwrap();
        assert!(disk.lookup(&key).is_none());

        // A fresh store repairs the entry.
        assert!(disk.store(&key, 8, 8, &frame(8, 8, 3)));
        assert_eq!(disk.lookup(&key).expect("repaired").rgba, frame(8, 8, 3));
    }

    fn walk(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    out.push(p);
                }
            }
        }
        out
    }

    #[test]
    fn budget_is_enforced_with_lru_eviction_and_no_temp_files_remain() {
        let scratch = Scratch::new("budget");
        let adj = AdjustmentSet::default();
        // Each frame is 32x32x4 = 4096 B (+20 B header); budget holds ~2.
        let disk = PreviewDisk::open(scratch.0.join("cache"), 10_000).unwrap();
        let sources: Vec<PathBuf> = (0..4)
            .map(|i| {
                let p = scratch.0.join(format!("s{i}.png"));
                std::fs::write(&p, format!("source-{i}")).unwrap();
                p
            })
            .collect();
        let keys: Vec<DiskKey> = sources.iter().map(|p| key_for(p, &adj, 32)).collect();
        for (i, key) in keys.iter().enumerate() {
            assert!(disk.store(key, 32, 32, &frame(32, 32, i as u8)));
            // Touch the first entry so it stays most recently used.
            let _ = disk.lookup(&keys[0]);
        }
        let stats = disk.inner().stats().unwrap();
        assert!(stats.total_bytes <= 10_000, "{stats:?}");
        assert!(disk.lookup(&keys[0]).is_some(), "recently used entry survives");
        assert!(disk.lookup(&keys[1]).is_none(), "least recently used entry is evicted");
        assert!(disk.lookup(&keys[3]).is_some(), "newest entry survives");
        assert!(
            walk(&scratch.0.join("cache").join("blobs"))
                .iter()
                .all(|p| p.extension().is_some_and(|e| e == "bin")),
            "no staging temp files are left behind"
        );
    }

    #[test]
    fn oversized_frames_and_zero_budget_are_skipped() {
        let scratch = Scratch::new("skip");
        let src = source_file(&scratch.0, b"source-v1");
        let key = key_for(&src, &AdjustmentSet::default(), 64);
        assert!(PreviewDisk::open(scratch.0.join("zero"), 0).is_none());
        let disk = PreviewDisk::open(scratch.0.join("small"), 1000).unwrap();
        assert!(!disk.store(&key, 16, 16, &frame(16, 16, 1)), "frame over half the budget");
        assert!(!disk.store(&key, 16, 16, &frame(8, 8, 1)), "length must match dimensions");
        assert!(disk.lookup(&key).is_none());
    }

    #[test]
    fn checksum_detects_single_bit_changes_at_every_position() {
        let base = frame(9, 3, 5);
        let sum = pixel_checksum(&base);
        for i in 0..base.len() {
            let mut changed = base.clone();
            changed[i] ^= 1;
            assert_ne!(pixel_checksum(&changed), sum, "byte {i}");
        }
    }
}
