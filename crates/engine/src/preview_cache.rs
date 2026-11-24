//! Frame cache for interactive previews.
//!
//! The pipeline is deterministic, so the same image, resolved parameters,
//! region and output size always produce the same bytes. A slider drag that
//! lands on a value already rendered (or a repaint of an unchanged frame) can
//! reuse the pixels instead of running the pipeline again.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

use crate::{AdjustmentSet, Histogram, ImageId, NormRect, RgbaImage};

/// Bumped by the RAW cache when the bytes behind a source path change. Every
/// cache drops its frames on the next access, so a rendered frame from the
/// replaced file can never be served again.
static SOURCE_EPOCH: AtomicU64 = AtomicU64::new(0);

pub(crate) fn invalidate_source_identity() {
    SOURCE_EPOCH.fetch_add(1, Ordering::Relaxed);
}

/// Serialises tests that inspect a cache with tests (in other modules) that
/// replace a source file and therefore bump [`SOURCE_EPOCH`].
#[cfg(test)]
pub(crate) fn test_source_epoch_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
pub(crate) fn source_epoch() -> u64 {
    SOURCE_EPOCH.load(Ordering::Relaxed)
}

/// Keeps every frame the viewer is likely to bounce between (drag back and
/// forth, before/after, resize).
pub(crate) const CAPACITY: usize = 8;
/// Eight 8192 px finals would be ~1.4 GB, so the cache also has a byte budget.
const MAX_BYTES: usize = 256 << 20;

/// Bump this whenever a change to decoding or rendering alters output pixels.
/// Cached frames outlive rebuilds, so without a bump an upgraded engine would
/// keep serving frames produced by the old algorithm.
const RENDER_VERSION: u32 = 1;

/// Everything that changes the rendered bytes. `generation` is deliberately
/// absent: a redraw of the same content must hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PreviewKey {
    build: u64,
    source_snapshot: Option<u64>,
    image: ImageId,
    params: u64,
    long_side: u32,
    region: [u32; 4],
    bypass: bool,
    mask_overlay: Option<u64>,
    fast: bool,
}

impl PreviewKey {
    /// Separate frames made from different in-memory source resolutions.
    pub(crate) fn with_source(mut self, snapshot: u64) -> Self {
        self.source_snapshot = Some(snapshot);
        self
    }
}

/// One cached frame. The pixel buffer is shared, so a hit costs an `Arc` clone.
#[derive(Clone)]
pub(crate) struct CachedFrame {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Arc<Vec<u8>>,
    pub(crate) histogram: Histogram,
}

impl CachedFrame {
    pub(crate) fn into_image(self) -> RgbaImage {
        RgbaImage::from_shared(self.width, self.height, self.pixels)
    }
}

/// LRU of rendered frames, most recently used last.
pub(crate) struct PreviewCache {
    entries: Vec<(PreviewKey, CachedFrame)>,
    capacity: usize,
    max_bytes: usize,
    bytes: usize,
    /// [`SOURCE_EPOCH`] at the last access; a newer epoch clears the cache.
    source_epoch: u64,
}

impl Default for PreviewCache {
    fn default() -> Self {
        Self::new(CAPACITY)
    }
}

impl PreviewCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self::with_budget(capacity, MAX_BYTES)
    }

    fn with_budget(capacity: usize, max_bytes: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            max_bytes,
            bytes: 0,
            source_epoch: SOURCE_EPOCH.load(Ordering::Relaxed),
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Drops every frame when a source file's bytes changed since the last
    /// access. Cheap to call on every hit.
    fn sync_source_epoch(&mut self) {
        let epoch = SOURCE_EPOCH.load(Ordering::Relaxed);
        if epoch != self.source_epoch {
            self.source_epoch = epoch;
            self.entries.clear();
            self.bytes = 0;
        }
    }

    pub(crate) fn get(&mut self, key: PreviewKey) -> Option<CachedFrame> {
        self.sync_source_epoch();
        let pos = self.entries.iter().position(|(k, _)| *k == key)?;
        let (key, frame) = self.entries.remove(pos);
        self.entries.push((key, frame.clone()));
        Some(frame)
    }

    pub(crate) fn insert(&mut self, key: PreviewKey, frame: CachedFrame) {
        self.sync_source_epoch();
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.bytes -= self.entries.remove(pos).1.pixels.len();
        }
        self.bytes += frame.pixels.len();
        self.entries.push((key, frame));
        while self.entries.len() > self.capacity || (self.bytes > self.max_bytes && self.entries.len() > 1) {
            self.bytes -= self.entries.remove(0).1.pixels.len();
        }
    }

    /// Drops every frame, for when the engine build behind a long-lived
    /// instance changes.
    #[allow(dead_code)] // callers live outside this crate-internal module
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }
}

/// Key for one preview request. `max_size` must already be the effective
/// output size (Draft clamps it), so a resize that keeps the output size hits.
pub(crate) fn key(
    image: ImageId,
    adj: &AdjustmentSet,
    region: NormRect,
    max_size: u32,
    bypass: bool,
    mask_overlay: Option<u64>,
    fast: bool,
) -> PreviewKey {
    PreviewKey {
        build: build_fingerprint(),
        source_snapshot: None,
        image,
        params: params_fingerprint(adj),
        long_side: max_size,
        region: [region.x.to_bits(), region.y.to_bits(), region.width.to_bits(), region.height.to_bits()],
        bypass,
        mask_overlay,
        fast,
    }
}

/// FNV-1a over the serialized adjustment set: stable across runs and Rust
/// versions, and no per-call `Debug` formatting. A near-miss on one slider
/// changes the bytes, so it always misses.
pub(crate) fn params_fingerprint(adj: &AdjustmentSet) -> u64 {
    struct Fnv(u64);
    impl std::io::Write for Fnv {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            for byte in bytes {
                self.0 ^= *byte as u64;
                self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut hash = Fnv(0xcbf2_9ce4_8422_2325u64);
    serde_json::to_writer(&mut hash, adj).unwrap_or_default();
    if let Some(producer) = adj.portrait_producer_inputs.as_deref() {
        for byte in producer.fingerprint().to_le_bytes() {
            hash.0 ^= byte as u64;
            hash.0 = hash.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash.0
}

/// Package version plus an optional build id, so a rebuilt engine never reuses
/// frames rendered by the previous build.
fn build_fingerprint() -> u64 {
    static FINGERPRINT: OnceLock<u64> = OnceLock::new();
    *FINGERPRINT.get_or_init(|| {
        let id = option_env!("AURORA_BUILD_ID").unwrap_or(env!("CARGO_PKG_VERSION"));
        build_fingerprint_of(id, RENDER_VERSION)
    })
}

fn build_fingerprint_of(id: &str, render_version: u32) -> u64 {
    let bytes = format!("{id}\0{render_version}");
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests that touch the process-wide source epoch run one at a time: a bump
    /// in one test must not clear a cache another test is inspecting.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        test_source_epoch_lock()
    }

    fn key_for(image: u64, exposure: f32) -> PreviewKey {
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = exposure;
        key(ImageId(image), &adj, NormRect::FULL, 1400, false, None, false)
    }

    #[test]
    fn streaming_params_fingerprint_matches_serialized_bytes() {
        let _serial = serial();
        fn reference(adj: &AdjustmentSet) -> u64 {
            let bytes = serde_json::to_vec(adj).unwrap();
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for byte in bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            hash
        }
        let mut exposure = AdjustmentSet::default();
        exposure.exposure.exposure = 0.735;
        exposure.extra.insert("test".into(), 0.4);
        let mut layered = AdjustmentSet::default();
        layered.layers.push(crate::Layer {
            id: 7,
            mask: crate::Mask {
                shapes: vec![crate::MaskShape::Linear(crate::LinearGradient {
                    op: crate::MaskOp::Add,
                    start: [0.2, 0.3],
                    end: [0.8, 0.7],
                })],
                ..Default::default()
            },
            ..Default::default()
        });
        for adj in [AdjustmentSet::default(), exposure, layered] {
            assert_eq!(params_fingerprint(&adj), reference(&adj));
        }
    }

    fn frame(tag: u8) -> CachedFrame {
        CachedFrame {
            width: 2,
            height: 2,
            pixels: Arc::new(vec![tag; 16]),
            histogram: Histogram::default(),
        }
    }

    #[test]
    fn one_parameter_change_misses() {
        let _serial = serial();
        assert_eq!(key_for(1, 0.5), key_for(1, 0.5));
        assert_ne!(key_for(1, 0.5), key_for(1, 0.500_001));
        assert_ne!(key_for(1, 0.5), key_for(2, 0.5));
    }

    #[test]
    fn lru_evicts_the_least_recently_used() {
        let _serial = serial();
        let mut cache = PreviewCache::new(2);
        cache.insert(key_for(1, 0.0), frame(1));
        cache.insert(key_for(2, 0.0), frame(2));
        // Touch the first one so the second becomes the eviction candidate.
        assert!(cache.get(key_for(1, 0.0)).is_some());
        cache.insert(key_for(3, 0.0), frame(3));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(key_for(2, 0.0)).is_none());
        assert!(cache.get(key_for(1, 0.0)).is_some());
        assert!(cache.get(key_for(3, 0.0)).is_some());
    }

    #[test]
    fn the_byte_budget_evicts_before_the_entry_limit() {
        let _serial = serial();
        let mut cache = PreviewCache::with_budget(8, 20);
        cache.insert(key_for(1, 0.0), frame(1));
        cache.insert(key_for(2, 0.0), frame(2));
        assert_eq!(cache.len(), 1);
        assert!(cache.get(key_for(1, 0.0)).is_none());
        assert!(cache.get(key_for(2, 0.0)).is_some());
    }

    #[test]
    fn the_same_build_fingerprint_hits() {
        let _serial = serial();
        let mut cache = PreviewCache::new(4);
        cache.insert(key_for(1, 0.0), frame(1));
        assert_eq!(key_for(1, 0.0).build, build_fingerprint());
        assert!(cache.get(key_for(1, 0.0)).is_some());
    }

    #[test]
    fn a_different_build_fingerprint_misses() {
        let _serial = serial();
        let mut cache = PreviewCache::new(4);
        let cached = key_for(1, 0.0);
        cache.insert(cached, frame(1));
        let other_build = PreviewKey { build: build_fingerprint_of("9.9.9", RENDER_VERSION), ..cached };
        assert_ne!(other_build.build, cached.build);
        assert!(cache.get(other_build).is_none());
    }

    #[test]
    fn a_source_identity_change_drops_rendered_frames() {
        let _serial = serial();
        let mut cache = PreviewCache::new(4);
        let key = key_for(1, 0.0);
        cache.insert(key, frame(1));
        assert!(cache.get(key).is_some());
        invalidate_source_identity();
        assert!(
            cache.get(key).is_none(),
            "a frame rendered from replaced source bytes must not be served"
        );
        cache.insert(key, frame(2));
        assert_eq!(cache.get(key).unwrap().pixels[0], 2);
    }

    #[test]
    fn the_build_fingerprint_tracks_the_version_text() {
        let _serial = serial();
        assert_eq!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.3", 1));
        assert_ne!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.4", 1));
        assert_ne!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.3", 2));
    }

    #[test]
    fn clear_drops_every_frame() {
        let _serial = serial();
        let mut cache = PreviewCache::new(4);
        cache.insert(key_for(1, 0.0), frame(1));
        cache.insert(key_for(2, 0.0), frame(2));
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get(key_for(1, 0.0)).is_none());
        assert!(cache.get(key_for(2, 0.0)).is_none());
    }
}
