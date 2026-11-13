//! Frame cache for interactive previews.
//!
//! The pipeline is deterministic, so the same image, resolved parameters,
//! region and output size always produce the same bytes. A slider drag that
//! lands on a value already rendered (or a repaint of an unchanged frame) can
//! reuse the pixels instead of running the pipeline again.

use std::sync::Arc;
use std::sync::OnceLock;

use crate::{AdjustmentSet, Histogram, ImageId, NormRect, RgbaImage};

/// Keeps every frame the viewer is likely to bounce between (drag back and
/// forth, before/after, resize).
pub(crate) const CAPACITY: usize = 8;
/// Eight 8192 px finals would be ~1.4 GB, so the cache also has a byte budget.
const MAX_BYTES: usize = 256 << 20;

/// Bump this whenever a change to decoding or rendering alters output pixels.
/// Cached frames outlive rebuilds, so without a bump an upgraded engine would
/// keep serving frames produced by the old algorithm.
const RENDER_VERSION: u32 = 2;

/// Everything that changes the rendered bytes. `generation` is deliberately
/// absent: a redraw of the same content must hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PreviewKey {
    build: u64,
    image: ImageId,
    params: u64,
    long_side: u32,
    region: [u32; 4],
    bypass: bool,
    mask_overlay: Option<u64>,
    fast: bool,
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
        Self { entries: Vec::with_capacity(capacity), capacity, max_bytes, bytes: 0 }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn get(&mut self, key: PreviewKey) -> Option<CachedFrame> {
        let pos = self.entries.iter().position(|(k, _)| *k == key)?;
        let (key, frame) = self.entries.remove(pos);
        self.entries.push((key, frame.clone()));
        Some(frame)
    }

    pub(crate) fn insert(&mut self, key: PreviewKey, frame: CachedFrame) {
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
fn params_fingerprint(adj: &AdjustmentSet) -> u64 {
    let bytes = serde_json::to_vec(adj).unwrap_or_default();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Package version plus an optional build id, so a rebuilt engine never reuses
/// frames rendered by the previous build.
fn build_fingerprint() -> u64 {
    static FINGERPRINT: OnceLock<u64> = OnceLock::new();
    *FINGERPRINT.get_or_init(|| {
        let id = option_env!("LUMEN_BUILD_ID").unwrap_or(env!("CARGO_PKG_VERSION"));
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

    fn key_for(image: u64, exposure: f32) -> PreviewKey {
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = exposure;
        key(ImageId(image), &adj, NormRect::FULL, 1400, false, None, false)
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
        assert_eq!(key_for(1, 0.5), key_for(1, 0.5));
        assert_ne!(key_for(1, 0.5), key_for(1, 0.500_001));
        assert_ne!(key_for(1, 0.5), key_for(2, 0.5));
    }

    #[test]
    fn lru_evicts_the_least_recently_used() {
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
        let mut cache = PreviewCache::with_budget(8, 20);
        cache.insert(key_for(1, 0.0), frame(1));
        cache.insert(key_for(2, 0.0), frame(2));
        assert_eq!(cache.len(), 1);
        assert!(cache.get(key_for(1, 0.0)).is_none());
        assert!(cache.get(key_for(2, 0.0)).is_some());
    }

    #[test]
    fn the_same_build_fingerprint_hits() {
        let mut cache = PreviewCache::new(4);
        cache.insert(key_for(1, 0.0), frame(1));
        assert_eq!(key_for(1, 0.0).build, build_fingerprint());
        assert!(cache.get(key_for(1, 0.0)).is_some());
    }

    #[test]
    fn a_different_build_fingerprint_misses() {
        let mut cache = PreviewCache::new(4);
        let cached = key_for(1, 0.0);
        cache.insert(cached, frame(1));
        let other_build = PreviewKey { build: build_fingerprint_of("9.9.9", RENDER_VERSION), ..cached };
        assert_ne!(other_build.build, cached.build);
        assert!(cache.get(other_build).is_none());
    }

    #[test]
    fn the_build_fingerprint_tracks_the_version_text() {
        assert_eq!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.3", 1));
        assert_ne!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.4", 1));
        assert_ne!(build_fingerprint_of("1.2.3", 1), build_fingerprint_of("1.2.3", 2));
    }

    #[test]
    fn clear_drops_every_frame() {
        let mut cache = PreviewCache::new(4);
        cache.insert(key_for(1, 0.0), frame(1));
        cache.insert(key_for(2, 0.0), frame(2));
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get(key_for(1, 0.0)).is_none());
        assert!(cache.get(key_for(2, 0.0)).is_none());
    }
}
