//! Bounded per-session frame cache keyed by [`RenderPlan`] identity, ROI,
//! precision, color space and backend equivalence tag.
//!
//! There is no process-global `static` here: each [`super::PlanExecutor`] (or
//! test) owns its own [`FrameCache`] instance with an explicit byte budget and
//! deterministic LRU eviction.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::render_plan::{BackendTag, CacheColorSpace, CachePrecision, NodeCacheKey};
use crate::NormRect;

use super::execute::RenderOutput;

/// Default per-session frame cache byte budget (128 MiB).
pub const DEFAULT_FRAME_CACHE_BUDGET_BYTES: usize = 128 * 1024 * 1024;

/// Stable key for one cached rendered frame.
///
/// Two requests share a cache entry only when every pixel-affecting dimension
/// matches: the compiled [`RenderPlan`] node key (which already incorporates
/// source identity, active subgraph topology, consumed parameter subsets, ROI,
/// color space, precision, algorithm version and backend equivalence), the
/// normalized region bits, target size, bypass/overlay/display flags, and
/// backend equivalence tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameCacheKey {
    /// Compiled plan output key (`NodeGraph::OUTPUT` with active subgraph folded
    /// into `revision`).
    pub plan_key: NodeCacheKey,
    /// Canonicalized bit patterns of `[x, y, width, height]` from `NormRect`.
    pub region_bits: [u32; 4],
    /// Output pixel ROI `[x, y, width, height]`.
    pub roi: [u32; 4],
    /// Requested maximum output dimension.
    pub max_size: u32,
    /// Output storage precision (8-bit preview vs 16-bit export).
    pub precision: CachePrecision,
    /// Output color space encoding.
    pub color_space: CacheColorSpace,
    /// Equivalence class from [`BackendTag::equivalence`]. Backends with
    /// different equivalence tags never share entries.
    pub backend_equivalence: u16,
    /// Whether adjustments were bypassed.
    pub bypass: bool,
    /// Optional mask overlay layer id.
    pub mask_overlay: Option<u64>,
    /// Whether display BGRA channel order was requested.
    pub want_bgra: bool,
    /// Whether half-res upsampled fast preview was requested.
    pub fast: bool,
}

impl FrameCacheKey {
    /// Construct a key from a compiled plan key and request parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plan_key: NodeCacheKey,
        region: NormRect,
        roi: [u32; 4],
        max_size: u32,
        precision: CachePrecision,
        color_space: CacheColorSpace,
        backend: BackendTag,
        bypass: bool,
        mask_overlay: Option<u64>,
        want_bgra: bool,
        fast: bool,
    ) -> Self {
        Self {
            plan_key,
            region_bits: [
                canonical_f32_bits(region.x),
                canonical_f32_bits(region.y),
                canonical_f32_bits(region.width),
                canonical_f32_bits(region.height),
            ],
            roi,
            max_size,
            precision,
            color_space,
            backend_equivalence: backend.equivalence,
            bypass,
            mask_overlay,
            want_bgra,
            fast,
        }
    }
}

#[inline]
fn canonical_f32_bits(v: f32) -> u32 {
    if v == 0.0 {
        0.0_f32.to_bits()
    } else if v.is_nan() {
        f32::NAN.to_bits()
    } else {
        v.to_bits()
    }
}

/// Snapshot of [`FrameCache`] occupancy and counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameCacheStats {
    pub entries: usize,
    pub used_bytes: usize,
    pub budget_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

struct CacheEntry {
    output: Arc<RenderOutput>,
    bytes: usize,
}

/// Per-session bounded LRU frame cache.
///
/// Entries are evicted least-recently-used first whenever `used_bytes` would
/// exceed `budget_bytes`. An entry larger than `budget_bytes` is never stored
/// (and does not evict existing entries).
pub struct FrameCache {
    budget_bytes: usize,
    used_bytes: usize,
    entries: HashMap<FrameCacheKey, CacheEntry>,
    /// Front is least-recently-used; back is most-recently-used.
    lru: VecDeque<FrameCacheKey>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl Default for FrameCache {
    fn default() -> Self {
        Self::new(DEFAULT_FRAME_CACHE_BUDGET_BYTES)
    }
}

impl FrameCache {
    /// Create a new frame cache with the given byte budget.
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            lru: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up a cached frame, promoting it to most-recently-used on hit.
    pub fn get(&mut self, key: &FrameCacheKey) -> Option<Arc<RenderOutput>> {
        if let Some(entry) = self.entries.get(key) {
            let output = Arc::clone(&entry.output);
            self.touch(key);
            self.hits = self.hits.saturating_add(1);
            Some(output)
        } else {
            self.misses = self.misses.saturating_add(1);
            None
        }
    }

    /// Check whether `key` is currently cached without changing LRU order or
    /// hit/miss counters.
    pub fn contains(&self, key: &FrameCacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Insert a frame into the cache, evicting LRU entries until within budget.
    ///
    /// Returns `true` if the entry was stored, or `false` if it was empty or
    /// exceeded `budget_bytes`.
    pub fn insert(&mut self, key: FrameCacheKey, output: Arc<RenderOutput>) -> bool {
        let bytes = output.byte_len();
        if bytes == 0 || bytes > self.budget_bytes {
            return false;
        }

        if let Some(existing) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(existing.bytes);
            if let Some(pos) = self.lru.iter().position(|k| k == &key) {
                self.lru.remove(pos);
            }
        }

        while self.used_bytes.saturating_add(bytes) > self.budget_bytes {
            let Some(oldest_key) = self.lru.pop_front() else {
                break;
            };
            if let Some(evicted) = self.entries.remove(&oldest_key) {
                self.used_bytes = self.used_bytes.saturating_sub(evicted.bytes);
                self.evictions = self.evictions.saturating_add(1);
            }
        }

        self.entries.insert(key, CacheEntry { output, bytes });
        self.lru.push_back(key);
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        true
    }

    /// Clear all cached frames and reset byte usage (counters are preserved).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.used_bytes = 0;
    }

    /// Number of cached frames currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total pixel bytes currently held by cached frames.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Configured maximum byte budget.
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Current occupancy and hit/miss/eviction counters.
    pub fn stats(&self) -> FrameCacheStats {
        FrameCacheStats {
            entries: self.entries.len(),
            used_bytes: self.used_bytes,
            budget_bytes: self.budget_bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }

    fn touch(&mut self, key: &FrameCacheKey) {
        if let Some(pos) = self.lru.iter().position(|k| k == key) {
            if pos + 1 != self.lru.len() {
                if let Some(k) = self.lru.remove(pos) {
                    self.lru.push_back(k);
                }
            }
        }
    }
}
