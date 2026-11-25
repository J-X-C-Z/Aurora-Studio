//! L3 stage cache: the developed display-domain frame, before detail.
//!
//! Design BO-10 §4 C2 puts the only worthwhile node-level cache here: after
//! sampling and the whole tone/colour chain, before noise reduction, sharpening,
//! structure, spots, layers and the portrait stage. Dragging those later stages
//! then reuses the developed frame instead of re-running sampling and tone.
//!
//! The key is the *develop revision* — the adjustment set with every detail-only
//! domain cleared — plus the sampling dependencies the frame is built from
//! (source snapshot, geometry, view region, output size, mip, interpolation,
//! halo, 8/16-bit path). A detail-only change therefore hits; any change the
//! develop chain reads misses.
//!
//! Scope and limits, stated rather than implied:
//! * Only the `use_f32_detail` path is cached. That is the path that carries
//!   noise reduction and sharpening, i.e. the drags this cache exists for.
//!   The 8-bit band path and the GPU two-pass path still render every frame.
//! * The GPU develop variant is part of the key. When it changes (a device
//!   appears or a launch fails) the cache is cleared, so a cached frame can
//!   never stand in for a render that would take a different numeric path.
//! * Entries are capped per frame and in total; a miss is always safe.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
#[cfg(test)]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::sync::Arc;

use crate::AdjustmentSet;

/// One entry larger than this is not retained. Previews are the target; a
/// 32 MiB f32 RGB entry is about 900k pixels, comfortably above 1080p.
pub(crate) const ENTRY_MAX_BYTES: usize = 32 << 20;
/// Total bytes kept across all entries.
pub(crate) const TOTAL_MAX_BYTES: usize = 192 << 20;
/// Hard entry-count cap, so many small frames cannot crowd out the byte budget.
const CAPACITY: usize = 8;
const PIXEL_BYTES: usize = std::mem::size_of::<[f32; 3]>();

/// Everything a developed frame is built from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StageKey {
    /// [`crate::mock::process::Source::snapshot_id`].
    pub(crate) source: u64,
    /// `hash_geometry` of the key geometry.
    pub(crate) geometry: u64,
    /// View region, as float bits.
    pub(crate) region: [u32; 4],
    pub(crate) out_w: u32,
    pub(crate) out_h: u32,
    pub(crate) level: usize,
    pub(crate) cubic: bool,
    /// Detail halo around the core (0 for a full-frame render).
    pub(crate) halo: usize,
    /// 8-bit (false) or 16-bit (true) develop path.
    pub(crate) wide: bool,
    /// The GPU develop variant the frame was filled with.
    pub(crate) gpu: bool,
    /// Develop revision of the adjustment set.
    pub(crate) develop: u64,
}

struct Entry {
    key: StageKey,
    pixels: Arc<Vec<[f32; 3]>>,
}

struct Cache {
    entries: VecDeque<Entry>,
    bytes: usize,
}

impl Cache {
    fn new() -> Self {
        Self { entries: VecDeque::new(), bytes: 0 }
    }
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Cache::new()))
}

/// `AURORA_STAGE_CACHE=0` turns the cache off without a rebuild. The tests use
/// the same switch to produce the "no cache" reference frame.
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if TEST_DISABLED.load(Ordering::Relaxed) {
        return false;
    }
    match std::env::var_os("AURORA_STAGE_CACHE") {
        Some(value) => {
            let value = value.to_string_lossy();
            !(value == "0" || value.eq_ignore_ascii_case("false"))
        }
        None => true,
    }
}

/// Process-wide switch for tests that need an uncached reference frame without
/// touching the process environment (which other tests share).
#[cfg(test)]
static TEST_DISABLED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn set_test_enabled(enabled: bool) {
    TEST_DISABLED.store(!enabled, Ordering::Relaxed);
}

/// Serialises tests that inspect or clear the process-wide cache.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
static HITS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static INSERTED: AtomicU64 = AtomicU64::new(0);

/// A cached developed frame, or `None`.
pub(crate) fn get(key: &StageKey) -> Option<Arc<Vec<[f32; 3]>>> {
    let mut store = cache().lock().unwrap_or_else(|error| error.into_inner());
    let position = store.entries.iter().position(|entry| entry.key == *key)?;
    // Promote to most-recently-used.
    let entry = store.entries.remove(position)?;
    let pixels = Arc::clone(&entry.pixels);
    store.entries.push_back(entry);
    #[cfg(test)]
    HITS.fetch_add(1, Ordering::Relaxed);
    Some(pixels)
}

/// Retains one developed frame. Silently declines frames over the entry cap.
pub(crate) fn insert(key: StageKey, pixels: &[[f32; 3]]) {
    let bytes = pixels.len().saturating_mul(PIXEL_BYTES);
    if bytes == 0 || bytes > ENTRY_MAX_BYTES {
        return;
    }
    let pixels = Arc::new(pixels.to_vec());
    let mut store = cache().lock().unwrap_or_else(|error| error.into_inner());
    if let Some(position) = store.entries.iter().position(|entry| entry.key == key) {
        let old = store.entries.remove(position).expect("index came from the same deque");
        store.bytes = store.bytes.saturating_sub(old.pixels.len().saturating_mul(PIXEL_BYTES));
    }
    store.bytes += bytes;
    store.entries.push_back(Entry { key, pixels });
    while store.entries.len() > CAPACITY
        || (store.bytes > TOTAL_MAX_BYTES && store.entries.len() > 1)
    {
        let Some(old) = store.entries.pop_front() else { break };
        store.bytes = store.bytes.saturating_sub(old.pixels.len().saturating_mul(PIXEL_BYTES));
    }
    #[cfg(test)]
    INSERTED.fetch_add(1, Ordering::Relaxed);
}

/// Drops every entry. Called when the GPU develop path falls back: the cached
/// frames were filled on the other numeric path and must not be reused.
pub(crate) fn clear() {
    let mut store = cache().lock().unwrap_or_else(|error| error.into_inner());
    store.entries.clear();
    store.bytes = 0;
}

#[cfg(test)]
pub(crate) fn stats() -> (u64, u64, usize, usize) {
    let store = cache().lock().unwrap_or_else(|error| error.into_inner());
    (
        HITS.load(Ordering::Relaxed),
        INSERTED.load(Ordering::Relaxed),
        store.entries.len(),
        store.bytes,
    )
}

#[cfg(test)]
pub(crate) fn reset_stats() {
    HITS.store(0, Ordering::Relaxed);
    INSERTED.store(0, Ordering::Relaxed);
}

/// The develop inputs of an adjustment set, by reference.
///
/// Serialising this instead of the whole set is what lets a detail-only drag
/// hit: the detail domains are absent, so the revision cannot move. Building it
/// costs nothing (no clone of layers, masks or spot raster data) and the
/// listing is deliberately explicit — a develop input that is missing here
/// would be a wrong hit, so each one is named and reviewed:
///
/// `build_plan` reads `base`, `white_balance`, `exposure`, `hdr`, `levels`,
/// `curves`, `clarity.clarity`/`method`, `dehaze`, `vignetting`,
/// `color_balance(_gains)`, `color_editor`, `black_white`; the point chain then
/// reads the LCC calibration (`lens.uniform_light*`) and the vignette centre's
/// `lens.optic_center`. Sampling and the banding frame come from `geometry`,
/// which is part of [`StageKey`] itself.
#[derive(serde::Serialize)]
struct DevelopView<'a> {
    base: &'a crate::BaseCharacteristics,
    white_balance: &'a crate::WhiteBalance,
    exposure: &'a crate::Exposure,
    hdr: &'a crate::HighDynamicRange,
    levels: &'a crate::Levels,
    curves: &'a crate::Curves,
    clarity: f32,
    clarity_method: crate::ClarityMethod,
    dehaze: &'a crate::Dehaze,
    vignetting: &'a crate::Vignetting,
    color_balance: &'a crate::ColorBalance,
    color_balance_gains: &'a Option<crate::ColorBalanceGains>,
    color_editor: &'a crate::ColorEditor,
    black_white: &'a crate::BlackWhite,
    uniform_light: f32,
    uniform_light_enabled: bool,
    optic_center: &'a Option<[f32; 2]>,
}

impl<'a> DevelopView<'a> {
    fn of(adj: &'a AdjustmentSet) -> Self {
        Self {
            base: &adj.base,
            white_balance: &adj.white_balance,
            exposure: &adj.exposure,
            hdr: &adj.hdr,
            levels: &adj.levels,
            curves: &adj.curves,
            clarity: adj.clarity.clarity,
            clarity_method: adj.clarity.method,
            dehaze: &adj.dehaze,
            vignetting: &adj.vignetting,
            color_balance: &adj.color_balance,
            color_balance_gains: &adj.color_balance_gains,
            color_editor: &adj.color_editor,
            black_white: &adj.black_white,
            uniform_light: adj.lens.uniform_light,
            uniform_light_enabled: adj.lens.uniform_light_enabled,
            optic_center: &adj.lens.optic_center,
        }
    }
}

/// FNV-1a over the develop view.
pub(crate) fn develop_revision(adj: &AdjustmentSet) -> u64 {
    struct Fnv(u64);

    impl std::io::Write for Fnv {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            for byte in bytes {
                self.0 ^= u64::from(*byte);
                self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let mut hash = Fnv(0xcbf2_9ce4_8422_2325u64);
    serde_json::to_writer(&mut hash, &DevelopView::of(adj)).unwrap_or_default();
    // `uniform_light_grid` is `#[serde(skip)]`, so the serialized view cannot
    // see it — yet its LCC gain is part of the develop point chain. Fold the
    // calibration in explicitly.
    if let Some(grid) = adj.lens.uniform_light_grid.as_deref() {
        let mut grid_hash = Hash64(hash.0);
        grid.header.hash(&mut grid_hash);
        grid.columns.hash(&mut grid_hash);
        grid.rows.hash(&mut grid_hash);
        for node in &grid.nodes {
            node.luminance.hash(&mut grid_hash);
            node.dx.hash(&mut grid_hash);
            node.dy.hash(&mut grid_hash);
        }
        hash.0 = grid_hash.finish();
    }
    hash.0
}

/// FNV-1a `Hasher`, for `Hash` impls that a serde writer cannot consume.
struct Hash64(u64);

impl std::hash::Hasher for Hash64 {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Geometry, NormRect};

    fn key(develop: u64) -> StageKey {
        StageKey {
            source: 7,
            geometry: 11,
            region: [0, 0, NormRect::FULL.width.to_bits(), NormRect::FULL.height.to_bits()],
            out_w: 8,
            out_h: 4,
            level: 0,
            cubic: false,
            halo: 0,
            wide: false,
            gpu: false,
            develop,
        }
    }

    fn frame(len: usize, value: f32) -> Vec<[f32; 3]> {
        vec![[value; 3]; len]
    }

    /// Serialising the develop view must be stable for an unchanged set and
    /// move when any develop-domain field moves.
    #[test]
    fn develop_revision_follows_develop_domains() {
        let mut adj = AdjustmentSet::default();
        let base = develop_revision(&adj);
        assert_eq!(base, develop_revision(&adj));

        adj.exposure.exposure = 0.5;
        let exposure = develop_revision(&adj);
        assert_ne!(base, exposure, "exposure is a develop input");

        adj.white_balance.kelvin = 5200.0;
        assert_ne!(exposure, develop_revision(&adj), "white balance is a develop input");

        adj.geometry = Geometry { rotation: 3.0, ..Default::default() };
        assert_ne!(exposure, develop_revision(&adj), "geometry feeds sampling");
    }

    /// A detail-only edit must leave the revision alone: that is the whole
    /// point of the cache.
    #[test]
    fn develop_revision_ignores_detail_only_domains() {
        let mut adj = AdjustmentSet::default();
        let base = develop_revision(&adj);

        adj.sharpening.amount = 120.0;
        assert_eq!(base, develop_revision(&adj), "sharpening runs after the cache point");

        adj.noise_reduction.color = 60.0;
        adj.noise_reduction.luminance = 40.0;
        assert_eq!(base, develop_revision(&adj), "noise reduction is a detail stage");

        adj.clarity.structure = 35.0;
        assert_eq!(base, develop_revision(&adj), "structure is a 5x5 detail stage");

        adj.lens.distortion = 25.0;
        adj.lens.purple_fringing = 10.0;
        adj.lens.dust_removal_enabled = true;
        assert_eq!(base, develop_revision(&adj), "lens warp/fringe/dust are detail stages");

        adj.film_grain.impact = 20.0;
        adj.moire.amount = 15.0;
        assert_eq!(base, develop_revision(&adj), "grain and moire are detail stages");
    }

    /// Clarity's low-frequency amount *does* run inside the develop loop.
    #[test]
    fn clarity_amount_stays_in_the_revision() {
        let mut adj = AdjustmentSet::default();
        let base = develop_revision(&adj);
        adj.clarity.clarity = 20.0;
        assert_ne!(base, develop_revision(&adj));
    }

    #[test]
    fn a_stored_frame_is_returned_for_the_same_key_only() {
        let _serial = serial();
        clear();
        reset_stats();
        let (hits, _inserted, len, bytes) = stats();
        assert_eq!(bytes, 0);
        assert_eq!(hits, 0);
        assert_eq!(len, 0);

        let k = key(1);
        insert(k, &frame(32, 0.25));
        let hit = get(&k).expect("same key must hit");
        assert_eq!(hit.len(), 32);
        assert_eq!(hit[0], [0.25; 3]);
        assert!(get(&key(2)).is_none(), "a different revision must miss");
        let mut other = k;
        other.out_w = 9;
        assert!(get(&other).is_none(), "a different output size must miss");
    }

    #[test]
    fn entries_over_the_byte_budget_are_evicted() {
        let _serial = serial();
        clear();
        // ~30 MiB per frame: under the per-entry cap, and the total budget
        // holds fewer than `CAPACITY` of them.
        let big = frame((2 << 20) + (1 << 19), 1.0);
        let per_entry = big.len() * PIXEL_BYTES;
        assert!(per_entry <= ENTRY_MAX_BYTES);
        let keep = TOTAL_MAX_BYTES / per_entry;
        assert!(keep >= 1 && keep < CAPACITY, "test frame must exercise the byte budget");
        for revision in 0..(keep + 2) as u64 {
            insert(key(revision), &big);
        }
        let (_, _, len, bytes) = stats();
        assert_eq!(len, keep);
        assert!(bytes <= TOTAL_MAX_BYTES);
        assert!(get(&key(0)).is_none(), "the oldest entry was evicted");
    }

    #[test]
    fn frames_over_the_entry_cap_are_not_retained() {
        let _serial = serial();
        clear();
        let too_big = ENTRY_MAX_BYTES / PIXEL_BYTES + 1;
        insert(key(3), &frame(too_big, 0.0));
        assert!(get(&key(3)).is_none());
    }

    /// The cache is process-wide; tests that inspect it must not interleave.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        test_lock()
    }
}
