//! Tile-level versioned render scheduler with halo splitting and stitching.
//!
//! Features:
//! - Priority queue ordering: `VisibleFast > VisibleBest > Prefetch > Background`.
//! - Version cancellation guard: advancing the version discards queued stale
//!   tasks and flips in-flight [`VersionGuard`] checks to cancelled.
//! - Same-tile task consolidation: duplicate `(version, pyramid_level, tile_rect)`
//!   tasks are merged and promoted to the highest requested priority and halo.
//! - Overlap/halo tile decomposition (`split_tiles_with_halo`) and lossless
//!   reassembly (`stitch_tiles`) for spatial filters (blur/convolution).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

/// Scheduling priority for tile rendering (`VisibleFast` is highest).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Background = 0,
    Prefetch = 1,
    VisibleBest = 2,
    VisibleFast = 3,
}

/// Axis-aligned pixel rectangle within an image plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl TileRect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub const fn right(self) -> u32 {
        self.x.saturating_add(self.width)
    }

    pub const fn bottom(self) -> u32 {
        self.y.saturating_add(self.height)
    }

    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// One scheduled tile render task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileTask {
    pub task_id: u64,
    pub version: u64,
    pub priority: Priority,
    pub pyramid_level: u8,
    pub tile_rect: TileRect,
    pub halo_px: u32,
}

/// Consolidation key for deduplicating queued tasks covering the exact same tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ConsolidationKey {
    version: u64,
    pyramid_level: u8,
    tile_rect: TileRect,
}

impl TileTask {
    const fn consolidation_key(&self) -> ConsolidationKey {
        ConsolidationKey {
            version: self.version,
            pyramid_level: self.pyramid_level,
            tile_rect: self.tile_rect,
        }
    }
}

/// Result of submitting a [`TileTask`] to [`TileScheduler`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// Task was enqueued as a new pending item.
    Enqueued { task_id: u64 },
    /// Task matched an already-queued tile and was merged into it.
    Consolidated {
        kept_task_id: u64,
        priority: Priority,
        halo_px: u32,
    },
    /// Task's version is already older than the scheduler's active version.
    CancelledStale,
}

/// Lock-free cancellation guard handed to a worker alongside a popped task.
#[derive(Clone, Debug)]
pub struct VersionGuard {
    task_version: u64,
    latest_version: Arc<AtomicU64>,
}

impl VersionGuard {
    /// Returns `true` when a newer version has superseded this task.
    pub fn is_cancelled(&self) -> bool {
        self.latest_version.load(Ordering::Acquire) > self.task_version
    }

    /// The version this guard was issued for.
    pub const fn task_version(&self) -> u64 {
        self.task_version
    }

    /// The current latest version observed by the scheduler.
    pub fn latest_version(&self) -> u64 {
        self.latest_version.load(Ordering::Acquire)
    }
}

/// Runtime counters for observability and testing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulerStats {
    pub enqueued: u64,
    pub consolidated: u64,
    pub cancelled: u64,
    pub dispatched: u64,
}

#[derive(Default)]
struct QueueState {
    pending: HashMap<ConsolidationKey, TileTask>,
    stats: SchedulerStats,
}

/// Thread-safe tile render scheduler with version cancellation and consolidation.
pub struct TileScheduler {
    latest_version: Arc<AtomicU64>,
    state: Mutex<QueueState>,
}

impl Default for TileScheduler {
    fn default() -> Self {
        Self::new(0)
    }
}

impl TileScheduler {
    pub fn new(initial_version: u64) -> Self {
        Self {
            latest_version: Arc::new(AtomicU64::new(initial_version)),
            state: Mutex::new(QueueState::default()),
        }
    }

    /// Current active version.
    pub fn current_version(&self) -> u64 {
        self.latest_version.load(Ordering::Acquire)
    }

    /// Advance to `new_version` (if strictly greater than current) and discard
    /// all queued tasks belonging to older versions. Returns the number of
    /// queued tasks discarded.
    pub fn advance_version(&self, new_version: u64) -> usize {
        let mut guard = self.state.lock();
        let cur = self.latest_version.load(Ordering::Acquire);
        if new_version <= cur {
            return 0;
        }
        self.latest_version.store(new_version, Ordering::Release);
        purge_stale_locked(&mut guard, new_version)
    }

    /// Increment the active version by 1 and discard all queued stale tasks.
    pub fn bump_version(&self) -> (u64, usize) {
        let mut guard = self.state.lock();
        let next = self.latest_version.fetch_add(1, Ordering::AcqRel) + 1;
        let dropped = purge_stale_locked(&mut guard, next);
        (next, dropped)
    }

    /// Submit a tile task.
    ///
    /// - If `task.version < current_version`, it is immediately discarded as stale.
    /// - If `task.version > current_version`, the scheduler advances to
    ///   `task.version` and discards all older queued tasks.
    /// - If a queued task already exists for `(version, pyramid_level, tile_rect)`,
    ///   the two are consolidated: priority is promoted to `max` and `halo_px`
    ///   to `max`.
    pub fn submit(&self, task: TileTask) -> SubmitOutcome {
        let mut guard = self.state.lock();
        let cur = self.latest_version.load(Ordering::Acquire);

        if task.version < cur {
            guard.stats.cancelled += 1;
            return SubmitOutcome::CancelledStale;
        }
        if task.version > cur {
            self.latest_version.store(task.version, Ordering::Release);
            purge_stale_locked(&mut guard, task.version);
        }

        let key = task.consolidation_key();
        if let Some(existing) = guard.pending.get_mut(&key) {
            if task.priority > existing.priority {
                existing.priority = task.priority;
            }
            if task.halo_px > existing.halo_px {
                existing.halo_px = task.halo_px;
            }
            let kept_task_id = existing.task_id;
            let priority = existing.priority;
            let halo_px = existing.halo_px;
            guard.stats.consolidated += 1;
            return SubmitOutcome::Consolidated {
                kept_task_id,
                priority,
                halo_px,
            };
        }

        let task_id = task.task_id;
        guard.pending.insert(key, task);
        guard.stats.enqueued += 1;
        SubmitOutcome::Enqueued { task_id }
    }

    /// Pop the highest-priority valid task (breaking ties by smallest `task_id`).
    pub fn pop_next(&self) -> Option<(TileTask, VersionGuard)> {
        let mut guard = self.state.lock();
        let cur = self.latest_version.load(Ordering::Acquire);
        purge_stale_locked(&mut guard, cur);

        let best_key = guard
            .pending
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| b.task_id.cmp(&a.task_id))
            })
            .map(|(k, _)| *k)?;

        let task = guard.pending.remove(&best_key)?;
        guard.stats.dispatched += 1;
        let version_guard = VersionGuard {
            task_version: task.version,
            latest_version: Arc::clone(&self.latest_version),
        };
        Some((task, version_guard))
    }

    /// Number of currently queued tasks.
    pub fn pending_len(&self) -> usize {
        self.state.lock().pending.len()
    }

    /// Snapshot of scheduler counters.
    pub fn stats(&self) -> SchedulerStats {
        self.state.lock().stats
    }
}

fn purge_stale_locked(guard: &mut QueueState, min_version: u64) -> usize {
    let before = guard.pending.len();
    guard.pending.retain(|_, t| t.version >= min_version);
    let dropped = before - guard.pending.len();
    guard.stats.cancelled += dropped as u64;
    dropped
}

/// Geometry of one tile decomposed with a spatial halo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileSpec {
    /// Non-overlapping output rectangle owned by this tile.
    pub core_rect: TileRect,
    /// Input rectangle expanded by `halo` and clamped to image bounds.
    pub padded_rect: TileRect,
    /// X offset of `core_rect` inside `padded_rect` (`core_rect.x - padded_rect.x`).
    pub inner_offset_x: u32,
    /// Y offset of `core_rect` inside `padded_rect` (`core_rect.y - padded_rect.y`).
    pub inner_offset_y: u32,
}

/// Rendered pixel buffer for one tile (either padded or already cropped to core).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedTile {
    pub spec: TileSpec,
    /// Pixel buffer in row-major order with `channels` bytes per pixel.
    /// Length must match either `spec.padded_rect` or `spec.core_rect`.
    pub pixels: Vec<u8>,
}

/// Errors returned when stitching tiles into a full frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StitchError {
    ZeroChannels,
    TileOutOfBounds(TileRect),
    InvalidBufferSize {
        tile: TileRect,
        actual: usize,
        expected_core: usize,
        expected_padded: usize,
    },
}

impl std::fmt::Display for StitchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroChannels => write!(f, "channel count must be non-zero"),
            Self::TileOutOfBounds(rect) => write!(f, "tile {rect:?} exceeds output dimensions"),
            Self::InvalidBufferSize {
                tile,
                actual,
                expected_core,
                expected_padded,
            } => write!(
                f,
                "tile {tile:?} buffer size {actual} matches neither core ({expected_core}) nor padded ({expected_padded})"
            ),
        }
    }
}

impl std::error::Error for StitchError {}

/// Split a `width × height` image into tiles of size `tile_size × tile_size`
/// with a clamped spatial halo of `halo` pixels on each side.
pub fn split_tiles_with_halo(
    width: u32,
    height: u32,
    tile_size: u32,
    halo: u32,
) -> Vec<TileSpec> {
    if width == 0 || height == 0 || tile_size == 0 {
        return Vec::new();
    }

    let mut tiles = Vec::new();
    let mut y = 0u32;
    while y < height {
        let core_h = tile_size.min(height - y);
        let pad_y0 = y.saturating_sub(halo);
        let pad_y1 = (y + core_h).saturating_add(halo).min(height);

        let mut x = 0u32;
        while x < width {
            let core_w = tile_size.min(width - x);
            let pad_x0 = x.saturating_sub(halo);
            let pad_x1 = (x + core_w).saturating_add(halo).min(width);

            tiles.push(TileSpec {
                core_rect: TileRect::new(x, y, core_w, core_h),
                padded_rect: TileRect::new(pad_x0, pad_y0, pad_x1 - pad_x0, pad_y1 - pad_y0),
                inner_offset_x: x - pad_x0,
                inner_offset_y: y - pad_y0,
            });

            x += core_w;
        }
        y += core_h;
    }
    tiles
}

/// Stitch rendered tiles back into a `width × height` interleaved buffer
/// (`channels` bytes per pixel, e.g. 4 for RGBA8).
///
/// Each [`RenderedTile`] may contain either its full `padded_rect` buffer (in
/// which case the halo is cropped away using `inner_offset_x/y`) or an
/// already-cropped `core_rect` buffer.
pub fn stitch_tiles(
    width: u32,
    height: u32,
    channels: usize,
    tiles: &[RenderedTile],
) -> Result<Vec<u8>, StitchError> {
    if channels == 0 {
        return Err(StitchError::ZeroChannels);
    }

    let out_w = width as usize;
    let out_h = height as usize;
    let mut output = vec![0u8; out_w * out_h * channels];

    for tile in tiles {
        let core = tile.spec.core_rect;
        let padded = tile.spec.padded_rect;
        if core.right() > width || core.bottom() > height {
            return Err(StitchError::TileOutOfBounds(core));
        }
        if core.is_empty() {
            continue;
        }

        let core_bytes = (core.width as usize) * (core.height as usize) * channels;
        let padded_bytes = (padded.width as usize) * (padded.height as usize) * channels;

        let (src_stride_px, off_x, off_y) = if tile.pixels.len() == padded_bytes {
            (
                padded.width as usize,
                tile.spec.inner_offset_x as usize,
                tile.spec.inner_offset_y as usize,
            )
        } else if tile.pixels.len() == core_bytes {
            (core.width as usize, 0usize, 0usize)
        } else {
            return Err(StitchError::InvalidBufferSize {
                tile: core,
                actual: tile.pixels.len(),
                expected_core: core_bytes,
                expected_padded: padded_bytes,
            });
        };

        let row_bytes = (core.width as usize) * channels;
        for row in 0..(core.height as usize) {
            let src_start = ((off_y + row) * src_stride_px + off_x) * channels;
            let dst_start = ((core.y as usize + row) * out_w + core.x as usize) * channels;
            output[dst_start..dst_start + row_bytes]
                .copy_from_slice(&tile.pixels[src_start..src_start + row_bytes]);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_task(
        task_id: u64,
        version: u64,
        priority: Priority,
        level: u8,
        rect: TileRect,
        halo_px: u32,
    ) -> TileTask {
        TileTask {
            task_id,
            version,
            priority,
            pyramid_level: level,
            tile_rect: rect,
            halo_px,
        }
    }

    #[test]
    fn priority_ordering_visible_fast_to_background() {
        assert!(Priority::VisibleFast > Priority::VisibleBest);
        assert!(Priority::VisibleBest > Priority::Prefetch);
        assert!(Priority::Prefetch > Priority::Background);

        let sched = TileScheduler::new(1);
        let r0 = TileRect::new(0, 0, 64, 64);
        let r1 = TileRect::new(64, 0, 64, 64);
        let r2 = TileRect::new(0, 64, 64, 64);
        let r3 = TileRect::new(64, 64, 64, 64);

        sched.submit(make_task(1, 1, Priority::Background, 0, r0, 1));
        sched.submit(make_task(2, 1, Priority::Prefetch, 0, r1, 1));
        sched.submit(make_task(3, 1, Priority::VisibleFast, 0, r2, 1));
        sched.submit(make_task(4, 1, Priority::VisibleBest, 0, r3, 1));

        let order: Vec<Priority> = std::iter::from_fn(|| sched.pop_next().map(|(t, _)| t.priority))
            .collect();
        assert_eq!(
            order,
            vec![
                Priority::VisibleFast,
                Priority::VisibleBest,
                Priority::Prefetch,
                Priority::Background,
            ]
        );
    }

    #[test]
    fn version_cancellation_discards_old_queued_and_signals_in_flight_guard() {
        let sched = TileScheduler::new(1);
        let r0 = TileRect::new(0, 0, 64, 64);
        let r1 = TileRect::new(64, 0, 64, 64);
        let r2 = TileRect::new(128, 0, 64, 64);

        assert_eq!(
            sched.submit(make_task(10, 1, Priority::VisibleFast, 0, r0, 2)),
            SubmitOutcome::Enqueued { task_id: 10 }
        );
        assert_eq!(
            sched.submit(make_task(11, 1, Priority::VisibleBest, 0, r1, 2)),
            SubmitOutcome::Enqueued { task_id: 11 }
        );

        // Pop one task so it is "in-flight".
        let (in_flight, guard) = sched.pop_next().expect("pop v1 task");
        assert_eq!(in_flight.task_id, 10);
        assert!(!guard.is_cancelled());
        assert_eq!(sched.pending_len(), 1);

        // New version arrives via submit -> discards queued v1 task and cancels in-flight guard.
        assert_eq!(
            sched.submit(make_task(20, 2, Priority::VisibleFast, 0, r2, 2)),
            SubmitOutcome::Enqueued { task_id: 20 }
        );
        assert!(guard.is_cancelled());
        assert_eq!(guard.latest_version(), 2);
        assert_eq!(sched.pending_len(), 1);

        // Submitting an older version task is immediately rejected as stale.
        assert_eq!(
            sched.submit(make_task(12, 1, Priority::VisibleFast, 0, r0, 2)),
            SubmitOutcome::CancelledStale
        );

        let (next, guard2) = sched.pop_next().expect("pop v2 task");
        assert_eq!(next.task_id, 20);
        assert_eq!(next.version, 2);
        assert!(!guard2.is_cancelled());
        assert!(sched.pop_next().is_none());

        // Bump version explicitly cancels guard2 as well.
        let (v3, dropped) = sched.bump_version();
        assert_eq!(v3, 3);
        assert_eq!(dropped, 0);
        assert!(guard2.is_cancelled());
    }

    #[test]
    fn consolidate_same_tile_deduplicates_and_promotes_priority_and_halo() {
        let sched = TileScheduler::new(5);
        let rect = TileRect::new(32, 64, 64, 64);

        assert_eq!(
            sched.submit(make_task(100, 5, Priority::Background, 1, rect, 1)),
            SubmitOutcome::Enqueued { task_id: 100 }
        );
        assert_eq!(
            sched.submit(make_task(101, 5, Priority::Prefetch, 1, rect, 4)),
            SubmitOutcome::Consolidated {
                kept_task_id: 100,
                priority: Priority::Prefetch,
                halo_px: 4,
            }
        );
        assert_eq!(
            sched.submit(make_task(102, 5, Priority::VisibleFast, 1, rect, 2)),
            SubmitOutcome::Consolidated {
                kept_task_id: 100,
                priority: Priority::VisibleFast,
                halo_px: 4,
            }
        );

        assert_eq!(sched.pending_len(), 1);
        let (task, _) = sched.pop_next().expect("consolidated task");
        assert_eq!(task.task_id, 100);
        assert_eq!(task.priority, Priority::VisibleFast);
        assert_eq!(task.halo_px, 4);
        assert_eq!(sched.stats().consolidated, 2);
    }

    /// Reference 3×3 box blur (clamp-to-edge boundary) on an interleaved image.
    fn box_blur_3x3(src: &[u8], width: u32, height: u32, channels: usize) -> Vec<u8> {
        let w = width as i32;
        let h = height as i32;
        let mut dst = vec![0u8; (width as usize) * (height as usize) * channels];

        for y in 0..h {
            for x in 0..w {
                let mut sums = [0u32; 4];
                for dy in -1..=1 {
                    let sy = (y + dy).clamp(0, h - 1) as usize;
                    for dx in -1..=1 {
                        let sx = (x + dx).clamp(0, w - 1) as usize;
                        let idx = (sy * (width as usize) + sx) * channels;
                        for c in 0..channels {
                            sums[c] += src[idx + c] as u32;
                        }
                    }
                }
                let out_idx = ((y as usize) * (width as usize) + (x as usize)) * channels;
                for c in 0..channels {
                    dst[out_idx + c] = ((sums[c] + 4) / 9) as u8;
                }
            }
        }
        dst
    }

    /// Reference 3×3 weighted convolution (clamp-to-edge) on an interleaved image.
    fn convolve_3x3(
        src: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        kernel: [[i32; 3]; 3],
        divisor: i32,
    ) -> Vec<u8> {
        let w = width as i32;
        let h = height as i32;
        let mut dst = vec![0u8; (width as usize) * (height as usize) * channels];

        for y in 0..h {
            for x in 0..w {
                let mut acc = [0i32; 4];
                for dy in -1..=1 {
                    let sy = (y + dy).clamp(0, h - 1) as usize;
                    for dx in -1..=1 {
                        let sx = (x + dx).clamp(0, w - 1) as usize;
                        let k = kernel[(dy + 1) as usize][(dx + 1) as usize];
                        let idx = (sy * (width as usize) + sx) * channels;
                        for c in 0..channels {
                            acc[c] += (src[idx + c] as i32) * k;
                        }
                    }
                }
                let out_idx = ((y as usize) * (width as usize) + (x as usize)) * channels;
                for c in 0..channels {
                    let v = (acc[c] / divisor).clamp(0, 255);
                    dst[out_idx + c] = v as u8;
                }
            }
        }
        dst
    }

    fn extract_region(
        src: &[u8],
        full_width: u32,
        rect: TileRect,
        channels: usize,
    ) -> Vec<u8> {
        let mut out = vec![0u8; (rect.width as usize) * (rect.height as usize) * channels];
        let row_bytes = (rect.width as usize) * channels;
        for row in 0..(rect.height as usize) {
            let src_start =
                ((rect.y as usize + row) * (full_width as usize) + rect.x as usize) * channels;
            let dst_start = row * row_bytes;
            out[dst_start..dst_start + row_bytes]
                .copy_from_slice(&src[src_start..src_start + row_bytes]);
        }
        out
    }

    fn synthetic_rgba(width: u32, height: u32) -> Vec<u8> {
        let mut img = vec![0u8; (width as usize) * (height as usize) * 4];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                // Deterministic high-frequency pattern with sharp edges to expose any seam error.
                let hash = x
                    .wrapping_mul(374_761_393)
                    .wrapping_add(y.wrapping_mul(668_265_263));
                img[idx] = ((x * 7 + y * 13 + (hash & 0xff)) % 256) as u8;
                img[idx + 1] = ((x * 19 + y * 5 + ((hash >> 8) & 0xff)) % 256) as u8;
                img[idx + 2] = ((x * 3 + y * 29 + ((hash >> 16) & 0xff)) % 256) as u8;
                img[idx + 3] = 255;
            }
        }
        img
    }

    #[test]
    fn halo_tiled_box_blur_and_convolution_match_full_frame_zero_diff() {
        // Cover odd/even, non-divisible dimensions, tiny edge cases, and multiple tile/halo sizes.
        let cases: &[(u32, u32, u32, u32)] = &[
            (1, 1, 8, 1),
            (2, 5, 3, 1),
            (13, 19, 4, 1),
            (37, 29, 8, 1),
            (37, 29, 17, 2),
            (65, 48, 16, 1),
            (100, 73, 32, 3),
        ];

        let gaussian_3x3 = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];

        for &(width, height, tile_size, halo) in cases {
            let src = synthetic_rgba(width, height);
            let specs = split_tiles_with_halo(width, height, tile_size, halo);
            assert!(!specs.is_empty());

            // 1. 3×3 box blur
            let full_blur = box_blur_3x3(&src, width, height, 4);
            let blur_tiles: Vec<RenderedTile> = specs
                .iter()
                .map(|&spec| {
                    let sub = extract_region(&src, width, spec.padded_rect, 4);
                    let blurred =
                        box_blur_3x3(&sub, spec.padded_rect.width, spec.padded_rect.height, 4);
                    RenderedTile {
                        spec,
                        pixels: blurred,
                    }
                })
                .collect();
            let stitched_blur =
                stitch_tiles(width, height, 4, &blur_tiles).expect("stitch blur tiles");
            assert_eq!(
                stitched_blur, full_blur,
                "3x3 box blur mismatch for {width}x{height} tile={tile_size} halo={halo}"
            );

            // 2. 3×3 weighted convolution
            let full_conv = convolve_3x3(&src, width, height, 4, gaussian_3x3, 16);
            let conv_tiles: Vec<RenderedTile> = specs
                .iter()
                .map(|&spec| {
                    let sub = extract_region(&src, width, spec.padded_rect, 4);
                    let conved = convolve_3x3(
                        &sub,
                        spec.padded_rect.width,
                        spec.padded_rect.height,
                        4,
                        gaussian_3x3,
                        16,
                    );
                    RenderedTile {
                        spec,
                        pixels: conved,
                    }
                })
                .collect();
            let stitched_conv =
                stitch_tiles(width, height, 4, &conv_tiles).expect("stitch conv tiles");
            assert_eq!(
                stitched_conv, full_conv,
                "3x3 convolution mismatch for {width}x{height} tile={tile_size} halo={halo}"
            );
        }
    }
}
