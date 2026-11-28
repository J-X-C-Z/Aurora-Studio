//! Viewer scopes: luminance waveform, RGB parade and vectorscope densities.
//!
//! `Waveform` rows run from level 0 (black, drawn at the bottom) upward, so
//! `at(level, column)` is the density of luma `level` in image column group
//! `column`. `Vectorscope` is indexed `y * size + x`, origin top-left, neutral in
//! the centre, Cb up and Cr right. Densities are 0–255, scaled so the busiest
//! cell of the map is 255.
//!
//! Two families live here:
//!
//! * the analytic maps above (`waveform`/`parades`/`vectorscope`), 8-bit integer
//!   luma, own bin geometry;
//! * the viewer maps below (`viewer_maps`), which reproduce the exact bins the
//!   scopes panel draws — the geometry the app has always used. Their reference
//!   implementation is `viewer_maps`; `native/cuda/scope_gpu.cu` mirrors it
//!   bin-for-bin, so a GPU frame and a CPU fallback frame show the same picture.

use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waveform {
    pub columns: usize,
    pub levels: usize,
    pub data: Vec<u8>,
}

impl Waveform {
    pub fn at(&self, level: usize, column: usize) -> u8 {
        self.data[level * self.columns + column]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vectorscope {
    pub size: usize,
    pub data: Vec<u8>,
}

impl Vectorscope {
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.size + x]
    }
}

/// Luminance waveform: `columns` groups across, `levels` rows of luma.
pub fn waveform(rgba: &[u8], w: usize, h: usize, columns: usize, levels: usize) -> Waveform {
    if columns == 0 || levels == 0 {
        return Waveform { columns, levels, data: Vec::new() };
    }
    let mut col_of_x = vec![0usize; w];
    for (x, slot) in col_of_x.iter_mut().enumerate() {
        *slot = x * columns / w;
    }
    let counts = accumulate(rgba, w, h, columns * levels, |bins, r, g, b, x| {
        let level = luma(r, g, b) as usize * levels / 256;
        bins[level * columns + col_of_x[x]] += 1;
    });
    Waveform { columns, levels, data: to_u8(&counts) }
}

/// RGB parades: three 256-wide waveforms side by side, R then G then B.
pub fn parades(rgba: &[u8], w: usize, h: usize) -> Waveform {
    const COLUMNS: usize = 256;
    const LEVELS: usize = 256;
    const TOTAL: usize = 3 * COLUMNS;
    let counts = accumulate(rgba, w, h, TOTAL * LEVELS, |bins, r, g, b, x| {
        for (channel, value) in [r, g, b].into_iter().enumerate() {
            let level = value as usize * LEVELS / 256;
            bins[level * TOTAL + channel * COLUMNS + x] += 1;
        }
    });
    Waveform { columns: TOTAL, levels: LEVELS, data: to_u8(&counts) }
}

/// Cb/Cr vectorscope on a `size`×`size` grid.
pub fn vectorscope(rgba: &[u8], w: usize, h: usize, size: usize) -> Vectorscope {
    if size == 0 {
        return Vectorscope { size, data: Vec::new() };
    }
    let half = (size / 2) as i32;
    let edge = size as i32 - 1;
    let counts = accumulate(rgba, w, h, size * size, |bins, r, g, b, _| {
        let r = r as i32;
        let g = g as i32;
        let b = b as i32;
        // BT.601 chroma, ±128 around neutral.
        let cb = (-117 * r - 394 * g + 512 * b) >> 10;
        let cr = (512 * r - 465 * g - 47 * b) >> 10;
        let x = (half + (cr * half).div_euclid(128)).clamp(0, edge);
        let y = (half - (cb * half).div_euclid(128)).clamp(0, edge);
        bins[(y * size as i32 + x) as usize] += 1;
    });
    Vectorscope { size, data: to_u8(&counts) }
}

fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((54 * r as u32 + 183 * g as u32 + 19 * b as u32) >> 8) as u8
}

/// Splits the rows into one band per rayon task, counts into per-band bins and
/// sums them. `add` receives the band's counters, the pixel and its column.
fn accumulate<F>(rgba: &[u8], w: usize, h: usize, bins: usize, add: F) -> Vec<u32>
where
    F: Fn(&mut [u32], u8, u8, u8, usize) + Sync,
{
    let len = w.saturating_mul(h).saturating_mul(4);
    if w == 0 || h == 0 || bins == 0 || rgba.len() < len {
        return vec![0; bins];
    }
    let stride = w * 4;
    let bands = (rayon::current_num_threads() * 4).max(1);
    let rows = (h + bands - 1) / bands;
    rgba[..len]
        .par_chunks(rows * stride)
        .map(|band| {
            let mut local = vec![0u32; bins];
            for row in band.chunks_exact(stride) {
                for (x, px) in row.chunks_exact(4).enumerate() {
                    add(&mut local, px[0], px[1], px[2], x);
                }
            }
            local
        })
        .reduce(
            || vec![0u32; bins],
            |mut a, b| {
                for (slot, value) in a.iter_mut().zip(b) {
                    *slot += value;
                }
                a
            },
        )
}

fn to_u8(counts: &[u32]) -> Vec<u8> {
    let max = counts.iter().copied().max().unwrap_or(0).max(1);
    counts.iter().map(|&c| (c as u64 * 255 / max as u64) as u8).collect()
}

// ---------------------------------------------------------------------------
// Viewer maps: the bins the scopes panel draws.
// ---------------------------------------------------------------------------

/// At most this many source pixels feed one viewer scope pass.
pub const VIEWER_MAX_SAMPLES: usize = 160_000;
/// Waveform width in output pixels: one column per output pixel.
pub const VIEWER_COLUMNS: usize = 256;
/// Waveform height in levels; level 0 is black and sits on the bottom row.
pub const VIEWER_LEVELS: usize = 160;
/// RGB parade lanes, drawn side by side inside `VIEWER_COLUMNS`.
pub const VIEWER_LANES: usize = 3;
/// Width of one parade lane in output pixels.
pub const VIEWER_LANE_WIDTH: usize = VIEWER_COLUMNS / VIEWER_LANES;
/// Vectorscope grid, `VIEWER_VECTOR * VIEWER_VECTOR` cells.
pub const VIEWER_VECTOR: usize = 160;
/// `wave | parade | vector`, in that order, as `u32` counts.
pub const VIEWER_WORDS: usize =
    VIEWER_LEVELS * VIEWER_COLUMNS * 2 + VIEWER_VECTOR * VIEWER_VECTOR;

/// Which maps one pass should fill. Kinds that are off stay zeroed, so a map
/// always has the same shape and a caller can switch kinds without a new pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ScopeWants {
    pub waveform: bool,
    pub parade: bool,
    pub vector: bool,
}

impl ScopeWants {
    pub const ALL: Self = Self { waveform: true, parade: true, vector: true };

    pub const WAVEFORM: Self = Self { waveform: true, parade: false, vector: false };
    pub const PARADE: Self = Self { waveform: false, parade: true, vector: false };
    pub const VECTOR: Self = Self { waveform: false, parade: false, vector: true };

    pub fn any(self) -> bool {
        self.waveform || self.parade || self.vector
    }

    /// Word span `[base, base + words)` one pass fills for `self`: the union of
    /// the wanted maps. This is the span a caller has to copy back — the panel
    /// asks for one kind, so it carries one map instead of all three.
    pub fn zone(self) -> (usize, usize) {
        if !self.any() {
            return (0, 0);
        }
        let kinds = self.mask();
        let base = mask::first_base(kinds);
        (base, mask::last_end(kinds) - base)
    }
}

/// Raw scope counts for one frame, laid out exactly like the CUDA kernel writes
/// them: `VIEWER_LEVELS * VIEWER_COLUMNS` waveform counts, the same number of
/// parade counts, then `VIEWER_VECTOR * VIEWER_VECTOR` vectorscope counts.
///
/// Rows run bottom-up (row 0 is black), matching the panel's drawing order, and
/// the counts are raw pixel counts — normalising to 0–255 is the drawing step's
/// job (`crate::scopes::viewer_tone`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScopeMaps {
    words: Vec<u32>,
    sampled: u32,
    /// Which maps a pass actually filled. A map built from one zone only
    /// answers for that zone; the others stay zero and report as not covered.
    wants: ScopeWants,
}

impl ScopeMaps {
    /// A map of the right shape with every count at zero.
    pub fn blank() -> Self {
        Self { words: vec![0; VIEWER_WORDS], sampled: 0, wants: ScopeWants::default() }
    }

    /// Wraps counts that are already in the layout above (e.g. copied back from
    /// the GPU). A short buffer is padded with zeros so accessors stay total.
    pub fn from_words(words: &[u32], sampled: u32) -> Self {
        let mut owned = vec![0; VIEWER_WORDS];
        let n = words.len().min(VIEWER_WORDS);
        owned[..n].copy_from_slice(&words[..n]);
        Self { words: owned, sampled, wants: ScopeWants::ALL }
    }

    /// Takes ownership of counts in the layout above, without copying them.
    pub fn from_owned_words(mut words: Vec<u32>, sampled: u32) -> Self {
        words.resize(VIEWER_WORDS, 0);
        Self { words, sampled, wants: ScopeWants::ALL }
    }

    /// A map that carries only the counts of `wants`, whose span starts at word
    /// `base` of the full layout. This is the GPU return path: one zone comes
    /// back and the other two maps stay zero instead of travelling as zeros.
    pub fn from_zone(base: usize, bins: &[u32], sampled: u32, wants: ScopeWants) -> Self {
        let mut words = vec![0u32; VIEWER_WORDS];
        if base < VIEWER_WORDS && wants.any() {
            let n = bins.len().min(VIEWER_WORDS - base);
            words[base..base + n].copy_from_slice(&bins[..n]);
        }
        Self { words, sampled, wants }
    }

    /// Which maps this map answers for; the rest of the layout is zero.
    pub fn wants(&self) -> ScopeWants {
        self.wants
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Waveform counts in drawing order (`level * columns + column`).
    pub fn wave_bins(&self) -> &[u32] {
        let end = VIEWER_LEVELS * VIEWER_COLUMNS;
        self.words.get(..end).unwrap_or(&[])
    }

    /// Parade counts in drawing order, three lanes of `VIEWER_LANE_WIDTH`.
    pub fn parade_bins(&self) -> &[u32] {
        let (start, end) = (VIEWER_LEVELS * VIEWER_COLUMNS, 2 * VIEWER_LEVELS * VIEWER_COLUMNS);
        self.words.get(start..end).unwrap_or(&[])
    }

    /// Vectorscope counts in drawing order (`y * VIEWER_VECTOR + x`).
    pub fn vector_bins(&self) -> &[u32] {
        self.words.get(2 * VIEWER_LEVELS * VIEWER_COLUMNS..).unwrap_or(&[])
    }

    pub fn sampled(&self) -> u32 {
        self.sampled
    }

    pub fn is_empty(&self) -> bool {
        self.sampled == 0
    }

    /// Waveform count at `row` and output column. Row 0 is the top of the panel
    /// (white); `VIEWER_LEVELS - 1` is black on the bottom line.
    pub fn wave_at(&self, row: usize, column: usize) -> u32 {
        self.at(0, row, column)
    }

    /// Parade count for channel `lane` (0 R, 1 G, 2 B) at `row`, same top-down
    /// row order as [`Self::wave_at`].
    pub fn parade_at(&self, lane: usize, row: usize, column: usize) -> u32 {
        if lane >= VIEWER_LANES || column >= VIEWER_LANE_WIDTH {
            return 0;
        }
        self.at(VIEWER_LEVELS * VIEWER_COLUMNS, row, lane * VIEWER_LANE_WIDTH + column)
    }

    /// Vectorscope count at `x` (Cb axis) and `y` (Cr axis, top-down).
    pub fn vector_at(&self, x: usize, y: usize) -> u32 {
        if x >= VIEWER_VECTOR || y >= VIEWER_VECTOR {
            return 0;
        }
        self.words
            .get(2 * VIEWER_LEVELS * VIEWER_COLUMNS + y * VIEWER_VECTOR + x)
            .copied()
            .unwrap_or(0)
    }

    fn at(&self, base: usize, row: usize, column: usize) -> u32 {
        if row >= VIEWER_LEVELS || column >= VIEWER_COLUMNS {
            return 0;
        }
        self.words.get(base + row * VIEWER_COLUMNS + column).copied().unwrap_or(0)
    }

    /// Busiest waveform cell; with `parade` set, the busiest parade cell instead.
    pub fn wave_peak(&self, parade: bool) -> u32 {
        let base = if parade { VIEWER_LEVELS * VIEWER_COLUMNS } else { 0 };
        let end = base + VIEWER_LEVELS * VIEWER_COLUMNS;
        self.words.get(base..end).and_then(|bins| bins.iter().copied().max()).unwrap_or(0)
    }

    /// Busiest vectorscope cell.
    pub fn vector_peak(&self) -> u32 {
        self.words
            .get(2 * VIEWER_LEVELS * VIEWER_COLUMNS..)
            .and_then(|bins| bins.iter().copied().max())
            .unwrap_or(0)
    }

    /// Adds a band accumulator that starts at word `base` of the map.
    fn accumulate_from(&mut self, base: usize, band: &[u32]) {
        for (slot, value) in self.words.iter_mut().skip(base).zip(band) {
            *slot += value;
        }
    }
}

/// Sampling stride for a `w`×`h` frame: at most `VIEWER_MAX_SAMPLES` pixels.
pub fn viewer_step(w: usize, h: usize) -> usize {
    (((w * h) as f32 / VIEWER_MAX_SAMPLES as f32).sqrt().ceil().max(1.0)) as usize
}

// ---------------------------------------------------------------------------
// Device counting seam: what a GPU backend has to implement.
// ---------------------------------------------------------------------------

/// A finished preview frame that is already in device memory.
///
/// The counts have to be taken from these pixels, not from a copy, or the panel
/// and the preview can disagree. Everything here is opaque to the backend
/// except the geometry and the byte order.
pub struct DeviceScopeFrame<'a, S: ?Sized> {
    /// Device address of the frame: `width * height` pixels of 4 bytes.
    pub pixels: u64,
    pub width: usize,
    pub height: usize,
    /// Display byte order of `pixels`: `true` is BGRA, `false` is RGBA. The
    /// alpha byte is ignored.
    pub bgra: bool,
    /// Borrow the renderer's real submission context. Never manufacture an owning
    /// stream from an integer or extend its lifetime beyond this dispatch.
    pub stream: &'a S,
    /// The backend's device object, so a counter that needs the device (Metal
    /// buffers, a DX12 device) can downcast it. `None` when the caller has no
    /// device handle to lend.
    pub device: Option<&'a dyn std::any::Any>,
}

/// Counts viewer scopes from pixels the renderer already owns.
///
/// This is the cross-platform seam for the scope stage: NVIDIA implements it in
/// `mock::scopes_gpu` (CUDA + PTX on the develop stream), and the DX12 (HLSL)
/// and Metal (MSL) backends implement the same trait over their own command
/// list/queue. The maps layout is the ABI: [`VIEWER_WORDS`] `u32` counts,
/// waveform `VIEWER_LEVELS * VIEWER_COLUMNS`, then the same number of parade
/// lanes, then `VIEWER_VECTOR * VIEWER_VECTOR` vectorscope cells, all with the
/// bin arithmetic of [`viewer_maps`].
pub trait DeviceScopeCounter: Send + Sync {
    type Submission: ?Sized;
    /// Backend name for logs, e.g. `"cuda"`.
    fn backend(&self) -> &'static str;

    /// Clears the `wants` zone of the `maps` device buffer, then counts `frame`
    /// into all three maps.
    ///
    /// Returns once the work is enqueued — the renderer's own fence covers it,
    /// which is why this takes the renderer's stream instead of making one.
    fn count(
        &self,
        frame: DeviceScopeFrame<'_,Self::Submission>,
        maps: u64,
        wants: ScopeWants,
    ) -> Result<(), ScopeCountError>;
}

/// Why a device scope pass could not run. The caller keeps counting on the CPU.
#[derive(Debug)]
pub enum ScopeCountError {
    /// No device/module/stream to count on.
    Unavailable(String),
    /// The geometry or the buffer did not fit the contract.
    Refused(String),
}

impl std::fmt::Display for ScopeCountError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(what) => write!(f, "device scopes unavailable: {what}"),
            Self::Refused(what) => write!(f, "device scopes refused: {what}"),
        }
    }
}

/// How many row bands one pass is split into. Every band owns one count array,
/// so the count is the per-frame cost of the reduce: it stays small enough that
/// summing the bands (which is serial) never outweighs the pixel work.
fn viewer_bands(rows: usize) -> usize {
    let threads = rayon::current_num_threads().clamp(1, 8);
    threads.min(rows.max(1)).min(6)
}

/// BT.709 luma in the exact order the panel has always used.
#[inline]
fn viewer_luma(r: u8, g: u8, b: u8) -> f32 {
    0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
}

/// `value` rounded half away from zero, written as `floor` plus a 0.5 comparison
/// on the fraction. `(value + 0.5).trunc()` is *not* equivalent — the addition
/// double-rounds, so `0.49999997` would jump to 1 — and the comparison form is
/// what `native/cuda/scope_gpu.cu` mirrors with `cvt.rmi` + `sub.rn` +
/// `setp.ge` + `selp`, giving identical bins.
#[inline]
pub fn viewer_round_away(value: f32) -> f32 {
    let magnitude = value.abs();
    let floor = magnitude.floor();
    let rounded = if magnitude - floor >= 0.5 { floor + 1.0 } else { floor };
    rounded.copysign(value)
}

/// The counting loop's rounding: the same sequence without the sign dance, for
/// the non-negative values this module produces (`(0.5 + Cb)`, `(0.5 − Cr)` and
/// `value / 255 * 159` are all ≥ 0). Pinned to the `f32::round()` the scopes
/// panel has always used by
/// `rounding_matches_the_panel_for_every_byte_triple`.
#[inline]
fn viewer_round_up(value: f32) -> f32 {
    let floor = value.floor();
    if value - floor >= 0.5 { floor + 1.0 } else { floor }
}

/// The level `value` (0–255) lands on.
#[inline]
fn viewer_level(value: f32) -> usize {
    let q = (value / 255.0) * (VIEWER_LEVELS - 1) as f32;
    debug_assert!(q >= 0.0);
    (viewer_round_up(q) as usize).min(VIEWER_LEVELS - 1)
}

/// Counts a frame into the three viewer maps in one strided pass.
///
/// The sampled pixels, the bin arithmetic and the rounding are the ones the
/// scopes panel has always used — `native/cuda/scope_gpu.cu` mirrors them
/// exactly, so `viewer_maps` is both the CPU fallback and the parity reference.
/// `bgra` selects byte order; the alpha byte is ignored.
pub fn viewer_maps(rgba: &[u8], w: usize, h: usize, bgra: bool, wants: ScopeWants) -> ScopeMaps {
    let mut maps = ScopeMaps::blank();
    let len = w.saturating_mul(h).saturating_mul(4);
    if w == 0 || h == 0 || rgba.len() < len || !wants.any() {
        return maps;
    }
    let kinds = wants.mask();
    let (words, sampled) = match kinds {
        mask::WAVEFORM => viewer_count::<{ mask::WAVEFORM }>(rgba, len, w, h, bgra),
        mask::PARADE => viewer_count::<{ mask::PARADE }>(rgba, len, w, h, bgra),
        mask::VECTOR => viewer_count::<{ mask::VECTOR }>(rgba, len, w, h, bgra),
        mask::WAVEFORM_PARADE => viewer_count::<{ mask::WAVEFORM_PARADE }>(rgba, len, w, h, bgra),
        mask::WAVEFORM_VECTOR => viewer_count::<{ mask::WAVEFORM_VECTOR }>(rgba, len, w, h, bgra),
        mask::PARADE_VECTOR => viewer_count::<{ mask::PARADE_VECTOR }>(rgba, len, w, h, bgra),
        _ => viewer_count::<{ mask::ALL }>(rgba, len, w, h, bgra),
    };
    maps.accumulate_from(mask::first_base(kinds), &words);
    maps.sampled = sampled;
    maps.wants = wants;
    maps
}

/// Bit flags over the wanted maps, also used as the const-generic parameter that
/// lets the counting loop drop the kinds it does not need.
pub(crate) mod mask {
    pub const WAVEFORM: u8 = 1;
    pub const PARADE: u8 = 2;
    pub const VECTOR: u8 = 4;
    pub const WAVEFORM_PARADE: u8 = 3;
    pub const WAVEFORM_VECTOR: u8 = 5;
    pub const PARADE_VECTOR: u8 = 6;
    pub const ALL: u8 = 7;

    /// Word where the band accumulators of `kinds` start: the first wanted map.
    pub const fn first_base(kinds: u8) -> usize {
        if kinds & WAVEFORM != 0 {
            0
        } else if kinds & PARADE != 0 {
            super::VIEWER_LEVELS * super::VIEWER_COLUMNS
        } else {
            2 * super::VIEWER_LEVELS * super::VIEWER_COLUMNS
        }
    }

    /// Word just past the last wanted map.
    pub const fn last_end(kinds: u8) -> usize {
        if kinds & VECTOR != 0 {
            super::VIEWER_WORDS
        } else if kinds & PARADE != 0 {
            2 * super::VIEWER_LEVELS * super::VIEWER_COLUMNS
        } else {
            super::VIEWER_LEVELS * super::VIEWER_COLUMNS
        }
    }
}

impl ScopeWants {
    fn mask(self) -> u8 {
        (if self.waveform { mask::WAVEFORM } else { 0 })
            | (if self.parade { mask::PARADE } else { 0 })
            | (if self.vector { mask::VECTOR } else { 0 })
    }
}

/// Counts one frame into `KINDS` with the kind tests folded away at compile time.
fn viewer_count<const KINDS: u8>(
    rgba: &[u8],
    len: usize,
    w: usize,
    h: usize,
    bgra: bool,
) -> (Vec<u32>, u32) {
    let step = viewer_step(w, h);
    let stride = w * 4;
    let (r_idx, b_idx) = if bgra { (2, 0) } else { (0, 2) };
    let (levels, columns, lane_w, vn) =
        (VIEWER_LEVELS, VIEWER_COLUMNS, VIEWER_LANE_WIDTH, VIEWER_VECTOR);
    let (parade_base, vector_base) = (levels * columns, 2 * levels * columns);
    let base = mask::first_base(KINDS);
    let end = mask::last_end(KINDS);
    let band_words = end - base;
    // Bands are whole numbers of sampled rows, so every band starts on a sampled
    // row and the counting loop never needs the row index: a waveform or parade
    // bin depends on the column and the value alone.
    let per_band = h.div_ceil(step).div_ceil(viewer_bands(h.div_ceil(step))).max(1);
    let band_bytes = step * per_band * stride;
    // Per-column targets and the 256-entry value table replace the per-pixel
    // integer division and level arithmetic; both depend on the frame geometry
    // instead of the pixels.
    let mut column_wave = Vec::with_capacity(w.div_ceil(step));
    let mut column_parade = Vec::with_capacity(w.div_ceil(step));
    for x in (0..w).step_by(step) {
        column_wave.push((x * columns / w).min(columns - 1));
        column_parade.push((x * lane_w / w).min(lane_w - 1));
    }
    let level_of_value: [usize; 256] =
        std::array::from_fn(|value| levels - 1 - viewer_level(value as f32));
    // Saturating: with only the vectorscope wanted, `base` sits past the parade
    // base, and an unconditional subtraction would underflow. The unused offset
    // is never read, because each map's writes are gated on its `KINDS` bit.
    let (parade_offset, vector_offset) =
        (parade_base.saturating_sub(base), vector_base.saturating_sub(base));
    let (column_wave, column_parade) = (&column_wave[..], &column_parade[..]);
    let columns_sampled = column_wave.len();
    let pixel_stride = step * 4;
    // The sample count is geometry, so it is arithmetic instead of a per-pixel
    // counter: one less loop-carried dependency in the hot loop.
    let sampled = (columns_sampled * h.div_ceil(step)) as u32;
    let words = rgba[..len]
        .par_chunks(band_bytes)
        .map(|band| {
            let mut local = vec![0u32; band_words];
            // Each group is `step` physical rows; only its first row is sampled.
            // The column table has one entry per sampled column, so the loop
            // walks it and derives the byte offset instead of stepping a range.
            for group in band.chunks(stride * step) {
                let row = &group[..stride];
                for i in 0..columns_sampled {
                    let o = i * pixel_stride;
                    let (r, g, b) = (row[o + r_idx], row[o + 1], row[o + b_idx]);
                    if KINDS & mask::WAVEFORM != 0 {
                        let level = levels - 1 - viewer_level(viewer_luma(r, g, b));
                        local[level * columns + column_wave[i]] += 1;
                    }
                    if KINDS & mask::PARADE != 0 {
                        let column = column_parade[i];
                        local[parade_offset + level_of_value[r as usize] * columns + column] += 1;
                        local[parade_offset + level_of_value[g as usize] * columns + lane_w + column] += 1;
                        local[parade_offset + level_of_value[b as usize] * columns + 2 * lane_w + column] += 1;
                    }
                    if KINDS & mask::VECTOR != 0 {
                        let luma = viewer_luma(r, g, b);
                        let cb = (b as f32 - luma) / 1.8556 / 255.0;
                        let cr = (r as f32 - luma) / 1.5748 / 255.0;
                        let vx = viewer_round_up((0.5 + cb) * (vn - 1) as f32);
                        let vy = viewer_round_up((0.5 - cr) * (vn - 1) as f32);
                        let vx = vx.clamp(0.0, (vn - 1) as f32) as usize;
                        let vy = vy.clamp(0.0, (vn - 1) as f32) as usize;
                        local[vector_offset + vy * vn + vx] += 1;
                    }
                }
            }
            local
        })
        .reduce(
            || vec![0u32; band_words],
            |mut acc, band| {
                for (slot, value) in acc.iter_mut().zip(band) {
                    *slot += value;
                }
                acc
            },
        );
    (words, sampled)
}

// ---------------------------------------------------------------------------
// Drawing: the density maps above become the BGRA images the panel shows.
// ---------------------------------------------------------------------------

/// Density 0–255 for `count`, scaled against the busiest cell. A square root
/// keeps sparse traces visible next to dense ones; every non-zero cell is at
/// least 12 % bright.
pub fn viewer_tone(count: u32, max: u32) -> f32 {
    if count == 0 {
        0.0
    } else {
        (count as f32 / max.max(1) as f32).sqrt().max(0.12)
    }
}

/// The waveform/parade panel as BGRA: `VIEWER_COLUMNS`×`VIEWER_LEVELS`, black
/// background, faint graticule at 0/25/50/75/100 %, one column per trace column.
/// `lanes` is 1 for the luma waveform and `VIEWER_LANES` for the RGB parade.
pub fn viewer_waveform_image(maps: &ScopeMaps, lanes: usize) -> Vec<u8> {
    let (w, h) = (VIEWER_COLUMNS, VIEWER_LEVELS);
    let lanes = if lanes == VIEWER_LANES { VIEWER_LANES } else { 1 };
    let lane_w = (w / lanes).max(1);
    let mut out = vec![0u8; w * h * 4];
    for px in out.chunks_exact_mut(4) {
        px[3] = 255;
    }
    // Graticule at 0, 25, 50, 75 and 100 %.
    for q in 0..=4 {
        let y = (h - 1) * q / 4;
        for x in 0..w {
            out[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[60, 60, 60, 255]);
        }
    }
    let max = maps.wave_peak(lanes > 1);
    let counts = if lanes > 1 { maps.parade_bins() } else { maps.wave_bins() };
    for (i, &count) in counts.iter().enumerate().take(w * h) {
        let tone = viewer_tone(count, max);
        if tone == 0.0 {
            continue;
        }
        let rgb = match (lanes, (i % w) / lane_w) {
            (1, _) => [0.75, 1.0, 0.8],
            (_, 0) => [1.0, 0.35, 0.35],
            (_, 1) => [0.35, 1.0, 0.45],
            _ => [0.4, 0.55, 1.0],
        };
        let px = &mut out[i * 4..i * 4 + 4];
        px[0] = (rgb[2] * tone * 255.0) as u8;
        px[1] = (rgb[1] * tone * 255.0) as u8;
        px[2] = (rgb[0] * tone * 255.0) as u8;
    }
    out
}

/// One vectorscope cell's decoration, which depends only on its position.
#[derive(Clone, Copy)]
struct VectorCell {
    /// Radii and axes to draw before the trace: outer circle, centre cross and
    /// the skin-tone line at about 123°.
    grid: bool,
    /// Trace colour for this hue, lifted toward white.
    hue: [f32; 3],
}

fn vector_cells(n: usize) -> &'static [VectorCell] {
    use std::sync::OnceLock;
    static CELLS: OnceLock<Vec<VectorCell>> = OnceLock::new();
    CELLS.get_or_init(|| {
        let c = (n - 1) as f32 / 2.0;
        (0..n)
            .flat_map(|y| {
                (0..n).map(move |x| {
                    let (dx, dy) = (x as f32 - c, c - y as f32);
                    let d = (dx * dx + (y as f32 - c) * (y as f32 - c)).sqrt();
                    let angle = dy.atan2(dx).to_degrees();
                    let skin = (angle - 123.0).abs() < 0.8 && d < c;
                    let grid = (d - c).abs() < 0.8
                        || ((x as f32 - c).abs() < 0.5 || (y as f32 - c).abs() < 0.5) && d < c
                        || skin;
                    let cb = x as f32 / (n - 1) as f32 - 0.5;
                    let cr = 0.5 - y as f32 / (n - 1) as f32;
                    let hue = [
                        (0.6 + 1.5748 * cr).clamp(0.0, 1.0),
                        (0.6 - 0.1873 * cb - 0.4681 * cr).clamp(0.0, 1.0),
                        (0.6 + 1.8556 * cb).clamp(0.0, 1.0),
                    ];
                    VectorCell { grid, hue }
                })
            })
            .collect()
    })
}

/// The vectorscope panel as BGRA, `VIEWER_VECTOR`×`VIEWER_VECTOR`.
pub fn viewer_vector_image(maps: &ScopeMaps) -> Vec<u8> {
    let n = VIEWER_VECTOR;
    let mut out = vec![0u8; n * n * 4];
    let max = maps.vector_peak();
    let counts = maps.vector_bins();
    for (i, cell) in vector_cells(n).iter().enumerate() {
        let px = &mut out[i * 4..i * 4 + 4];
        px[3] = 255;
        if cell.grid {
            px.copy_from_slice(&[60, 60, 60, 255]);
        }
        let tone = viewer_tone(counts.get(i).copied().unwrap_or(0), max);
        if tone > 0.0 {
            px[0] = (cell.hue[2] * tone * 255.0) as u8;
            px[1] = (cell.hue[1] * tone * 255.0) as u8;
            px[2] = (cell.hue[0] * tone * 255.0) as u8;
        }
    }
    out
}

/// Describes where two count arrays disagree, without dumping every bin.
///
/// Test diagnostics only: a 107 520-word assert_eq dump buries the one line a
/// reader needs, so the parity tests compare through this instead.
#[cfg(test)]
pub(crate) fn bins_differ(gpu: &[u32], cpu: &[u32]) -> Option<String> {
    let wave = VIEWER_LEVELS * VIEWER_COLUMNS;
    let mut bins = 0usize;
    let mut per_zone = [0usize; 3];
    let mut first = Vec::new();
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        if g != c {
            bins += 1;
            let zone = if i < wave {
                0
            } else if i < 2 * wave {
                1
            } else {
                2
            };
            per_zone[zone] += 1;
            if first.len() < 6 {
                first.push(format!("word {i} (zone {zone}): gpu {g} cpu {c}"));
            }
        }
    }
    (bins > 0).then(|| {
        format!(
            "{bins} bins differ (wave {}, parade {}, vector {}): {}",
            per_zone[0],
            per_zone[1],
            per_zone[2],
            first.join("; ")
        )
    })
}

/// Counts a frame with a plain global stride: one pass over `(0..h).step_by(step)`
/// and `(0..w).step_by(step)`, exactly the arithmetic the panel has always used.
///
/// Slower than [`viewer_maps`] and sharing none of its band bookkeeping, so the
/// parity tests use it as the second opinion: `viewer_maps` and the CUDA kernel
/// both have to agree with this, and a disagreement then belongs to whichever
/// side deviates. Test-only.
#[cfg(test)]
pub(crate) fn naive_viewer_maps(px: &[u8], w: usize, h: usize, bgra: bool) -> Vec<u32> {
    let step = viewer_step(w, h);
    let (ri, bi) = if bgra { (2usize, 0usize) } else { (0, 2) };
    let (levels, columns, lane_w, vn) =
        (VIEWER_LEVELS, VIEWER_COLUMNS, VIEWER_LANE_WIDTH, VIEWER_VECTOR);
    let (parade_base, vector_base) = (levels * columns, 2 * levels * columns);
    let mut out = vec![0u32; VIEWER_WORDS];
    if w == 0 || h == 0 || px.len() < w * h * 4 {
        return out;
    }
    for y in (0..h).step_by(step) {
        for x in (0..w).step_by(step) {
            let o = (y * w + x) * 4;
            let (r, g, b) = (px[o + ri], px[o + 1], px[o + bi]);
            let luma = viewer_luma(r, g, b);
            let row = levels - 1 - viewer_level(luma);
            out[row * columns + (x * columns / w).min(columns - 1)] += 1;
            for (lane, value) in [r, g, b].into_iter().enumerate() {
                let row = levels - 1 - viewer_level(value as f32);
                let col = (x * lane_w / w).min(lane_w - 1);
                out[parade_base + row * columns + lane * lane_w + col] += 1;
            }
            let cb = (b as f32 - luma) / 1.8556 / 255.0;
            let cr = (r as f32 - luma) / 1.5748 / 255.0;
            let vx = viewer_round_up((0.5 + cb) * (vn - 1) as f32).min((vn - 1) as f32) as usize;
            let vy = viewer_round_up((0.5 - cr) * (vn - 1) as f32).min((vn - 1) as f32) as usize;
            out[vector_base + vy * vn + vx] += 1;
        }
    }
    out
}

#[cfg(test)]
mod viewer_tests {
    use super::*;

    /// The banded counter and the panel's plain global stride have to agree:
    /// the panel path is the reference the CUDA kernel mirrors, and this is the
    /// check that keeps the band bookkeeping honest for awkward geometries.
    #[test]
    fn viewer_maps_matches_a_global_stride_reference() {
        for (w, h) in [(613usize, 409usize), (2000, 1333), (1600, 1067), (1280, 853)] {
            let mut px = vec![0u8; w * h * 4];
            for y in 0..h {
                for x in 0..w {
                    let base = 18.0 + 205.0 * (0.35 * x as f32 / w as f32 + 0.5 * y as f32 / h as f32);
                    let noise = ((x * 37 + y * 11) % 13) as f32 - 6.0;
                    let grey = (base + noise).clamp(0.0, 255.0);
                    let r = (grey + 40.0 * x as f32 / w as f32).min(255.0) as u8;
                    let b = (grey - 25.0 * x as f32 / w as f32).max(0.0) as u8;
                    let o = (y * w + x) * 4;
                    px[o] = b;
                    px[o + 1] = grey as u8;
                    px[o + 2] = r;
                    px[o + 3] = 255;
                }
            }
            for bgra in [true, false] {
                let maps = viewer_maps(&px, w, h, bgra, ScopeWants::ALL);
                let naive = naive_viewer_maps(&px, w, h, bgra);
                assert_eq!(maps.words(), &naive[..], "{w}x{h} bgra {bgra}");
            }
        }
    }

    fn flat(w: usize, h: usize, px: [u8; 4]) -> Vec<u8> {
        px.iter().copied().cycle().take(w * h * 4).collect()
    }

    fn all() -> ScopeWants {
        ScopeWants::ALL
    }

    #[test]
    fn rounding_matches_f32_round_for_non_negative_inputs() {
        // Every level the panel can ask for, plus the values either side of each
        // half-level boundary.
        for level in 0..=255u32 {
            let v = level as f32;
            let expect = ((v / 255.0) * (VIEWER_LEVELS - 1) as f32).round() as usize;
            assert_eq!(viewer_level(v), expect, "level {level}");
        }
        for k in 0..160u32 {
            for offset in [-1e-3f32, -1e-6, 0.0, 1e-6, 1e-3] {
                let q = k as f32 + 0.5 + offset;
                let value = q / (VIEWER_LEVELS - 1) as f32 * 255.0;
                assert_eq!(viewer_round_away(q), q.round(), "q {q}");
            }
        }
        assert_eq!(viewer_round_away(-0.5), -1.0);
        assert_eq!(viewer_round_away(-1.4), -1.0);
    }

    #[test]
    fn rounding_matches_the_panel_for_every_byte_triple() {
        // The panel rounds with `f32::round()`; the counting loop uses
        // `(q + 0.5).trunc()` so the CUDA kernel can reproduce it exactly. Prove
        // the two agree everywhere the maps can be reached from 8-bit input.
        let panel_level = |v: f32| ((v / 255.0) * (VIEWER_LEVELS - 1) as f32).round() as usize;
        let mut checked = 0u32;
        for r in 0..=255u8 {
            for g in 0..=255u8 {
                for b in (0..=255u8).step_by(17).chain(Some(255)) {
                    let luma = viewer_luma(r, g, b);
                    assert_eq!(
                        viewer_level(luma),
                        panel_level(luma),
                        "waveform level for {r}/{g}/{b}"
                    );
                    let cb = (b as f32 - luma) / 1.8556 / 255.0;
                    let cr = (r as f32 - luma) / 1.5748 / 255.0;
                    let qx = (0.5 + cb) * (VIEWER_VECTOR - 1) as f32;
                    let qy = (0.5 - cr) * (VIEWER_VECTOR - 1) as f32;
                    assert!(qx >= 0.0 && qy >= 0.0, "non-negative chroma for {r}/{g}/{b}");
                    assert_eq!(viewer_round_up(qx), qx.round(), "Cb cell for {r}/{g}/{b}");
                    assert_eq!(viewer_round_up(qy), qy.round(), "Cr cell for {r}/{g}/{b}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 900_000, "swept {checked} triples");
        // Every half-level boundary at full f32 granularity on both sides.
        for k in 0..(VIEWER_LEVELS as u32 - 1) {
            let centre = k as f32 + 0.5;
            for step in -64i32..=64 {
                let q = f32::from_bits((centre.to_bits() as i32 + step) as u32);
                if q < 0.0 {
                    continue;
                }
                assert_eq!(viewer_round_up(q), q.round(), "q {q} near {centre}");
            }
        }
    }

    #[test]
    fn sampled_pixels_match_the_stride() {
        let (w, h) = (2000, 1333);
        let maps = viewer_maps(&flat(w, h, [90, 140, 200, 255]), w, h, true, all());
        let step = viewer_step(w, h);
        let expect = (w.div_ceil(step) * h.div_ceil(step)) as u32;
        assert_eq!(maps.sampled(), expect);
        let total: u32 = (0..VIEWER_LEVELS)
            .map(|level| (0..VIEWER_COLUMNS).map(|c| maps.wave_at(level, c)).sum::<u32>())
            .sum();
        assert_eq!(total, maps.sampled(), "every sampled pixel lands in one waveform bin");
    }

    #[test]
    fn white_sits_on_the_top_row_and_grey_in_the_middle() {
        let row_total = |maps: &ScopeMaps, row: usize| -> u32 {
            (0..VIEWER_COLUMNS).map(|c| maps.wave_at(row, c)).sum()
        };
        let white = viewer_maps(&flat(16, 16, [255, 255, 255, 255]), 16, 16, true, all());
        assert_eq!(row_total(&white, 0), white.sampled(), "white is the top row");
        let black = viewer_maps(&flat(16, 16, [0, 0, 0, 255]), 16, 16, true, all());
        assert_eq!(row_total(&black, VIEWER_LEVELS - 1), black.sampled(), "black is the bottom row");
        let grey = viewer_maps(&flat(16, 16, [128, 128, 128, 255]), 16, 16, true, all());
        // 128 -> 128/255*159 = 79.8 -> level 80, i.e. row 159 - 80.
        assert_eq!(row_total(&grey, VIEWER_LEVELS - 1 - 80), grey.sampled());
    }

    #[test]
    fn parade_lanes_keep_their_channels_apart() {
        // RGBA order, so this pixel is r=255 g=128 b=0.
        let maps = viewer_maps(&flat(32, 32, [255, 128, 0, 255]), 32, 32, false, all());
        let lane_total = |lane: usize, row: usize| -> u32 {
            (0..VIEWER_LANE_WIDTH).map(|c| maps.parade_at(lane, row, c)).sum()
        };
        assert_eq!(lane_total(0, 0), maps.sampled(), "red lane, top row");
        assert_eq!(lane_total(1, VIEWER_LEVELS - 1 - 80), maps.sampled(), "green lane, level 80");
        assert_eq!(lane_total(2, VIEWER_LEVELS - 1), maps.sampled(), "blue lane, bottom row");
        assert_eq!(maps.parade_at(0, VIEWER_LEVELS - 1, 0), 0);
        assert_eq!(maps.parade_at(3, 0, 0), 0, "out-of-range lane reads zero");
    }

    #[test]
    fn neutral_grey_sits_in_the_vectorscope_centre() {
        let maps = viewer_maps(&flat(16, 16, [128, 128, 128, 255]), 16, 16, true, all());
        assert_eq!(maps.vector_at(VIEWER_VECTOR / 2, VIEWER_VECTOR / 2), maps.sampled());
        let red = viewer_maps(&flat(16, 16, [255, 0, 0, 255]), 16, 16, true, all());
        let peak = (0..VIEWER_VECTOR)
            .flat_map(|y| (0..VIEWER_VECTOR).map(move |x| (x, y)))
            .max_by_key(|&(x, y)| red.vector_at(x, y))
            .unwrap();
        assert!(peak.0 >= VIEWER_VECTOR / 2, "red sits on the Cr side, got {peak:?}");
    }

    #[test]
    fn byte_order_only_changes_where_channels_are_read_from() {
        let rgba: Vec<u8> = (0..64usize * 64 * 4).map(|i| (i * 37 % 251) as u8).collect();
        let mut bgra = rgba.clone();
        for px in bgra.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        let a = viewer_maps(&rgba, 64, 64, false, all());
        let b = viewer_maps(&bgra, 64, 64, true, all());
        assert_eq!(a, b);
    }

    #[test]
    fn kinds_that_are_off_stay_zeroed() {
        let maps = viewer_maps(&flat(16, 16, [10, 20, 30, 255]), 16, 16, true, ScopeWants {
            waveform: true,
            ..ScopeWants::default()
        });
        let wave_total: u32 = maps.wave_bins().iter().sum();
        assert!(wave_total > 0, "waveform counts were asked for");
        let parade: u32 = (0..VIEWER_LANES)
            .map(|lane| {
                (0..VIEWER_LEVELS)
                    .map(|level| (0..VIEWER_LANE_WIDTH).map(|c| maps.parade_at(lane, level, c)).sum::<u32>())
                    .sum::<u32>()
            })
            .sum();
        assert_eq!(parade, 0);
        assert_eq!(maps.vector_at(0, 0), 0);
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        assert_eq!(viewer_maps(&[], 0, 0, true, all()).sampled(), 0);
        assert_eq!(viewer_maps(&[1, 2, 3, 4], 4, 4, true, all()).sampled(), 0);
        assert!(viewer_maps(&flat(4, 4, [1, 2, 3, 255]), 4, 4, true, ScopeWants::default()).is_empty());
    }

    #[test]
    fn from_words_round_trips_and_pads() {
        let maps = viewer_maps(&flat(8, 8, [7, 8, 9, 255]), 8, 8, true, all());
        let restored = ScopeMaps::from_words(maps.words(), maps.sampled());
        assert_eq!(restored, maps);
        let padded = ScopeMaps::from_words(&[1, 2, 3], 2);
        assert_eq!(padded.wave_at(0, 0), 1);
        assert_eq!(padded.wave_at(0, 1), 2);
        assert_eq!(padded.wave_at(159, 255), 0);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, px: [u8; 4]) -> Vec<u8> {
        px.iter().copied().cycle().take(w * h * 4).collect()
    }

    fn lit_rows(wave: &Waveform) -> Vec<usize> {
        (0..wave.levels)
            .filter(|&level| (0..wave.columns).any(|c| wave.at(level, c) > 0))
            .collect()
    }

    #[test]
    fn black_lands_on_bottom_row() {
        let wave = waveform(&image(8, 8, [0, 0, 0, 255]), 8, 8, 8, 256);
        assert_eq!(lit_rows(&wave), vec![0]);
        assert_eq!(wave.at(0, 3), 255);
    }

    #[test]
    fn white_lands_on_top_row() {
        let wave = waveform(&image(8, 8, [255, 255, 255, 255]), 8, 8, 8, 256);
        assert_eq!(lit_rows(&wave), vec![255]);
        assert_eq!(wave.at(255, 0), 255);
    }

    #[test]
    fn mid_gray_sits_at_middle_level() {
        let wave = waveform(&image(4, 4, [128, 128, 128, 255]), 4, 4, 4, 256);
        assert_eq!(lit_rows(&wave), vec![128]);
    }

    #[test]
    fn gray_vectorscope_is_centered() {
        let scope = vectorscope(&image(8, 8, [128, 128, 128, 255]), 8, 8, 64);
        assert_eq!(scope.data.iter().copied().max(), Some(255));
        assert_eq!(scope.at(32, 32), 255);
    }

    #[test]
    fn saturated_colors_point_the_right_way() {
        let peak = |px: [u8; 4]| {
            let scope = vectorscope(&image(8, 8, px), 8, 8, 64);
            let i = scope
                .data
                .iter()
                .enumerate()
                .max_by_key(|entry| *entry.1)
                .map(|entry| entry.0)
                .unwrap();
            ((i % 64) as i32, (i / 64) as i32)
        };
        let (x, _) = peak([255, 0, 0, 255]);
        assert!(x > 32, "red sits right of centre, got x={x}");
        let (x, y) = peak([0, 0, 255, 255]);
        assert!(y < 32, "blue sits above centre, got {x}/{y}");
        let (x, y) = peak([0, 255, 0, 255]);
        assert!(x < 32 && y > 32, "green sits left and below centre, got {x}/{y}");
    }

    #[test]
    fn empty_input_is_safe() {
        assert!(waveform(&[], 0, 0, 16, 16).data.iter().all(|&v| v == 0));
        assert!(waveform(&[], 4, 4, 16, 16).data.iter().all(|&v| v == 0));
        assert!(vectorscope(&[], 4, 4, 16).data.iter().all(|&v| v == 0));
        assert!(parades(&[], 0, 0).data.iter().all(|&v| v == 0));
    }

    #[test]
    fn size_parameters_are_respected() {
        assert!(waveform(&image(4, 4, [1, 2, 3, 0]), 4, 4, 0, 8).data.is_empty());
        assert!(waveform(&image(4, 4, [1, 2, 3, 0]), 4, 4, 8, 0).data.is_empty());
        assert!(vectorscope(&image(4, 4, [1, 2, 3, 0]), 4, 4, 0).data.is_empty());
        let one = waveform(&image(4, 4, [0, 0, 0, 255]), 4, 4, 1, 1);
        assert_eq!((one.columns, one.levels, one.at(0, 0)), (1, 1, 255));
        assert_eq!(vectorscope(&image(4, 4, [255, 0, 0, 255]), 4, 4, 1).at(0, 0), 255);
    }

    #[test]
    fn parades_has_three_sections() {
        let wave = parades(&image(8, 8, [255, 128, 0, 255]), 8, 8);
        assert_eq!((wave.columns, wave.levels), (768, 256));
        assert_eq!(wave.at(255, 0), 255, "red section");
        assert_eq!(wave.at(128, 256), 255, "green section");
        assert_eq!(wave.at(0, 512), 255, "blue section");
    }

    #[test]
    #[ignore]
    fn perf_probe() {
        let (w, h) = (2000, 1333);
        let rgba = image(w, h, [90, 140, 200, 255]);
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let wave = waveform(&rgba, w, h, 512, 256);
            let wave_ms = t.elapsed();
            let t = std::time::Instant::now();
            let scope = vectorscope(&rgba, w, h, 256);
            let vec_ms = t.elapsed();
            println!(
                "waveform {wave_ms:?} ({} B)  vectorscope {vec_ms:?} ({} B)",
                wave.data.len(),
                scope.data.len()
            );
        }
    }
}
