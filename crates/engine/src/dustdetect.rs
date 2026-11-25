//! AI dust detection: `dust.onnx`, the model behind Capture One's
//! `IC_DetectDust` automatic dust picking.
//!
//! Contract recovered from `ImgCoreDll.dll` (see `analysis/ai_models.md`):
//! input `[1,3,416,416]` float32 RGB in 0..255; output `[1,3549,6]`.  The
//! three grids are strides 8/16/32, 52²/26²/13² cells.  A row is
//! `cx_in_cell, cy_in_cell, log(w/stride), log(h/stride), objectness, class`.
//! The image is scaled by `f = 416/min_side + 0.1` (floor 0.5 under 80 MP,
//! 0.25 above), tiled 416 with 100 px overlap, and the last tile is flush with
//! the right/bottom edge.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64 as TickCounter, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::aimask::{Runtime, Session};
use crate::{Spot, SpotKind};

pub const INPUT_SIDE: usize = 416;
pub const ANCHORS: usize = 52 * 52 + 26 * 26 + 13 * 13;
pub const MAX_SPOTS: usize = 200;
/// `DAT_182a2bc70`; the old 0.3 dropped faint dust.
pub const DEFAULT_CONFIDENCE: f32 = 0.15;
const TILE_STEP: usize = INPUT_SIDE - 100;
pub(crate) const MIN_RADIUS_PX: f32 = 1.5;
pub(crate) const MAX_RADIUS_PX: f32 = 100.0;
pub(crate) const NMS_IOU: f32 = 0.45;
const STRIDES: [usize; 3] = [8, 16, 32];

/// `AURORA_DUST_ONNX=1` turns automatic detection on.
pub fn enabled() -> bool {
    std::env::var("AURORA_DUST_ONNX").ok().as_deref() == Some("1")
}

pub fn min_confidence() -> f32 {
    std::env::var("AURORA_DUST_CONF")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(DEFAULT_CONFIDENCE)
}

/// Image-to-canvas scale.  `None` means C1 itself rejects the image as too
/// small (`Dust detection rejected image because it is too small`).
fn spot_scale(w: u32, h: u32) -> Option<f32> {
    let min_side = w.min(h) as f32;
    let floor = if (w as u64) * (h as u64) < 80_000_001 { 0.5 } else { 0.25 };
    let f = (INPUT_SIDE as f32 / min_side + 0.1).max(floor);
    (f <= 1.0).then_some(f)
}

pub(crate) fn spot_tiles(n: usize) -> Vec<usize> {
    if n <= INPUT_SIDE {
        return vec![0];
    }
    let mut out: Vec<usize> = (0..=n - INPUT_SIDE).step_by(TILE_STEP).collect();
    if out.last().copied() != Some(n - INPUT_SIDE) {
        out.push(n - INPUT_SIDE);
    }
    out
}

pub(crate) struct RawSpot {
    pub score: f32,
    pub cx: f32,
    pub cy: f32,
    pub side: f32,
}

/// Everything, apart from the model, that decides the pixels fed to the dust
/// model.
///
/// The four fingerprints are the `preview_cache` parameter groups that run
/// before dust reads the frame (characterization, geometry, develop and
/// detail). `local` (spots/layers/nodes) and `portrait` (retouch/extras) run
/// after it and stay out of the key, so moving a spot or a layer still hits.
/// Film grain lives in `develop`: the detail stage applies it before dust, and
/// it is deterministic (a fixed hash offset, no random seed), so its parameters
/// fully describe its contribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DustInputKey {
    /// The immutable decoded frame (`Source::snapshot_id`).
    pub source: u64,
    pub characterization: u64,
    pub geometry: u64,
    pub develop: u64,
    pub detail: u64,
    /// Viewport rectangle and output size.
    pub region: [u32; 4],
    pub out: [u32; 2],
    /// Detail radii and grain sigma are multiplied by this scale.
    pub scale_bits: u32,
    /// Mip level the source is sampled from.
    pub level: u32,
    /// A wide (16-bit export) render drives the 8-bit preview buffer through a
    /// different detail sequence.
    pub wide: bool,
}

/// Model identity: the resolved path plus the file's metadata and
/// first/middle/last KiB. A model replacement changes this even when the path
/// is reused, so stale sessions and detections cannot survive an ONNX update.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ModelIdentity {
    path: PathBuf,
    fingerprint: Vec<u8>,
}

fn model_identity() -> Result<ModelIdentity> {
    let path = model_path()?;
    let fingerprint = model_fingerprint(&path);
    Ok(ModelIdentity { path, fingerprint })
}

/// Content fingerprint of the model file: metadata plus head/middle/tail KiB.
/// Read on every call, including cache hits: the read *is* the replacement
/// check. Memoising the bytes by path/length/mtime would skip it but would also
/// miss a rewrite that preserves length and mtime, so the stronger check stays.
fn model_fingerprint(path: &Path) -> Vec<u8> {
    let mut out = Vec::new();
    let Ok(meta) = std::fs::metadata(path) else { return out };
    out.extend_from_slice(&meta.len().to_le_bytes());
    out.extend_from_slice(&meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|t| t.as_secs().to_le_bytes()).unwrap_or([0; 8]));
    use std::io::{Read, Seek, SeekFrom};
    if let Ok(mut file) = std::fs::File::open(path) {
        for at in [SeekFrom::Start(0), SeekFrom::End(-1024), SeekFrom::End(-2048)] {
            let mut bytes = [0u8; 1024];
            if file.seek(at).is_ok() && file.read(&mut bytes).unwrap_or(0) > 0 {
                out.extend_from_slice(&bytes);
            }
        }
    }
    out
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    input: DustInputKey,
    threshold_bits: u32,
    model: ModelIdentity,
}

struct CacheEntry {
    key: CacheKey,
    spots: Arc<Vec<Spot>>,
    last_used: u64,
}

const DUST_CACHE_ENTRIES: usize = 8;

fn cache() -> &'static Mutex<Vec<CacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn next_tick() -> u64 {
    static TICK: TickCounter = TickCounter::new(1);
    TICK.fetch_add(1, Ordering::Relaxed)
}

/// Least-recently-used lookup: a hit refreshes the entry's recency.
fn cache_find(entries: &mut [CacheEntry], key: &CacheKey, now: u64) -> Option<Arc<Vec<Spot>>> {
    let entry = entries.iter_mut().find(|entry| &entry.key == key)?;
    entry.last_used = now;
    Some(Arc::clone(&entry.spots))
}

/// Inserts with the newest recency, replacing an equal key. When full it evicts
/// the least recently used entry instead of the oldest inserted (FIFO before).
fn cache_store(entries: &mut Vec<CacheEntry>, key: CacheKey, spots: Arc<Vec<Spot>>, now: u64) {
    entries.retain(|entry| entry.key != key);
    if entries.len() >= DUST_CACHE_ENTRIES {
        if let Some(index) = entries
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(index, _)| index)
        {
            entries.remove(index);
        }
    }
    entries.push(CacheEntry { key, spots, last_used: now });
}

fn cache_insert(key: CacheKey, spots: Arc<Vec<Spot>>) {
    let mut entries = cache().lock().unwrap_or_else(|error| error.into_inner());
    cache_store(&mut entries, key, spots, next_tick());
}

/// One in-flight detection. `done` is signalled after `result` is written.
struct Flight<V> {
    result: Mutex<Option<Result<V, String>>>,
    done: Condvar,
}

/// Runs `work` once per key; concurrent callers with the same key wait for the
/// leader and receive a clone of its result. The finished flight is removed, so
/// a process that renders many frames does not accumulate one entry per frame:
/// the leader stores its answer in the LRU before returning, so a caller that
/// arrives after removal finds the cache or starts one fresh attempt.
///
/// Failures are shared with the waiters but never cached, and a panicking
/// leader becomes an error instead of hanging the waiters.
///
/// `detect_dust` itself fans out with rayon. A waiter blocks on a condvar and
/// leaves the pool; the leader still finishes because rayon's joins run the
/// remaining work inline on the leader, so this cannot deadlock.
fn single_flight<K, V>(
    store: &OnceLock<Mutex<HashMap<K, Arc<Flight<V>>>>>,
    key: K,
    work: impl FnOnce() -> Result<V, String>,
) -> Result<V, String>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    let flights = store.get_or_init(|| Mutex::new(HashMap::new()));
    let (flight, leader) = {
        let mut flights = flights.lock().unwrap_or_else(|error| error.into_inner());
        match flights.get(&key) {
            Some(flight) => (Arc::clone(flight), false),
            None => {
                let flight = Arc::new(Flight { result: Mutex::new(None), done: Condvar::new() });
                flights.insert(key.clone(), Arc::clone(&flight));
                (flight, true)
            }
        }
    };
    if leader {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
            .unwrap_or_else(|_| Err("dust detection panicked".to_string()));
        *flight.result.lock().unwrap_or_else(|error| error.into_inner()) = Some(outcome);
        // Publish before the key disappears: a later caller then finds either
        // this flight's answer or the LRU entry, never an empty gap.
        flights.lock().unwrap_or_else(|error| error.into_inner()).remove(&key);
        flight.done.notify_all();
    }
    let mut result = flight.result.lock().unwrap_or_else(|error| error.into_inner());
    while result.is_none() {
        result = flight
            .done
            .wait(result)
            .unwrap_or_else(|error| error.into_inner());
    }
    result
        .as_ref()
        .expect("single_flight result is written before any waiter returns")
        .clone()
}

fn dust_flights() -> &'static OnceLock<Mutex<HashMap<CacheKey, Arc<Flight<Arc<Vec<Spot>>>>>>> {
    static FLIGHTS: OnceLock<Mutex<HashMap<CacheKey, Arc<Flight<Arc<Vec<Spot>>>>>>> =
        OnceLock::new();
    &FLIGHTS
}

/// Cached automatic dust detection.
///
/// The key covers the source snapshot, every parameter group that runs before
/// dust reads the frame, the viewport and output geometry, the confidence
/// threshold and the model identity. Colour and tone sliders are part of it:
/// the earlier key omitted them and could serve spots computed from different
/// pixels. Spots, retouch and layers are absent because they are applied after
/// detection.
pub fn detect_dust_cached(
    input: DustInputKey,
    rgba: &[u8],
    w: u32,
    h: u32,
    min_confidence: f32,
) -> Result<Arc<Vec<Spot>>> {
    let key = CacheKey {
        input,
        threshold_bits: min_confidence.to_bits(),
        model: model_identity()?,
    };
    {
        let mut entries = cache().lock().unwrap_or_else(|error| error.into_inner());
        if let Some(found) = cache_find(&mut entries, &key, next_tick()) {
            return Ok(found);
        }
    }
    let cache_key = key.clone();
    let spots = single_flight(dust_flights(), key, || {
        let spots = Arc::new(
            detect_dust_with_model(&cache_key.model, rgba, w, h, min_confidence)
                .map_err(|error| format!("{error:#}"))?,
        );
        cache_insert(cache_key.clone(), Arc::clone(&spots));
        Ok(spots)
    })
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok(spots)
}

/// Detects dust spots in a developed RGBA frame.  Returned coordinates are
/// normalised like every other `Spot`; `radius` is a fraction of the long side.
pub fn detect_dust(rgba: &[u8], w: u32, h: u32, min_confidence: f32) -> Result<Vec<Spot>> {
    let model = model_identity()?;
    detect_dust_with_model(&model, rgba, w, h, min_confidence)
}

fn detect_dust_with_model(
    model: &ModelIdentity,
    rgba: &[u8],
    w: u32,
    h: u32,
    min_confidence: f32,
) -> Result<Vec<Spot>> {
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let need = w as usize * h as usize * 4;
    if rgba.len() < need {
        bail!("dust detection: buffer has {} bytes, need {need} for {w}x{h}", rgba.len());
    }
    let Some(scale) = spot_scale(w, h) else {
        bail!("Dust detection rejected image because it is too small");
    };
    let canvas_w = ((w as f32 * scale).ceil() as usize).max(INPUT_SIDE);
    let canvas_h = ((h as f32 * scale).ceil() as usize).max(INPUT_SIDE);
    let canvas = resize_canvas(rgba, w as usize, h as usize, canvas_w, canvas_h);
    let session = session_for(model)?;
    let runtime = Runtime::get()
        .ok_or_else(|| anyhow::anyhow!("ONNX Runtime is not available for dust detection"))?;
    let tiles_x = spot_tiles(canvas_w);
    let tiles_y = spot_tiles(canvas_h);
    let tensor_len = 3 * INPUT_SIDE * INPUT_SIDE;
    let found: Result<Vec<Vec<RawSpot>>> = tiles_y
        .par_iter()
        .map(|&ty| {
            let mut tile_input = vec![0.0f32; tensor_len];
            let mut row = Vec::new();
            for &tx in &tiles_x {
                fill_tile_tensor(&mut tile_input, &canvas, canvas_w, canvas_h, tx, ty);
                let planes = runtime.run(
                    &session,
                    &mut tile_input,
                    &[1, 3, INPUT_SIDE as i64, INPUT_SIDE as i64],
                )?;
                let rows = planes
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("dust model returned no output"))?;
                row.extend(decode_tile(rows, tx, ty, scale, min_confidence)?);
            }
            Ok(row)
        })
        .collect();
    let mut found: Vec<RawSpot> = found?.into_iter().flatten().collect();
    found.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<RawSpot> = Vec::new();
    for spot in found {
        if kept.len() >= MAX_SPOTS {
            break;
        }
        let radius = spot.side * 0.5;
        if !(MIN_RADIUS_PX..=MAX_RADIUS_PX).contains(&radius) {
            continue;
        }
        if kept.iter().any(|k| iou(&spot, k) > NMS_IOU) {
            continue;
        }
        kept.push(spot);
    }
    let long = w.max(h) as f32;
    Ok(kept
        .into_iter()
        .enumerate()
        .map(|(i, s)| Spot {
            id: i as u64 + 1,
            kind: SpotKind::Dust,
            x: (s.cx / w as f32).clamp(0.0, 1.0),
            y: (s.cy / h as f32).clamp(0.0, 1.0),
            radius: (s.side * 0.5 / long).clamp(0.0, 1.0),
        })
        .collect())
}

struct SessionSlot {
    identity: ModelIdentity,
    outcome: Result<Session, String>,
}

fn session_slot() -> &'static Mutex<Option<SessionSlot>> {
    static SLOT: OnceLock<Mutex<Option<SessionSlot>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

fn session_error(error: String) -> anyhow::Error {
    anyhow::anyhow!("dust detection unavailable: {error}")
}

/// The session for `identity`, reopened when the model file changes.
///
/// The old `OnceLock` kept the first session for the life of the process, so
/// replacing `dust.onnx` kept running the old graph even after the cache key
/// had moved on. The outcome (including a failure) is stored per identity, so
/// a broken model is not reopened on every render while a replaced one is.
fn session_for(identity: &ModelIdentity) -> Result<Session> {
    {
        let slot = session_slot().lock().unwrap_or_else(|error| error.into_inner());
        if let Some(slot) = slot.as_ref() {
            if slot.identity == *identity {
                return slot.outcome.clone().map_err(session_error);
            }
        }
    }
    // Open outside the lock; `Runtime::open_session` is itself single-flight
    // per model key, so concurrent callers share one graph.
    let outcome = open_dust_session(&identity.path);
    let mut slot = session_slot().lock().unwrap_or_else(|error| error.into_inner());
    if let Some(existing) = slot.as_ref() {
        if existing.identity == *identity {
            return existing.outcome.clone().map_err(session_error);
        }
    }
    *slot = Some(SessionSlot { identity: identity.clone(), outcome: outcome.clone() });
    outcome.map_err(session_error)
}

fn open_dust_session(path: &Path) -> Result<Session, String> {
    let runtime = Runtime::get().ok_or_else(|| "ONNX Runtime is not available".to_string())?;
    let session = runtime.open_session(path).map_err(|error| format!("{error:#}"))?;
    let dims = &session.input.dims;
    if dims != &[1, 3, INPUT_SIDE as i64, INPUT_SIDE as i64] || session.input.element_type != 1 {
        return Err(format!("dust.onnx: unexpected input {dims:?}"));
    }
    if session.outputs.len() != 1 || session.outputs[0].name.is_empty() {
        return Err("dust.onnx: expected one output".into());
    }
    Ok(session)
}

/// `AURORA_DUST_MODEL` wins, then `AURORA_AIMODELS`, then the Capture One
/// install.  The real model lives in `AIModels/Common/DustRemoval/dust.onnx`.
fn model_path() -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = std::env::var_os("AURORA_DUST_MODEL") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(dir) = std::env::var_os("AURORA_AIMODELS") {
        candidates.push(PathBuf::from(&dir).join("dust.onnx"));
        candidates.push(PathBuf::from(&dir).join("Common/DustRemoval/dust.onnx"));
        candidates.push(PathBuf::from(&dir).join("AIModels/Common/DustRemoval/dust.onnx"));
    }
    for root in crate::aimask::c1_roots() {
        candidates.push(root.join("AIModels/Common/DustRemoval/dust.onnx"));
        candidates.push(root.join("AIModels/Common/dust.onnx"));
        candidates.push(root.join("dust.onnx"));
    }
    if let Some(path) = candidates.iter().find(|p| p.is_file()) {
        return Ok(path.clone());
    }
    bail!(
        "dust.onnx not found; set AURORA_DUST_MODEL to the model file (looked in {})",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// Bilinear resize to the C1 canvas, RGB 0..255 (the model's trained range).
pub(crate) fn resize_canvas(rgba: &[u8], w: usize, h: usize, cw: usize, ch: usize) -> Vec<f32> {
    let mut canvas = vec![0.0f32; cw * ch * 3];
    canvas
        .par_chunks_mut(cw * 3)
        .enumerate()
        .for_each(|(y, line)| {
            let sy = (y as f32 + 0.5) * h as f32 / ch as f32 - 0.5;
            let y0 = sy.floor().max(0.0) as usize;
            let y1 = (y0 + 1).min(h - 1);
            let fy = (sy - y0 as f32).clamp(0.0, 1.0);
            for (x, pixel) in line.chunks_exact_mut(3).enumerate() {
                let sx = (x as f32 + 0.5) * w as f32 / cw as f32 - 0.5;
                let x0 = sx.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(w - 1);
                let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                let at = |xx: usize, yy: usize, c: usize| rgba[(yy * w + xx) * 4 + c] as f32;
                for c in 0..3 {
                    let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
                    let bottom = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
                    pixel[c] = top * (1.0 - fy) + bottom * fy;
                }
            }
        });
    canvas
}

/// Fills a reusable tile tensor in place (the caller zero-fills once and keeps
/// it for the rest of the row), avoiding a 1.9 MiB allocation per inference.
fn fill_tile_tensor(out: &mut [f32], canvas: &[f32], cw: usize, ch: usize, tx: usize, ty: usize) {
    out.fill(0.0);
    for y in 0..INPUT_SIDE {
        let sy = ty + y;
        if sy >= ch {
            continue;
        }
        for x in 0..INPUT_SIDE {
            let sx = tx + x;
            if sx >= cw {
                continue;
            }
            let si = (sy * cw + sx) * 3;
            let di = y * INPUT_SIDE + x;
            for c in 0..3 {
                out[c * INPUT_SIDE * INPUT_SIDE + di] = canvas[si + c];
            }
        }
    }
}

pub(crate) fn tile_tensor(canvas: &[f32], cw: usize, ch: usize, tx: usize, ty: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; 3 * INPUT_SIDE * INPUT_SIDE];
    fill_tile_tensor(&mut out, canvas, cw, ch, tx, ty);
    out
}

/// `FUN_181435ef0`: anchors 8/16/32, `w=exp(col2)*stride`, `conf=col4*col5`.
pub(crate) fn decode_tile(
    rows: &[f32],
    x_off: usize,
    y_off: usize,
    scale: f32,
    min_confidence: f32,
) -> Result<Vec<RawSpot>> {
    if rows.len() < ANCHORS * 6 {
        bail!("dust model returned {} values, expected {}", rows.len(), ANCHORS * 6);
    }
    let threshold = if min_confidence.is_finite() { min_confidence } else { 0.0 };
    let mut found = Vec::new();
    let mut index = 0usize;
    for stride in STRIDES {
        let grid = INPUT_SIDE / stride;
        for gy in 0..grid {
            for gx in 0..grid {
                let p = &rows[index * 6..index * 6 + 6];
                index += 1;
                let score = p[4] * p[5];
                if !score.is_finite() || score <= threshold {
                    continue;
                }
                let bw = p[2].exp() * stride as f32;
                let bh = p[3].exp() * stride as f32;
                if !bw.is_finite() || !bh.is_finite() || bw <= 0.0 || bh <= 0.0 {
                    continue;
                }
                let cx = (x_off as f32 + (gx as f32 + p[0]) * stride as f32) / scale;
                let cy = (y_off as f32 + (gy as f32 + p[1]) * stride as f32) / scale;
                let side = bw.max(bh) / scale;
                if cx.is_finite() && cy.is_finite() && side.is_finite() && side > 0.0 {
                    found.push(RawSpot { score, cx, cy, side });
                }
            }
        }
    }
    Ok(found)
}

pub(crate) fn iou(a: &RawSpot, b: &RawSpot) -> f32 {
    let ah = a.side * 0.5;
    let bh = b.side * 0.5;
    let w = (a.cx + ah).min(b.cx + bh) - (a.cx - ah).max(b.cx - bh);
    let h = (a.cy + ah).min(b.cy + bh) - (a.cy - ah).max(b.cy - bh);
    if w <= 0.0 || h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    inter / (a.side * a.side + b.side * b.side - inter).max(f32::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_input(source: u64) -> DustInputKey {
        DustInputKey {
            source,
            characterization: 1,
            geometry: 2,
            develop: 3,
            detail: 4,
            region: [0, 0, 1.0f32.to_bits(), 1.0f32.to_bits()],
            out: [64, 64],
            scale_bits: 1.0f32.to_bits(),
            level: 0,
            wide: false,
        }
    }

    fn key_for(source: u64) -> CacheKey {
        CacheKey {
            input: key_input(source),
            threshold_bits: DEFAULT_CONFIDENCE.to_bits(),
            model: ModelIdentity {
                path: PathBuf::from("dust.onnx"),
                fingerprint: vec![7, 7, 7],
            },
        }
    }

    /// The key must move when a stage applied before dust changes, and stay put
    /// for the stages applied after it.
    #[test]
    fn dust_input_fingerprint_tracks_pre_dust_stages_only() {
        let base = crate::AdjustmentSet::default();
        let baseline = crate::preview_cache::dust_input_fingerprint(&base);

        // Film grain is applied by the detail stage before dust.
        let mut grain = base.clone();
        grain.film_grain.impact = 60.0;
        assert_ne!(
            crate::preview_cache::dust_input_fingerprint(&grain).develop,
            baseline.develop,
            "film grain changes the pixels fed to the model"
        );

        // A colour/tone slider changes the developed pixels too.
        let mut exposure = base.clone();
        exposure.exposure.exposure = 0.5;
        assert_ne!(
            crate::preview_cache::dust_input_fingerprint(&exposure).develop,
            baseline.develop
        );

        // Spots and portrait extras are applied after detection.
        let mut spots = base.clone();
        spots.spots.push(Spot { id: 1, kind: SpotKind::Dust, x: 0.1, y: 0.1, radius: 0.1 });
        assert_eq!(crate::preview_cache::dust_input_fingerprint(&spots), baseline);

        let mut extra = base.clone();
        extra.extra.insert("pc:21093".into(), 1.0);
        assert_eq!(crate::preview_cache::dust_input_fingerprint(&extra), baseline);
    }

    #[test]
    fn dust_cache_evicts_the_least_recently_used_entry() {
        let mut entries = Vec::new();
        for source in 1..=DUST_CACHE_ENTRIES as u64 {
            cache_store(&mut entries, key_for(source), Arc::new(Vec::new()), source);
        }
        assert_eq!(entries.len(), DUST_CACHE_ENTRIES);
        // Refresh key 1, so key 2 becomes the oldest.
        assert!(cache_find(&mut entries, &key_for(1), 100).is_some());
        cache_store(&mut entries, key_for(99), Arc::new(Vec::new()), 101);
        assert_eq!(entries.len(), DUST_CACHE_ENTRIES);
        assert!(
            cache_find(&mut entries, &key_for(2), 102).is_none(),
            "least recently used entry must be evicted"
        );
        assert!(
            cache_find(&mut entries, &key_for(1), 103).is_some(),
            "recently used entry must survive"
        );
    }

    #[test]
    fn dust_cache_replaces_an_equal_key_without_growing() {
        let mut entries = Vec::new();
        cache_store(&mut entries, key_for(1), Arc::new(Vec::new()), 1);
        let spots = Arc::new(vec![Spot {
            id: 1,
            kind: SpotKind::Dust,
            x: 0.1,
            y: 0.2,
            radius: 0.3,
        }]);
        cache_store(&mut entries, key_for(1), Arc::clone(&spots), 2);
        assert_eq!(entries.len(), 1);
        let found = cache_find(&mut entries, &key_for(1), 3).expect("entry");
        assert!(Arc::ptr_eq(&found, &spots));
    }

    #[test]
    fn single_flight_runs_one_detection_for_concurrent_callers() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        let store: OnceLock<Mutex<HashMap<String, Arc<Flight<u32>>>>> = OnceLock::new();
        let runs = AtomicUsize::new(0);
        let barrier = std::sync::Barrier::new(8);
        let results: Vec<u32> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        single_flight(&store, "same-key".to_string(), || {
                            runs.fetch_add(1, AtomicOrdering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(150));
                            Ok(7u32)
                        })
                        .expect("shared result")
                    })
                })
                .collect();
            handles.into_iter().map(|handle| handle.join().unwrap()).collect()
        });
        assert_eq!(runs.load(AtomicOrdering::SeqCst), 1, "one leader must run the work");
        assert!(results.iter().all(|value| *value == 7));
    }

    #[test]
    fn single_flight_shares_failure_but_retries_the_next_call() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        let store: OnceLock<Mutex<HashMap<String, Arc<Flight<u32>>>>> = OnceLock::new();
        let calls = AtomicUsize::new(0);
        let first: Result<u32, String> = single_flight(&store, "err-key".to_string(), || {
            calls.fetch_add(1, AtomicOrdering::SeqCst);
            Err("boom".to_string())
        });
        assert!(first.is_err());
        let second = single_flight(&store, "err-key".to_string(), || {
            calls.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(3u32)
        });
        assert_eq!(second.unwrap(), 3);
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[test]
    fn single_flight_turns_a_panicking_leader_into_an_error() {
        let store: OnceLock<Mutex<HashMap<String, Arc<Flight<u32>>>>> = OnceLock::new();
        let outcome: Result<u32, String> =
            single_flight(&store, "panic-key".to_string(), || panic!("boom"));
        assert!(outcome.is_err());
        let retry = single_flight(&store, "panic-key".to_string(), || Ok(5u32));
        assert_eq!(retry.unwrap(), 5);
    }

    fn rows_with(detections: &[(usize, [f32; 6])]) -> Vec<f32> {
        let mut rows = vec![0.0f32; ANCHORS * 6];
        for (index, row) in detections {
            rows[index * 6..index * 6 + 6].copy_from_slice(row);
        }
        rows
    }

    #[test]
    fn empty_result_when_nothing_passes_the_threshold() {
        let rows = rows_with(&[(0, [0.5, 0.5, 0.0, 0.0, 0.1, 0.1])]);
        assert!(decode_tile(&rows, 0, 0, 1.0, 0.15).unwrap().is_empty());
        assert_eq!(decode_tile(&rows, 0, 0, 1.0, 0.0).unwrap().len(), 1);
    }

    #[test]
    fn exp_size_and_cell_offsets_are_decoded() {
        let rows = rows_with(&[(5 * 52 + 10, [0.5, 0.5, 0.0, 0.0, 0.9, 0.9])]);
        let spots = decode_tile(&rows, 0, 0, 1.0, 0.15).unwrap();
        assert_eq!(spots.len(), 1);
        assert!((spots[0].cx - 10.5 * 8.0).abs() < 1e-4);
        assert!((spots[0].cy - 5.5 * 8.0).abs() < 1e-4);
        assert!((spots[0].side - 8.0).abs() < 1e-4);
    }

    #[test]
    fn tile_offsets_and_scale_return_to_image_pixels() {
        let rows = rows_with(&[(0, [0.5, 0.5, 0.0, 0.0, 0.9, 0.9])]);
        let spots = decode_tile(&rows, 316, 100, 0.5, 0.15).unwrap();
        assert!((spots[0].cx - (316.0 + 4.0) / 0.5).abs() < 1e-3);
        assert!((spots[0].cy - (100.0 + 4.0) / 0.5).abs() < 1e-3);
        assert!((spots[0].side - 16.0).abs() < 1e-3);
    }

    #[test]
    fn non_positive_boxes_are_rejected() {
        let rows = rows_with(&[(0, [0.5, 0.5, f32::NEG_INFINITY, 0.0, 0.9, 0.9])]);
        assert!(decode_tile(&rows, 0, 0, 1.0, 0.0).unwrap().is_empty());
    }

    #[test]
    fn overlapping_tiles_are_deduplicated() {
        let a = RawSpot { score: 0.9, cx: 100.0, cy: 100.0, side: 20.0 };
        let b = RawSpot { score: 0.8, cx: 102.0, cy: 99.0, side: 20.0 };
        assert!(iou(&a, &b) > NMS_IOU);
    }

    #[test]
    fn scale_and_tiles_match_c1() {
        assert!(spot_scale(416, 416).is_none());
        let f = spot_scale(6048, 4032).unwrap();
        assert!((f - 0.5).abs() < 1e-6);
        let tiles = spot_tiles(2016);
        assert_eq!(tiles.first(), Some(&0));
        assert_eq!(tiles.last(), Some(&(2016 - 416)));
    }

    #[test]
    fn short_buffer_is_an_error() {
        assert!(detect_dust(&[0u8; 16], 8, 8, 0.15).is_err());
    }

    /// Real model: 600×600 with three dark dots, C1 scale/tile path.
    #[test]
    #[ignore]
    fn detect_synthetic_dots_with_the_real_model() {
        let (w, h) = (600u32, 600u32);
        let mut buf = vec![153u8; (w * h * 4) as usize];
        let mut dots = Vec::new();
        for (cx, cy) in [(150i64, 150i64), (300, 150), (450, 450)] {
            dots.push((cx, cy));
            for y in cy - 12..=cy + 12 {
                for x in cx - 12..=cx + 12 {
                    if (x - cx).pow(2) + (y - cy).pow(2) <= 144 {
                        let p = ((y as u32 * w + x as u32) * 4) as usize;
                        buf[p..p + 3].fill(0);
                    }
                }
            }
        }
        let start = std::time::Instant::now();
        let spots = detect_dust(&buf, w, h, 0.15).expect("detection");
        println!("detected {} spots in {:?}", spots.len(), start.elapsed());
        for s in &spots {
            println!("  x={:.4} y={:.4} r={:.4} (px {:.1})", s.x, s.y, s.radius, s.radius * 600.0);
        }
        for (cx, cy) in dots {
            assert!(
                spots.iter().any(|s| (s.x * 600.0 - cx as f32).abs() < 25.0
                    && (s.y * 600.0 - cy as f32).abs() < 25.0),
                "missing dot at ({cx},{cy})"
            );
        }
    }

    #[test]
    #[ignore]
    fn flat_image_has_no_spots_with_the_real_model() {
        let buf = vec![153u8; 600 * 600 * 4];
        let spots = detect_dust(&buf, 600, 600, 0.15).expect("detection");
        assert!(spots.is_empty(), "flat frame produced {spots:?}");
    }

    #[test]
    #[ignore]
    fn detect_real_9868_frame() {
        let path = r"C:\WorkSpace\Aurora Studio\90_work\build-cache\z6iii-desktop-smoke-a88d98b0\output\DSC_9868.tif";
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let image = image::open(path).unwrap().to_rgba8();
        let (w, h) = image.dimensions();
        let spots = detect_dust(image.as_raw(), w, h, 0.15).expect("detection");
        println!("real 9868: {} spots in {w}x{h}", spots.len());
        for s in spots.iter().take(20) {
            println!("  x={:.4} y={:.4} r={:.4}", s.x, s.y, s.radius);
        }
    }
}
