//! On-disk cache of decoded RAW previews.
//!
//! libraw takes seconds on a full sensor frame, and reopening a file used to pay
//! that again. This keeps the downscaled decode as a lossless PNG, keyed by the
//! source path, size, mtime and requested long side.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use image::{ImageFormat, RgbaImage as ImageBuffer};

use crate::mock::process::Source;
use crate::raw::{DecodeProvenance, DecodeRoute};
use crate::RgbaImage;

/// Only files with this prefix are ever evicted; the caller shares the folder.
const PREFIX: &str = "raw-";
// Bump when the persisted layout changes. B9 persists the source content
// identity and the route digest next to the B8 stage-mix fingerprint, so B8 and
// older files miss naturally.
const RAW16_MAGIC: &[u8; 8] = b"AUR16B9\0";
/// magic(8) width(4) height(4) body bytes(8) image hash(8) body crc32(4)
/// pad(4) pipeline fingerprint(8) source len(8) source content(8)
/// source mtime(8) route digest(8) identity flags(1) backend(1) opencl(1)
/// pixel_space(1) flags(1) reserved(11, must be zero).
const HEADER16: usize = 96;
const MAX_ENTRIES: usize = 64;
const MAX_BYTES: u64 = 1 << 30;
/// Full-sensor entries get their own budget: they are what a second open of
/// the app would otherwise decode again, but they are also the large files.
const MAX_NATIVE_ENTRIES: usize = 16;
const MAX_NATIVE_BYTES: u64 = 2 << 30;
/// Below this many free bytes on the cache volume, an automatic native cache
/// write is skipped instead of filling the disk.
const NATIVE_DISK_RESERVE: u64 = 1 << 30;
/// Decodes at least this long (or this many pixels) are worth a native copy.
const NATIVE_PERSIST_MIN_MILLIS: u64 = 250;
const NATIVE_PERSIST_MIN_BYTES: u64 = 48 << 20;
/// Bytes read at each of [`IDENTITY_SAMPLES`] evenly spread windows when the
/// source is too large to hash whole.
const IDENTITY_WINDOW: usize = 16 << 10;
/// Windows at 0, 1/8, ..., 7/8 and the tail. The middle window straddles the
/// exact file midpoint, so a changed header, middle or tail always misses.
const IDENTITY_SAMPLES: u64 = 9;
/// Files at or below this size are hashed whole: cheaper than the bookkeeping
/// and immune to changes between windows.
const IDENTITY_FULL_BELOW: u64 = 2 << 20;
/// Default bytes of full-sensor frames kept after cold proxy decodes. The fit
/// view already paid for that decode and then discarded the native pixels,
/// which made the first 1:1 zoom decode the same sensor again. 256 MiB covers a
/// 24 MP RGBA16 frame; `AURORA_RAW_PENDING_BYTES` overrides it.
const MAX_PENDING_NATIVE_BYTES: usize = 256 << 20;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A decoded frame: width, height, RGBA16 pixels, provenance.
type Frame16 = (u32, u32, Vec<u16>, DecodeProvenance);

/// Decoded pixels for `path`, at most `long_side` on the longest edge.
///
/// A cache hit copies the PNG; a miss decodes, downscales and stores the result.
/// An unwritable cache directory is silently ignored: the answer is still the decode.
pub fn get_or_decode(path: &Path, dir: &Path, long_side: u32) -> Result<RgbaImage> {
    get_or_decode_with(path, dir, long_side, |p| {
        let decoded = crate::raw::decode(p)?;
        Ok(RgbaImage::new(
            decoded.width(),
            decoded.height(),
            decoded.into_raw(),
        ))
    })
}

/// The per-user cache directory for lossless linear RAW frames.
///
/// `AURORA_RAW_CACHE_DIR` overrides it for tests or portable installs, and
/// `AURORA_DISABLE_RAW_CACHE` keeps every decode in memory only.
pub fn default_raw16_dir() -> Option<PathBuf> {
    if std::env::var_os("AURORA_DISABLE_RAW_CACHE").is_some() {
        return None;
    }
    if let Some(dir) = std::env::var_os("AURORA_RAW_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(target_os = "windows")]
    {
        return std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .map(|dir| dir.join("Aurora Studio").join("Cache").join("raw"));
    }
    // macOS convention: `~/Library/Caches`. The explicit overrides above still
    // win, so portable installs and tests can redirect the cache.
    #[cfg(target_os = "macos")]
    {
        return std::env::var_os("HOME").map(PathBuf::from).map(|dir| {
            dir.join("Library")
                .join("Caches")
                .join("Aurora Studio")
                .join("raw")
        });
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|dir| PathBuf::from(dir).join(".cache")))
            .map(|dir| dir.join("aurora-studio").join("raw"))
    }
}

/// Lossless 16-bit linear RAW decode, cached across process runs.
///
/// The returned dimensions are already capped to `long_side`. Cache failures
/// only fall back to a fresh decode; they never change the pixels.
pub fn get_or_decode16(path: &Path, long_side: u32) -> Option<(u32, u32, Vec<u16>)> {
    get_or_decode16_traced(path, long_side).map(|(w, h, px, _)| (w, h, px))
}

/// [`get_or_decode16`] plus the provenance stored with the cached samples.
pub fn get_or_decode16_traced(
    path: &Path,
    long_side: u32,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    let route = crate::raw::decode_route(path);
    let Some(dir) = default_raw16_dir() else {
        let identity = tracked_identity(path).unwrap_or(SourceIdentity::UNKNOWN);
        let mut failed = None;
        return decode_frame16(path, long_side, route, identity, &mut failed);
    };
    get_or_decode16_with_route(path, &dir, long_side, route, |path| {
        crate::raw::decode16_traced_routed(path, route)
    })
}

pub(crate) fn get_or_decode_source16(
    path: &Path,
    long_side: u32,
) -> Option<(Source, DecodeProvenance)> {
    let failed = std::cell::RefCell::new(None);
    get_or_decode_source16_with_failed(path, long_side, &failed)
}

/// [`get_or_decode_source16`] that also returns the container bytes a failed
/// 16-bit decode already read.
///
/// The caller's 8-bit fallback reuses those bytes, so one cold load reads the
/// file once instead of once per backend. A cache hit leaves them `None`.
pub(crate) fn get_or_decode_source16_with_bytes(
    path: &Path,
    long_side: u32,
) -> (Option<(Source, DecodeProvenance)>, Option<crate::raw::FailedRaw16>) {
    let failed = std::cell::RefCell::new(None);
    let source = get_or_decode_source16_with_failed(path, long_side, &failed);
    (source, failed.into_inner())
}

fn get_or_decode_source16_with_failed(
    path: &Path,
    long_side: u32,
    failed: &std::cell::RefCell<Option<crate::raw::FailedRaw16>>,
) -> Option<(Source, DecodeProvenance)> {
    let route = crate::raw::decode_route(path);
    let Some(dir) = default_raw16_dir() else {
        return decode_source16(path, long_side, route, &mut *failed.borrow_mut());
    };
    let Some(identity) = tracked_identity(path) else {
        return decode_source16(path, long_side, route, &mut *failed.borrow_mut());
    };
    let key = cache_name16_for_identity(path, long_side, route, identity);
    let file = dir.join(&key);
    // Only the full-sensor decode gets a deferred mip tail. Proxy decodes stay
    // eager: they are small and immediately used by fit-view sampling. This
    // defers `Source` pixels only; the pending native frame is moved into
    // `source_of` and no decoder buffer is retained twice.
    let lazy_mips = long_side == 0;
    let source_of = |(width, height, pixels, provenance): Frame16| {
        (rgba16_source(width, height, &pixels, lazy_mips), provenance)
    };
    if let Some(source) = read_cached_source16_with_mips(&file, &route, &identity, lazy_mips) {
        return Some(source);
    }
    if let Some(frame) = take_pending_native(path, long_side, route, identity) {
        return Some(source_of(frame));
    }
    if let Some(frame) = cached_full_frame16(path, &dir, long_side, route, identity) {
        return Some(source_of(frame));
    }

    let mut decode = Some(route);
    for _ in 0..3 {
        match acquire_flight(&key) {
            FlightTicket::Leader(flight) => {
                let result = (|| {
                    // A previous leader for this key may have stored the entry
                    // just before we acquired the lead.
                    if let Some(source) =
                        read_cached_source16_with_mips(&file, &route, &identity, lazy_mips)
                    {
                        return Some(source);
                    }
                    if let Some(frame) = take_pending_native(path, long_side, route, identity) {
                        return Some(source_of(frame));
                    }
                    if let Some(frame) =
                        cached_full_frame16(path, &dir, long_side, route, identity)
                    {
                        return Some(source_of(frame));
                    }
                    let native_file =
                        dir.join(cache_name16_for_identity(path, 0, route, identity));
                    let image = decode_and_fit(
                        path,
                        Some(&native_file),
                        &file,
                        long_side,
                        &route,
                        identity,
                        |path| {
                            let route = decode.take()?;
                            let mut leftover = None;
                            let frame =
                                crate::raw::decode16_traced_into(path, route, &mut leftover);
                            *failed.borrow_mut() = leftover;
                            frame
                        },
                    )?;
                    Some(source_of(image))
                })();
                release_flight(&key, &flight);
                return result;
            }
            FlightTicket::Follower(flight) => {
                count_waiter();
                flight.wait();
                if let Some(source) =
                    read_cached_source16_with_mips(&file, &route, &identity, lazy_mips)
                {
                    return Some(source);
                }
                if let Some(frame) = take_pending_native(path, long_side, route, identity) {
                    return Some(source_of(frame));
                }
            }
        }
    }
    let native_file = dir.join(cache_name16_for_identity(path, 0, route, identity));
    let image = decode_and_fit(
        path,
        Some(&native_file),
        &file,
        long_side,
        &route,
        identity,
        |path| {
            let route = decode.take()?;
            let mut leftover = None;
            let frame = crate::raw::decode16_traced_into(path, route, &mut leftover);
            *failed.borrow_mut() = leftover;
            frame
        },
    )?;
    Some(source_of(image))
}

fn rgba16_source(width: u32, height: u32, pixels: &[u16], lazy_mips: bool) -> Source {
    if lazy_mips {
        Source::from_rgba16_native(width, height, pixels)
    } else {
        Source::from_rgba16(width, height, pixels)
    }
}

fn decode_source16(
    path: &Path,
    long_side: u32,
    route: DecodeRoute,
    failed: &mut Option<crate::raw::FailedRaw16>,
) -> Option<(Source, DecodeProvenance)> {
    let identity = tracked_identity(path).unwrap_or(SourceIdentity::UNKNOWN);
    let image = decode_frame16(path, long_side, route, identity, failed)?;
    Some((rgba16_source(image.0, image.1, &image.2, long_side == 0), image.3))
}

fn get_or_decode16_with<F>(
    path: &Path,
    dir: &Path,
    long_side: u32,
    decode: F,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)>
where
    F: FnOnce(&Path) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)>,
{
    let route = crate::raw::decode_route(path);
    get_or_decode16_with_route(path, dir, long_side, route, decode)
}

fn get_or_decode16_with_route<F>(
    path: &Path,
    dir: &Path,
    long_side: u32,
    route: DecodeRoute,
    decode: F,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)>
where
    F: FnOnce(&Path) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)>,
{
    let Some(identity) = tracked_identity(path) else {
        let (w, h, pixels, provenance) = decode(path)?;
        let (width, height, fitted) = fit16(w, h, pixels, long_side);
        return Some((width, height, fitted, provenance));
    };
    let key = cache_name16_for_identity(path, long_side, route, identity);
    let file = dir.join(&key);
    let mut recheck = || {
        read_cached16_binary(&file, &route, &identity)
            .or_else(|| take_pending_native(path, long_side, route, identity))
            .or_else(|| cached_full_frame16(path, dir, long_side, route, identity))
    };
    if let Some(hit) = recheck() {
        return Some(hit);
    }
    let mut decode = Some(decode);
    let native_file = dir.join(cache_name16_for_identity(path, 0, route, identity));
    let mut work = || {
        decode_and_fit(
            path,
            Some(&native_file),
            &file,
            long_side,
            &route,
            identity,
            |path| decode.take()?(path),
        )
    };
    with_single_flight(&key, &mut recheck, &mut work)
}

fn get_or_decode_with<F>(path: &Path, dir: &Path, long_side: u32, decode: F) -> Result<RgbaImage>
where
    F: FnOnce(&Path) -> Result<RgbaImage>,
{
    let key = cache_key(path, dir, long_side);
    if let Ok(file) = &key {
        if let Ok(image) = read_cached(file) {
            return Ok(image);
        }
    }
    let decoded = decode(path)?;
    let image = fit(decoded, long_side);
    if let Ok(file) = key {
        store(&file, &image);
    }
    Ok(image)
}

/// Cache key from the file identity plus every input that chooses a decoder.
///
/// `route` carries the stage-mix pipeline fingerprint, C1 switch, Z6III
/// detection, libraw availability, OpenCL state and demosaic filter, so a
/// backend change or a bump of a stage that route runs lands on a different
/// file instead of reusing pixels from another pipeline.
fn cache_key16_for_route(
    path: &Path,
    dir: &Path,
    long_side: u32,
    route: DecodeRoute,
) -> Option<PathBuf> {
    let identity = source_identity(path)?;
    Some(dir.join(cache_name16_for_identity(path, long_side, route, identity)))
}

/// The flat name from the content identity and decoder route.
///
/// The content hash means a same-path, same-length replacement with a restored
/// mtime lands on a different file instead of reusing the old pixels.
fn cache_name16_for_identity(
    path: &Path,
    long_side: u32,
    route: DecodeRoute,
    identity: SourceIdentity,
) -> String {
    let mut a = DefaultHasher::new();
    let mut b = DefaultHasher::new();
    (
        RAW16_MAGIC,
        path,
        identity.len,
        identity.mtime_nanos,
        identity.content,
        identity.full,
        long_side,
        route,
    )
        .hash(&mut a);
    (0xa7u8, path, identity.content, identity.mtime_nanos, long_side, identity.len, route)
        .hash(&mut b);
    format!(
        "{PREFIX}16b-{:016x}{:016x}-{long_side}.bin",
        a.finish(),
        b.finish()
    )
}

/// Quick content identity of a RAW source, read alongside its metadata.
///
/// Path + length + mtime alone cannot distinguish a same-size replacement that
/// preserved the timestamp (sync tools, unpack-over semantics). The content
/// fingerprint covers the file header, the tail and seven evenly spread
/// windows; small files (`AURORA_RAW_CACHE_IDENTITY=full` forces this for every
/// file) are hashed whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SourceIdentity {
    len: u64,
    mtime_nanos: u64,
    content: u64,
    full: bool,
}

impl SourceIdentity {
    /// Used when the metadata cannot be read: the pending frame is then keyed
    /// by path alone.
    const UNKNOWN: Self = Self { len: 0, mtime_nanos: 0, content: 0, full: false };

    fn flags(self) -> u8 {
        self.full as u8
    }
}

const IDENTITY_PRIME: u64 = 0x0000_0100_0000_01b3;
const IDENTITY_DOMAIN: u64 = 0xA9C1_3E00_0000_0001;

fn identity_mode_full() -> bool {
    std::env::var("AURORA_RAW_CACHE_IDENTITY")
        .is_ok_and(|value| value.eq_ignore_ascii_case("full"))
}

fn fnv_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(IDENTITY_PRIME);
    }
    hash
}

fn source_identity(path: &Path) -> Option<SourceIdentity> {
    let meta = fs::metadata(path).ok()?;
    let len = meta.len();
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
    let full = identity_mode_full() || len <= IDENTITY_FULL_BELOW;
    let content = content_fingerprint(path, len, full)?;
    Some(SourceIdentity { len, mtime_nanos, content, full })
}

fn content_fingerprint(path: &Path, len: u64, full: bool) -> Option<u64> {
    use std::io::{Seek, SeekFrom};

    let mut file = fs::File::open(path).ok()?;
    let mut hash = fnv_bytes(IDENTITY_DOMAIN, &len.to_le_bytes());
    if full {
        let mut buffer = vec![0u8; 1 << 20];
        let mut remaining = len;
        while remaining > 0 {
            let take = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..take]).ok()?;
            hash = fnv_bytes(hash, &buffer[..take]);
            remaining -= take as u64;
        }
        return Some(hash ^ (hash >> 32));
    }
    let window = (IDENTITY_WINDOW as u64).min(len);
    let last = len - window;
    let mut buffer = vec![0u8; window as usize];
    for sample in 0..IDENTITY_SAMPLES {
        let offset = (last * sample) / (IDENTITY_SAMPLES - 1);
        hash = fnv_bytes(hash, &offset.to_le_bytes());
        file.seek(SeekFrom::Start(offset)).ok()?;
        file.read_exact(&mut buffer).ok()?;
        hash = fnv_bytes(hash, &buffer);
    }
    Some(hash ^ (hash >> 32))
}

/// Stable hash of the route fields, persisted so a renamed or copied cache file
/// cannot serve a different decoder.
fn route_digest(route: &DecodeRoute) -> u64 {
    let mut hasher = DefaultHasher::new();
    route.hash(&mut hasher);
    hasher.finish()
}

/// [`source_identity`] for a lookup that also invalidates stale pending frames
/// and rendered previews when the bytes behind a path changed.
fn tracked_identity(path: &Path) -> Option<SourceIdentity> {
    let identity = source_identity(path)?;
    if note_source_identity(path, identity) {
        invalidate_source(path);
    }
    Some(identity)
}

/// Content fingerprint for callers that keep decoded sources of their own (the
/// file engine's native source slot). `None` when the file cannot be read.
///
/// This is the invalidation hook for `FileEngine::state.native_raw`: compare
/// the token it was built from and drop the slot when it changes.
#[allow(dead_code)] // the file-engine wiring lands with the merge of BO-1
pub fn source_identity_token(path: &Path) -> Option<u64> {
    source_identity(path).map(|identity| identity.content)
}

/// Remembers the last identity seen for a path. Returns true when the bytes
/// behind a known path changed.
fn note_source_identity(path: &Path, identity: SourceIdentity) -> bool {
    static SEEN: Mutex<Option<std::collections::HashMap<PathBuf, SourceIdentity>>> =
        Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let map = seen.get_or_insert_with(std::collections::HashMap::new);
    if map.len() > 256 {
        map.clear();
    }
    match map.insert(path.to_path_buf(), identity) {
        Some(previous) if previous != identity => {
            log::info!(
                "raw cache: source identity changed for a tracked path; \
                 dropping its pending frame and rendered previews"
            );
            true
        }
        _ => false,
    }
}

fn invalidate_source(path: &Path) {
    {
        let mut pending = PENDING_NATIVE.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove_path(path);
    }
    crate::preview_cache::invalidate_source_identity();
}

#[cfg(test)]
fn read_cached16_png(file: &Path) -> Option<(u32, u32, Vec<u16>)> {
    let image = image::open(file).ok()?.to_rgba16();
    let expected = image.width() as usize * image.height() as usize * 4;
    (image.as_raw().len() == expected).then(|| {
        (image.width(), image.height(), image.into_raw())
    })
}

fn fit16(width: u32, height: u32, pixels: Vec<u16>, long_side: u32) -> (u32, u32, Vec<u16>) {
    match resized16(width, height, &pixels, long_side) {
        Some((target_width, target_height, resized)) => (target_width, target_height, resized),
        None => (width, height, pixels),
    }
}

fn resized16(
    width: u32,
    height: u32,
    pixels: &[u16],
    long_side: u32,
) -> Option<(u32, u32, Vec<u16>)> {
    let longest = width.max(height);
    if long_side == 0 || longest <= long_side {
        return None;
    }
    let scale = long_side as f32 / longest as f32;
    let short = ((if width <= height { width } else { height }) as f32 * scale)
        .round()
        .max(1.0) as u32;
    let (target_width, target_height) =
        if width >= height { (long_side, short) } else { (short, long_side) };
    Some((
        target_width,
        target_height,
        crate::recipe::resize_rgba16(pixels, width, height, target_width, target_height),
    ))
}

/// Downscales a full sensor frame and keeps its native samples for the first
/// zoom request. The pending frame never triggers extra work.
fn fit16_keep_native(
    native_file: Option<&Path>,
    native_key: Option<String>,
    path: &Path,
    width: u32,
    height: u32,
    pixels: Vec<u16>,
    long_side: u32,
    provenance: DecodeProvenance,
    identity: SourceIdentity,
    route: &DecodeRoute,
    decode_millis: Option<u64>,
    explicit_native: bool,
) -> Frame16 {
    let Some((target_width, target_height, proxy)) =
        resized16(width, height, &pixels, long_side)
    else {
        return (width, height, pixels, provenance);
    };
    // A cold proxy decode is the only place that can persist the native frame
    // for the next session. Pixels read back from disk (`None`) are already
    // persisted and are not rewritten.
    if let (Some(file), Some(millis)) = (native_file, decode_millis) {
        let input = NativePersistInput {
            explicit_native,
            decode_millis: millis,
            frame_bytes: (pixels.len() as u64).saturating_mul(2),
            // Only the Z6III reconstruction makes a re-decode expensive; a
            // LibRaw fallback on the same route is cheap and must not trigger
            // the expensive-path policy.
            z6iii: provenance.unlocks_c1_z6iii_linear(),
            free_disk: free_disk_bytes(file.parent().unwrap_or(Path::new("."))),
        };
        persist_native_entry(
            file,
            width,
            height,
            &pixels,
            provenance,
            identity,
            route,
            input.z6iii,
            input,
        );
    }
    remember_native(native_key, path, width, height, pixels, provenance, identity, route);
    (target_width, target_height, proxy, provenance)
}

/// Decodes under one decode slot, fits the result, and applies the proxy/native
/// persistence split: proxies always persist, native frames only when
/// [`native_persist_decision`] says the write is worth it.
fn decode_and_fit<F>(
    path: &Path,
    native_file: Option<&Path>,
    probe_file: &Path,
    long_side: u32,
    route: &DecodeRoute,
    identity: SourceIdentity,
    decode: F,
) -> Option<Frame16>
where
    F: FnOnce(&Path) -> Option<Frame16>,
{
    let started = std::time::Instant::now();
    let (width, height, pixels, provenance) = {
        let _slot = enter_decode_slot();
        count_decode();
        decode(path)?
    };
    let millis = started.elapsed().as_millis() as u64;
    let native_key = native_file
        .and_then(Path::file_name)
        .map(|name| name.to_string_lossy().into_owned());
    let image = fit16_keep_native(
        native_file,
        native_key,
        path,
        width,
        height,
        pixels,
        long_side,
        provenance,
        identity,
        route,
        Some(millis),
        long_side == 0,
    );
    if long_side == 0 {
        // The requested frame is the native frame, so `fit16_keep_native`
        // returned early and persistence is decided here.
        let z6iii = provenance.unlocks_c1_z6iii_linear();
        let input = NativePersistInput {
            explicit_native: true,
            decode_millis: millis,
            frame_bytes: (image.2.len() as u64).saturating_mul(2),
            z6iii,
            free_disk: free_disk_bytes(
                native_file.and_then(Path::parent).unwrap_or(Path::new(".")),
            ),
        };
        persist_native_entry(
            probe_file,
            image.0,
            image.1,
            &image.2,
            image.3,
            identity,
            route,
            z6iii,
            input,
        );
    } else {
        store16_binary(probe_file, image.0, image.1, &image.2, image.3, identity, route);
    }
    Some(image)
}

/// Counters for the pending-native LRU and its single-flight decode gate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PendingNativeStats {
    /// 1:1 requests served from a retained native frame.
    pub hits: u64,
    /// 1:1 requests that had to decode.
    pub misses: u64,
    pub inserts: u64,
    /// A newer frame replaced one already retained under the same key.
    pub overwrites: u64,
    /// A frame was dropped because the byte budget was exceeded.
    pub evictions: u64,
    pub rejected_too_large: u64,
    /// Decodes that actually ran under the gate.
    pub decodes: u64,
    /// Requests that waited for another request's decode of the same key.
    pub decode_waiters: u64,
    pub peak_bytes: usize,
    pub bytes: usize,
    pub entries: usize,
}

impl PendingNativeStats {
    const fn empty() -> Self {
        Self {
            hits: 0,
            misses: 0,
            inserts: 0,
            overwrites: 0,
            evictions: 0,
            rejected_too_large: 0,
            decodes: 0,
            decode_waiters: 0,
            peak_bytes: 0,
            bytes: 0,
            entries: 0,
        }
    }
}

struct PendingEntry {
    path: PathBuf,
    key: String,
    width: u32,
    height: u32,
    pixels: Vec<u16>,
    provenance: DecodeProvenance,
    identity: SourceIdentity,
}

impl PendingEntry {
    fn bytes(&self) -> usize {
        self.pixels.len().saturating_mul(2)
    }
}

/// Least-recently-used table of full-sensor frames retained for the first 1:1
/// request of each image, bounded by total frame bytes rather than entry count.
struct PendingNative {
    entries: std::collections::VecDeque<PendingEntry>,
    bytes: usize,
    stats: PendingNativeStats,
}

impl PendingNative {
    const fn new() -> Self {
        Self { entries: std::collections::VecDeque::new(), bytes: 0, stats: PendingNativeStats::empty() }
    }

    /// Drops every retained frame decoded from `path`. Returns how many were
    /// removed.
    fn remove_path(&mut self, path: &Path) -> usize {
        let before = self.entries.len();
        let mut removed_bytes = 0;
        self.entries.retain(|entry| {
            let keep = entry.path != path;
            if !keep {
                removed_bytes += entry.bytes();
            }
            keep
        });
        self.bytes = self.bytes.saturating_sub(removed_bytes);
        self.stats.bytes = self.bytes;
        self.stats.entries = self.entries.len();
        before - self.entries.len()
    }

    #[allow(clippy::too_many_arguments)]
    fn insert(
        &mut self,
        native_key: Option<String>,
        path: &Path,
        width: u32,
        height: u32,
        pixels: Vec<u16>,
        provenance: DecodeProvenance,
        identity: SourceIdentity,
        route: &DecodeRoute,
        budget: usize,
    ) {
        let Some(key) = native_key else { return };
        if !provenance.consistent_with(route) {
            return;
        }
        let frame_bytes = pixels.len().saturating_mul(2);
        if frame_bytes > budget {
            self.stats.rejected_too_large += 1;
            return;
        }
        if let Some(pos) = self.entries.iter().position(|entry| entry.key == key) {
            self.bytes -= self.entries.remove(pos).map_or(0, |entry| entry.bytes());
            self.stats.overwrites += 1;
        }
        self.entries.push_back(PendingEntry {
            path: path.to_path_buf(),
            key,
            width,
            height,
            pixels,
            provenance,
            identity,
        });
        self.bytes += frame_bytes;
        self.stats.inserts += 1;
        // Evict least-recently-used frames until the budget fits; never drop
        // the frame that was just inserted.
        while self.bytes > budget && self.entries.len() > 1 {
            if let Some(entry) = self.entries.pop_front() {
                self.bytes = self.bytes.saturating_sub(entry.bytes());
                self.stats.evictions += 1;
            }
        }
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.bytes);
        self.stats.bytes = self.bytes;
        self.stats.entries = self.entries.len();
    }

    fn take(
        &mut self,
        path: &Path,
        long_side: u32,
        route: DecodeRoute,
        identity: SourceIdentity,
    ) -> Option<Frame16> {
        if long_side != 0 {
            return None;
        }
        let key = cache_name16_for_identity(path, 0, route, identity);
        let pos = self.entries.iter().position(|entry| {
            entry.key == key
                && entry.path == path
                && entry.identity == identity
                && entry.provenance.consistent_with(&route)
        });
        let Some(pos) = pos else {
            self.stats.misses += 1;
            return None;
        };
        let entry = self.entries.remove(pos)?;
        self.bytes = self.bytes.saturating_sub(entry.bytes());
        self.stats.hits += 1;
        self.stats.bytes = self.bytes;
        self.stats.entries = self.entries.len();
        Some((entry.width, entry.height, entry.pixels, entry.provenance))
    }
}

static PENDING_NATIVE: Mutex<PendingNative> = Mutex::new(PendingNative::new());

/// Retained-frame budget in bytes. `AURORA_RAW_PENDING_BYTES` overrides the
/// 256 MiB default; a zero or unparsable value keeps the default.
fn pending_native_budget_bytes() -> usize {
    std::env::var("AURORA_RAW_PENDING_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|bytes| *bytes > 0)
        .unwrap_or(MAX_PENDING_NATIVE_BYTES)
}

/// Snapshot of the pending cache counters and current occupancy. Exposed for
/// diagnostics and the A/B/C acceptance tests.
#[allow(dead_code)]
pub fn pending_native_stats() -> PendingNativeStats {
    let pending = PENDING_NATIVE.lock().unwrap_or_else(|e| e.into_inner());
    PendingNativeStats {
        bytes: pending.bytes,
        entries: pending.entries.len(),
        ..pending.stats
    }
}

/// Clears the counters but keeps the retained frames.
#[allow(dead_code)]
pub fn reset_pending_native_stats() {
    let mut pending = PENDING_NATIVE.lock().unwrap_or_else(|e| e.into_inner());
    pending.stats = PendingNativeStats::empty();
    pending.stats.bytes = pending.bytes;
    pending.stats.entries = pending.entries.len();
    pending.stats.peak_bytes = pending.bytes;
}

fn remember_native(
    native_key: Option<String>,
    path: &Path,
    width: u32,
    height: u32,
    pixels: Vec<u16>,
    provenance: DecodeProvenance,
    identity: SourceIdentity,
    route: &DecodeRoute,
) {
    let budget = pending_native_budget_bytes();
    PENDING_NATIVE.lock().unwrap_or_else(|e| e.into_inner()).insert(
        native_key, path, width, height, pixels, provenance, identity, route, budget,
    );
}

fn take_pending_native(
    path: &Path,
    long_side: u32,
    route: DecodeRoute,
    identity: SourceIdentity,
) -> Option<Frame16> {
    PENDING_NATIVE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take(path, long_side, route, identity)
}

/// One in-flight decode per cache key.
struct Flight {
    done: Mutex<bool>,
    ready: std::sync::Condvar,
}

impl Flight {
    fn new() -> Self {
        Self { done: Mutex::new(false), ready: std::sync::Condvar::new() }
    }

    fn wait(&self) {
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            done = self.ready.wait(done).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn finish(&self) {
        *self.done.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.ready.notify_all();
    }
}

enum FlightTicket {
    Leader(std::sync::Arc<Flight>),
    Follower(std::sync::Arc<Flight>),
}

fn flights() -> &'static Mutex<std::collections::HashMap<String, std::sync::Arc<Flight>>> {
    static FLIGHTS: std::sync::OnceLock<
        Mutex<std::collections::HashMap<String, std::sync::Arc<Flight>>>,
    > = std::sync::OnceLock::new();
    FLIGHTS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn acquire_flight(key: &str) -> FlightTicket {
    let mut map = flights().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(flight) = map.get(key) {
        FlightTicket::Follower(flight.clone())
    } else {
        let flight = std::sync::Arc::new(Flight::new());
        map.insert(key.to_string(), flight.clone());
        FlightTicket::Leader(flight)
    }
}

fn release_flight(key: &str, flight: &std::sync::Arc<Flight>) {
    // Remove first: a new request after this point becomes a leader instead of
    // observing a finished flight.
    flights().lock().unwrap_or_else(|e| e.into_inner()).remove(key);
    flight.finish();
}

/// Decodes that may run at once across different cache keys.
/// `AURORA_RAW_DECODE_CONCURRENCY` overrides the default of 2.
fn decode_concurrency() -> usize {
    std::env::var("AURORA_RAW_DECODE_CONCURRENCY")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 8)
}

struct DecodeSlot;

impl Drop for DecodeSlot {
    fn drop(&mut self) {
        let mut active = DECODE_ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        *active = active.saturating_sub(1);
        DECODE_SLOT_READY.notify_one();
    }
}

static DECODE_ACTIVE: Mutex<usize> = Mutex::new(0);
static DECODE_SLOT_READY: std::sync::Condvar = std::sync::Condvar::new();

fn enter_decode_slot() -> DecodeSlot {
    let limit = decode_concurrency();
    let mut active = DECODE_ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
    while *active >= limit {
        active = DECODE_SLOT_READY.wait(active).unwrap_or_else(|e| e.into_inner());
    }
    *active += 1;
    DecodeSlot
}

fn count_decode() {
    PENDING_NATIVE.lock().unwrap_or_else(|e| e.into_inner()).stats.decodes += 1;
}

fn count_waiter() {
    PENDING_NATIVE.lock().unwrap_or_else(|e| e.into_inner()).stats.decode_waiters += 1;
}

/// Runs `work` as the single leader for `key`, or waits for the current leader
/// and retries `recheck`. Both closures may run at most a few times; `work` is
/// only entered by the leader, holding one decode slot.
fn with_single_flight(
    key: &str,
    recheck: &mut dyn FnMut() -> Option<Frame16>,
    work: &mut dyn FnMut() -> Option<Frame16>,
) -> Option<Frame16> {
    for _ in 0..3 {
        match acquire_flight(key) {
            FlightTicket::Leader(flight) => {
                let result = recheck().or_else(|| work());
                release_flight(key, &flight);
                return result;
            }
            FlightTicket::Follower(flight) => {
                count_waiter();
                flight.wait();
                if let Some(hit) = recheck() {
                    return Some(hit);
                }
                // The leader decoded nothing (failure or cancellation); take
                // the lead ourselves.
            }
        }
    }
    work()
}

fn cached_full_frame16(
    path: &Path,
    dir: &Path,
    long_side: u32,
    route: DecodeRoute,
    identity: SourceIdentity,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    if long_side == 0 {
        return None;
    }
    let native_key = cache_name16_for_identity(path, 0, route, identity);
    let native_file = dir.join(&native_key);
    let (width, height, pixels, provenance) =
        read_cached16_binary(&native_file, &route, &identity)?;
    Some(fit16_keep_native(
        Some(&native_file),
        Some(native_key),
        path,
        width,
        height,
        pixels,
        long_side,
        provenance,
        identity,
        &route,
        // Already persisted; never count a disk read as a fresh decode.
        None,
        false,
    ))
}

fn decode_frame16(
    path: &Path,
    long_side: u32,
    route: DecodeRoute,
    identity: SourceIdentity,
    failed: &mut Option<crate::raw::FailedRaw16>,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    if let Some(pending) = take_pending_native(path, long_side, route, identity) {
        *failed = None;
        return Some(pending);
    }
    let native_key = cache_name16_for_identity(path, 0, route, identity);
    let (width, height, pixels, provenance) = {
        let _slot = enter_decode_slot();
        count_decode();
        crate::raw::decode16_traced_into(path, route, failed)?
    };
    Some(fit16_keep_native(
        // No cache directory: nothing to persist, but the pending frame is
        // still keyed by the native name.
        None,
        Some(native_key),
        path,
        width,
        height,
        pixels,
        long_side,
        provenance,
        identity,
        &route,
        None,
        false,
    ))
}

/// The validated fields every reader needs from the entry header.
struct Raw16Header {
    width: u32,
    height: u32,
    body_bytes: usize,
    image_hash: u64,
    crc32: u32,
    provenance: DecodeProvenance,
}

/// Validates magic, length, source identity, route digest, stage fingerprint and
/// provenance codes. A truncated, tampered or older entry returns `None`, which
/// makes the caller rebuild it.
fn parse_header(
    header: &[u8],
    file_len: u64,
    route: &DecodeRoute,
    identity: &SourceIdentity,
) -> Option<Raw16Header> {
    if header.len() != HEADER16 || &header[..8] != RAW16_MAGIC {
        return None;
    }
    let width = u32::from_le_bytes(header[8..12].try_into().ok()?);
    let height = u32::from_le_bytes(header[12..16].try_into().ok()?);
    let body_bytes = u64::from_le_bytes(header[16..24].try_into().ok()?) as usize;
    let expected_bytes = width as usize * height as usize * 6;
    if width == 0
        || height == 0
        || body_bytes != expected_bytes
        || file_len != (HEADER16 + body_bytes) as u64
    {
        return None;
    }
    if header[36..40] != [0u8; 4] || header[85..96] != [0u8; 11] {
        return None;
    }
    let stored_len = u64::from_le_bytes(header[48..56].try_into().ok()?);
    let stored_content = u64::from_le_bytes(header[56..64].try_into().ok()?);
    let stored_mtime = u64::from_le_bytes(header[64..72].try_into().ok()?);
    let stored_route = u64::from_le_bytes(header[72..80].try_into().ok()?);
    if stored_len != identity.len
        || stored_content != identity.content
        || stored_mtime != identity.mtime_nanos
        || header[80] != identity.flags()
        || stored_route != route_digest(route)
    {
        return None;
    }
    let provenance = DecodeProvenance::from_codes(
        u64::from_le_bytes(header[40..48].try_into().ok()?),
        header[81],
        header[82],
        header[83],
        header[84],
    )?;
    if !provenance.consistent_with(route) {
        return None;
    }
    Some(Raw16Header {
        width,
        height,
        body_bytes,
        image_hash: u64::from_le_bytes(header[24..32].try_into().ok()?),
        crc32: u32::from_le_bytes(header[32..36].try_into().ok()?),
        provenance,
    })
}

fn body_matches(header: &Raw16Header, body: &[u8]) -> bool {
    crc32fast::hash(body) == header.crc32
        && image_hash(header.width, header.height, body) == header.image_hash
}

/// Reads one RAW16 entry and rejects it unless its source identity, route and
/// provenance all match the current request.
///
/// The compact RGB16 body is expanded into the RGBA16 result from the back, so
/// a full-sensor entry needs one frame buffer instead of two.
fn read_cached16_binary(
    file: &Path,
    route: &DecodeRoute,
    identity: &SourceIdentity,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    let mut reader = fs::File::open(file).ok()?;
    let size = reader.metadata().ok()?.len();
    let mut header = [0u8; HEADER16];
    reader.read_exact(&mut header).ok()?;
    let header = parse_header(&header, size, route, identity)?;
    let pixel_count = header.width as usize * header.height as usize;
    let mut pixels = vec![0u16; pixel_count * 4];
    // SAFETY: `u8` has weaker alignment than `u16`, and the complete zeroed
    // allocation is written from disk before any `u16` is read.
    let (before, body, after) = unsafe { pixels.align_to_mut::<u8>() };
    if !before.is_empty() || !after.is_empty() {
        return None;
    }
    reader.read_exact(&mut body[..header.body_bytes]).ok()?;
    if !body_matches(&header, &body[..header.body_bytes]) {
        return None;
    }
    // Every source byte sits below its destination, so a backward pass reads
    // each pixel before the copy can overwrite it.
    for index in (0..pixel_count).rev() {
        let source = index * 6;
        let target = index * 8;
        let mut sample = [0u8; 6];
        sample.copy_from_slice(&body[source..source + 6]);
        body[target..target + 6].copy_from_slice(&sample);
        body[target + 6] = 0xFF;
        body[target + 7] = 0xFF;
    }
    #[cfg(target_endian = "big")]
    for sample in &mut pixels {
        *sample = u16::from_le(*sample);
    }
    Some((header.width, header.height, pixels, header.provenance))
}

#[cfg(target_endian = "little")]
fn read_cached_source16_with_mips(
    file: &Path,
    route: &DecodeRoute,
    identity: &SourceIdentity,
    lazy_mips: bool,
) -> Option<(Source, DecodeProvenance)> {
    use memmap2::MmapOptions;

    let file = fs::File::open(file).ok()?;
    // SAFETY: the cache file is read-only for the lifetime of the mapping and
    // all mutations replace it atomically through rename.
    let map = unsafe { MmapOptions::new().map(&file).ok()? };
    if map.len() < HEADER16 {
        return None;
    }
    let header = parse_header(&map[..HEADER16], map.len() as u64, route, identity)?;
    let body = &map[HEADER16..];
    // SAFETY: the mapping starts page-aligned and HEADER16 is 96 bytes, so the
    // body is u16-aligned. `align_to` checks and reports any residual bytes.
    let (before, pixels, after) = unsafe { body.align_to::<u16>() };
    if !before.is_empty() || !after.is_empty() || pixels.len() != header.body_bytes / 2 {
        return None;
    }
    let (valid, source) = rayon::join(
        || body_matches(&header, body),
        || {
            if lazy_mips {
                Source::from_rgb16_native(header.width, header.height, pixels)
            } else {
                Source::from_rgb16(header.width, header.height, pixels)
            }
        },
    );
    valid.then_some((source, header.provenance))
}

#[cfg(not(target_endian = "little"))]
fn read_cached_source16_with_mips(
    file: &Path,
    route: &DecodeRoute,
    identity: &SourceIdentity,
    lazy_mips: bool,
) -> Option<(Source, DecodeProvenance)> {
    let (width, height, pixels, provenance) = read_cached16_binary(file, route, identity)?;
    Some((rgba16_source(width, height, &pixels, lazy_mips), provenance))
}

/// Eager reader kept for the existing cache tests. Production callers choose
/// deferred mips explicitly through [`read_cached_source16_with_mips`].
#[cfg(test)]
fn read_cached_source16(
    file: &Path,
    route: &DecodeRoute,
    identity: &SourceIdentity,
) -> Option<(Source, DecodeProvenance)> {
    read_cached_source16_with_mips(file, route, identity, false)
}

#[cfg(test)]
fn store16_png(file: &Path, width: u32, height: u32, pixels: &[u16]) {
    use rayon::prelude::*;

    let Some(dir) = file.parent() else { return };
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let temp = dir.join(format!(
        ".tmp16-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut bytes = vec![0u8; pixels.len() * 2];
    bytes
        .par_chunks_exact_mut(2)
        .zip(pixels.par_iter())
        .for_each(|(dst, src)| dst.copy_from_slice(&src.to_ne_bytes()));
    let saved = image::save_buffer_with_format(
        &temp,
        &bytes,
        width,
        height,
        image::ExtendedColorType::Rgba16,
        ImageFormat::Png,
    );
    if saved.is_err() || fs::rename(&temp, file).is_err() {
        let _ = fs::remove_file(&temp);
        return;
    }
    enforce_limits(dir);
}

fn store16_binary(
    file: &Path,
    width: u32,
    height: u32,
    pixels: &[u16],
    provenance: DecodeProvenance,
    identity: SourceIdentity,
    route: &DecodeRoute,
) -> bool {
    let Some(dir) = file.parent() else { return false };
    if fs::create_dir_all(dir).is_err() {
        return false;
    }
    let temp = dir.join(format!(
        ".tmp16b-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    // The body goes out first and the header follows, so a full-sensor frame
    // never needs a body-sized buffer just to be cached.
    let body = fs::File::create(&temp)
        .ok()
        .and_then(|mut out| {
            out.write_all(&[0u8; HEADER16]).ok()?;
            let (len, crc32, hash) = write16_body(&mut out, pixels, width, height)?;
            let mut header = [0u8; HEADER16];
            write16_header(
                &mut header,
                width,
                height,
                len,
                hash,
                crc32,
                provenance,
                identity,
                route,
            );
            out.seek(SeekFrom::Start(0)).ok()?;
            out.write_all(&header).ok()?;
            Some(())
        });
    if body.is_none() || fs::rename(&temp, file).is_err() {
        let _ = fs::remove_file(&temp);
        return false;
    }
    enforce_limits(dir);
    true
}

/// Streams the compact RGB16 body and returns its byte length, CRC-32 and image
/// hash, hashing the same 256 KiB windows `image_hash` would see on a buffered
/// body so cached bytes and checksums stay identical.
fn write16_body(
    out: &mut fs::File,
    pixels: &[u16],
    width: u32,
    height: u32,
) -> Option<(usize, u32, u64)> {
    use rayon::prelude::*;

    // 131,072 pixels are 786,432 body bytes, exactly three 256 KiB hash
    // windows, so every batch ends on a window boundary.
    const UNIT_PIXELS: usize = 131_072;
    const BATCH_UNITS: usize = 12;
    const HASH_WINDOW: usize = 256 << 10;
    let batch_pixels = UNIT_PIXELS * BATCH_UNITS;
    let mut scratch = vec![0u8; batch_pixels * 6];
    let mut crc = crc32fast::Hasher::new();
    let mut parts: Vec<u64> = Vec::new();
    let mut body_len = 0usize;
    for chunk in pixels.chunks(batch_pixels * 4) {
        let packed = scratch.get_mut(..chunk.len() / 4 * 6)?;
        packed
            .par_chunks_exact_mut(6)
            .zip(chunk.par_chunks_exact(4))
            .for_each(|(dst, src)| {
                dst[0..2].copy_from_slice(&src[0].to_le_bytes());
                dst[2..4].copy_from_slice(&src[1].to_le_bytes());
                dst[4..6].copy_from_slice(&src[2].to_le_bytes());
            });
        body_len += packed.len();
        out.write_all(packed).ok()?;
        crc.update(packed);
        parts.extend(packed.par_chunks(HASH_WINDOW).map(chunk_hash).collect::<Vec<u64>>());
    }
    Some((body_len, crc.finalize(), fold_hash(width, height, &parts)))
}

#[allow(clippy::too_many_arguments)]
fn write16_header(
    header: &mut [u8; HEADER16],
    width: u32,
    height: u32,
    body_bytes: usize,
    image_hash: u64,
    crc32: u32,
    provenance: DecodeProvenance,
    identity: SourceIdentity,
    route: &DecodeRoute,
) {
    header[..8].copy_from_slice(RAW16_MAGIC);
    header[8..12].copy_from_slice(&width.to_le_bytes());
    header[12..16].copy_from_slice(&height.to_le_bytes());
    header[16..24].copy_from_slice(&(body_bytes as u64).to_le_bytes());
    header[24..32].copy_from_slice(&image_hash.to_le_bytes());
    header[32..36].copy_from_slice(&crc32.to_le_bytes());
    header[36..40].copy_from_slice(&[0u8; 4]);
    header[40..48].copy_from_slice(&provenance.pipeline_fingerprint.to_le_bytes());
    header[48..56].copy_from_slice(&identity.len.to_le_bytes());
    header[56..64].copy_from_slice(&identity.content.to_le_bytes());
    header[64..72].copy_from_slice(&identity.mtime_nanos.to_le_bytes());
    header[72..80].copy_from_slice(&route_digest(route).to_le_bytes());
    header[80] = identity.flags();
    header[81] = provenance.backend_code();
    header[82] = provenance.opencl_code();
    header[83] = provenance.pixel_space_code();
    header[84] = provenance.flags();
    header[85..].copy_from_slice(&[0u8; 11]);
}

fn image_hash(width: u32, height: u32, body: &[u8]) -> u64 {
    use rayon::prelude::*;

    const CHUNK: usize = 256 << 10;
    let parts: Vec<u64> = body.par_chunks(CHUNK).map(chunk_hash).collect();
    fold_hash(width, height, &parts)
}

fn chunk_hash(body: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for chunk in body.chunks_exact(8) {
        hash ^= u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"));
        hash = hash.rotate_left(27).wrapping_mul(0x9E37_79B1_85EB_CA87);
        hash = hash.wrapping_add(0xD1B5_4A32_D192_ED03);
    }
    let tail = body.len() % 8;
    if tail != 0 {
        let mut word = [0u8; 8];
        word[..tail].copy_from_slice(&body[body.len() - tail..]);
        hash ^= u64::from_le_bytes(word);
        hash = hash.wrapping_mul(0x9E37_79B1_85EB_CA87);
    }
    hash
}

fn fold_hash(width: u32, height: u32, parts: &[u64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ u64::from(width).rotate_left(17);
    hash ^= u64::from(height).rotate_left(41);
    hash = hash.wrapping_mul(0x9E37_79B1_85EB_CA87);
    for part in parts {
        hash ^= *part;
        hash = hash.rotate_left(27).wrapping_mul(0x9E37_79B1_85EB_CA87);
        hash = hash.wrapping_add(0xD1B5_4A32_D192_ED03);
    }
    hash ^ (hash >> 32)
}

/// Flat file name from every input that changes the pixels. Two hashers make an
/// accidental alias between two files effectively impossible.
fn cache_key(path: &Path, dir: &Path, long_side: u32) -> Result<PathBuf> {
    let meta = fs::metadata(path)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let mut a = DefaultHasher::new();
    let mut b = DefaultHasher::new();
    (1u8, path, meta.len(), mtime, long_side).hash(&mut a);
    (0xa5u8, path, mtime, long_side, meta.len()).hash(&mut b);
    let name = format!("{PREFIX}{:016x}{:016x}-{long_side}.png", a.finish(), b.finish());
    Ok(dir.join(name))
}

fn read_cached(file: &Path) -> Result<RgbaImage> {
    let image = image::open(file)?.to_rgba8();
    Ok(RgbaImage::new(image.width(), image.height(), image.into_raw()))
}

fn fit(mut image: RgbaImage, long_side: u32) -> RgbaImage {
    let longest = image.width.max(image.height);
    if long_side == 0 || longest <= long_side {
        return image;
    }
    let (width, height) = if image.width >= image.height {
        (long_side, scale(image.height, long_side, image.width))
    } else {
        (scale(image.width, long_side, image.height), long_side)
    };
    let buffer = ImageBuffer::from_raw(image.width, image.height, image.pixels.as_ref().clone())
        .expect("RGBA buffer matches its dimensions");
    let resized = image::imageops::resize(&buffer, width, height, image::imageops::FilterType::Triangle);
    image = RgbaImage::new(width, height, resized.into_raw());
    image
}

fn scale(value: u32, target: u32, total: u32) -> u32 {
    ((u64::from(value) * u64::from(target) / u64::from(total)).max(1)) as u32
}

/// Best effort: any failure leaves the decoded image as the only result.
fn store(file: &Path, image: &RgbaImage) {
    let Some(dir) = file.parent() else { return };
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let temp = dir.join(format!(
        ".tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let saved = image::save_buffer_with_format(
        &temp,
        image.pixels.as_slice(),
        image.width,
        image.height,
        image::ExtendedColorType::Rgba8,
        ImageFormat::Png,
    );
    if saved.is_err() || fs::rename(&temp, file).is_err() {
        let _ = fs::remove_file(&temp);
        return;
    }
    enforce_limits(dir);
}

/// Full-sensor entries end in `-0.bin` (long side 0); proxies carry the
/// requested long side and are the cheap, always-worth-keeping half.
fn is_native_entry(name: &str) -> bool {
    name.starts_with(PREFIX) && name.ends_with("-0.bin")
}

fn max_native_bytes() -> u64 {
    std::env::var("AURORA_RAW_NATIVE_CACHE_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|bytes| *bytes > 0)
        .unwrap_or(MAX_NATIVE_BYTES)
}

/// Applies the proxy and native budgets independently, so a folder full of
/// native frames cannot evict the proxies the viewer needs.
fn enforce_limits(dir: &Path) {
    let (mut proxies, mut natives) = scan_cache_entries(dir);
    enforce_items(&mut proxies, MAX_ENTRIES, MAX_BYTES);
    enforce_items(&mut natives, MAX_NATIVE_ENTRIES, max_native_bytes());
}

/// One directory pass, partitioned by budget kind. Same name and regular-file
/// checks as before, so foreign files in the shared folder stay untouched.
fn scan_cache_entries(
    dir: &Path,
) -> (Vec<(PathBuf, u64, SystemTime)>, Vec<(PathBuf, u64, SystemTime)>) {
    let mut proxies = Vec::new();
    let mut natives = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else { return (proxies, natives) };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(PREFIX) || !(name.ends_with(".png") || name.ends_with(".bin")) {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let item = (entry.path(), meta.len(), meta.modified().unwrap_or(UNIX_EPOCH));
        if is_native_entry(&name) {
            natives.push(item);
        } else {
            proxies.push(item);
        }
    }
    (proxies, natives)
}

/// Removes the oldest entries of one kind until both budgets fit.
fn enforce_items(items: &mut Vec<(PathBuf, u64, SystemTime)>, max_entries: usize, max_bytes: u64) {
    items.sort_by_key(|(_, _, modified)| *modified);
    let mut count = items.len();
    let mut bytes: u64 = items.iter().map(|(_, size, _)| size).sum();
    for (file, size, _) in items.iter() {
        if count <= max_entries && bytes <= max_bytes {
            break;
        }
        if fs::remove_file(file).is_ok() {
            count -= 1;
            bytes = bytes.saturating_sub(*size);
        }
    }
}

/// Single-kind view kept for the eviction tests.
#[cfg(test)]
fn enforce_budget(dir: &Path, native: bool, max_entries: usize, max_bytes: u64) {
    let (mut proxies, mut natives) = scan_cache_entries(dir);
    enforce_items(if native { &mut natives } else { &mut proxies }, max_entries, max_bytes);
}

/// Whether a freshly decoded full-sensor frame is also worth an on-disk copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NativeCachePolicy {
    /// Never write the native frame.
    Off,
    /// Write it when the decode was expensive or the caller explicitly asked
    /// for the native size.
    Auto,
    /// Always write it while the disk has room.
    Always,
}

impl NativeCachePolicy {
    fn from_env() -> Self {
        match std::env::var("AURORA_RAW_NATIVE_CACHE").ok().as_deref() {
            Some("0") | Some("off") | Some("OFF") => Self::Off,
            Some("1") | Some("on") | Some("always") => Self::Always,
            _ => Self::Auto,
        }
    }
}

fn native_persist_min_millis() -> u64 {
    std::env::var("AURORA_RAW_NATIVE_CACHE_MIN_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(NATIVE_PERSIST_MIN_MILLIS)
}

fn z6iii_native_persist_enabled() -> bool {
    !std::env::var("AURORA_RAW_NATIVE_CACHE_Z6III")
        .is_ok_and(|value| value == "0" || value.eq_ignore_ascii_case("off"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativePersist {
    Write,
    Skip(NativeSkipReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeSkipReason {
    PolicyOff,
    Z6iiiDisabled,
    CheapDecode,
    LowDisk,
}

#[derive(Clone, Copy, Debug)]
struct NativePersistInput {
    explicit_native: bool,
    decode_millis: u64,
    frame_bytes: u64,
    z6iii: bool,
    free_disk: Option<u64>,
}

/// Pure decision, so the policy matrix is testable without the file system.
fn native_persist_decision(
    policy: NativeCachePolicy,
    z6iii_enabled: bool,
    min_millis: u64,
    input: NativePersistInput,
) -> NativePersist {
    if policy == NativeCachePolicy::Off {
        return NativePersist::Skip(NativeSkipReason::PolicyOff);
    }
    if input.z6iii && !z6iii_enabled {
        return NativePersist::Skip(NativeSkipReason::Z6iiiDisabled);
    }
    let worth_it = input.explicit_native
        || input.z6iii
        || input.decode_millis >= min_millis
        || input.frame_bytes >= NATIVE_PERSIST_MIN_BYTES;
    if policy == NativeCachePolicy::Auto && !worth_it {
        return NativePersist::Skip(NativeSkipReason::CheapDecode);
    }
    if let Some(free) = input.free_disk {
        if free < input.frame_bytes.saturating_add(NATIVE_DISK_RESERVE) {
            return NativePersist::Skip(NativeSkipReason::LowDisk);
        }
    }
    NativePersist::Write
}

/// Free bytes on the volume holding `dir` (or its nearest existing parent).
#[cfg(windows)]
fn free_disk_bytes(dir: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            free_to_caller: *mut u64,
            total: *mut u64,
            total_free: *mut u64,
        ) -> i32;
    }

    let existing = dir.ancestors().find(|path| path.exists())?;
    let wide: Vec<u16> = existing.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut free = 0u64;
    // SAFETY: `wide` is a NUL-terminated UTF-16 path that outlives the call;
    // the two totals are allowed to be null.
    let ok = unsafe {
        GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut())
    };
    (ok != 0).then_some(free)
}

#[cfg(not(windows))]
fn free_disk_bytes(_dir: &Path) -> Option<u64> {
    None
}

/// Counters for the native-cache policy decisions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeCacheStats {
    pub written: u64,
    pub skipped_policy_off: u64,
    pub skipped_z6iii_disabled: u64,
    pub skipped_cheap: u64,
    pub skipped_low_disk: u64,
    pub write_failed: u64,
}

impl NativeCacheStats {
    const fn empty() -> Self {
        Self {
            written: 0,
            skipped_policy_off: 0,
            skipped_z6iii_disabled: 0,
            skipped_cheap: 0,
            skipped_low_disk: 0,
            write_failed: 0,
        }
    }
}

static NATIVE_STATS: Mutex<NativeCacheStats> = Mutex::new(NativeCacheStats::empty());

#[allow(dead_code)]
pub fn native_cache_stats() -> NativeCacheStats {
    *NATIVE_STATS.lock().unwrap_or_else(|e| e.into_inner())
}

#[allow(dead_code)]
pub fn reset_native_cache_stats() {
    *NATIVE_STATS.lock().unwrap_or_else(|e| e.into_inner()) = NativeCacheStats::empty();
}

/// Decides and performs the native copy for a fresh decode. Proxies are stored
/// unconditionally elsewhere; only full-sensor entries pass through here.
#[allow(clippy::too_many_arguments)]
fn persist_native_entry(
    file: &Path,
    width: u32,
    height: u32,
    pixels: &[u16],
    provenance: DecodeProvenance,
    identity: SourceIdentity,
    route: &DecodeRoute,
    z6iii: bool,
    input: NativePersistInput,
) {
    let decision = native_persist_decision(
        NativeCachePolicy::from_env(),
        z6iii_native_persist_enabled(),
        native_persist_min_millis(),
        input,
    );
    {
        let mut stats = NATIVE_STATS.lock().unwrap_or_else(|e| e.into_inner());
        match decision {
            NativePersist::Write => stats.written += 1,
            NativePersist::Skip(reason) => match reason {
                NativeSkipReason::PolicyOff => stats.skipped_policy_off += 1,
                NativeSkipReason::Z6iiiDisabled => stats.skipped_z6iii_disabled += 1,
                NativeSkipReason::CheapDecode => stats.skipped_cheap += 1,
                NativeSkipReason::LowDisk => stats.skipped_low_disk += 1,
            },
        }
    }
    if decision == NativePersist::Write
        && !store16_binary(file, width, height, pixels, provenance, identity, route)
    {
        NATIVE_STATS.lock().unwrap_or_else(|e| e.into_inner()).write_failed += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn scratch(name: &str) -> PathBuf {
        let base = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir).join("raw-cache-tests");
        let dir = base.join(format!(
            "{name}-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn solid(width: u32, height: u32, byte: u8) -> RgbaImage {
        RgbaImage::new(width, height, vec![byte; (width * height * 4) as usize])
    }

    fn cache_files(dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(PREFIX))
            .collect();
        files.sort();
        files
    }

    fn source(dir: &Path, bytes: &[u8]) -> PathBuf {
        let path = dir.join("source.nef");
        fs::write(&path, bytes).unwrap();
        path
    }

    /// A route a synthetic LibRaw-only entry is consistent with.
    fn test_route(c1_raw_enabled: bool, z6iii_nef: bool) -> DecodeRoute {
        let mut route = DecodeRoute {
            pipeline_fingerprint: 0,
            c1_raw_enabled,
            z6iii_nef,
            z6iii_feature: true,
            libraw_available: true,
            opencl: crate::raw::OpenClRouteState::Available,
            demosaic: crate::raw::DEMOSAIC_DEFAULT,
        };
        route.pipeline_fingerprint =
            crate::raw::DECODE_STAGE_REVISIONS.pipeline_fingerprint(&route);
        route
    }

    fn test_identity(content: u64) -> SourceIdentity {
        SourceIdentity { len: 4096, mtime_nanos: 7, content, full: true }
    }

    fn store_libraw(
        file: &Path,
        width: u32,
        height: u32,
        pixels: &[u16],
        identity: SourceIdentity,
        route: &DecodeRoute,
    ) {
        store16_binary(file, width, height, pixels, DecodeProvenance::libraw16(), identity, route);
    }

    #[test]
    fn raw16_obsolete_cache_is_rejected_by_both_readers() {
        let dir = scratch("legacy-raw16");
        let file = dir.join("raw-test.bin");
        let route = test_route(true, false);
        let identity = test_identity(1);
        store_libraw(&file, 2, 1, &[100, 200, 300, 65535, 400, 500, 600, 65535], identity, &route);
        assert!(read_cached16_binary(&file, &route, &identity).is_some());
        assert!(read_cached_source16(&file, &route, &identity).is_some());
        // B8 carried the stage fingerprint but no source identity; B7 carried
        // the single algorithm revision; B6 had no provenance header at all.
        for magic in [b"AUR16B8\0", b"AUR16B7\0", b"AUR16B6\0", b"AUR16B5\0"] {
            let mut bytes = fs::read(&file).unwrap();
            bytes[..8].copy_from_slice(magic);
            fs::write(&file, bytes).unwrap();
            assert!(read_cached16_binary(&file, &route, &identity).is_none(), "{magic:?}");
            assert!(read_cached_source16(&file, &route, &identity).is_none(), "{magic:?}");
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn route_fingerprint_changes_the_raw16_key() {
        let dir = scratch("route-key");
        let path = source(&dir, b"route");
        let base = test_route(true, true);
        let key = |route| cache_key16_for_route(&path, &dir, 80, route).unwrap();
        let baseline = key(base);

        let mut c1_off = base;
        c1_off.c1_raw_enabled = false;
        let mut z6iii_off = base;
        z6iii_off.z6iii_nef = false;
        let mut no_libraw = base;
        no_libraw.libraw_available = false;
        let mut opencl_flipped = base;
        opencl_flipped.opencl = crate::raw::OpenClRouteState::Unavailable;
        let mut fingerprint = base;
        fingerprint.pipeline_fingerprint += 1;
        let mut demosaic = base;
        demosaic.demosaic = 0;
        for route in [c1_off, z6iii_off, no_libraw, opencl_flipped, fingerprint, demosaic] {
            assert_ne!(baseline, key(route), "a route change must land on a new key");
        }
        assert_eq!(baseline, key(base), "the same route must be stable");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cache_key_changes_when_the_z6iii_flag_flips() {
        let dir = scratch("z6iii-flag-key");
        let path = source(&dir, b"z6iii-flag");
        let z6iii = test_route(true, true);
        let generic = test_route(true, false);
        let z6iii_key = cache_key16_for_route(&path, &dir, 0, z6iii).unwrap();
        let generic_key = cache_key16_for_route(&path, &dir, 0, generic).unwrap();
        assert_ne!(
            z6iii_key, generic_key,
            "a Z6III route must not reuse the generic Bayer entry"
        );
        assert_eq!(
            z6iii_key,
            cache_key16_for_route(&path, &dir, 0, z6iii).unwrap(),
            "the same route must be stable"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn raw16_header_rejects_route_mismatches_and_stale_fingerprints() {
        let dir = scratch("header-route");
        let file = dir.join("raw-header.bin");
        let pixels = vec![7u16; 4 * 4];
        let provenance = DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiOpenCl);
        let z6iii_route = test_route(true, true);
        let identity = test_identity(2);
        store16_binary(&file, 4, 1, &pixels, provenance, identity, &z6iii_route);
        let read = read_cached16_binary(&file, &z6iii_route, &identity).expect("consistent route");
        assert_eq!(read.3, provenance);
        assert!(read_cached_source16(&file, &z6iii_route, &identity).is_some());

        let not_z6iii = test_route(true, false);
        assert!(
            read_cached16_binary(&file, &not_z6iii, &identity).is_none(),
            "a Z6III pre-matrix entry must not serve a non-Z6III route"
        );
        assert!(read_cached_source16(&file, &not_z6iii, &identity).is_none());

        let c1_off = test_route(false, true);
        assert!(
            read_cached16_binary(&file, &c1_off, &identity).is_none(),
            "camera-linear 16-bit must not serve a LibRaw-only route"
        );

        let mut bytes = fs::read(&file).unwrap();
        bytes[40..48]
            .copy_from_slice(&provenance.pipeline_fingerprint.wrapping_add(1).to_le_bytes());
        fs::write(&file, bytes).unwrap();
        assert!(read_cached16_binary(&file, &z6iii_route, &identity).is_none());
        assert!(read_cached_source16(&file, &z6iii_route, &identity).is_none());

        // The reserved words must stay zero: a header from a newer layout is
        // not reinterpreted as this one.
        let mut bytes = fs::read(&file).unwrap();
        bytes[40..48].copy_from_slice(&provenance.pipeline_fingerprint.to_le_bytes());
        bytes[85] = 1;
        fs::write(&file, bytes).unwrap();
        assert!(read_cached16_binary(&file, &z6iii_route, &identity).is_none());
        assert!(read_cached_source16(&file, &z6iii_route, &identity).is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn raw16_header_keeps_the_opencl_failure_state() {
        let dir = scratch("opencl-state");
        let file = dir.join("raw-opencl.bin");
        let pixels = vec![9u16; 2 * 4];
        let provenance =
            DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiCpuOpenClFailed);
        let route = test_route(true, true);
        let identity = test_identity(3);
        store16_binary(&file, 2, 1, &pixels, provenance, identity, &route);
        let read = read_cached16_binary(&file, &route, &identity).expect("read");
        assert_eq!(read.3.opencl, crate::raw::OpenClStatus::Failed);
        assert_eq!(read.3.backend, crate::raw::DecodeBackend::AuroraCpuChain);
        assert!(read.3.unlocks_c1_z6iii_linear());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn first_call_writes_and_second_hits() {
        let dir = scratch("hit");
        let path = source(&dir, b"raw-one");
        let calls = Cell::new(0);
        let decode = |_: &Path| {
            calls.set(calls.get() + 1);
            Ok(solid(80, 40, 7))
        };
        let first = get_or_decode_with(&path, &dir, 80, decode).unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(cache_files(&dir).len(), 1);

        let cached = get_or_decode_with(&path, &dir, 80, |_| {
            calls.set(calls.get() + 1);
            Ok(solid(80, 40, 99))
        })
        .unwrap();
        assert_eq!(calls.get(), 1, "the second call must not decode");
        assert_eq!(cached.pixels, first.pixels);
    }

    #[test]
    fn raw16_cache_round_trips_exact_linear_samples_and_hits() {
        let dir = scratch("raw16-hit");
        let path = source(&dir, b"raw16-one");
        let pixels: Vec<u16> = (0..80u32 * 40 * 4)
            .map(|i| if i % 4 == 3 { 65535 } else { i.wrapping_mul(40503) as u16 })
            .collect();
        let expected = fit16(80, 40, pixels.clone(), 80);
        let provenance = DecodeProvenance::libraw16();
        let calls = Cell::new(0);
        let first = get_or_decode16_with(&path, &dir, 80, |_| {
            calls.set(calls.get() + 1);
            Some((80, 40, pixels.clone(), provenance))
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!((first.0, first.1, first.2.clone()), expected);
        assert_eq!(first.3, provenance);
        assert_eq!(cache_files(&dir).len(), 1);

        let cached = get_or_decode16_with(&path, &dir, 80, |_| {
            calls.set(calls.get() + 1);
            None
        })
        .unwrap();
        assert_eq!(calls.get(), 1, "the second call must hit the 16-bit cache");
        assert_eq!(cached, first, "the provenance must survive the cache");
    }

    #[test]
    fn mapped_source_cache_preserves_render_pixels() {
        let dir = scratch("raw16-source-map");
        let file = dir.join("sample.bin");
        let width = 73u32;
        let height = 41u32;
        let pixels: Vec<u16> = (0..width as usize * height as usize * 4)
            .map(|i| i.wrapping_mul(40503) as u16)
            .collect();
        let route = test_route(true, false);
        let identity = test_identity(4);
        store_libraw(&file, width, height, &pixels, identity, &route);
        let mapped = read_cached_source16(&file, &route, &identity).expect("mapped source").0;
        let expected = Source::from_rgba16(width, height, &pixels);
        let render = |src: &Source| {
            crate::render(
                src,
                &crate::AdjustmentSet::default(),
                crate::NormRect::FULL,
                width.max(height),
                false,
                None,
            )
            .rgba
        };
        assert_eq!(render(&mapped), render(&expected));
    }

    #[test]
    fn mapped_native_source_defers_the_tail_without_changing_pixels() {
        let dir = scratch("raw16-source-map-native");
        let file = dir.join("sample.bin");
        let (width, height) = (1024u32, 768u32);
        let pixels: Vec<u16> = (0..width as usize * height as usize * 4)
            .map(|i| i.wrapping_mul(40503) as u16)
            .collect();
        let route = test_route(true, false);
        let identity = test_identity(5);
        store_libraw(&file, width, height, &pixels, identity, &route);
        let lazy = read_cached_source16_with_mips(&file, &route, &identity, true)
            .expect("mapped source")
            .0;
        let eager = Source::from_rgba16(width, height, &pixels);
        assert_eq!(lazy.levels.len(), eager.levels.len());
        assert!(
            lazy.levels[1..].iter().all(|level| level.data.is_empty()),
            "the native map must not build the tail up front"
        );
        for max_size in [width, width / 2, width / 4] {
            let render = |src: &Source| {
                crate::render(
                    src,
                    &crate::AdjustmentSet::default(),
                    crate::NormRect::FULL,
                    max_size,
                    false,
                    None,
                )
                .rgba
            };
            assert_eq!(render(&lazy), render(&eager), "mapped lazy pixels at {max_size}");
        }
    }

    #[test]
    fn raw16_cache_invalidates_changed_files_and_rebuilds_corruption() {
        let _serial = crate::preview_cache::test_source_epoch_lock();
        let dir = scratch("raw16-invalid");
        let path = source(&dir, b"a");
        let provenance = DecodeProvenance::libraw16();
        let first = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((20, 10, vec![1000; 20 * 10 * 4], provenance))
        })
        .unwrap();
        assert_eq!(first.2[0], 1000);

        fs::write(&path, b"changed-size").unwrap();
        let second = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((20, 10, vec![50000; 20 * 10 * 4], provenance))
        })
        .unwrap();
        assert_eq!(second.2[0], 50000);

        let file = cache_files(&dir)
            .into_iter()
            .max_by_key(|p| fs::metadata(p).unwrap().modified().unwrap())
            .unwrap();
        fs::write(&file, b"broken").unwrap();
        let rebuilt = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((20, 10, vec![12345; 20 * 10 * 4], provenance))
        })
        .unwrap();
        assert_eq!(rebuilt.2[0], 12345);
    }

    #[test]
    fn same_path_same_length_and_restored_mtime_still_redecodes() {
        let _serial = crate::preview_cache::test_source_epoch_lock();
        let dir = scratch("raw16-same-len");
        let path = dir.join("source.nef");
        let len = 4096usize;
        let original = vec![b'a'; len];
        fs::write(&path, &original).unwrap();
        let modified = fs::metadata(&path).unwrap().modified().unwrap();
        let provenance = DecodeProvenance::libraw16();
        let calls = Cell::new(0);

        let first = get_or_decode16_with(&path, &dir, 20, |_| {
            calls.set(calls.get() + 1);
            Some((20, 10, vec![1000; 20 * 10 * 4], provenance))
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(first.2[0], 1000);

        // Same length, different bytes, original mtime restored: only the
        // content fingerprint can tell these apart.
        let mut replaced = original;
        replaced[len / 2] = b'b';
        fs::write(&path, &replaced).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), len as u64);
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().modified().unwrap(),
            modified,
            "the test must actually restore the mtime"
        );

        let second = get_or_decode16_with(&path, &dir, 20, |_| {
            calls.set(calls.get() + 1);
            Some((20, 10, vec![2000; 20 * 10 * 4], provenance))
        })
        .unwrap();
        assert_eq!(calls.get(), 2, "a same-length replacement must re-decode");
        assert_eq!(second.2[0], 2000);
    }

    #[test]
    fn header_middle_and_tail_changes_each_miss() {
        let _serial = crate::preview_cache::test_source_epoch_lock();
        let dir = scratch("raw16-windows");
        let path = dir.join("source.nef");
        // Above the whole-file threshold, so the sampled-window path runs.
        let len = (IDENTITY_FULL_BELOW * 2 + 4096) as usize;
        let mut bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &bytes).unwrap();
        let provenance = DecodeProvenance::libraw16();
        let calls = Cell::new(0);

        for (round, offset) in [0usize, len / 2, len - 1].into_iter().enumerate() {
            let tag = (round + 1) as u16;
            let image = get_or_decode16_with(&path, &dir, 20, |_| {
                calls.set(calls.get() + 1);
                Some((20, 10, vec![tag; 20 * 10 * 4], provenance))
            })
            .unwrap();
            assert_eq!(image.2[0], tag);
            assert_eq!(calls.get(), round + 1, "change at byte {offset} must re-decode");
            bytes[offset] ^= 0xff;
            fs::write(&path, &bytes).unwrap();
        }
    }

    #[test]
    fn cache_header_from_another_source_is_rejected() {
        let dir = scratch("raw16-foreign-header");
        let file = dir.join("entry.bin");
        let route = test_route(true, false);
        let identity = test_identity(11);
        store_libraw(&file, 2, 1, &[1, 2, 3, 65535, 4, 5, 6, 65535], identity, &route);
        let mut other = identity;
        other.content = 12;
        assert!(read_cached16_binary(&file, &route, &other).is_none());
        assert!(read_cached_source16(&file, &route, &other).is_none());
        let mut other = identity;
        other.len += 1;
        assert!(read_cached16_binary(&file, &route, &other).is_none());
        let mut other = identity;
        other.full = false;
        assert!(read_cached16_binary(&file, &route, &other).is_none());

        // Route digest mismatch: the file was copied from another route.
        let other_route = test_route(true, true);
        assert!(read_cached16_binary(&file, &other_route, &identity).is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn truncated_and_tampered_cache_entries_are_rebuilt() {
        let dir = scratch("raw16-tamper");
        let path = source(&dir, b"tamper-source");
        let provenance = DecodeProvenance::libraw16();
        let pixels: Vec<u16> = (0..80 * 40 * 4).map(|i| i as u16).collect();
        let first = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((80, 40, pixels.clone(), provenance))
        })
        .unwrap();
        assert_eq!((first.0, first.1), (20, 10));

        let file = cache_files(&dir).pop().unwrap();
        let bytes = fs::read(&file).unwrap();
        fs::write(&file, &bytes[..HEADER16 + 64]).unwrap();
        let rebuilt = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((80, 40, vec![7; 80 * 40 * 4], provenance))
        })
        .unwrap();
        assert_eq!(rebuilt.2[0], 7, "a truncated entry must be rebuilt");

        let file = cache_files(&dir)
            .into_iter()
            .max_by_key(|p| fs::metadata(p).unwrap().modified().unwrap())
            .unwrap();
        let mut bytes = fs::read(&file).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x5a;
        fs::write(&file, bytes).unwrap();
        let rebuilt = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((80, 40, vec![8; 80 * 40 * 4], provenance))
        })
        .unwrap();
        assert_eq!(rebuilt.2[0], 8, "a tampered body must be rebuilt");
    }

    #[test]
    fn replacing_a_tracked_source_drops_pending_and_bumps_preview_epoch() {
        let _serial = crate::preview_cache::test_source_epoch_lock();
        let dir = scratch("raw16-epoch");
        let path = dir.join("source.nef");
        fs::write(&path, vec![b'a'; 4096]).unwrap();
        let provenance = DecodeProvenance::libraw16();
        let before = crate::preview_cache::source_epoch();
        let calls = Cell::new(0);
        get_or_decode16_with(&path, &dir, 20, |_| {
            calls.set(calls.get() + 1);
            Some((80, 40, vec![1000; 80 * 40 * 4], provenance))
        })
        .unwrap();
        assert_eq!(crate::preview_cache::source_epoch(), before, "first sighting is not a change");

        // Same length, so the pending slot's key differs only by content.
        fs::write(&path, vec![b'b'; 4096]).unwrap();
        let full = get_or_decode16_with(&path, &dir, 0, |_| {
            calls.set(calls.get() + 1);
            Some((80, 40, vec![50000; 80 * 40 * 4], provenance))
        })
        .unwrap();
        assert_eq!(calls.get(), 2, "the replaced file must not reuse the pending frame");
        assert_eq!(full.2[0], 50000);
        assert!(
            crate::preview_cache::source_epoch() > before,
            "a replaced source must invalidate rendered previews"
        );
    }

    #[test]
    #[ignore = "profiles lossless 16-bit RAW cache with a real NEF"]
    fn bench_real_nef_raw16_cache() {
        use std::time::Instant;

        let path = std::env::var_os("AURORA_TEST_RAW")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    r"C:\Users\Jxcz2\Desktop\样张\portrait-test\_DSC4133.NEF",
                )
            });
        if !path.is_file() {
            return;
        }
        let dir = scratch("raw16-real-nef");

        let started = Instant::now();
        let first = get_or_decode16_with(&path, &dir, 4096, |path| {
            crate::raw::decode16_traced(path)
        })
        .expect("first decode");
        let cold_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let second = get_or_decode16_with(&path, &dir, 4096, |_| {
            panic!("cache hit must not invoke LibRaw")
        })
        .expect("cache hit");
        let warm_ms = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(first, second, "cached pixels must be exact");
        let file_bytes = cache_files(&dir).iter().map(|p| fs::metadata(p).unwrap().len()).sum::<u64>();
        println!(
            "NEF_RAW16_CACHE size={}x{} cold={cold_ms:.1} ms warm={warm_ms:.1} ms \
             cache_bytes={file_bytes} speedup={:.2}x",
            first.0,
            first.1,
            cold_ms / warm_ms
        );
    }

    #[test]
    #[ignore = "profiles 16-bit cache formats"]
    fn bench_raw16_cache_formats_11mp() {
        use std::time::Instant;

        let dir = scratch("raw16-formats");
        let width = 4096u32;
        let height = 2729u32;
        let pixels: Vec<u16> = (0..width as usize * height as usize * 4)
            .map(|i| i.wrapping_mul(40503) as u16)
            .collect();
        let png = dir.join("sample.png");
        let bin = dir.join("sample.bin");

        let started = Instant::now();
        store16_png(&png, width, height, &pixels);
        let png_write = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let png_read_data = read_cached16_png(&png).expect("PNG read");
        let png_read = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(png_read_data.2, pixels);

        let bin_route = test_route(true, false);
        let bin_identity = test_identity(5);
        let started = Instant::now();
        store_libraw(&bin, width, height, &pixels, bin_identity, &bin_route);
        let bin_write = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let bin_read_data = read_cached16_binary(&bin, &bin_route, &bin_identity)
            .expect("binary read");
        let bin_read = started.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(bin_read_data.2, pixels);

        println!(
            "RAW16_CACHE_FORMAT png_write={png_write:.1} png_read={png_read:.1} \
             png_bytes={} bin_write={bin_write:.1} bin_read={bin_read:.1} bin_bytes={} \
             read_speedup={:.2}x",
            fs::metadata(&png).unwrap().len(),
            fs::metadata(&bin).unwrap().len(),
            png_read / bin_read
        );
    }

    #[test]
    fn changed_file_invalidates() {
        let dir = scratch("invalidate");
        let path = source(&dir, b"1");
        let decode = |path: &Path| {
            let tag = fs::read(path).unwrap()[0];
            Ok(solid(20, 10, tag))
        };
        let first = get_or_decode_with(&path, &dir, 20, decode).unwrap();
        fs::write(&path, b"22").unwrap();
        let second = get_or_decode_with(&path, &dir, 20, decode).unwrap();
        assert_ne!(first.pixels, second.pixels);
        assert_eq!(second.pixels[0], b'2');
        assert_eq!(cache_files(&dir).len(), 2);
    }

    #[test]
    fn unwritable_cache_directory_falls_back_without_error() {
        // A regular file where the directory should be is unwritable on every OS,
        // which is what the cache has to survive.
        let dir = scratch("unwritable");
        let blocked = dir.join("blocked");
        fs::write(&blocked, b"not a directory").unwrap();
        let path = source(&dir, b"raw");
        let image = get_or_decode_with(&path, &blocked, 16, |_| Ok(solid(32, 16, 3)));
        assert_eq!(image.unwrap().width, 16);
    }

    #[test]
    fn different_long_side_does_not_alias() {
        let dir = scratch("longside");
        let path = source(&dir, b"raw");
        let decode = |_: &Path| Ok(solid(100, 50, 11));
        let big = get_or_decode_with(&path, &dir, 64, decode).unwrap();
        let small = get_or_decode_with(&path, &dir, 32, decode).unwrap();
        assert_eq!((big.width, big.height), (64, 32));
        assert_eq!((small.width, small.height), (32, 16));
        assert_eq!(cache_files(&dir).len(), 2);
    }

    #[test]
    fn corrupt_cache_is_rebuilt() {
        let dir = scratch("corrupt");
        let path = source(&dir, b"raw");
        get_or_decode_with(&path, &dir, 40, |_| Ok(solid(40, 20, 5))).unwrap();
        let file = cache_files(&dir).pop().unwrap();
        fs::write(&file, b"not a png").unwrap();

        let calls = Cell::new(0);
        let image = get_or_decode_with(&path, &dir, 40, |_| {
            calls.set(calls.get() + 1);
            Ok(solid(40, 20, 6))
        })
        .unwrap();
        assert_eq!(calls.get(), 1, "a corrupt file must decode again");
        assert_eq!(image.pixels[0], 6);
        assert!(image::open(&file).is_ok(), "the cache file must be rewritten");
    }

    #[test]
    fn eviction_keeps_count_and_bytes() {
        let dir = scratch("evict");
        for i in 0..3 {
            fs::write(dir.join(format!("{PREFIX}{i}.png")), vec![0u8; 10]).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        enforce_budget(&dir, false, 2, u64::MAX);
        assert_eq!(cache_files(&dir).len(), 2);
        enforce_budget(&dir, false, 10, 15);
        assert_eq!(cache_files(&dir).len(), 1);
    }

    #[test]
    fn native_and_proxy_budgets_are_separate() {
        let dir = scratch("evict-split");
        // Two proxies and two native entries; a proxy-only budget must not
        // touch the native files.
        for name in ["raw-a-100.png", "raw-b-100.png", "raw-a-0.bin", "raw-b-0.bin"] {
            fs::write(dir.join(name), vec![0u8; 10]).unwrap();
        }
        enforce_budget(&dir, false, 0, u64::MAX);
        let remaining: Vec<String> = cache_files(&dir)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(remaining, vec!["raw-a-0.bin".to_string(), "raw-b-0.bin".to_string()]);
        enforce_budget(&dir, true, 1, u64::MAX);
        assert_eq!(cache_files(&dir).len(), 1);
    }

    #[test]
    fn one_scan_applies_both_budgets_and_spares_foreign_files() {
        let dir = scratch("one-scan");
        for name in ["raw-p1-100.bin", "raw-p2-100.bin", "raw-p3-100.bin", "raw-n1-0.bin", "raw-n2-0.bin"] {
            fs::write(dir.join(name), vec![0u8; 10]).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        fs::write(dir.join("external.bin"), b"keep").unwrap();
        fs::create_dir(dir.join("raw-dir-0.bin")).unwrap();

        let (mut proxies, mut natives) = scan_cache_entries(&dir);
        assert_eq!((proxies.len(), natives.len()), (3, 2));
        enforce_items(&mut proxies, 2, u64::MAX);
        enforce_items(&mut natives, 1, u64::MAX);

        let mut remaining: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|entry| entry.path().is_file())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(PREFIX))
            .collect();
        remaining.sort();
        assert_eq!(remaining, vec!["raw-n2-0.bin", "raw-p2-100.bin", "raw-p3-100.bin"]);
        assert!(dir.join("external.bin").is_file());
        assert!(dir.join("raw-dir-0.bin").is_dir());
    }

    #[test]
    fn proxy_decode_feeds_the_first_full_size_request_without_decoding() {
        let _serial = crate::preview_cache::test_source_epoch_lock();
        let dir = scratch("raw16-pending");
        let path = source(&dir, b"raw16-pending");
        let pixels: Vec<u16> = (0..80u32 * 40 * 4)
            .map(|i| if i % 4 == 3 { 65535 } else { i.wrapping_mul(40503) as u16 })
            .collect();
        let expected_proxy = fit16(80, 40, pixels.clone(), 40);
        let provenance = DecodeProvenance::libraw16();
        let calls = Cell::new(0);
        let proxy = get_or_decode16_with(&path, &dir, 40, |_| {
            calls.set(calls.get() + 1);
            Some((80, 40, pixels.clone(), provenance))
        })
        .expect("proxy decode");
        assert_eq!((proxy.0, proxy.1, proxy.2.clone()), expected_proxy);
        assert_eq!(calls.get(), 1);
        assert_eq!(cache_files(&dir).len(), 1, "only the proxy is persisted");

        let full = get_or_decode16_with(&path, &dir, 0, |_| {
            calls.set(calls.get() + 1);
            None
        })
        .expect("full-size request must reuse the cold decode");
        assert_eq!(calls.get(), 1, "the first 1:1 view must not decode again");
        assert_eq!((full.0, full.1, full.2), (80, 40, pixels));
        assert_eq!(full.3, provenance);
        assert_eq!(cache_files(&dir).len(), 1, "consuming the frame must not write the native cache");
    }

    #[test]
    fn cached_full_frame_feeds_a_proxy_without_decoding() {
        let _serial = crate::preview_cache::test_source_epoch_lock();
        let dir = scratch("raw16-native-fallback");
        let path = source(&dir, b"raw16-native");
        let pixels: Vec<u16> = (0..80u32 * 40 * 4)
            .map(|i| if i % 4 == 3 { 65535 } else { i.wrapping_mul(40503) as u16 })
            .collect();
        let provenance = DecodeProvenance::libraw16();
        let calls = Cell::new(0);
        let full = get_or_decode16_with(&path, &dir, 0, |_| {
            calls.set(calls.get() + 1);
            Some((80, 40, pixels.clone(), provenance))
        })
        .expect("native cache decode");
        assert_eq!((full.0, full.1, full.2.clone()), (80, 40, pixels.clone()));
        assert_eq!(calls.get(), 1);

        let proxy = get_or_decode16_with(&path, &dir, 40, |_| {
            calls.set(calls.get() + 1);
            None
        })
        .expect("proxy from the native cache");
        assert_eq!(calls.get(), 1, "a native cache hit must not decode");
        assert_eq!((proxy.0, proxy.1, proxy.2), fit16(80, 40, pixels, 40));
        assert_eq!(proxy.3, provenance);
    }

    #[test]
    fn replaced_source_is_not_served_from_the_pending_frame() {
        let _serial = crate::preview_cache::test_source_epoch_lock();
        let dir = scratch("raw16-pending-stale");
        let path = source(&dir, b"first");
        let provenance = DecodeProvenance::libraw16();
        get_or_decode16_with(&path, &dir, 40, |_| {
            Some((80, 40, vec![1000; 80 * 40 * 4], provenance))
        })
        .expect("proxy decode");
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::write(&path, b"second").unwrap();
        let calls = Cell::new(0);
        let full = get_or_decode16_with(&path, &dir, 0, |_| {
            calls.set(calls.get() + 1);
            Some((80, 40, vec![50000; 80 * 40 * 4], provenance))
        })
        .expect("full decode after replacement");
        assert_eq!(calls.get(), 1, "a replaced file must not use stale pixels");
        assert_eq!(full.2[0], 50000);
    }

    /// Manual measurement against a real file: set `AURORA_RAW_CACHE_BENCH`.
    #[test]
    #[ignore = "manual timing, needs AURORA_RAW_CACHE_BENCH"]
    fn bench_real_raw() {
        let Ok(path) = std::env::var("AURORA_RAW_CACHE_BENCH") else { return };
        let path = PathBuf::from(path);
        let dir = scratch("bench");
        let started = std::time::Instant::now();
        let first = get_or_decode(&path, &dir, 2560).unwrap();
        let cold = started.elapsed();
        let started = std::time::Instant::now();
        let second = get_or_decode(&path, &dir, 2560).unwrap();
        let warm = started.elapsed();
        assert_eq!(first.pixels, second.pixels);
        let file = cache_files(&dir).pop().unwrap();
        eprintln!(
            "cold={}ms warm={}ms png={}B size={}x{}",
            cold.as_millis(),
            warm.as_millis(),
            fs::metadata(file).unwrap().len(),
            second.width,
            second.height
        );
    }

    #[test]
    fn pending_lru_evicts_by_bytes_and_counts() {
        let route = test_route(true, false);
        let identity = test_identity(21);
        let provenance = DecodeProvenance::libraw16();
        let path = PathBuf::from("lru.nef");
        let mut pending = PendingNative::new();
        // 100 u16 = 200 bytes per frame.
        for key in ["raw-a-0.bin", "raw-b-0.bin"] {
            pending.insert(
                Some(key.to_string()), &path, 10, 10, vec![0; 100], provenance, identity, &route,
                400,
            );
        }
        assert_eq!(pending.entries.len(), 2);
        assert_eq!(pending.bytes, 400);

        pending.insert(
            Some("raw-c-0.bin".to_string()), &path, 10, 10, vec![0; 100], provenance, identity,
            &route, 400,
        );
        assert_eq!(pending.entries.len(), 2, "the byte budget bounds the table");
        assert_eq!(pending.stats.evictions, 1);
        assert_eq!(pending.entries.front().unwrap().key, "raw-b-0.bin");

        pending.insert(
            Some("raw-b-0.bin".to_string()), &path, 10, 10, vec![0; 100], provenance, identity,
            &route, 400,
        );
        assert_eq!(pending.stats.overwrites, 1);
        assert_eq!(pending.bytes, 400);

        pending.insert(
            Some("raw-d-0.bin".to_string()), &path, 10, 10, vec![0; 300], provenance, identity,
            &route, 400,
        );
        assert_eq!(pending.stats.rejected_too_large, 1);
        assert_eq!(pending.entries.len(), 2);
        assert_eq!(pending.stats.peak_bytes, 400);
    }

    #[test]
    fn pending_take_matches_only_the_same_identity_and_size() {
        let route = test_route(true, false);
        let identity = test_identity(22);
        let provenance = DecodeProvenance::libraw16();
        let path = PathBuf::from("take.nef");
        let name = cache_name16_for_identity(&path, 0, route, identity);
        let mut pending = PendingNative::new();
        pending.insert(
            Some(name), &path, 10, 10, vec![0; 100], provenance, identity, &route, 1 << 20,
        );
        assert!(pending.take(&path, 20, route, identity).is_none(), "only 1:1 takes");
        let mut other = identity;
        other.content = 99;
        assert!(pending.take(&path, 0, route, other).is_none());
        // Proxy requests are not counted; only 1:1 misses are.
        assert_eq!(pending.stats.misses, 1);
        let frame = pending.take(&path, 0, route, identity).expect("hit");
        assert_eq!(frame.0, 10);
        assert_eq!(pending.stats.hits, 1);
        assert!(pending.entries.is_empty());
    }

    #[test]
    fn three_images_alternate_proxy_and_full_size_without_cross_overwrites() {
        let dir = scratch("raw16-abc");
        let provenance = DecodeProvenance::libraw16();
        let mut paths = Vec::new();
        let mut frames = Vec::new();
        for tag in 0..3u16 {
            let path = dir.join(format!("img{tag}.nef"));
            fs::write(&path, format!("raw-image-{tag}")).unwrap();
            frames.push(vec![tag + 1; 80 * 40 * 4]);
            paths.push(path);
        }
        for (index, path) in paths.iter().enumerate() {
            let pixels = frames[index].clone();
            let proxy = get_or_decode16_with(path, &dir, 40, move |_| {
                Some((80, 40, pixels, provenance))
            })
            .expect("proxy decode");
            assert_eq!((proxy.0, proxy.1), (40, 20));
        }
        // 1:1 in reverse order: every proxy decode must still have its native
        // frame, and one image's work must not drop another's.
        for index in [2usize, 0, 1] {
            let pixels = frames[index].clone();
            let calls = Cell::new(0);
            let full = get_or_decode16_with(&paths[index], &dir, 0, |_| {
                calls.set(calls.get() + 1);
                Some((80, 40, pixels.clone(), provenance))
            })
            .expect("full-size request");
            assert_eq!(calls.get(), 0, "image {index} must reuse its pending frame");
            assert_eq!(full.2, frames[index]);
        }
    }

    #[test]
    fn concurrent_requests_for_one_key_decode_once() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        use std::sync::{Arc, Barrier};

        let dir = scratch("raw16-single-flight");
        let path = source(&dir, b"flight");
        let calls = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));
        let provenance = DecodeProvenance::libraw16();
        // The cache stores RGB only and reconstructs alpha as opaque, so the
        // synthetic frame must be opaque for the waiter's disk hit to match.
        let pixels: Vec<u16> = (0..80 * 40 * 4)
            .map(|i| if i % 4 == 3 { 65535 } else { i as u16 })
            .collect();
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let dir = dir.clone();
                let path = path.clone();
                let calls = calls.clone();
                let barrier = barrier.clone();
                let pixels = pixels.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    get_or_decode16_with(&path, &dir, 20, |_| {
                        calls.fetch_add(1, AtomicOrdering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        Some((80, 40, pixels.clone(), provenance))
                    })
                    .expect("decode")
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1, "one decode per key");
        assert_eq!(results[0], results[1], "waiters see the leader's pixels");
    }

    #[test]
    fn a_slow_decode_for_one_image_leaves_another_images_frame_intact() {
        let dir = scratch("raw16-late");
        let provenance = DecodeProvenance::libraw16();
        let slow_path = dir.join("slow.nef");
        let fast_path = dir.join("fast.nef");
        fs::write(&slow_path, b"slow-image").unwrap();
        fs::write(&fast_path, b"fast-image").unwrap();
        let slow_pixels = vec![7u16; 80 * 40 * 4];
        let fast_pixels = vec![9u16; 80 * 40 * 4];

        let slow = {
            let dir = dir.clone();
            let path = slow_path.clone();
            let pixels = slow_pixels.clone();
            std::thread::spawn(move || {
                get_or_decode16_with(&path, &dir, 40, |_| {
                    std::thread::sleep(std::time::Duration::from_millis(150));
                    Some((80, 40, pixels, provenance))
                })
                .expect("slow proxy decode")
            })
        };
        let proxy = get_or_decode16_with(&fast_path, &dir, 40, |_| {
            Some((80, 40, fast_pixels.clone(), provenance))
        })
        .expect("fast proxy decode");
        assert_eq!((proxy.0, proxy.1), (40, 20));
        slow.join().unwrap();

        // The slow request finished after the fast one; the fast image's native
        // frame must still be there for its first 1:1.
        let calls = Cell::new(0);
        let full = get_or_decode16_with(&fast_path, &dir, 0, |_| {
            calls.set(calls.get() + 1);
            Some((80, 40, fast_pixels.clone(), provenance))
        })
        .expect("full-size request");
        assert_eq!(calls.get(), 0, "a late unrelated decode must not evict this frame");
        assert_eq!(full.2, fast_pixels);
    }

    /// Runs `f` with `AURORA_RAW_NATIVE_CACHE` (and the Z6III switch) pinned,
    /// restoring the environment afterwards. The cache tests that depend on the
    /// policy hold the same lock, so no other test observes the override.
    fn with_native_policy<T>(
        policy: Option<&str>,
        z6iii: Option<&str>,
        f: impl FnOnce() -> T,
    ) -> T {
        let _serial = crate::preview_cache::test_source_epoch_lock();
        let previous_policy = std::env::var_os("AURORA_RAW_NATIVE_CACHE");
        let previous_z6iii = std::env::var_os("AURORA_RAW_NATIVE_CACHE_Z6III");
        // SAFETY: every test that reads these variables or writes native cache
        // entries holds the same serial lock.
        unsafe {
            match policy {
                Some(value) => std::env::set_var("AURORA_RAW_NATIVE_CACHE", value),
                None => std::env::remove_var("AURORA_RAW_NATIVE_CACHE"),
            }
            match z6iii {
                Some(value) => std::env::set_var("AURORA_RAW_NATIVE_CACHE_Z6III", value),
                None => std::env::remove_var("AURORA_RAW_NATIVE_CACHE_Z6III"),
            }
        }
        let result = f();
        // SAFETY: as above; the lock is still held.
        unsafe {
            match previous_policy {
                Some(value) => std::env::set_var("AURORA_RAW_NATIVE_CACHE", value),
                None => std::env::remove_var("AURORA_RAW_NATIVE_CACHE"),
            }
            match previous_z6iii {
                Some(value) => std::env::set_var("AURORA_RAW_NATIVE_CACHE_Z6III", value),
                None => std::env::remove_var("AURORA_RAW_NATIVE_CACHE_Z6III"),
            }
        }
        result
    }

    fn z6iii_proxy_entry(dir: &Path) -> (PathBuf, DecodeRoute, DecodeProvenance, Vec<u16>) {
        let path = dir.join("z6iii.nef");
        fs::write(&path, b"z6iii-he-source").unwrap();
        let route = test_route(true, true);
        let provenance = DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiOpenCl);
        (path, route, provenance, vec![4321u16; 80 * 40 * 4])
    }

    #[test]
    fn z6iii_proxy_decode_persists_the_native_frame_by_default() {
        with_native_policy(None, None, || {
            let dir = scratch("raw16-z6iii-native");
            let (path, route, provenance, pixels) = z6iii_proxy_entry(&dir);
            reset_native_cache_stats();
            get_or_decode16_with_route(&path, &dir, 40, route, |_| {
                Some((80, 40, pixels.clone(), provenance))
            })
            .expect("proxy decode");
            assert_eq!(
                cache_files(&dir).len(),
                2,
                "the proxy and the expensive native frame are both persisted"
            );
            assert!(native_cache_stats().written >= 1);
        });
    }

    #[test]
    fn native_cache_off_keeps_only_the_proxy() {
        with_native_policy(Some("off"), None, || {
            let dir = scratch("raw16-native-off");
            let (path, route, provenance, pixels) = z6iii_proxy_entry(&dir);
            reset_native_cache_stats();
            get_or_decode16_with_route(&path, &dir, 40, route, |_| {
                Some((80, 40, pixels.clone(), provenance))
            })
            .expect("proxy decode");
            assert_eq!(cache_files(&dir).len(), 1);
            assert!(native_cache_stats().skipped_policy_off >= 1);
        });
    }

    #[test]
    fn z6iii_native_switch_keeps_only_the_proxy() {
        with_native_policy(None, Some("0"), || {
            let dir = scratch("raw16-z6iii-switch");
            let (path, route, provenance, pixels) = z6iii_proxy_entry(&dir);
            reset_native_cache_stats();
            get_or_decode16_with_route(&path, &dir, 40, route, |_| {
                Some((80, 40, pixels.clone(), provenance))
            })
            .expect("proxy decode");
            assert_eq!(cache_files(&dir).len(), 1);
            assert!(native_cache_stats().skipped_z6iii_disabled >= 1);
        });
    }

    #[test]
    fn native_policy_matrix() {
        let base = NativePersistInput {
            explicit_native: false,
            decode_millis: 10,
            frame_bytes: 1000,
            z6iii: false,
            free_disk: Some(u64::MAX),
        };
        let write = NativePersist::Write;
        assert_eq!(
            native_persist_decision(NativeCachePolicy::Off, true, 250, base),
            NativePersist::Skip(NativeSkipReason::PolicyOff)
        );
        assert_eq!(
            native_persist_decision(NativeCachePolicy::Auto, true, 250, base),
            NativePersist::Skip(NativeSkipReason::CheapDecode)
        );
        assert_eq!(
            native_persist_decision(
                NativeCachePolicy::Auto,
                true,
                250,
                NativePersistInput { explicit_native: true, ..base }
            ),
            write
        );
        assert_eq!(
            native_persist_decision(
                NativeCachePolicy::Auto,
                true,
                250,
                NativePersistInput { decode_millis: 300, ..base }
            ),
            write
        );
        assert_eq!(
            native_persist_decision(
                NativeCachePolicy::Auto,
                true,
                250,
                NativePersistInput { z6iii: true, ..base }
            ),
            write
        );
        assert_eq!(
            native_persist_decision(
                NativeCachePolicy::Auto,
                false,
                250,
                NativePersistInput { z6iii: true, ..base }
            ),
            NativePersist::Skip(NativeSkipReason::Z6iiiDisabled)
        );
        assert_eq!(
            native_persist_decision(NativeCachePolicy::Always, true, 250, base),
            write
        );
        assert_eq!(
            native_persist_decision(
                NativeCachePolicy::Always,
                true,
                250,
                NativePersistInput {
                    free_disk: Some(1000 + NATIVE_DISK_RESERVE - 1),
                    ..base
                }
            ),
            NativePersist::Skip(NativeSkipReason::LowDisk)
        );
        assert_eq!(
            native_persist_decision(
                NativeCachePolicy::Auto,
                true,
                250,
                NativePersistInput { frame_bytes: NATIVE_PERSIST_MIN_BYTES, ..base }
            ),
            write
        );
    }
}
