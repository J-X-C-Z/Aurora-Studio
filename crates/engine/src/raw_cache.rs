//! On-disk cache of decoded RAW previews.
//!
//! libraw takes seconds on a full sensor frame, and reopening a file used to pay
//! that again. This keeps the downscaled decode as a lossless PNG, keyed by the
//! source path, size, mtime and requested long side.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Read;
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
// Bump when the persisted layout changes. B7 adds the provenance header
// (revision, backend, OpenCL status, pixel space) at bytes 32..40, so B6 files
// miss naturally.
const RAW16_MAGIC: &[u8; 8] = b"AUR16B7\0";
/// magic(8) width(4) height(4) body bytes(8) image hash(8) revision(4)
/// backend(1) opencl(1) pixel_space(1) flags(1).
const HEADER16: usize = 40;
const MAX_ENTRIES: usize = 64;
const MAX_BYTES: u64 = 1 << 30;
/// One full-sensor frame kept after a cold proxy decode. The fit view already
/// paid for that decode and then discarded the native pixels, which made the
/// first 1:1 zoom decode the same sensor again. 256 MiB covers a 24 MP RGBA16
/// frame; larger sensors keep the old decode-on-zoom path.
const MAX_PENDING_NATIVE_BYTES: usize = 256 << 20;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static PENDING_NATIVE: Mutex<Option<(String, u32, u32, Vec<u16>, DecodeProvenance)>> =
    Mutex::new(None);

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
    #[cfg(not(target_os = "windows"))]
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
        return decode_frame16(path, long_side, route);
    };
    get_or_decode16_with(path, &dir, long_side, crate::raw::decode16_traced)
}

pub(crate) fn get_or_decode_source16(
    path: &Path,
    long_side: u32,
) -> Option<(Source, DecodeProvenance)> {
    let route = crate::raw::decode_route(path);
    let Some(dir) = default_raw16_dir() else {
        return decode_source16(path, long_side, route);
    };
    let key = cache_key16_for_route(path, &dir, long_side, route);
    if let Some(source) = key.as_deref().and_then(|file| read_cached_source16(file, &route)) {
        return Some(source);
    }
    if let Some((width, height, pixels, provenance)) = take_pending_native(path, long_side, route) {
        return Some((Source::from_rgba16(width, height, &pixels), provenance));
    }
    let image = match cached_full_frame16(path, &dir, long_side, route) {
        Some(image) => image,
        None => decode_frame16(path, long_side, route)?,
    };
    if let Some(file) = key {
        store16_binary(&file, image.0, image.1, &image.2, image.3);
    }
    Some((Source::from_rgba16(image.0, image.1, &image.2), image.3))
}

fn decode_source16(
    path: &Path,
    long_side: u32,
    route: DecodeRoute,
) -> Option<(Source, DecodeProvenance)> {
    let image = decode_frame16(path, long_side, route)?;
    Some((Source::from_rgba16(image.0, image.1, &image.2), image.3))
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
    let key = cache_key16_for_route(path, dir, long_side, route);
    if let Some(file) = key
        .as_deref()
        .and_then(|file| read_cached16_binary(file, &route))
    {
        return Some(file);
    }
    if let Some(image) = take_pending_native(path, long_side, route) {
        return Some(image);
    }
    let image = match cached_full_frame16(path, dir, long_side, route) {
        Some(image) => image,
        None => {
            let native_key = cache_name16_for_route(path, 0, route);
            let (w, h, pixels, provenance) = decode(path)?;
            fit16_keep_native(native_key, w, h, pixels, long_side, provenance)
        }
    };
    if let Some(file) = key {
        store16_binary(&file, image.0, image.1, &image.2, image.3);
    }
    Some(image)
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
/// `route` carries the algorithm revision, C1 switch, Z6III detection, libraw
/// availability, OpenCL state and demosaic filter, so any backend change lands
/// on a different file instead of reusing pixels from another route.
fn cache_key16_for_route(
    path: &Path,
    dir: &Path,
    long_side: u32,
    route: DecodeRoute,
) -> Option<PathBuf> {
    Some(dir.join(cache_name16_for_route(path, long_side, route)?))
}

/// The flat name from the file identity and decoder route.
fn cache_name16_for_route(path: &Path, long_side: u32, route: DecodeRoute) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let mut a = DefaultHasher::new();
    let mut b = DefaultHasher::new();
    (RAW16_MAGIC, path, meta.len(), mtime, long_side, route).hash(&mut a);
    (0xa7u8, path, mtime, long_side, meta.len(), route).hash(&mut b);
    Some(format!(
        "{PREFIX}16b-{:016x}{:016x}-{long_side}.bin",
        a.finish(),
        b.finish()
    ))
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
/// zoom request. The pending frame is one slot and never triggers extra work.
fn fit16_keep_native(
    native_key: Option<String>,
    width: u32,
    height: u32,
    pixels: Vec<u16>,
    long_side: u32,
    provenance: DecodeProvenance,
) -> (u32, u32, Vec<u16>, DecodeProvenance) {
    let Some((target_width, target_height, proxy)) =
        resized16(width, height, &pixels, long_side)
    else {
        return (width, height, pixels, provenance);
    };
    remember_native(native_key, width, height, pixels, provenance);
    (target_width, target_height, proxy, provenance)
}

fn remember_native(
    native_key: Option<String>,
    width: u32,
    height: u32,
    pixels: Vec<u16>,
    provenance: DecodeProvenance,
) {
    if pixels.capacity().saturating_mul(2) > MAX_PENDING_NATIVE_BYTES {
        return;
    }
    let Some(key) = native_key else { return };
    *PENDING_NATIVE.lock().unwrap_or_else(|e| e.into_inner()) =
        Some((key, width, height, pixels, provenance));
}

fn take_pending_native(
    path: &Path,
    long_side: u32,
    route: DecodeRoute,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    if long_side != 0 {
        return None;
    }
    let key = cache_name16_for_route(path, 0, route)?;
    let mut pending = PENDING_NATIVE.lock().unwrap_or_else(|e| e.into_inner());
    if pending
        .as_ref()
        .is_some_and(|(stored, _, _, _, provenance)| {
            *stored == key && provenance.consistent_with(&route)
        })
    {
        pending
            .take()
            .map(|(_, width, height, pixels, provenance)| (width, height, pixels, provenance))
    } else {
        None
    }
}

fn cached_full_frame16(
    path: &Path,
    dir: &Path,
    long_side: u32,
    route: DecodeRoute,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    if long_side == 0 {
        return None;
    }
    let native_key = cache_name16_for_route(path, 0, route)?;
    let (width, height, pixels, provenance) =
        read_cached16_binary(&dir.join(&native_key), &route)?;
    Some(fit16_keep_native(
        Some(native_key),
        width,
        height,
        pixels,
        long_side,
        provenance,
    ))
}

fn decode_frame16(
    path: &Path,
    long_side: u32,
    route: DecodeRoute,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    if let Some(pending) = take_pending_native(path, long_side, route) {
        return Some(pending);
    }
    let native_key = cache_name16_for_route(path, 0, route);
    let (width, height, pixels, provenance) = crate::raw::decode16_traced(path)?;
    Some(fit16_keep_native(
        native_key,
        width,
        height,
        pixels,
        long_side,
        provenance,
    ))
}

/// Reads one RAW16 entry and rejects it unless its provenance belongs to
/// `route`. Revision, magic, enum codes and route consistency are all checked.
fn read_cached16_binary(
    file: &Path,
    route: &DecodeRoute,
) -> Option<(u32, u32, Vec<u16>, DecodeProvenance)> {
    use rayon::prelude::*;

    let mut reader = fs::File::open(file).ok()?;
    let size = reader.metadata().ok()?.len();
    let mut header = [0u8; HEADER16];
    reader.read_exact(&mut header).ok()?;
    if &header[..8] != RAW16_MAGIC {
        return None;
    }
    let width = u32::from_le_bytes(header[8..12].try_into().ok()?);
    let height = u32::from_le_bytes(header[12..16].try_into().ok()?);
    let bytes = u64::from_le_bytes(header[16..24].try_into().ok()?) as usize;
    let expected_bytes = width as usize * height as usize * 6;
    if width == 0
        || height == 0
        || bytes != expected_bytes
        || size != (HEADER16 + bytes) as u64
    {
        return None;
    }
    let provenance = DecodeProvenance::from_codes(
        u32::from_le_bytes(header[32..36].try_into().ok()?),
        header[36],
        header[37],
        header[38],
        header[39],
    )?;
    if !provenance.consistent_with(route) {
        return None;
    }
    let stored_hash = u64::from_le_bytes(header[24..32].try_into().ok()?);
    let mut rgb = vec![0u16; expected_bytes / 2];
    // SAFETY: `u8` has weaker alignment than `u16`, and the complete zeroed
    // allocation is written from disk before any `u16` is read.
    let (before, body, after) = unsafe { rgb.align_to_mut::<u8>() };
    if !before.is_empty() || !after.is_empty() {
        return None;
    }
    reader.read_exact(&mut body[..expected_bytes]).ok()?;
    if image_hash(width, height, &body[..expected_bytes]) != stored_hash {
        return None;
    }
    #[cfg(target_endian = "big")]
    for sample in &mut rgb {
        *sample = u16::from_le(*sample);
    }
    let mut pixels = vec![65535u16; width as usize * height as usize * 4];
    pixels
        .par_chunks_exact_mut(4)
        .zip(rgb.par_chunks_exact(3))
        .for_each(|(dst, src)| dst[..3].copy_from_slice(src));
    Some((width, height, pixels, provenance))
}

#[cfg(target_endian = "little")]
fn read_cached_source16(
    file: &Path,
    route: &DecodeRoute,
) -> Option<(Source, DecodeProvenance)> {
    use memmap2::MmapOptions;

    let file = fs::File::open(file).ok()?;
    // SAFETY: the cache file is read-only for the lifetime of the mapping and
    // all mutations replace it atomically through rename.
    let map = unsafe { MmapOptions::new().map(&file).ok()? };
    if map.len() < HEADER16 || &map[..8] != RAW16_MAGIC {
        return None;
    }
    let width = u32::from_le_bytes(map[8..12].try_into().ok()?);
    let height = u32::from_le_bytes(map[12..16].try_into().ok()?);
    let bytes = u64::from_le_bytes(map[16..24].try_into().ok()?) as usize;
    let expected_bytes = width as usize * height as usize * 6;
    if width == 0 || height == 0 || bytes != expected_bytes || map.len() != HEADER16 + bytes {
        return None;
    }
    let provenance = DecodeProvenance::from_codes(
        u32::from_le_bytes(map[32..36].try_into().ok()?),
        map[36],
        map[37],
        map[38],
        map[39],
    )?;
    if !provenance.consistent_with(route) {
        return None;
    }
    let stored_hash = u64::from_le_bytes(map[24..32].try_into().ok()?);
    let body = &map[HEADER16..];
    // SAFETY: the mapping starts page-aligned and HEADER16 is 40 bytes, so the
    // body is u16-aligned. `align_to` checks and reports any residual bytes.
    let (before, pixels, after) = unsafe { body.align_to::<u16>() };
    if !before.is_empty() || !after.is_empty() || pixels.len() != expected_bytes / 2 {
        return None;
    }
    let (valid, source) = rayon::join(
        || image_hash(width, height, body) == stored_hash,
        || Source::from_rgb16(width, height, pixels),
    );
    valid.then_some((source, provenance))
}

#[cfg(not(target_endian = "little"))]
fn read_cached_source16(
    file: &Path,
    route: &DecodeRoute,
) -> Option<(Source, DecodeProvenance)> {
    let (width, height, pixels, provenance) = read_cached16_binary(file, route)?;
    Some((Source::from_rgba16(width, height, &pixels), provenance))
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
    enforce_limits(dir, MAX_ENTRIES, MAX_BYTES);
}

fn store16_binary(
    file: &Path,
    width: u32,
    height: u32,
    pixels: &[u16],
    provenance: DecodeProvenance,
) {
    use rayon::prelude::*;

    let Some(dir) = file.parent() else { return };
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let temp = dir.join(format!(
        ".tmp16b-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut body = vec![0u8; pixels.len() / 4 * 6];
    body.par_chunks_exact_mut(6)
        .zip(pixels.par_chunks_exact(4))
        .for_each(|(dst, src)| {
            dst[0..2].copy_from_slice(&src[0].to_le_bytes());
            dst[2..4].copy_from_slice(&src[1].to_le_bytes());
            dst[4..6].copy_from_slice(&src[2].to_le_bytes());
        });
    let mut data = Vec::with_capacity(HEADER16 + body.len());
    data.extend_from_slice(RAW16_MAGIC);
    data.extend_from_slice(&width.to_le_bytes());
    data.extend_from_slice(&height.to_le_bytes());
    data.extend_from_slice(&(body.len() as u64).to_le_bytes());
    data.extend_from_slice(&image_hash(width, height, &body).to_le_bytes());
    data.extend_from_slice(&provenance.revision.to_le_bytes());
    data.push(provenance.backend_code());
    data.push(provenance.opencl_code());
    data.push(provenance.pixel_space_code());
    data.push(provenance.flags());
    data.extend_from_slice(&body);
    if fs::write(&temp, &data).is_err() || fs::rename(&temp, file).is_err() {
        let _ = fs::remove_file(&temp);
        return;
    }
    enforce_limits(dir, MAX_ENTRIES, MAX_BYTES);
}

fn image_hash(width: u32, height: u32, body: &[u8]) -> u64 {
    use rayon::prelude::*;

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

    const CHUNK: usize = 256 << 10;
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ u64::from(width).rotate_left(17);
    hash ^= u64::from(height).rotate_left(41);
    hash = hash.wrapping_mul(0x9E37_79B1_85EB_CA87);
    let parts: Vec<u64> = body.par_chunks(CHUNK).map(chunk_hash).collect();
    for part in parts {
        hash ^= part;
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
    enforce_limits(dir, MAX_ENTRIES, MAX_BYTES);
}

/// Removes the oldest entries until both budgets fit.
fn enforce_limits(dir: &Path, max_entries: usize, max_bytes: u64) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut items: Vec<(PathBuf, u64, SystemTime)> = entries
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with(PREFIX) && (name.ends_with(".png") || name.ends_with(".bin"))
        })
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            if !meta.is_file() {
                return None;
            }
            let modified = meta.modified().unwrap_or(UNIX_EPOCH);
            Some((e.path(), meta.len(), modified))
        })
        .collect();
    items.sort_by_key(|(_, _, modified)| *modified);
    let mut count = items.len();
    let mut bytes: u64 = items.iter().map(|(_, size, _)| size).sum();
    for (file, size, _) in items {
        if count <= max_entries && bytes <= max_bytes {
            break;
        }
        if fs::remove_file(&file).is_ok() {
            count -= 1;
            bytes = bytes.saturating_sub(size);
        }
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
        DecodeRoute {
            revision: crate::raw::DECODE_ALGORITHM_REVISION,
            c1_raw_enabled,
            z6iii_nef,
            z6iii_feature: true,
            libraw_available: true,
            opencl: crate::raw::OpenClRouteState::Available,
            demosaic: crate::raw::DEMOSAIC_DEFAULT,
        }
    }

    fn store_libraw(file: &Path, width: u32, height: u32, pixels: &[u16]) {
        store16_binary(file, width, height, pixels, DecodeProvenance::libraw16());
    }

    #[test]
    fn raw16_obsolete_cache_is_rejected_by_both_readers() {
        let dir = scratch("legacy-raw16");
        let file = dir.join("raw-test.bin");
        let route = test_route(true, false);
        store_libraw(&file, 2, 1, &[100, 200, 300, 65535, 400, 500, 600, 65535]);
        assert!(read_cached16_binary(&file, &route).is_some());
        assert!(read_cached_source16(&file, &route).is_some());
        // B6 had no provenance header and must miss; B5 is older still.
        for magic in [b"AUR16B6\0", b"AUR16B5\0"] {
            let mut bytes = fs::read(&file).unwrap();
            bytes[..8].copy_from_slice(magic);
            fs::write(&file, bytes).unwrap();
            assert!(read_cached16_binary(&file, &route).is_none(), "{magic:?}");
            assert!(read_cached_source16(&file, &route).is_none(), "{magic:?}");
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
        let mut revision = base;
        revision.revision += 1;
        let mut demosaic = base;
        demosaic.demosaic = 0;
        for route in [c1_off, z6iii_off, no_libraw, opencl_flipped, revision, demosaic] {
            assert_ne!(baseline, key(route), "a route change must land on a new key");
        }
        assert_eq!(baseline, key(base), "the same route must be stable");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn raw16_header_rejects_route_mismatches_and_stale_revisions() {
        let dir = scratch("header-route");
        let file = dir.join("raw-header.bin");
        let pixels = vec![7u16; 4 * 4];
        let provenance = DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiOpenCl);
        store16_binary(&file, 4, 1, &pixels, provenance);
        let z6iii_route = test_route(true, true);
        let read = read_cached16_binary(&file, &z6iii_route).expect("consistent route");
        assert_eq!(read.3, provenance);
        assert!(read_cached_source16(&file, &z6iii_route).is_some());

        let not_z6iii = test_route(true, false);
        assert!(
            read_cached16_binary(&file, &not_z6iii).is_none(),
            "a Z6III pre-matrix entry must not serve a non-Z6III route"
        );
        assert!(read_cached_source16(&file, &not_z6iii).is_none());

        let c1_off = test_route(false, true);
        assert!(
            read_cached16_binary(&file, &c1_off).is_none(),
            "camera-linear 16-bit must not serve a LibRaw-only route"
        );

        let mut bytes = fs::read(&file).unwrap();
        bytes[32..36].copy_from_slice(&(provenance.revision + 1).to_le_bytes());
        fs::write(&file, bytes).unwrap();
        assert!(read_cached16_binary(&file, &z6iii_route).is_none());
        assert!(read_cached_source16(&file, &z6iii_route).is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn raw16_header_keeps_the_opencl_failure_state() {
        let dir = scratch("opencl-state");
        let file = dir.join("raw-opencl.bin");
        let pixels = vec![9u16; 2 * 4];
        let provenance =
            DecodeProvenance::aurora(crate::aurora_raw::AuroraRoute::Z6iiiCpuOpenClFailed);
        store16_binary(&file, 2, 1, &pixels, provenance);
        let route = test_route(true, true);
        let read = read_cached16_binary(&file, &route).expect("read");
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
        store_libraw(&file, width, height, &pixels);
        let route = test_route(true, false);
        let mapped = read_cached_source16(&file, &route).expect("mapped source").0;
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
    fn raw16_cache_invalidates_changed_files_and_rebuilds_corruption() {
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

        let started = Instant::now();
        store_libraw(&bin, width, height, &pixels);
        let bin_write = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let bin_route = test_route(true, false);
        let bin_read_data = read_cached16_binary(&bin, &bin_route).expect("binary read");
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
        enforce_limits(&dir, 2, u64::MAX);
        assert_eq!(cache_files(&dir).len(), 2);
        enforce_limits(&dir, 10, 15);
        assert_eq!(cache_files(&dir).len(), 1);
    }

    #[test]
    fn proxy_decode_feeds_the_first_full_size_request_without_decoding() {
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
}
