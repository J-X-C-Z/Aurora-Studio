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
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use image::{ImageFormat, RgbaImage as ImageBuffer};

use crate::mock::process::Source;
use crate::RgbaImage;

/// Only files with this prefix are ever evicted; the caller shares the folder.
const PREFIX: &str = "raw-";
// Bump when decoded pixels change, even if the persisted layout is unchanged.
// B5 may contain fixed chroma filtering baked into Z6III camera RGB.
const RAW16_MAGIC: &[u8; 8] = b"AUR16B6\0";
const MAX_ENTRIES: usize = 64;
const MAX_BYTES: u64 = 1 << 30;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

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
    let Some(dir) = default_raw16_dir() else {
        return crate::raw::decode16(path).map(|(w, h, px)| fit16(w, h, px, long_side));
    };
    get_or_decode16_with(path, &dir, long_side, |path| crate::raw::decode16(path))
}

pub(crate) fn get_or_decode_source16(path: &Path, long_side: u32) -> Option<Source> {
    let Some(dir) = default_raw16_dir() else {
        return decode_source16(path, long_side);
    };
    let key = cache_key16(path, &dir, long_side);
    if let Some(source) = key.as_deref().and_then(read_cached_source16) {
        return Some(source);
    }
    let (w, h, pixels) = crate::raw::decode16(path)?;
    let image = fit16(w, h, pixels, long_side);
    if let Some(file) = key {
        store16_binary(&file, image.0, image.1, &image.2);
    }
    Some(Source::from_rgba16(image.0, image.1, &image.2))
}

fn decode_source16(path: &Path, long_side: u32) -> Option<Source> {
    let (w, h, pixels) = crate::raw::decode16(path)?;
    let image = fit16(w, h, pixels, long_side);
    Some(Source::from_rgba16(image.0, image.1, &image.2))
}

fn get_or_decode16_with<F>(
    path: &Path,
    dir: &Path,
    long_side: u32,
    decode: F,
) -> Option<(u32, u32, Vec<u16>)>
where
    F: FnOnce(&Path) -> Option<(u32, u32, Vec<u16>)>,
{
    let key = cache_key16(path, dir, long_side);
    if let Some(file) = key.as_deref().and_then(read_cached16_binary) {
        return Some(file);
    }
    let (w, h, pixels) = decode(path)?;
    let image = fit16(w, h, pixels, long_side);
    if let Some(file) = key {
        store16_binary(&file, image.0, image.1, &image.2);
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

fn cache_key16(path: &Path, dir: &Path, long_side: u32) -> Option<PathBuf> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let demosaic = crate::raw::demosaic_quality();
    let mut a = DefaultHasher::new();
    let mut b = DefaultHasher::new();
    let source = crate::raw::c1_raw_enabled();
    (RAW16_MAGIC, path, meta.len(), mtime, long_side, demosaic, source).hash(&mut a);
    (0xa7u8, path, mtime, demosaic, long_side, meta.len(), source).hash(&mut b);
    Some(dir.join(format!(
        "{PREFIX}16b-{:016x}{:016x}-{long_side}.bin",
        a.finish(),
        b.finish()
    )))
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
    let longest = width.max(height);
    if long_side == 0 || longest <= long_side {
        return (width, height, pixels);
    }
    let scale = long_side as f32 / longest as f32;
    let short = ((if width <= height { width } else { height }) as f32 * scale)
        .round()
        .max(1.0) as u32;
    let (target_width, target_height) =
        if width >= height { (long_side, short) } else { (short, long_side) };
    (
        target_width,
        target_height,
        crate::recipe::resize_rgba16(
            &pixels,
            width,
            height,
            target_width,
            target_height,
        ),
    )
}

fn read_cached16_binary(file: &Path) -> Option<(u32, u32, Vec<u16>)> {
    use rayon::prelude::*;

    const HEADER: usize = 32;
    let mut reader = fs::File::open(file).ok()?;
    let size = reader.metadata().ok()?.len();
    let mut header = [0u8; HEADER];
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
        || size != (HEADER + bytes) as u64
    {
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
    Some((width, height, pixels))
}

#[cfg(target_endian = "little")]
fn read_cached_source16(file: &Path) -> Option<Source> {
    use memmap2::MmapOptions;

    const HEADER: usize = 32;
    let file = fs::File::open(file).ok()?;
    // SAFETY: the cache file is read-only for the lifetime of the mapping and
    // all mutations replace it atomically through rename.
    let map = unsafe { MmapOptions::new().map(&file).ok()? };
    if map.len() < HEADER || &map[..8] != RAW16_MAGIC {
        return None;
    }
    let width = u32::from_le_bytes(map[8..12].try_into().ok()?);
    let height = u32::from_le_bytes(map[12..16].try_into().ok()?);
    let bytes = u64::from_le_bytes(map[16..24].try_into().ok()?) as usize;
    let expected_bytes = width as usize * height as usize * 6;
    if width == 0 || height == 0 || bytes != expected_bytes || map.len() != HEADER + bytes {
        return None;
    }
    let stored_hash = u64::from_le_bytes(map[24..32].try_into().ok()?);
    let body = &map[HEADER..];
    // SAFETY: the mapping starts page-aligned and HEADER is 32 bytes, so the
    // body is u16-aligned. `align_to` checks and reports any residual bytes.
    let (before, pixels, after) = unsafe { body.align_to::<u16>() };
    if !before.is_empty() || !after.is_empty() || pixels.len() != expected_bytes / 2 {
        return None;
    }
    let (valid, source) = rayon::join(
        || image_hash(width, height, body) == stored_hash,
        || Source::from_rgb16(width, height, pixels),
    );
    valid.then_some(source)
}

#[cfg(not(target_endian = "little"))]
fn read_cached_source16(file: &Path) -> Option<Source> {
    let (width, height, pixels) = read_cached16_binary(file)?;
    Some(Source::from_rgba16(width, height, &pixels))
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

fn store16_binary(file: &Path, width: u32, height: u32, pixels: &[u16]) {
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
    let mut data = Vec::with_capacity(32 + body.len());
    data.extend_from_slice(RAW16_MAGIC);
    data.extend_from_slice(&width.to_le_bytes());
    data.extend_from_slice(&height.to_le_bytes());
    data.extend_from_slice(&(body.len() as u64).to_le_bytes());
    data.extend_from_slice(&image_hash(width, height, &body).to_le_bytes());
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

    #[test]
    fn raw16_obsolete_cache_is_rejected_by_both_readers() {
        let dir = scratch("legacy-raw16");
        let file = dir.join("raw-test.bin");
        store16_binary(&file, 2, 1, &[100, 200, 300, 65535, 400, 500, 600, 65535]);
        assert!(read_cached16_binary(&file).is_some());
        assert!(read_cached_source16(&file).is_some());
        let mut bytes = fs::read(&file).unwrap();
        bytes[..8].copy_from_slice(b"AUR16B5\0");
        fs::write(&file, bytes).unwrap();
        assert!(read_cached16_binary(&file).is_none());
        assert!(read_cached_source16(&file).is_none());
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
        let calls = Cell::new(0);
        let first = get_or_decode16_with(&path, &dir, 80, |_| {
            calls.set(calls.get() + 1);
            Some((80, 40, pixels.clone()))
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(first, expected);
        assert_eq!(cache_files(&dir).len(), 1);

        let cached = get_or_decode16_with(&path, &dir, 80, |_| {
            calls.set(calls.get() + 1);
            None
        })
        .unwrap();
        assert_eq!(calls.get(), 1, "the second call must hit the 16-bit cache");
        assert_eq!(cached, first);
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
        store16_binary(&file, width, height, &pixels);
        let mapped = read_cached_source16(&file).expect("mapped source");
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
        let first = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((20, 10, vec![1000; 20 * 10 * 4]))
        })
        .unwrap();
        assert_eq!(first.2[0], 1000);

        fs::write(&path, b"changed-size").unwrap();
        let second = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((20, 10, vec![50000; 20 * 10 * 4]))
        })
        .unwrap();
        assert_eq!(second.2[0], 50000);

        let file = cache_files(&dir)
            .into_iter()
            .max_by_key(|p| fs::metadata(p).unwrap().modified().unwrap())
            .unwrap();
        fs::write(&file, b"broken").unwrap();
        let rebuilt = get_or_decode16_with(&path, &dir, 20, |_| {
            Some((20, 10, vec![12345; 20 * 10 * 4]))
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
            crate::raw::decode16(path)
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
        store16_binary(&bin, width, height, &pixels);
        let bin_write = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let bin_read_data = read_cached16_binary(&bin).expect("binary read");
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
