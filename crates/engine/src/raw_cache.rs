//! On-disk cache of decoded RAW previews.
//!
//! libraw takes seconds on a full sensor frame, and reopening a file used to pay
//! that again. This keeps the downscaled decode as a lossless PNG, keyed by the
//! source path, size, mtime and requested long side.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use image::{ImageFormat, RgbaImage as ImageBuffer};

use crate::RgbaImage;

/// Only files with this prefix are ever evicted; the caller shares the folder.
const PREFIX: &str = "raw-";
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
            name.starts_with(PREFIX) && name.ends_with(".png")
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
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/raw-cache-tests");
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

    /// Manual measurement against a real file: set `LUMEN_RAW_CACHE_BENCH`.
    #[test]
    #[ignore = "manual timing, needs LUMEN_RAW_CACHE_BENCH"]
    fn bench_real_raw() {
        let Ok(path) = std::env::var("LUMEN_RAW_CACHE_BENCH") else { return };
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
