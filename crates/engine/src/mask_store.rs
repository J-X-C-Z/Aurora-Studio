//! AI masks that survive a restart, as 8-bit grayscale PGM under
//! `<catalog>/.lumen/masks/<hash>.pgm`.
//!
//! The key is built from the image's path relative to the catalog, its variant
//! number and the mask kind, never from `ImageId` (that changes every launch).
//! Every write goes to a `.tmp` file and is renamed into place. Unreadable,
//! truncated or missing files load as `None`; nothing here panics on bad input.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::aimask::MaskPlane;
use crate::AiMaskKind;

const MASK_DIR: &str = ".lumen/masks";
/// PGM is header + one byte per pixel; larger files are not a mask this engine
/// wrote, and rejecting them keeps a corrupt header from allocating gigabytes.
const MAX_PIXELS: usize = 16 * 1024 * 1024;

/// Stable cache key: relative path + variant number + kind.
pub fn key(rel_path: &str, variant: u32, kind: AiMaskKind) -> String {
    format!("mask-v1:{}#v{variant}:{kind:?}", rel_path.replace('\\', "/"))
}

fn mask_path(root: &Path, key: &str) -> PathBuf {
    root.join(MASK_DIR).join(format!("{}.pgm", fnv1a_hex(key)))
}

/// Writes `plane` as a single-channel PGM, atomically.
pub fn save(root: &Path, key: &str, plane: &MaskPlane) -> Result<()> {
    if plane.w == 0 || plane.h == 0 || plane.w.checked_mul(plane.h) != Some(plane.data.len()) {
        bail!("mask {key} does not match its {}x{} plane", plane.w, plane.h);
    }
    let dir = root.join(MASK_DIR);
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = mask_path(root, key);
    let tmp = path.with_extension("pgm.tmp");

    let mut pixels = Vec::with_capacity(plane.data.len());
    pixels.extend(plane.data.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));

    let mut file = fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    write!(file, "P5\n{} {}\n255\n", plane.w, plane.h)?;
    file.write_all(&pixels)?;
    file.sync_all()?;
    drop(file);

    // Windows rename does not replace an existing file.
    if fs::rename(&tmp, &path).is_err() {
        fs::remove_file(&path).ok();
        fs::rename(&tmp, &path).with_context(|| format!("publishing {}", path.display()))?;
    }
    Ok(())
}

/// Reads a saved mask; any I/O or format problem is `None`.
pub fn load(root: &Path, key: &str) -> Option<Arc<MaskPlane>> {
    let bytes = fs::read(mask_path(root, key)).ok()?;
    let (w, h, data) = parse_pgm(&bytes)?;
    Some(Arc::new(MaskPlane::new(w, h, data)))
}

/// Deletes a saved mask, if there is one.
pub fn forget(root: &Path, key: &str) {
    let _ = fs::remove_file(mask_path(root, key));
}

fn parse_pgm(bytes: &[u8]) -> Option<(usize, usize, Vec<f32>)> {
    if !bytes.starts_with(b"P5") {
        return None;
    }
    let mut at = 2;
    let mut token = || -> Option<usize> {
        while at < bytes.len() && bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        let start = at;
        while at < bytes.len() && bytes[at].is_ascii_digit() {
            at += 1;
        }
        (start != at).then(|| std::str::from_utf8(&bytes[start..at]).ok()?.parse().ok())?
    };
    let w = token()?;
    let h = token()?;
    if token()? != 255 {
        return None;
    }
    // Exactly one separator after maxval; pixel data may itself start with a
    // whitespace byte, so only one is skipped.
    if at < bytes.len() && bytes[at].is_ascii_whitespace() {
        at += 1;
    }
    let pixels = w.checked_mul(h)?;
    if w == 0 || h == 0 || pixels > MAX_PIXELS || bytes.len() - at != pixels {
        return None;
    }
    let data = bytes[at..].iter().map(|b| *b as f32 / 255.0).collect();
    Some((w, h, data))
}

/// The same FNV-1a 64 used by `store` for adjustment filenames.
fn fnv1a_hex(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-maskstore-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn plane(w: usize, h: usize, data: Vec<f32>) -> MaskPlane {
        MaskPlane::new(w, h, data)
    }

    fn gradient(w: usize, h: usize) -> MaskPlane {
        let data = (0..w * h).map(|i| (i % 251) as f32 / 250.0).collect();
        plane(w, h, data)
    }

    #[test]
    fn roundtrip_keeps_the_mask() {
        let dir = temp_dir("roundtrip");
        let key = key("sub/shot.NEF", 1, AiMaskKind::Subject);
        let saved = gradient(37, 21);
        save(&dir, &key, &saved).unwrap();
        let loaded = load(&dir, &key).expect("saved mask loads");
        assert_eq!((loaded.w, loaded.h), (37, 21));
        for (a, b) in saved.data.iter().zip(&loaded.data) {
            assert!((a - b).abs() <= 1.0 / 255.0, "8-bit roundtrip drift: {a} vs {b}");
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupt_file_loads_as_none() {
        let dir = temp_dir("corrupt");
        let key = key("shot.NEF", 1, AiMaskKind::Sky);
        fs::create_dir_all(dir.join(MASK_DIR)).unwrap();
        let path = mask_path(&dir, &key);
        fs::write(&path, b"P5\n10 10\n255\nnot enough pixels").unwrap();
        assert!(load(&dir, &key).is_none());
        fs::write(&path, b"this is not a pgm").unwrap();
        assert!(load(&dir, &key).is_none());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn key_changes_with_variant_and_kind() {
        let base = key("sub/shot.NEF", 1, AiMaskKind::Subject);
        assert_eq!(base, key("sub\\shot.NEF", 1, AiMaskKind::Subject));
        assert_ne!(base, key("sub/shot.NEF", 2, AiMaskKind::Subject));
        assert_ne!(base, key("sub/shot.NEF", 1, AiMaskKind::Skin));

        let dir = temp_dir("key-change");
        save(&dir, &base, &gradient(8, 4)).unwrap();
        assert!(load(&dir, &key("sub/shot.NEF", 1, AiMaskKind::Skin)).is_none());
        assert!(load(&dir, &key("other.NEF", 1, AiMaskKind::Subject)).is_none());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn forget_removes_the_saved_mask() {
        let dir = temp_dir("forget");
        let key = key("shot.NEF", 1, AiMaskKind::Hair);
        save(&dir, &key, &gradient(5, 5)).unwrap();
        assert!(load(&dir, &key).is_some());
        forget(&dir, &key);
        assert!(load(&dir, &key).is_none());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unwritable_mask_directory_does_not_panic() {
        let dir = temp_dir("unwritable");
        // A regular file where the mask directory should be makes every write
        // fail on all platforms, without relying on read-only permissions.
        fs::create_dir_all(dir.join(".lumen")).unwrap();
        fs::write(dir.join(MASK_DIR), b"blocker").unwrap();
        let key = key("shot.NEF", 1, AiMaskKind::Subject);
        assert!(save(&dir, &key, &gradient(4, 4)).is_err());
        assert!(load(&dir, &key).is_none());
        forget(&dir, &key); // must not panic either
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mismatched_plane_is_rejected() {
        let dir = temp_dir("mismatch");
        let key = key("shot.NEF", 1, AiMaskKind::Subject);
        assert!(save(&dir, &key, &plane(3, 3, vec![0.5; 8])).is_err());
        assert!(save(&dir, &key, &plane(0, 0, Vec::new())).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Two independent `cargo test` runs: `LUMEN_MASK_BENCH=warm` builds a
    /// synthetic plane and saves it; `cold` only loads it.
    #[test]
    #[ignore = "manual two-process timing, needs LUMEN_MASK_BENCH"]
    fn bench_persist_versus_load() {
        let dir = std::env::temp_dir().join("lumen-mask-bench");
        fs::create_dir_all(&dir).unwrap();
        let key = key("bench/synthetic.NEF", 1, AiMaskKind::Subject);
        let phase = std::env::var("LUMEN_MASK_BENCH").unwrap_or_else(|_| "warm".into());
        let start = std::time::Instant::now();
        if phase == "cold" {
            let plane = load(&dir, &key).expect("warm run saved the mask");
            println!(
                "cold: load {}x{} in {:.2} ms",
                plane.w,
                plane.h,
                start.elapsed().as_secs_f64() * 1000.0
            );
        } else {
            // Stand-in for a model run: a soft radial field over 1024x683.
            let (w, h) = (1024usize, 683usize);
            let data = (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                    let d = ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt();
                    (1.0 - d * 2.4).clamp(0.0, 1.0)
                })
                .collect();
            let plane = plane(w, h, data);
            let infer = start.elapsed().as_secs_f64() * 1000.0;
            let save_start = std::time::Instant::now();
            save(&dir, &key, &plane).unwrap();
            let save_ms = save_start.elapsed().as_secs_f64() * 1000.0;
            let reload_start = std::time::Instant::now();
            assert!(load(&dir, &key).is_some());
            let reload_ms = reload_start.elapsed().as_secs_f64() * 1000.0;
            println!(
                "warm: synthetic mask {infer:.2} ms, save {save_ms:.2} ms, reload {reload_ms:.2} ms"
            );
        }
    }
}
