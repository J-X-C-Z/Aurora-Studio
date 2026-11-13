//! Capture One film curves (`.fcrv`), the response curve behind `ZFILMCURVE`.
//!
//! Layout, verified against all 4166 shipped files:
//! `MM`/`II` byte-order mark, `u32` version, `u32` point count, then `count`
//! big/little-endian `(u32 x, u32 y)` pairs where `0xFFFF_FFFF` is 1.0, then a
//! `u32` tag, then two length-prefixed UTF-8 strings (description, name).
//! After the strings: `u32 a`, `f32 scale`, `u32 b`, `u32 c`, then
//! `u32 version, u32 n, points`, then `u32 n, points` (some files end with an
//! extra `f32` gamma). Those extra tables are returned as extra curves because
//! their role is not established.
//!
//! The embedded name is descriptive only; Capture One selects a curve by file
//! name (`ZFILMCURVE` stores it), so callers must key on the path, not the name.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{bail, Result};

const FULL: f32 = u32::MAX as f32;

#[derive(Debug, Clone, PartialEq)]
pub struct FilmCurve {
    pub name: String,
    pub points: Vec<(f32, f32)>,
}

impl FilmCurve {
    /// Piecewise-linear lookup, clamped at both ends.
    pub fn sample(&self, x: f32) -> f32 {
        let Some(&(x0, y0)) = self.points.first() else {
            return x;
        };
        if x <= x0 {
            return y0;
        }
        for w in self.points.windows(2) {
            let (ax, ay) = w[0];
            let (bx, by) = w[1];
            if x <= bx {
                return if bx > ax {
                    ay + (by - ay) * (x - ax) / (bx - ax)
                } else {
                    by
                };
            }
        }
        self.points.last().map(|p| p.1).unwrap_or(x)
    }
}

struct Cursor<'a> {
    d: &'a [u8],
    o: usize,
    be: bool,
}

impl<'a> Cursor<'a> {
    fn u32(&mut self) -> Result<u32> {
        let b = self.d.get(self.o..self.o + 4).ok_or_else(|| anyhow::anyhow!("truncated u32"))?;
        self.o += 4;
        let a = [b[0], b[1], b[2], b[3]];
        Ok(if self.be {
            u32::from_be_bytes(a)
        } else {
            u32::from_le_bytes(a)
        })
    }

    fn points(&mut self) -> Result<Vec<(u32, u32)>> {
        let n = self.u32()? as usize;
        if n == 0 || n > 4096 {
            bail!("point count {n} out of range");
        }
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push((self.u32()?, self.u32()?));
        }
        Ok(pts)
    }
}

/// Curves in one `.fcrv`. Entry 0 is the film curve; later entries are the
/// unframed auxiliary tables, named `<name> #n` so they are never mistaken for
/// separate profiles.
pub fn parse_film_curves(bytes: &[u8]) -> Result<Vec<FilmCurve>> {
    let be = match bytes.get(..2) {
        Some(b"MM") => true,
        Some(b"II") => false,
        _ => bail!("not a film curve: missing MM/II marker"),
    };
    let mut c = Cursor { d: bytes, o: 2, be };
    let _version = c.u32()?;
    let primary = c.points()?;
    let _tag = c.u32()?;
    let mut strings = Vec::with_capacity(2);
    for _ in 0..2 {
        let n = c.u32()? as usize;
        let raw = bytes.get(c.o..c.o + n).ok_or_else(|| anyhow::anyhow!("truncated string"))?;
        c.o += n;
        // Old (v2/v4) files count the NUL terminator in the length.
        strings.push(String::from_utf8_lossy(raw).trim_end_matches('\0').to_owned());
    }
    let name = strings.swap_remove(1);

    let mut curves = vec![FilmCurve {
        name: name.clone(),
        points: to_unit(&primary),
    }];

    // Trailer; absent in a few short files, so every step is best-effort.
    if c.o + 20 <= bytes.len() {
        c.o += 16; // a, f32 scale, b, c
        if c.u32().is_ok() {
            let extra = c.points();
            let third = c.points();
            for pts in [extra, third].into_iter().flatten() {
                if pts.len() > 1 {
                    curves.push(FilmCurve {
                        name: format!("{name} #{}", curves.len()),
                        points: to_unit(&pts),
                    });
                }
            }
        }
    }
    Ok(curves)
}

fn to_unit(pts: &[(u32, u32)]) -> Vec<(f32, f32)> {
    pts.iter().map(|&(x, y)| (x as f32 / FULL, y as f32 / FULL)).collect()
}

static DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Directory holding Capture One's `.fcrv` files. `use_dir` wins, then
/// `LUMEN_FILM_CURVES`, then this machine's install.
pub fn curve_dir() -> PathBuf {
    if let Some(dir) = DIR.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return dir.clone();
    }
    std::env::var_os("LUMEN_FILM_CURVES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files\Capture One\Capture One\Film Curves"))
}


/// Serialises the tests that repoint the curve directory, since the index is global.
#[cfg(test)]
pub static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Points the curve index at `dir`; replaces any earlier choice.
#[cfg(test)]
pub fn use_dir(dir: impl Into<PathBuf>) {
    *DIR.lock().unwrap_or_else(|e| e.into_inner()) = Some(dir.into());
}

/// File-stem -> primary curve, built once. `ZFILMCURVE` stores the file name,
/// so the key is the stem (`Foo.fcrv` -> `Foo`).
static INDEX: Mutex<Option<(PathBuf, HashMap<String, FilmCurve>)>> = Mutex::new(None);

/// Runs `f` over the index for the current directory, loading it once. Keyed on
/// the path so a later `use_dir` still takes effect.
fn with_curves<R>(f: impl FnOnce(&HashMap<String, FilmCurve>) -> R) -> R {
    let dir = curve_dir();
    let mut guard = INDEX.lock().unwrap_or_else(|e| e.into_inner());
    if guard.as_ref().is_none_or(|(d, _)| *d != dir) {
        *guard = Some((dir.clone(), load_index(&dir).unwrap_or_default()));
    }
    f(&guard.as_ref().expect("index just loaded").1)
}

fn load_index(dir: &Path) -> Result<HashMap<String, FilmCurve>> {
    SCANS.fetch_add(1, Ordering::Relaxed);
    let mut out = HashMap::new();
    for entry in std::fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("fcrv") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if let Ok(bytes) = std::fs::read(&path) {
            if let Ok(mut parsed) = parse_film_curves(&bytes) {
                if let Some(primary) = parsed.drain(..).next() {
                    out.insert(stem.to_owned(), primary);
                }
            }
        }
    }
    Ok(out)
}

/// Number of directory scans so far; a render must not add to it.
#[cfg(test)]
pub fn scan_count() -> usize {
    SCANS.load(Ordering::Relaxed)
}

static SCANS: AtomicUsize = AtomicUsize::new(0);

/// 256-entry table of the named curve over 0..1, or `None` when unknown.
pub fn curve_lut(name: &str) -> Option<[f32; 256]> {
    with_curves(|index| {
        let c = index.get(name)?;
        let mut lut = [0.0f32; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = c.sample(i as f32 / 255.0);
        }
        Some(lut)
    })
}

/// Every curve stem in the installed index, sorted by name.
pub fn names() -> Vec<String> {
    with_curves(|index| {
        let mut names: Vec<String> = index.keys().cloned().collect();
        names.sort();
        names
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn be(version: u32, pts: &[(u32, u32)], tag: u32, desc: &str, name: &str) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"MM");
        v.extend_from_slice(&version.to_be_bytes());
        v.extend_from_slice(&(pts.len() as u32).to_be_bytes());
        for (x, y) in pts {
            v.extend_from_slice(&x.to_be_bytes());
            v.extend_from_slice(&y.to_be_bytes());
        }
        v.extend_from_slice(&tag.to_be_bytes());
        for s in [desc, name] {
            v.extend_from_slice(&(s.len() as u32).to_be_bytes());
            v.extend_from_slice(s.as_bytes());
        }
        v
    }

    #[test]
    fn parses_identity_curve() {
        let bytes = be(6, &[(0, 0), (u32::MAX, u32::MAX)], 1, "Linear response", "Linear response");
        let c = parse_film_curves(&bytes).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].name, "Linear response");
        assert_eq!(c[0].points, vec![(0.0, 0.0), (1.0, 1.0)]);
        assert!((c[0].sample(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn reads_little_endian_marker() {
        let mut v = Vec::new();
        v.extend_from_slice(b"II");
        for word in [6u32, 2, 0, 0, u32::MAX, u32::MAX, 1] {
            v.extend_from_slice(&word.to_le_bytes());
        }
        for s in ["d", "Low contrast"] {
            v.extend_from_slice(&(s.len() as u32).to_le_bytes());
            v.extend_from_slice(s.as_bytes());
        }
        let c = parse_film_curves(&v).unwrap();
        assert_eq!(c[0].name, "Low contrast");
        assert_eq!(c[0].points, vec![(0.0, 0.0), (1.0, 1.0)]);
    }

    #[test]
    fn rejects_foreign_bytes() {
        assert!(parse_film_curves(b"PK\x03\x04").is_err());
        assert!(parse_film_curves(b"MM").is_err());
    }

    #[test]
    fn truncated_points_do_not_panic() {
        let bytes = be(6, &[(0, 0), (u32::MAX, u32::MAX)], 1, "d", "n");
        assert!(parse_film_curves(&bytes[..20]).is_err());
    }

    #[test]
    fn sample_clamps_outside_range() {
        let c = FilmCurve { name: "c".into(), points: vec![(0.25, 0.1), (0.75, 0.9)] };
        assert_eq!(c.sample(0.0), 0.1);
        assert_eq!(c.sample(1.0), 0.9);
        assert!((c.sample(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empty_curve_samples_identity() {
        let c = FilmCurve { name: "e".into(), points: vec![] };
        assert_eq!(c.sample(0.3), 0.3);
    }

    #[test]
    fn unknown_name_has_no_lut() {
        assert!(curve_lut("definitely not a curve").is_none());
    }

    /// Needs the Capture One install, so it is skipped by default.
    #[test]
    #[ignore]
    fn parses_the_installed_curve_set() {
        let dir = r"C:\Program Files\Capture One\Capture One\Film Curves";
        let mut n = 0;
        for e in std::fs::read_dir(dir).expect("Capture One film curves").flatten() {
            if e.path().extension().and_then(|x| x.to_str()) != Some("fcrv") {
                continue;
            }
            let c = parse_film_curves(&std::fs::read(e.path()).unwrap()).unwrap();
            assert!(!c[0].name.is_empty());
            assert!(c[0].points.len() >= 2);
            n += 1;
        }
        assert!(n > 4000, "only {n} curves");
    }
}
