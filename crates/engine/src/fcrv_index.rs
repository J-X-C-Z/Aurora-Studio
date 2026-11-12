//! Index every shipped Capture One film curve (`.fcrv`) for the frontend.
//!
//! `name` is the file stem because that is what `ZFILMCURVE` stores; the name
//! embedded in the file is descriptive only ("Linear response" in every file
//! checked). `points` are the primary curve's control points, quantised to four
//! decimals when written to JSON.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::filmcurve::parse_film_curves;

#[derive(Debug, Clone, PartialEq)]
pub struct CurveRow {
    pub name: String,
    pub file: String,
    pub points: Vec<(f32, f32)>,
    pub monotonic: bool,
}

#[derive(Debug, Default)]
pub struct CurveIndex {
    pub total: usize,
    pub parsed: usize,
    /// Curves whose x or y decreases; a curve counts once.
    pub non_monotonic: usize,
    /// Files that failed to read or parse, capped at ten.
    pub failures: Vec<String>,
    /// Number of failed files, before the cap.
    pub failed: usize,
    /// `(point count, file count)`, ascending by point count.
    pub point_counts: Vec<(usize, usize)>,
    pub rows: Vec<CurveRow>,
}

/// True when x never decreases and y never decreases.
pub fn is_monotonic(points: &[(f32, f32)]) -> bool {
    points.windows(2).all(|w| w[1].0 >= w[0].0 && w[1].1 >= w[0].1)
}

/// Parses every `.fcrv` in `dir`, sorted by file name. A bad file is recorded
/// and skipped, never fatal.
pub fn scan(dir: &Path) -> CurveIndex {
    let mut index = CurveIndex::default();
    let mut entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.flatten().collect(),
        Err(_) => return index,
    };
    entries.retain(|e| e.path().extension().and_then(|x| x.to_str()) == Some("fcrv"));
    entries.sort_by_key(|e| e.file_name().to_string_lossy().to_lowercase());

    let mut counts = BTreeMap::new();
    for entry in entries {
        index.total += 1;
        let file = entry.file_name().to_string_lossy().into_owned();
        let stem = match Path::new(&file).file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_owned(),
            None => {
                index.failed += 1;
                if index.failures.len() < 10 {
                    index.failures.push(format!("{file}: no UTF-8 stem"));
                }
                continue;
            }
        };
        let parsed = std::fs::read(entry.path())
            .map_err(|e| e.to_string())
            .and_then(|b| parse_film_curves(&b).map_err(|e| e.to_string()));
        let mut curves = match parsed {
            Ok(c) => c,
            Err(e) => {
                index.failed += 1;
                if index.failures.len() < 10 {
                    index.failures.push(format!("{file}: {e}"));
                }
                continue;
            }
        };
        let Some(primary) = curves.drain(..).next() else {
            index.failed += 1;
            if index.failures.len() < 10 {
                index.failures.push(format!("{file}: no curve"));
            }
            continue;
        };
        let monotonic = is_monotonic(&primary.points);
        if !monotonic {
            index.non_monotonic += 1;
        }
        *counts.entry(primary.points.len()).or_insert(0usize) += 1;
        index.rows.push(CurveRow {
            name: stem,
            file,
            points: primary.points,
            monotonic,
        });
        index.parsed += 1;
    }
    index.point_counts = counts.into_iter().collect();
    // Sort the exported list by stem, not file name: `ACROS.fcrv` must precede
    // `ACROS +Ye FILTER.fcrv`, which a full-name sort gets wrong at the dot.
    index.rows.sort_by_cached_key(|r| r.name.to_lowercase());
    index
}

/// Compact JSON array of `{name, file, points, monotonic}`, points rounded to
/// four decimals. Callers hash the resulting bytes.
pub fn contract_json(index: &CurveIndex) -> String {
    let mut out = String::with_capacity(index.rows.len() * 512);
    out.push('[');
    for (i, row) in index.rows.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let name = serde_json::to_string(&row.name).unwrap_or_else(|_| "\"\"".to_owned());
        let file = serde_json::to_string(&row.file).unwrap_or_else(|_| "\"\"".to_owned());
        let _ = write!(out, "{{\"name\":{name},\"file\":{file},\"points\":[");
        for (j, (x, y)) in row.points.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push('[');
            push_num(&mut out, *x);
            out.push(',');
            push_num(&mut out, *y);
            out.push(']');
        }
        let _ = write!(out, "],\"monotonic\":{}}}", row.monotonic);
    }
    out.push(']');
    out
}

/// Shortest decimal that round-trips the four-decimal value.
fn push_num(out: &mut String, v: f32) {
    let q = (v * 10_000.0).round() / 10_000.0;
    let mut s = format!("{q:.4}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    if s.is_empty() || s == "-0" {
        s.push('0');
    }
    out.push_str(&s);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const INSTALLED: &str = r"C:\Program Files\Capture One\Capture One\Film Curves";

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-fcrv-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(feature = "external-fixtures")]
    #[test]
    fn installed_directory_enumerates_every_curve() {
        let dir = PathBuf::from(INSTALLED);
        if !dir.exists() {
            return;
        }
        let index = scan(&dir);
        assert!(index.total > 4000, "only {} files", index.total);
        assert_eq!(index.parsed + index.failed, index.total);
        assert_eq!(index.rows.len(), index.parsed);
        assert!(index.rows.iter().any(|r| r.name == "NikonZ6III-Auto"));
    }


    #[cfg(feature = "external-fixtures")]
    #[test]
    fn nikon_z_series_maps_to_auto_curve() {
        let _serial = crate::filmcurve::test_lock();
        if !Path::new(INSTALLED).exists() {
            return;
        }
        assert_eq!(
            crate::camera_curve::default_curve("NIKON Z6_3").as_deref(),
            Some("NikonZ6III-Auto")
        );
        assert_eq!(
            crate::camera_curve::default_curve("NIKON Z 7_2").as_deref(),
            Some("NikonZ7II-Auto")
        );
    }


    #[test]
    fn json_quantises_to_four_decimals() {
        let index = CurveIndex {
            rows: vec![CurveRow {
                name: "C".into(),
                file: "C.fcrv".into(),
                points: vec![(0.123456, 0.0), (1.0, 0.99999)],
                monotonic: true,
            }],
            ..CurveIndex::default()
        };
        let json = contract_json(&index);
        assert!(json.contains("\"points\":[[0.1235,0],[1,1]]"), "{json}");
        assert!(json.contains("\"monotonic\":true"));
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/fcrv_index-tests.rs");
}
