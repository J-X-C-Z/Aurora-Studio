//! Writes our adjustments back into a Capture One catalog's `ZVARIANTLAYER`.
//!
//! Only the full-snapshot row (`Z_ENT = 18`) is edited. `17` is the delta on
//! top of it, `20` the composite and `23` per-mask rows; Capture One recomputes
//! those, so touching them would fight the app.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::AdjustmentSet;

/// `ZVARIANTLAYER.Z_ENT` of the full per-variant snapshot.
pub const FULL_SNAPSHOT_ENT: i64 = 18;

#[derive(Clone, Debug, PartialEq)]
pub enum WriteOutcome {
    /// Every supported column already held the value. `warnings` still names
    /// what was refused (for example a clamped crop).
    Unchanged { warnings: Vec<String> },
    Written { columns: usize, warnings: Vec<String> },
}

/// What [`crate::FileEngine::save_to_c1_catalog`] did.
#[derive(Clone, Debug, PartialEq)]
pub enum CatalogSaveOutcome {
    /// The engine was not opened on a Capture One catalog.
    NotACatalog,
    Unchanged { warnings: Vec<String> },
    Saved { columns: usize, backup: PathBuf },
}

fn num(value: f32) -> String {
    if value == 0.0 {
        return "0".to_owned();
    }
    let text = format!("{value}");
    text
}

fn vec3(value: [f32; 3]) -> String {
    value.iter().map(|v| num(*v)).collect::<Vec<_>>().join(";")
}

fn vec4(value: [f32; 4]) -> String {
    value.iter().map(|v| num(*v)).collect::<Vec<_>>().join(";")
}

fn curve(points: &[crate::CurvePoint]) -> String {
    points
        .iter()
        .map(|point| format!("{},{}", num(point.x), num(point.y)))
        .collect::<Vec<_>>()
        .join(";")
}

/// Inverse of [`crate::c1_model::crop_on_canvas_with_clipping`] for the
/// unclamped case (`crop_outside_image = 1`). Clamped crops need the same
/// `crop_limit` mode as the file, which this build cannot guarantee, so they
/// are refused rather than approximated.
pub fn crop_px(
    rect: crate::NormRect,
    geometry: &crate::Geometry,
    source: Option<(u32, u32)>,
) -> Option<[f32; 4]> {
    if !geometry.crop_outside_image {
        return None;
    }
    let [canvas_width, canvas_height] = geometry.canvas;
    if canvas_width <= 0.0 || canvas_height <= 0.0 {
        return None;
    }
    let turn = geometry.orientation % 4;
    let (source_width, source_height) = source
        .filter(|&(w, h)| w > 0 && h > 0)
        .map(|(w, h)| (w as f32, h as f32))
        .unwrap_or(if turn % 2 == 0 {
            (canvas_width, canvas_height)
        } else {
            (canvas_height, canvas_width)
        });
    let width = rect.width * canvas_width;
    let height = rect.height * canvas_height;
    let (x, y) = (
        rect.x * canvas_width + width / 2.0 - canvas_width / 2.0,
        rect.y * canvas_height + height / 2.0 - canvas_height / 2.0,
    );
    let (sin, cos) = geometry.rotation.to_radians().sin_cos();
    let (x, y) = (x * cos + y * sin, -x * sin + y * cos);
    let (mut x, mut y) = (x, y);
    if geometry.flip_vertical {
        y = -y;
    }
    if geometry.flip_horizontal {
        x = -x;
    }
    let (raw_x, raw_y) = match turn {
        1 => (y, -x),
        2 => (-x, -y),
        3 => (-y, x),
        _ => (x, y),
    };
    Some([raw_x + source_width / 2.0, raw_y + source_height / 2.0, width, height])
}

/// `(column, value)` pairs this build can encode.
pub fn columns(a: &AdjustmentSet) -> Vec<(&'static str, String)> {
    let mut out = vec![
        ("ZEXPOSURE", num(a.exposure.exposure)),
        ("ZCONTRAST", num(a.exposure.contrast)),
        ("ZBRIGHTNESS", num(a.exposure.brightness)),
        ("ZSATURATION", num(a.exposure.saturation)),
        ("ZHIGHLIGHTRECOVERY", num(a.hdr.highlights)),
        ("ZSHADOWRECOVERY", num(a.hdr.shadows)),
        ("ZWHITERECOVERY", num(a.hdr.whites)),
        ("ZBLACKRECOVERY", num(a.hdr.blacks)),
        ("ZCLARITY", num(a.clarity.clarity)),
        ("ZCLARITYSTRUCTURE", num(a.clarity.structure)),
        ("ZUSMAMOUNT", num(a.sharpening.amount)),
        ("ZUSMRADIUS", num(a.sharpening.radius)),
        ("ZUSMTHRESHOLD", num(a.sharpening.threshold)),
        ("ZUSMHALOCONTROL", num(a.sharpening.halo_suppression)),
        ("ZNOISEREDUCTIONAMOUNT", num(a.noise_reduction.luminance)),
        ("ZNOISEDETAILSAMOUNT", num(a.noise_reduction.details)),
        ("ZAIDENOISEAMOUNT", num(a.noise_reduction.color)),
        ("ZDEHAZEAMOUNT", num(a.dehaze.amount)),
        ("ZDEHAZELEVELLOW", vec3(a.dehaze.level_low)),
        ("ZDEHAZELEVELHIGH", vec3(a.dehaze.level_high)),
        ("ZROTATION", num(a.geometry.rotation)),
        ("ZGRADATIONCURVE", curve(&a.curves.rgb.points)),
        ("ZGRADATIONCURVERED", curve(&a.curves.red.points)),
        ("ZGRADATIONCURVEGREEN", curve(&a.curves.green.points)),
        ("ZGRADATIONCURVEBLUE", curve(&a.curves.blue.points)),
        ("ZGRADATIONCURVEY", curve(&a.curves.luma.points)),
    ];
    if let Some(gains) = a.white_balance.gains {
        out.push(("ZWHITEBALANCE", vec3(gains)));
    }
    out
}

/// Tools this build refuses to encode, named only when they are in use.
pub fn skipped(a: &AdjustmentSet) -> Vec<String> {
    let default = AdjustmentSet::default();
    let mut out = Vec::new();
    if a.levels != default.levels {
        out.push(
            "Levels: no non-neutral ZLEVELS* row exists in either catalog (sample: 4 rows, \
             user catalog: 9 rows, all 0;0;0;0 / 1;1;1;1), so the four-slot order cannot be \
             proven. Needs a variant with one channel's levels moved."
                .to_owned(),
        );
    }
    if a.geometry.crop != default.geometry.crop
        || a.geometry.flip_horizontal
        || a.geometry.flip_vertical
    {
        out.push(
            "Crop / flip: the canvas lives on ZVARIANT.ZCANVASSIZE (4043.054;6056.673), not \
             on ZVARIANTLAYER, so the layer row alone cannot be inverted. The forward path \
             also clamps through crop_limit (Mode1/Mode8/Mode10), so an exact inverse has to \
             reproduce that clamp. _DSC5673: 2525.8076;2348.0876;3328.8486;4993.2734 with \
             6048x4032 decodes today; writing it back needs the ZVARIANT join and a \
             clamp-aware inverse, which is not yet proven."
                .to_owned(),
        );
    }
    if a.lens != default.lens {
        out.push(
            "Lens corrections: ZLENSDISTORTION is points|a|b|c|d with the identity row \
             `|0.0|1|0|0.00000` and the full row `...|100.0|1|100|-1.00000`. Both rows carry \
             1 in the second slot, so it cannot be an enable flag; the remaining slots \
             (amount / 100 / -1) cannot be separated from two samples. ZLENSOPTICCENTER is \
             the profile-derived optical centre (3067.851562|1920.515625), not a user edit."
                .to_owned(),
        );
    }
    if a.black_white != default.black_white {
        out.push("Black & White".to_owned());
    }
    if a.color_balance != default.color_balance || a.color_balance_gains != default.color_balance_gains {
        out.push("Color Balance".to_owned());
    }
    if a.color_editor != default.color_editor {
        out.push("Color Editor".to_owned());
    }
    if a.film_grain != default.film_grain {
        out.push("Film Grain".to_owned());
    }
    out
}

/// The columns whose tool group really differs from `current`.
///
/// Comparing decoded values rather than raw text keeps a precision or spelling
/// difference from being mistaken for an edit.
pub fn changed_columns(current: &AdjustmentSet, next: &AdjustmentSet) -> Vec<(&'static str, String)> {
    changed_columns_with_source(current, next, None).0
}

/// [`changed_columns`] plus the variant's source size, needed for `ZCROP`.
/// The second value is the reason the crop was refused, when it was.
pub fn changed_columns_with_source(
    current: &AdjustmentSet,
    next: &AdjustmentSet,
    source: Option<(u32, u32)>,
) -> (Vec<(&'static str, String)>, Option<String>) {
    let encoded: std::collections::HashMap<&'static str, String> = columns(next).into_iter().collect();
    let mut out: Vec<(&'static str, String)> = Vec::new();
    let mut crop_refusal = None;
    let mut take = |columns: &[&'static str]| {
        for column in columns {
            if let Some(value) = encoded.get(column) {
                out.push((*column, value.clone()));
            }
        }
    };
    if current.exposure != next.exposure {
        take(&["ZEXPOSURE", "ZCONTRAST", "ZBRIGHTNESS", "ZSATURATION"]);
    }
    if current.hdr != next.hdr {
        take(&["ZHIGHLIGHTRECOVERY", "ZSHADOWRECOVERY", "ZWHITERECOVERY", "ZBLACKRECOVERY"]);
    }
    if current.white_balance.gains != next.white_balance.gains && next.white_balance.gains.is_some() {
        take(&["ZWHITEBALANCE"]);
    }
    if current.clarity != next.clarity {
        take(&["ZCLARITY", "ZCLARITYSTRUCTURE"]);
    }
    if current.sharpening != next.sharpening {
        take(&["ZUSMAMOUNT", "ZUSMRADIUS", "ZUSMTHRESHOLD", "ZUSMHALOCONTROL"]);
    }
    let noise = |a: &crate::NoiseReduction| (a.luminance, a.details, a.color);
    if noise(&current.noise_reduction) != noise(&next.noise_reduction) {
        take(&["ZNOISEREDUCTIONAMOUNT", "ZNOISEDETAILSAMOUNT", "ZAIDENOISEAMOUNT"]);
    }
    if current.dehaze != next.dehaze {
        take(&["ZDEHAZEAMOUNT", "ZDEHAZELEVELLOW", "ZDEHAZELEVELHIGH"]);
    }
    if current.curves != next.curves {
        take(&[
            "ZGRADATIONCURVE",
            "ZGRADATIONCURVERED",
            "ZGRADATIONCURVEGREEN",
            "ZGRADATIONCURVEBLUE",
            "ZGRADATIONCURVEY",
        ]);
    }
    if current.geometry.rotation != next.geometry.rotation {
        take(&["ZROTATION"]);
    }
    if current.geometry.crop != next.geometry.crop {
        match crop_px(next.geometry.crop, &next.geometry, source) {
            Some(crop) => out.push(("ZCROP", vec4(crop))),
            None => {
                crop_refusal = Some(if !next.geometry.crop_outside_image {
                    "Crop: crop_outside_image=0 (clamped); the crop_limit mode inverse is not proven"
                        .to_owned()
                } else {
                    "Crop: the canvas size is unknown, so the pixel rect cannot be computed".to_owned()
                });
            }
        }
    }
    (out, crop_refusal)
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

fn integrity(conn: &rusqlite::Connection) -> Result<()> {
    let verdict: String = conn
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .context("PRAGMA integrity_check")?;
    if verdict != "ok" {
        bail!("integrity_check: {verdict}");
    }
    Ok(())
}

/// Writes `next` into the variant's full-snapshot row.
///
/// The database is copied to `<catalog>.bak` first, `PRAGMA integrity_check`
/// runs before and after, and any failure restores the backup.
pub fn write_variant(path: &Path, variant_pk: i64, next: &AdjustmentSet) -> Result<WriteOutcome> {
    write_variant_with_source(path, variant_pk, next, None)
}

/// [`write_variant`] with the source size `ZCROP` needs (`ZIMAGE.ZWIDTH/HEIGHT`).
pub fn write_variant_with_source(
    path: &Path,
    variant_pk: i64,
    next: &AdjustmentSet,
    source: Option<(u32, u32)>,
) -> Result<WriteOutcome> {
    let mut conn = rusqlite::Connection::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    integrity(&conn)?;

    let (row_pk, current) = {
        let mut stmt = conn.prepare(
            "SELECT * FROM ZVARIANTLAYER WHERE ZVARIANT = ?1 AND Z_ENT = ?2 ORDER BY Z_PK DESC LIMIT 1",
        )?;
        let column_names: Vec<String> = stmt.column_names().iter().map(|name| name.to_string()).collect();
        let mut rows = stmt.query(rusqlite::params![variant_pk, FULL_SNAPSHOT_ENT])?;
        let Some(row) = rows.next()? else {
            bail!("variant {variant_pk} has no Z_ENT={FULL_SNAPSHOT_ENT} row");
        };
        let pk: i64 = row.get("Z_PK")?;
        let mut current = std::collections::HashMap::new();
        for (index, name) in column_names.iter().enumerate() {
            let value = match row.get_ref(index)? {
                rusqlite::types::ValueRef::Text(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
                rusqlite::types::ValueRef::Integer(v) => Some(v.to_string()),
                rusqlite::types::ValueRef::Real(v) => Some(v.to_string()),
                _ => None,
            };
            current.insert(name.clone(), value);
        }
        let borrowed: std::collections::HashMap<&str, Option<String>> =
            current.iter().map(|(key, value)| (key.as_str(), value.clone())).collect();
        let settings = crate::c1_model::from_layer_row(&borrowed);
        (pk, crate::c1_model::to_adjustment_set(&settings))
    };

    let (changes, crop_refusal) = changed_columns_with_source(&current, next, source);
    if changes.is_empty() {
        return Ok(WriteOutcome::Unchanged {
            warnings: crop_refusal.into_iter().collect(),
        });
    }

    let backup = sibling(path, ".bak");
    std::fs::copy(path, &backup)
        .with_context(|| format!("backing up to {}", backup.display()))?;

    let result = (|| -> Result<()> {
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        for (column, value) in &changes {
            // `column` comes from the static table above, never from input.
            tx.execute(
                &format!("UPDATE ZVARIANTLAYER SET {column} = ?1 WHERE Z_PK = ?2 AND Z_ENT = ?3"),
                rusqlite::params![value, row_pk, FULL_SNAPSHOT_ENT],
            )?;
        }
        tx.commit()?;
        integrity(&conn)
    })();

    if let Err(error) = result {
        drop(conn);
        let _ = std::fs::copy(&backup, path);
        return Err(error).with_context(|| format!("writing {}; restored from backup", path.display()));
    }

    let mut warnings = skipped(next);
    if let Some(reason) = crop_refusal {
        warnings.push(reason);
    }
    Ok(WriteOutcome::Written { columns: changes.len(), warnings })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageEngine as _;

    fn copy_sample(name: &str) -> PathBuf {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-sample.cocatalogdb");
        let dir = std::env::temp_dir().join(format!("aurora-catalog-write-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("sample.cocatalogdb");
        std::fs::copy(&source, &path).expect("copy sample catalog");
        path
    }

    /// A variant that really owns a `Z_ENT = 18` row.
    fn snapshot_variant(path: &Path) -> (i64, AdjustmentSet) {
        let import = crate::c1import::C1Import::open(path).expect("open catalog");
        let conn = rusqlite::Connection::open(path).expect("sqlite");
        let pk: i64 = conn
            .query_row(
                "SELECT ZVARIANT FROM ZVARIANTLAYER WHERE Z_ENT = 18 ORDER BY Z_PK DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("a full-snapshot row");
        let variant = import
            .catalog
            .variants
            .iter()
            .find(|variant| variant.pk == pk)
            .expect("catalog knows the variant");
        (pk, variant.adjustments.clone())
    }

    #[test]
    fn exposure_round_trips_through_the_catalog() {
        let path = copy_sample("roundtrip");
        let (pk, current) = snapshot_variant(&path);
        let mut next = current.clone();
        next.exposure.exposure = 0.5;

        let outcome = write_variant(&path, pk, &next).expect("write");
        assert!(matches!(outcome, WriteOutcome::Written { .. }), "{outcome:?}");
        assert!(sibling(&path, ".bak").is_file(), "the backup is written first");

        let reread = crate::c1import::C1Import::open(&path).expect("reopen");
        let variant = reread.catalog.variants.iter().find(|v| v.pk == pk).expect("variant");
        assert!((variant.adjustments.exposure.exposure - 0.5).abs() < 1e-4);
        assert_eq!(
            std::fs::read(sibling(&path, ".bak")).expect("backup"),
            std::fs::read(copy_sample("roundtrip-original")).expect("original"),
            "the backup holds the original bytes"
        );
    }

    #[test]
    fn an_unchanged_variant_is_not_rewritten() {
        let path = copy_sample("unchanged");
        let (pk, current) = snapshot_variant(&path);
        let before = std::fs::read(&path).expect("bytes");
        let outcome = write_variant(&path, pk, &current).expect("write");
        assert_eq!(outcome, WriteOutcome::Unchanged { warnings: Vec::new() });
        assert_eq!(std::fs::read(&path).expect("bytes"), before, "the file must not change");
        assert!(!sibling(&path, ".bak").exists(), "no backup without a write");
    }

    #[test]
    fn a_corrupt_catalog_is_refused_before_any_write() {
        let path = copy_sample("corrupt");
        std::fs::write(&path, b"not a sqlite database").expect("corrupt");
        let outcome = write_variant(&path, 1, &AdjustmentSet::default());
        assert!(outcome.is_err());
        assert_eq!(std::fs::read(&path).expect("bytes"), b"not a sqlite database");
        assert!(!sibling(&path, ".bak").exists(), "refused before the backup step");
    }

    #[test]
    fn a_clamped_crop_write_leaves_the_row_alone_and_warns() {
        let path = copy_sample("clamped-crop");
        let (pk, current) = snapshot_variant(&path);
        let mut next = current.clone();
        next.geometry.crop_outside_image = false;
        next.geometry.crop = crate::NormRect { x: 0.1, y: 0.1, width: 0.5, height: 0.5 };
        next.geometry.canvas = [4043.054, 6056.673];
        let before = std::fs::read(&path).expect("bytes");
        let outcome = write_variant_with_source(&path, pk, &next, Some((6048, 4032))).expect("write");
        let WriteOutcome::Unchanged { warnings } = outcome else {
            panic!("expected Unchanged, got {outcome:?}");
        };
        assert!(
            warnings.iter().any(|warning| warning.contains("crop_outside_image=0")),
            "{warnings:?}"
        );
        assert_eq!(std::fs::read(&path).expect("bytes"), before, "the row must not change");
    }

    #[test]
    fn an_unclamped_crop_is_written_as_pixels() {
        let path = copy_sample("crop");
        let (pk, current) = snapshot_variant(&path);
        let mut next = current.clone();
        next.geometry.crop_outside_image = true;
        next.geometry.orientation = 0;
        next.geometry.canvas = [6048.0, 4032.0];
        next.geometry.crop = crate::NormRect { x: 0.25, y: 0.25, width: 0.5, height: 0.25 };
        let expected = crop_px(next.geometry.crop, &next.geometry, Some((6048, 4032))).expect("inverse");
        let outcome = write_variant_with_source(&path, pk, &next, Some((6048, 4032))).expect("write");
        let WriteOutcome::Written { columns, .. } = outcome else {
            panic!("expected Written, got {outcome:?}");
        };
        assert!(columns >= 1);

        let conn = rusqlite::Connection::open(&path).expect("sqlite");
        let stored: String = conn
            .query_row(
                "SELECT ZCROP FROM ZVARIANTLAYER WHERE ZVARIANT = ?1 AND Z_ENT = 18 ORDER BY Z_PK DESC LIMIT 1",
                [pk],
                |row| row.get(0),
            )
            .expect("ZCROP");
        let values: Vec<f32> = stored
            .split(';')
            .filter_map(|part| part.trim().parse().ok())
            .collect();
        assert_eq!(values.len(), 4, "{stored}");
        for (a, b) in expected.iter().zip(values.iter()) {
            assert!((a - b).abs() < 0.01, "{expected:?} vs {values:?}");
        }
    }

    fn crop_geometry(canvas: [f32; 2], turn: u8) -> crate::Geometry {
        crate::Geometry { canvas, orientation: turn, crop_outside_image: true, ..Default::default() }
    }

    #[test]
    fn crop_inverts_the_unclamped_canvas_mapping() {
        let crop = [2525.8076, 2348.0876, 3328.8486, 4993.2734];
        for turn in 0..4u8 {
            let geometry = crop_geometry([6048.0, 4032.0], turn);
            let rect = crate::c1_model::crop_on_canvas_with_clipping(
                crop,
                &geometry,
                Some((6048, 4032)),
                2,
            );
            let back = crop_px(rect, &geometry, Some((6048, 4032))).expect("unclamped inverse");
            for (a, b) in crop.iter().zip(back.iter()) {
                assert!((a - b).abs() < 0.01, "turn {turn}: {crop:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn a_clamped_crop_is_refused() {
        let mut geometry = crop_geometry([6048.0, 4032.0], 0);
        geometry.crop_outside_image = false;
        assert!(crop_px(crate::NormRect::FULL, &geometry, Some((6048, 4032))).is_none());
    }

    #[test]
    fn the_engine_entry_point_writes_and_rereads() {
        let path = copy_sample("engine");
        let engine = crate::FileEngine::open_c1_catalog(&path).expect("catalog engine");
        let info = engine.images().first().expect("an imported image").clone();
        let mut next = engine.load_adjustments(info.id);
        next.exposure.exposure = 0.75;
        let outcome = engine.save_to_c1_catalog(info.id, &next).expect("save");
        assert!(
            matches!(outcome, crate::c1_writeback::CatalogSaveOutcome::Saved { .. }),
            "{outcome:?}"
        );
        let import = crate::c1import::C1Import::open(&path).expect("reopen");
        let row = import
            .catalog
            .images
            .iter()
            .find(|row| row.file_name == info.file_name)
            .expect("catalog row");
        let variant = import
            .catalog
            .variants
            .iter()
            .find(|variant| variant.image_pk == row.pk)
            .expect("variant");
        assert!((variant.adjustments.exposure.exposure - 0.75).abs() < 1e-4);
    }

    #[test]
    fn the_engine_entry_point_reports_unchanged() {
        let path = copy_sample("engine-unchanged");
        let engine = crate::FileEngine::open_c1_catalog(&path).expect("catalog engine");
        let info = engine.images().first().expect("an imported image").clone();
        let current = engine.load_adjustments(info.id);
        let before = std::fs::read(&path).expect("bytes");
        let outcome = engine.save_to_c1_catalog(info.id, &current).expect("save");
        assert!(
            matches!(outcome, crate::c1_writeback::CatalogSaveOutcome::Unchanged { .. }),
            "{outcome:?}"
        );
        assert_eq!(std::fs::read(&path).expect("bytes"), before);
    }

    #[test]
    fn a_plain_folder_engine_is_not_a_catalog() {
        let dir = std::env::temp_dir().join(format!("aurora-not-a-catalog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let engine = crate::FileEngine::open(&dir).expect("folder engine");
        let outcome = engine
            .save_to_c1_catalog(crate::ImageId(1), &AdjustmentSet::default())
            .expect("save");
        assert_eq!(outcome, crate::c1_writeback::CatalogSaveOutcome::NotACatalog);
        let _ = std::fs::remove_dir_all(dir);
    }
}
