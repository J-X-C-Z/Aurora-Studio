//! Writing the current library as a Capture One session bundle.
//!
//! The bundle is `<Name>.cosessiondb/<Name>.cosessiondb` (SQLite) beside the
//! session folders. Images are hard linked (or copied when the file system
//! refuses a link) into `Capture/` and referenced by relative path
//! (`Capture/<file>`), never by an absolute one, so the bundle can be moved.
//!
//! Rating, colour tag and keywords go into `ZVARIANTMETADATA`, linked from
//! `ZVARIANTLAYER.ZMETADATA`; the keyword hierarchy is written to `ZKEYWORD` and
//! the metadata row keeps the plain names C1 stores (comma separated).
//!
//! Only the three tables the Portrait reader requires are guaranteed; Capture One
//! opening this bundle has not been verified, so this is a session *structure*,
//! not a proven C1 export.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

use crate::c1_masks::C1MaskLayer;
use crate::c1_model::C1VariantSettings;
use crate::c1import::C1Import;
use crate::catalog::CollectionKind;
use crate::sqlite_write::{write_tables, Cell, SynthTable};

/// The folders a Capture One session carries; media would land in `Capture`.
const SESSION_FOLDERS: [&str; 6] =
    ["Capture", "Selects", "Trash", "Output", "Cache", "Adjustments"];
/// Every image points at this folder, relative to the database.
const MEDIA_DIR: &str = "Capture";

/// What an export wrote, for the caller's report.
#[derive(Debug, Clone)]
pub struct SessionExport {
    pub session_dir: PathBuf,
    pub database: PathBuf,
    pub images: usize,
    pub variants: usize,
    /// Media files hard linked into `Capture/`.
    pub media_linked: usize,
    /// Media files copied because a hard link was refused.
    pub media_copied: usize,
    /// Images whose original file was missing; their row keeps the relative
    /// name but no file exists for it.
    pub media_missing: usize,
    /// Media files also placed in `Selects/` because a variant is rated >= 1.
    pub media_selected: usize,
    /// Media files placed in `Trash/` because `ZIMAGE.ZISTRASHED` is set.
    pub media_trashed: usize,
    /// `.comask` sidecars copied byte-for-byte out of the source catalog.
    pub masks_copied: usize,
    /// Images with mask layers whose `<name>.comask` is absent from the source.
    pub mask_missing: usize,
    /// Images with mask layers when the source has no `Adjustments/LAM` at all.
    pub mask_absent: usize,
}

/// Writes the library behind `import` to `target` and returns what was written.
///
/// `target` must not exist: the bundle is staged beside it and renamed into
/// place, so a failed export never leaves half a session behind.
pub fn export_session(import: &C1Import, target: &Path) -> Result<SessionExport> {
    if target.exists() {
        bail!("{} already exists; refusing to overwrite it", target.display());
    }
    let name = target
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("{} has no usable folder name", target.display()))?;
    let parent = target.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("creating {}", parent.display()))?;

    let staging = parent.join(format!(".{name}.partial-{}", std::process::id()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging)
            .with_context(|| format!("clearing {}", staging.display()))?;
    }
    let final_root = if target.is_absolute() {
        target.to_path_buf()
    } else {
        std::env::current_dir()?.join(target)
    };
    let (database, media, masks) = match write_bundle(import, &staging, name, &final_root) {
        Ok(written) => written,
        Err(err) => {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(err);
        }
    };
    std::fs::rename(&staging, target).with_context(|| {
        format!("moving {} to {}", staging.display(), target.display())
    })?;
    Ok(SessionExport {
        session_dir: target.to_path_buf(),
        database: target.join(database),
        images: import.images().len(),
        variants: import.variants().len(),
        media_linked: media.linked,
        media_copied: media.copied,
        media_missing: media.missing,
        media_selected: media.selected,
        media_trashed: media.trashed,
        masks_copied: masks.copied,
        mask_missing: masks.missing,
        mask_absent: masks.absent,
    })
}

/// The database file name inside the bundle: the folder's own name, as C1 uses.
fn database_name(bundle: &str) -> String {
    if bundle.to_ascii_lowercase().ends_with(".cosessiondb") {
        bundle.to_string()
    } else {
        "Session.cosessiondb".to_string()
    }
}

/// How the media of one export was placed in `Capture/`.
#[derive(Debug, Default)]
struct Media {
    linked: usize,
    copied: usize,
    missing: usize,
    selected: usize,
    trashed: usize,
}

fn write_bundle(
    import: &C1Import,
    dir: &Path,
    bundle: &str,
    final_root: &Path,
) -> Result<(String, Media, MaskCopy)> {
    for folder in SESSION_FOLDERS {
        std::fs::create_dir_all(dir.join(folder))
            .with_context(|| format!("creating {folder} in {}", dir.display()))?;
    }
    let source = SourceRows::load(import)?;
    let plan = media_plan(import, &source);
    let media = place_media(&plan, dir)?;
    let masks = copy_masks(import, dir)?;
    let database = database_name(bundle);
    write_tables(
        &dir.join(&database),
        &session_tables(import, &plan, final_root, &source)?,
    )
        .with_context(|| format!("writing {database}"))?;
    Ok((database, media, masks))
}

/// What [`copy_masks`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MaskCopy {
    copied: usize,
    missing: usize,
    absent: usize,
}

/// Copies every `.comask` under the source catalog's `Adjustments/LAM` tree to
/// the same relative path in the session bundle.
///
/// Only untouched masks are covered: the bytes come from C1, so C1 reopens
/// them (`IC_MaskLoad64` rc=0). A mask edited inside Lumen still needs the
/// writer (r75's open item), never this copy path.
fn copy_masks(import: &C1Import, dir: &Path) -> Result<MaskCopy> {
    let source_root = import.catalog_path.parent().unwrap_or_else(|| Path::new("."));
    let lam = source_root.join("Adjustments").join("LAM");
    let names: Vec<String> = images_with_masks(import)
        .iter()
        .map(|i| i.file_name.to_ascii_lowercase())
        .collect();
    copy_masks_from(&lam, dir, &names)
}

/// [`copy_masks`] without the `C1Import` plumbing, so it can be tested alone.
fn copy_masks_from(lam: &Path, dir: &Path, masked_images: &[String]) -> Result<MaskCopy> {
    let mut copy = MaskCopy::default();
    if !lam.is_dir() {
        copy.absent = masked_images.len();
        return Ok(copy);
    }
    let mut files = Vec::new();
    collect_comask(lam, &mut files)?;
    for source in &files {
        let Ok(rel) = source.strip_prefix(lam) else { continue };
        let target = dir.join("Adjustments").join("LAM").join(rel);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::copy(source, &target)
            .with_context(|| format!("copying {}", source.display()))?;
        copy.copied += 1;
    }
    let names: std::collections::HashSet<String> = files
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
        .map(|n| n.to_ascii_lowercase())
        .collect();
    for image in masked_images {
        if !names.contains(&format!("{image}.comask")) {
            copy.missing += 1;
        }
    }
    Ok(copy)
}

/// Images that carry at least one local mask layer with a UUID.
fn images_with_masks(import: &C1Import) -> Vec<&crate::c1catalog::C1Image> {
    let mut pks = std::collections::HashSet::new();
    for variant in import.variants() {
        if variant.local_layers.iter().any(|l| l.mask_uuid.is_some()) {
            pks.insert(variant.image_pk);
        }
    }
    import.images().iter().filter(|i| pks.contains(&i.pk)).collect()
}

/// Recursive walk that only collects `.comask` files.
fn collect_comask(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_comask(&path, out)?;
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("comask"))
        {
            out.push(path);
        }
    }
    Ok(())
}

/// One image's place in the bundle.
struct MediaItem {
    name: String,
    source: PathBuf,
    selected: bool,
    trashed: bool,
}

type MediaPlan = Vec<MediaItem>;

/// The bundle file names, with a suffix for images that share a name but not a
/// source file (two cards can both hold `_DSC0001.NEF`).
fn media_plan(import: &C1Import, source: &SourceRows) -> MediaPlan {
    let mut plan = MediaPlan::new();
    let mut taken: HashMap<String, ()> = HashMap::new();
    for image in import.images() {
        let mut name = if image.file_name.is_empty() {
            format!("image_{}.bin", image.pk)
        } else {
            image.file_name.clone()
        };
        if taken.contains_key(&name.to_lowercase()) {
            name = suffixed(&name, image.pk);
        }
        taken.insert(name.to_lowercase(), ());
        let trashed = source
            .image(image.pk)
            .and_then(|row| row_i64(row, "ZISTRASHED"))
            == Some(1);
        let selected = !trashed
            && import
                .variants_of(image.pk)
                .iter()
                .any(|variant| variant.rating > 0);
        plan.push(MediaItem {
            name,
            source: PathBuf::from(&image.abs_path),
            selected,
            trashed,
        });
    }
    plan
}

fn suffixed(name: &str, pk: i64) -> String {
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => format!("{stem}~{pk}.{ext}"),
        _ => format!("{name}~{pk}"),
    }
}

/// Hard links every original into `Capture/`; a refused link is copied. The
/// source file is only ever read.
fn place_media(plan: &MediaPlan, dir: &Path) -> Result<Media> {
    let media_dir = dir.join(MEDIA_DIR);
    let selects_dir = dir.join("Selects");
    let trash_dir = dir.join("Trash");
    let mut media = Media::default();
    for item in plan {
        if !item.source.is_file() {
            media.missing += 1;
            continue;
        }
        place_one(&item.source, &media_dir.join(&item.name), &mut media)?;
        if item.trashed {
            place_one(&item.source, &trash_dir.join(&item.name), &mut media)?;
            media.trashed += 1;
        } else if item.selected {
            place_one(&item.source, &selects_dir.join(&item.name), &mut media)?;
            media.selected += 1;
        }
    }
    Ok(media)
}

/// Link (or copy on refusal) one source file; only the Capture call updates
/// `linked`/`copied`, so the existing media counts keep their meaning.
fn place_one(source: &Path, dest: &Path, media: &mut Media) -> Result<()> {
    if std::fs::hard_link(source, dest).is_ok() {
        if dest.parent().and_then(Path::file_name) == Some(std::ffi::OsStr::new(MEDIA_DIR)) {
            media.linked += 1;
        }
        return Ok(());
    }
    std::fs::copy(source, dest)
        .with_context(|| format!("copying {} to {}", source.display(), dest.display()))?;
    if dest.parent().and_then(Path::file_name) == Some(std::ffi::OsStr::new(MEDIA_DIR)) {
        media.copied += 1;
    }
    Ok(())
}

/// The source rows behind the columns the model does not carry yet.
struct SourceRows {
    images: Vec<HashMap<String, Option<String>>>,
    layers: Vec<HashMap<String, Option<String>>>,
    metadata: Vec<HashMap<String, Option<String>>>,
    entities: Vec<HashMap<String, Option<String>>>,
    collection_properties: Vec<HashMap<String, Option<String>>>,
    selected_variants: Vec<HashMap<String, Option<String>>>,
    compare_variants: Vec<HashMap<String, Option<String>>>,
    variant_members: Vec<HashMap<String, Option<String>>>,
    version_info: Vec<HashMap<String, Option<String>>>,
    document_settings: Vec<HashMap<String, Option<String>>>,
    output_recipes: Vec<HashMap<String, Option<String>>>,
    process_history: Vec<HashMap<String, Option<String>>>,
}

impl SourceRows {
    fn load(import: &C1Import) -> Result<Self> {
        Ok(Self {
            images: crate::c1catalog::export_table_rows(&import.catalog_path, "ZIMAGE")?,
            layers: crate::c1catalog::export_table_rows(&import.catalog_path, "ZVARIANTLAYER")?,
            metadata: crate::c1catalog::export_table_rows(&import.catalog_path, "ZVARIANTMETADATA")?,
            entities: crate::c1catalog::export_table_rows(&import.catalog_path, "ZENTITIES")?,
            collection_properties: crate::c1catalog::export_table_rows(
                &import.catalog_path,
                "ZIMAGEINCOLLECTIONPROPERTIES",
            )?,
            selected_variants: crate::c1catalog::export_table_rows(
                &import.catalog_path,
                "ZSELECTEDVARIANTS",
            )?,
            compare_variants: crate::c1catalog::export_table_rows(
                &import.catalog_path, "ZCOMPAREVARIANTS",
            )?,
            variant_members: crate::c1catalog::export_table_rows(
                &import.catalog_path, "ZVARIANTINCOLLECTION",
            )?,
            version_info: crate::c1catalog::export_table_rows(&import.catalog_path, "ZVERSIONINFO")?,
            document_settings: crate::c1catalog::export_table_rows(
                &import.catalog_path,
                "ZDOCUMENTSETTING",
            )?,
            output_recipes: crate::c1catalog::export_table_rows(
                &import.catalog_path,
                "ZENABLEDOUTPUTRECIPE",
            )?,
            process_history: crate::c1catalog::export_table_rows(
                &import.catalog_path,
                "ZPROCESSHISTORY",
            )?,
        })
    }

    fn image(&self, pk: i64) -> Option<&HashMap<String, Option<String>>> {
        self.images.iter().find(|row| row_i64(row, "Z_PK") == Some(pk))
    }

    fn layer(&self, variant: i64, ent: i64) -> Option<&HashMap<String, Option<String>>> {
        self.layers.iter().find(|row| {
            row_i64(row, "ZVARIANT") == Some(variant) && row_i64(row, "Z_ENT") == Some(ent)
        })
    }

    /// The base metadata row is the one the variant's `Z_ENT=18` layer points at.
    fn metadata_for_variant(&self, variant: i64) -> Option<&HashMap<String, Option<String>>> {
        let meta_pk = row_i64(self.layer(variant, 18)?, "ZMETADATA")?;
        self.metadata.iter().find(|row| row_i64(row, "Z_PK") == Some(meta_pk))
    }
}

fn row_text<'a>(row: &'a HashMap<String, Option<String>>, key: &str) -> Option<&'a str> {
    row.get(key).and_then(|value| value.as_deref())
}

fn row_f64(row: &HashMap<String, Option<String>>, key: &str) -> Option<f64> {
    row_text(row, key)?.trim().parse().ok()
}

fn row_i64(row: &HashMap<String, Option<String>>, key: &str) -> Option<i64> {
    row_f64(row, key).map(|value| value as i64)
}

fn row_bool(row: &HashMap<String, Option<String>>, key: &str) -> Option<i64> {
    row_f64(row, key).map(|value| i64::from(value != 0.0))
}

fn src_text(row: Option<&HashMap<String, Option<String>>>, key: &str) -> Cell {
    row.and_then(|row| row_text(row, key)).map(Cell::text).unwrap_or(Cell::Null)
}

fn src_int(row: Option<&HashMap<String, Option<String>>>, key: &str) -> Cell {
    row.and_then(|row| row_i64(row, key)).map(Cell::Int).unwrap_or(Cell::Null)
}

fn src_real(row: Option<&HashMap<String, Option<String>>>, key: &str) -> Cell {
    row.and_then(|row| row_f64(row, key))
        .filter(|value| value.is_finite())
        .map(Cell::Real)
        .unwrap_or(Cell::Null)
}

#[derive(Clone)]
struct PathParts {
    original: String,
    win: String,
    mac: String,
    rel: String,
}

fn root_prefix(path: &Path) -> String {
    let mut root = path.to_string_lossy().into_owned();
    if !root.ends_with(['\\', '/']) {
        root.push(std::path::MAIN_SEPARATOR);
    }
    root
}

/// Splits a catalog path into the same `win/mac root + relative` shape C1 uses.
fn split_location(path: &str, session_root: &str) -> PathParts {
    let bytes = path.as_bytes();
    if path.len() >= 3 && bytes[1] == b':' && matches!(bytes[2], b'\\' | b'/') {
        return PathParts {
            original: path.to_string(),
            win: format!("{}:\\", &path[..1]),
            mac: String::new(),
            rel: path[3..].trim_start_matches(['\\', '/']).to_string(),
        };
    }
    if let Some(rest) = path.strip_prefix("\\\\") {
        let mut parts = rest.splitn(2, ['\\', '/']);
        let server = parts.next().unwrap_or("");
        let after_server = parts.next().unwrap_or("");
        let (share, rel) = after_server.split_once(['\\', '/']).unwrap_or((after_server, ""));
        return PathParts {
            original: path.to_string(),
            win: format!("\\\\{server}\\{share}\\"),
            mac: String::new(),
            rel: rel.trim_start_matches(['\\', '/']).to_string(),
        };
    }
    if path.starts_with('/') {
        return PathParts {
            original: path.to_string(),
            win: String::new(),
            mac: "/".to_string(),
            rel: path.trim_start_matches('/').to_string(),
        };
    }
    PathParts {
        original: path.to_string(),
        win: session_root.to_string(),
        mac: String::new(),
        rel: path.trim_start_matches(['\\', '/']).to_string(),
    }
}

#[derive(Clone, Copy)]
enum CopyKind {
    Pk,
    Int,
    Real,
    Text,
}

fn copied_table(
    name: &str,
    columns: &[&str],
    rows: &[HashMap<String, Option<String>>],
    kinds: &[CopyKind],
) -> SynthTable {
    let mut table = SynthTable::new(name, columns);
    for row in rows {
        let Some(pk) = row_i64(row, "Z_PK") else { continue };
        let cells = columns
            .iter()
            .zip(kinds)
            .map(|(definition, kind)| {
                let column = definition.split_whitespace().next().unwrap_or("");
                match kind {
                    CopyKind::Pk => Cell::Null,
                    CopyKind::Int => src_int(Some(row), column),
                    CopyKind::Real => src_real(Some(row), column),
                    CopyKind::Text => src_text(Some(row), column),
                }
            })
            .collect();
        table = table.row_with_id(pk, cells);
    }
    table
}

fn entities_table(source: &SourceRows) -> SynthTable {
    let mut table = SynthTable::new(
        "ZENTITIES",
        &["Z_ENT INTEGER PRIMARY KEY", "ZNAME TEXT"],
    );
    let mut rows: Vec<&HashMap<String, Option<String>>> = source.entities.iter().collect();
    rows.sort_by_key(|row| row_i64(row, "Z_ENT").unwrap_or(0));
    for row in rows {
        let Some(ent) = row_i64(row, "Z_ENT") else { continue };
        table = table.row_with_id(ent, vec![Cell::Null, src_text(Some(row), "ZNAME")]);
    }
    table
}

fn session_tables(
    import: &C1Import,
    plan: &MediaPlan,
    final_root: &Path,
    source: &SourceRows,
) -> Result<Vec<SynthTable>> {
    let session_root = root_prefix(final_root);
    let mut locations = SynthTable::new(
        "ZPATHLOCATION",
        &[
            "Z_ENT INTEGER",
            "Z_PK INTEGER PRIMARY KEY",
            "ZWINROOT TEXT",
            "ZMACROOT TEXT",
            "ZRELATIVEPATH TEXT",
            "ZISRELATIVE BOOLEAN",
            "ZVOLUME TEXT",
            "ZWINATTRIBUTE TEXT",
        ],
    )
    .row_with_id(
        1,
        vec![
            Cell::Int(38),
            Cell::Null,
            Cell::text(&session_root),
            Cell::text(""),
            Cell::text(MEDIA_DIR),
            Cell::Int(0),
            Cell::Null,
            Cell::Null,
        ],
    );

    // A folder collection's name is derived from the path it points at, so
    // each distinct path becomes a ZPATHLOCATION row; identical paths share
    // one row. Collections without a path keep ZFOLDERLOCATION NULL.
    let mut folder_location: HashMap<i64, i64> = HashMap::new();
    let mut folder_rows: Vec<(i64, PathParts)> = Vec::new();
    for collection in &import.catalog.collections {
        let Some(path) = &collection.folder_path else { continue };
        let location = match folder_rows.iter().find(|(_, p)| &p.original == path) {
            Some((pk, _)) => *pk,
            None => {
                let pk = 2 + folder_rows.len() as i64;
                folder_rows.push((pk, split_location(path, &session_root)));
                pk
            }
        };
        folder_location.insert(collection.pk, location);
    }
    for (pk, parts) in &folder_rows {
        locations = locations.row_with_id(
            *pk,
            vec![
                Cell::Int(38),
                Cell::Null,
                Cell::text(&parts.win),
                Cell::text(&parts.mac),
                Cell::text(&parts.rel),
                Cell::Int(0),
                Cell::Null,
                Cell::Null,
            ],
        );
    }

    let mut images = SynthTable::new(
        "ZIMAGE",
        &[
            "Z_PK INTEGER PRIMARY KEY",
            "ZIMAGEUUID TEXT",
            "ZIMAGEFILENAME TEXT",
            "ZIMAGELOCATION INTEGER",
            "ZWIDTH INTEGER",
            "ZHEIGHT INTEGER",
            "ZISO INTEGER",
            "ZCAMERA_MAKE TEXT",
            "ZCAMERA_MODEL TEXT",
            "ZCAMERA_LENS TEXT",
            "ZEXP_APERTURE REAL",
            "ZEXP_SHUTTERSPEED REAL",
            "ZEXP_FOCALLENGTH REAL",
            "ZEXP_DATE REAL",
            "ZIMPORTDATE REAL",
            "ZEXP_DRIVEMODE INTEGER",
            "ZEXP_EXPOSUREPROG INTEGER",
            "ZEXP_FLASHMODE INTEGER",
            "ZEXP_METERINGMODE INTEGER",
            "ZEXP_EXPOSURECOMPENSATION TEXT",
            "ZEXP_COLORSPACE TEXT",
            "ZEXP_EXPOSUREMODE INTEGER",
            "ZCAMERA_SOFTWARE TEXT",
            "ZCAMERA_SERIAL TEXT",
            "ZCAMERA_OWNER TEXT",
            "ZPIXELWIDTH REAL",
            "ZPIXELHEIGHT REAL",
        ],
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or(0.0);
    for (image, item) in import.images().iter().zip(plan) {
        let source = source.image(image.pk);
        let import_date = source
            .and_then(|row| row_f64(row, "ZIMPORTDATE"))
            .map(Cell::Real)
            .unwrap_or(Cell::Real(now));
        images = images.row_with_id(
            image.pk,
            vec![
                Cell::Null,
                Cell::text(&image.uuid),
                Cell::text(&item.name),
                Cell::Int(1),
                Cell::Int(i64::from(image.width)),
                Cell::Int(i64::from(image.height)),
                image.iso.map(Cell::Int).unwrap_or(Cell::Null),
                Cell::text(&image.camera_make),
                Cell::text(&image.camera_model),
                Cell::text(&image.camera_lens),
                image.aperture.map(real).unwrap_or(Cell::Null),
                image.shutter.map(real).unwrap_or(Cell::Null),
                image.focal_length.map(real).unwrap_or(Cell::Null),
                image.shot_at.map(|t| Cell::Real(t as f64)).unwrap_or(Cell::Null),
                import_date,
                src_int(source, "ZEXP_DRIVEMODE"),
                src_int(source, "ZEXP_EXPOSUREPROG"),
                src_int(source, "ZEXP_FLASHMODE"),
                src_int(source, "ZEXP_METERINGMODE"),
                src_text(source, "ZEXP_EXPOSURECOMPENSATION"),
                src_text(source, "ZEXP_COLORSPACE"),
                src_int(source, "ZEXP_EXPOSUREMODE"),
                src_text(source, "ZCAMERA_SOFTWARE"),
                src_text(source, "ZCAMERA_SERIAL"),
                src_text(source, "ZCAMERA_OWNER"),
                src_real(source, "ZPIXELWIDTH"),
                src_real(source, "ZPIXELHEIGHT"),
            ],
        );
    }

    let (variants, layers) = variants_and_layers(import, source)?;
    let (collections, members, root_pk) = collection_tables(import, &folder_location);
    let document = SynthTable::new(
        "ZDOCUMENTCONTENT",
        &["Z_PK INTEGER PRIMARY KEY", "ZDOCUMENTTYPE INTEGER", "ZROOTCOLLECTION INTEGER"],
    )
    .row_with_id(1, vec![Cell::Null, Cell::Int(2), root_pk.map(Cell::Int).unwrap_or(Cell::Null)]);
    Ok(vec![
        document,
        locations,
        images,
        variants,
        collections,
        members,
        metadata(import, source),
        keywords(import),
        layers,
        retouch_table(import),
        entities_table(source),
        copied_table(
            "ZIMAGEINCOLLECTIONPROPERTIES",
            &[
                "Z_ENT INTEGER",
                "Z_PK INTEGER PRIMARY KEY",
                "ZCOLLECTION INTEGER",
                "ZIMAGE INTEGER",
                "ZCOLLAPSED BOOLEAN",
                "ZMANUALSORTINDEX INTEGER",
            ],
            &source.collection_properties,
            &[
                CopyKind::Int,
                CopyKind::Pk,
                CopyKind::Int,
                CopyKind::Int,
                CopyKind::Int,
                CopyKind::Int,
            ],
        ),
        copied_table(
            "ZSELECTEDVARIANTS",
            &[
                "Z_ENT INTEGER",
                "Z_PK INTEGER PRIMARY KEY",
                "ZSELECTEDINCOLLECTIONS INTEGER",
                "ZSELECTEDVARIANTS INTEGER",
            ],
            &source.selected_variants,
            &[CopyKind::Int, CopyKind::Pk, CopyKind::Int, CopyKind::Int],
        ),
        copied_table(
            "ZCOMPAREVARIANTS",
            &[
                "Z_ENT INTEGER",
                "Z_PK INTEGER PRIMARY KEY",
                "ZCOMPAREINCOLLECTIONS INTEGER",
                "ZCOMPAREVARIANTS INTEGER",
            ],
            &source.compare_variants,
            &[CopyKind::Int, CopyKind::Pk, CopyKind::Int, CopyKind::Int],
        ),
        copied_table(
            "ZVARIANTINCOLLECTION",
            &[
                "Z_ENT INTEGER NOT NULL",
                "Z_PK INTEGER PRIMARY KEY NOT NULL",
                "ZCOLLECTION INTEGER NOT NULL",
                "ZVARIANT INTEGER NOT NULL",
            ],
            &source.variant_members,
            &[CopyKind::Int, CopyKind::Pk, CopyKind::Int, CopyKind::Int],
        ),
        copied_table(
            "ZVERSIONINFO",
            &[
                "Z_ENT INTEGER",
                "Z_PK INTEGER PRIMARY KEY",
                "ZAUTHOR TEXT",
                "ZCOMPATIBILITY TEXT",
                "ZCOMPATIBLEVERSION INTEGER",
                "ZFORMAT TEXT",
                "ZVERSION INTEGER",
            ],
            &source.version_info,
            &[
                CopyKind::Int,
                CopyKind::Pk,
                CopyKind::Text,
                CopyKind::Text,
                CopyKind::Int,
                CopyKind::Text,
                CopyKind::Int,
            ],
        ),
        copied_table(
            "ZDOCUMENTSETTING",
            &[
                "Z_ENT INTEGER",
                "Z_PK INTEGER PRIMARY KEY",
                "ZSETTINGKEY TEXT",
                "ZSETTINGVALUE TEXT",
            ],
            &source.document_settings,
            &[CopyKind::Int, CopyKind::Pk, CopyKind::Text, CopyKind::Text],
        ),
        copied_table(
            "ZENABLEDOUTPUTRECIPE",
            &[
                "Z_ENT INTEGER",
                "Z_PK INTEGER PRIMARY KEY",
                "ZNAME TEXT",
                "ZENABLEDINPROJECT INTEGER",
            ],
            &source.output_recipes,
            &[CopyKind::Int, CopyKind::Pk, CopyKind::Text, CopyKind::Int],
        ),
        copied_table(
            "ZPROCESSHISTORY",
            &[
                "Z_ENT INTEGER",
                "Z_PK INTEGER PRIMARY KEY",
                "ZDATE REAL",
                "ZURL TEXT",
                "ZVARIANT INTEGER",
            ],
            &source.process_history,
            &[
                CopyKind::Int,
                CopyKind::Pk,
                CopyKind::Real,
                CopyKind::Text,
                CopyKind::Int,
            ],
        ),
    ])
}

/// `ZCOLLECTION` plus the `ZIMAGEINCOLLECTION` membership rows.
///
/// `ZFOLDERLOCATION` points at the collection's `ZPATHLOCATION` row when it has
/// one, so C1 can derive the folder name; otherwise it stays NULL.
fn collection_tables(
    import: &C1Import,
    folder_location: &HashMap<i64, i64>,
) -> (SynthTable, SynthTable, Option<i64>) {
    let collections = &import.catalog.collections;
    let root_pk = collections.iter().find(|c| c.parent.is_none()).map(|c| c.pk);
    let role = |kind: CollectionKind| {
        collections.iter().find(|c| c.kind == kind).map(|c| c.pk)
    };
    let all_images = role(CollectionKind::AllImages);
    let trash = role(CollectionKind::Trash);
    let recent_captures = role(CollectionKind::RecentCaptures);
    let recent_imports = role(CollectionKind::RecentImports);
    let role_cell = |is_root: bool, pk: Option<i64>| match (is_root, pk) {
        (true, Some(pk)) => Cell::Int(pk),
        _ => Cell::Null,
    };

    let mut table = SynthTable::new(
        "ZCOLLECTION",
        &[
            "Z_ENT INTEGER",
            "Z_PK INTEGER PRIMARY KEY",
            "ZUUID TEXT",
            "ZNAME TEXT",
            "ZPARENT INTEGER",
            "ZPROJECT INTEGER",
            "ZCOLLECTIONINDEX INTEGER",
            "ZFOLDERLOCATION INTEGER",
            "ZALLIMAGESCOLLECTION INTEGER",
            "ZTRASHCOLLECTION INTEGER",
            "ZLASTCAPTURESCOLLECTION INTEGER",
            "ZLASTIMPORTSCOLLECTION INTEGER",
        ],
    );
    for c in collections {
        let is_root = Some(c.pk) == root_pk;
        table = table.row_with_id(
            c.pk,
            vec![
                Cell::Int(c.ent),
                Cell::Null,
                Cell::text(&c.uuid),
                Cell::text(&c.name),
                Cell::Int(c.parent.unwrap_or(-1)),
                if is_root { Cell::Int(-1) } else { Cell::Null },
                Cell::Int(c.index),
                folder_location.get(&c.pk).map(|p| Cell::Int(*p)).unwrap_or(Cell::Null),
                role_cell(is_root, all_images),
                role_cell(is_root, trash),
                role_cell(is_root, recent_captures),
                role_cell(is_root, recent_imports),
            ],
        );
    }

    let mut members = SynthTable::new(
        "ZIMAGEINCOLLECTION",
        &["Z_ENT INTEGER", "Z_PK INTEGER PRIMARY KEY", "ZCOLLECTION INTEGER", "ZIMAGE INTEGER"],
    );
    let mut rowid = 1i64;
    for c in collections {
        for image in &c.images {
            members = members.row_with_id(
                rowid,
                vec![Cell::Int(47), Cell::Null, Cell::Int(c.pk), Cell::Int(*image)],
            );
            rowid += 1;
        }
    }
    (table, members, root_pk)
}

/// One `ZVARIANTMETADATA` row per variant: the rating, colour tag and keywords
/// C1 links to it from `ZVARIANTLAYER.ZMETADATA`.
fn metadata(import: &C1Import, source: &SourceRows) -> SynthTable {
    let mut table = SynthTable::new(
        "ZVARIANTMETADATA",
        &[
            "Z_ENT INTEGER",
            "Z_PK INTEGER PRIMARY KEY",
            "ZLAYER INTEGER",
            "ZBASIC_RATING INTEGER",
            "ZCOLOR_TAG_INDEX INTEGER",
            "ZCONTENT_KEYWORDS TEXT",
            "ZCONTACT_CREATOR TEXT",
            "ZIMAGE_DATECREATED TEXT",
        ],
    );
    let paths = import.keyword_paths();
    for variant in import.variants() {
        let source_meta = source.metadata_for_variant(variant.pk);
        // C1 stores the keyword names, not their tree paths; the hierarchy is
        // rebuilt from ZKEYWORD on read.
        let keywords: Vec<String> =
            variant.keywords.iter().map(|k| plain_keyword(&paths, k)).collect();
        table = table.row_with_id(
            variant.pk,
            vec![
                Cell::Int(13),
                Cell::Null,
                Cell::Int(variant.pk),
                Cell::Int(variant.rating),
                Cell::Int(variant.color_tag),
                if keywords.is_empty() {
                    Cell::Null
                } else {
                    Cell::Text(keywords.join(","))
                },
                src_text(source_meta, "ZCONTACT_CREATOR"),
                src_text(source_meta, "ZIMAGE_DATECREATED"),
            ],
        );
    }
    table
}

fn plain_keyword(paths: &HashMap<String, String>, keyword: &str) -> String {
    paths
        .iter()
        .find(|(_, path)| path.as_str() == keyword)
        .map(|(name, _)| name.clone())
        .unwrap_or_else(|| keyword.to_string())
}

/// C1's keyword tree; the reader rebuilds `parent/child` paths from `ZPARENT`.
fn keywords(import: &C1Import) -> SynthTable {
    let mut table = SynthTable::new(
        "ZKEYWORD",
        &["Z_ENT INTEGER", "Z_PK INTEGER PRIMARY KEY", "ZNAME TEXT", "ZPARENT INTEGER"],
    );
    for keyword in &import.catalog.keywords {
        table = table.row_with_id(
            keyword.pk,
            vec![
                Cell::Int(44),
                Cell::Null,
                Cell::text(&keyword.name),
                keyword.parent.map(Cell::Int).unwrap_or(Cell::Null),
            ],
        );
    }
    table
}

/// `ZVARIANTLAYER` columns: the link columns plus the edits the reader applies.
const LAYER_COLUMNS: &[&str] = &[
    "Z_PK INTEGER PRIMARY KEY",
    "ZINDEX INTEGER",
    "Z_ENT INTEGER",
    "ZVARIANT INTEGER",
    "ZMETADATA INTEGER",
    "ZMASKUUID TEXT",
    "ZNAME TEXT",
    "ZENABLED INTEGER",
    "ZOPACITY TEXT",
    "ZEXPOSURE TEXT",
    "ZCONTRAST TEXT",
    "ZBRIGHTNESS TEXT",
    "ZSATURATION TEXT",
    "ZWHITEBALANCE TEXT",
    "ZLEVELSSHADOW TEXT",
    "ZLEVELSHIGHLIGHT TEXT",
    "ZMIDTONE TEXT",
    "ZLEVELSTARGETSHADOW TEXT",
    "ZLEVELSTARGETHIGHLIGHT TEXT",
    "ZCLARITY TEXT",
    "ZCLARITYSTRUCTURE TEXT",
    "ZCLARITYMETHOD TEXT",
    "ZHIGHLIGHTRECOVERY TEXT",
    "ZSHADOWRECOVERY TEXT",
    "ZWHITERECOVERY TEXT",
    "ZBLACKRECOVERY TEXT",
    "ZUSMAMOUNT TEXT",
    "ZUSMRADIUS TEXT",
    "ZUSMTHRESHOLD TEXT",
    "ZUSMHALOCONTROL TEXT",
    "ZNOISEREDUCTIONAMOUNT TEXT",
    "ZNOISEDETAILSAMOUNT TEXT",
    "ZAIDENOISEAMOUNT TEXT",
    "ZDENOISEMETHOD INTEGER",
    "ZDEHAZEAMOUNT TEXT",
    "ZCROP TEXT",
    "ZCROPOUTSIDEIMAGE INTEGER",
    "ZCROPASPECTRATIONAME TEXT",
    "ZROTATION TEXT",
    "ZICCPROFILE TEXT",
    "ZFILMCURVE TEXT",
    "ZGEOMETRYKEYSTONEFOCALLENGTH REAL",
    "ZLENSOPTICCENTER TEXT",
    "ZLENSOPTICCENTERENABLED INTEGER",
    "ZLEVELSTARGETENABLE INTEGER",
    "ZRETOUCHINGBLEMISHREMOVALAMOUNT REAL",
    "ZRETOUCHINGFACESCULPTINGCONTOURING REAL",
    "ZRETOUCHINGSKINEVENINGAMOUNT REAL",
    "ZRETOUCHINGOPACITY REAL",
    "ZRETOUCHINGINCLUDENECKINBLEMISHREMOVAL INTEGER",
    "ZRETOUCHINGINCLUDENECKINSKINEVENING INTEGER",
];

/// The pass-through A columns on `ZVARIANTLAYER`.
#[derive(Clone, Default)]
struct LayerExtras {
    optical_enabled: Option<i64>,
    target_enable: Option<i64>,
    blemish: Option<f64>,
    face_sculpting: Option<f64>,
    skin_evening: Option<f64>,
    opacity: Option<f64>,
    neck_blemish: Option<i64>,
    neck_skin: Option<i64>,
}

impl LayerExtras {
    fn from_row(row: Option<&HashMap<String, Option<String>>>) -> Self {
        Self {
            optical_enabled: row.and_then(|row| row_bool(row, "ZLENSOPTICCENTERENABLED")),
            target_enable: row.and_then(|row| row_bool(row, "ZLEVELSTARGETENABLE")),
            blemish: row.and_then(|row| row_f64(row, "ZRETOUCHINGBLEMISHREMOVALAMOUNT")),
            face_sculpting: row.and_then(|row| row_f64(row, "ZRETOUCHINGFACESCULPTINGCONTOURING")),
            skin_evening: row.and_then(|row| row_f64(row, "ZRETOUCHINGSKINEVENINGAMOUNT")),
            opacity: row.and_then(|row| row_f64(row, "ZRETOUCHINGOPACITY")),
            neck_blemish: row.and_then(|row| row_bool(row, "ZRETOUCHINGINCLUDENECKINBLEMISHREMOVAL")),
            neck_skin: row.and_then(|row| row_bool(row, "ZRETOUCHINGINCLUDENECKINSKINEVENING")),
        }
    }

    /// `defaults` is true on the base/combined snapshot rows, false on the
    /// sparse `Z_ENT=17` delta, where a missing amount stays NULL.
    fn cells(&self, defaults: bool) -> Vec<Cell> {
        let int = |value: Option<i64>, fallback: i64| {
            value.map(Cell::Int).unwrap_or(if defaults { Cell::Int(fallback) } else { Cell::Null })
        };
        let real = |value: Option<f64>, fallback: f64| {
            value.map(Cell::Real).unwrap_or(if defaults { Cell::Real(fallback) } else { Cell::Null })
        };
        vec![
            int(self.optical_enabled, 0),
            int(self.target_enable, 1),
            real(self.blemish, 0.0),
            real(self.face_sculpting, 0.0),
            real(self.skin_evening, 0.0),
            real(self.opacity, 100.0),
            int(self.neck_blemish, 1),
            int(self.neck_skin, 1),
        ]
    }
}

/// One full-snapshot layer (`Z_ENT` 18), the combined row (20) and the sparse
/// adjustment delta (17) per variant, plus local mask layers (23).
fn variants_and_layers(import: &C1Import, source: &SourceRows) -> Result<(SynthTable, SynthTable)> {
    let mut variants = SynthTable::new(
        "ZVARIANT",
        &[
            "Z_PK INTEGER PRIMARY KEY",
            "ZIMAGE INTEGER",
            "ZINDEX INTEGER",
            "ZCANVASSIZE TEXT",
            "ZVARIANTUUID TEXT",
            "ZADJUSTMENTLAYER INTEGER",
            "ZDEFAULTLAYER INTEGER",
            "ZCOMBINEDSETTINGS INTEGER",
        ],
    );
    let variant_rows = crate::c1catalog::export_table_rows(&import.catalog_path, "ZVARIANT")?;
    let variant_rows: HashMap<_, _> = variant_rows
        .iter()
        .filter_map(|row| Some((row_i64(row, "Z_PK")?, row)))
        .collect();
    let mut table = SynthTable::new("ZVARIANTLAYER", LAYER_COLUMNS);
    let mut pk = import.variants().iter().map(|v| v.pk).max().unwrap_or(0) + 1;
    for variant in import.variants() {
        let combined_pk = pk;
        let delta_pk = pk + 1;
        pk += 2;
        // These references point to rows allocated by this export.
        variants = variants.row_with_id(
            variant.pk,
            vec![
                Cell::Null,
                Cell::Int(variant.image_pk),
                Cell::Int(variant.index),
                variant
                    .canvas
                    .map(|(w, h)| Cell::Text(format!("{w};{h}")))
                    .unwrap_or(Cell::Null),
                src_text(variant_rows.get(&variant.pk).copied(), "ZVARIANTUUID"),
                Cell::Int(delta_pk),
                Cell::Int(variant.pk),
                Cell::Int(combined_pk),
            ],
        );
        let base_row = source.layer(variant.pk, 18);
        let combined_row = source.layer(variant.pk, 20);
        let base = LayerExtras::from_row(base_row);
        let combined = combined_row
            .map(|row| LayerExtras::from_row(Some(row)))
            .unwrap_or_else(|| base.clone());
        let delta = LayerExtras::from_row(source.layer(variant.pk, 17));
        table = table.row_with_id(
            variant.pk,
            layer_cells(
                variant.pk,
                18,
                variant.pk,
                Some(variant.pk),
                &variant.settings,
                &base,
                base_row,
                true,
            ),
        );
        table = table.row_with_id(
            combined_pk,
            layer_cells(
                combined_pk, 20, variant.pk, None, &variant.settings, &combined, combined_row, true,
            ),
        );
        table = table.row_with_id(delta_pk, delta_layer_cells(delta_pk, variant.pk, &delta));
    }
    // Local mask layers are references (`Z_ENT` 23 + `ZMASKUUID`); the mask
    // pixels live in a `.comask` sidecar and are not copied.
    for variant in import.variants() {
        for layer in &variant.local_layers {
            table = table.row_with_id(pk, local_layer_cells(variant.pk, layer));
            pk += 1;
        }
    }
    Ok((variants, table))
}

fn local_layer_cells(variant_pk: i64, layer: &C1MaskLayer) -> Vec<Cell> {
    let mut cells = vec![
        Cell::Null,
        Cell::Int(layer.index),
        Cell::Int(23),
        Cell::Int(variant_pk),
        Cell::Null,
        layer.mask_uuid.as_deref().map(Cell::text).unwrap_or(Cell::Null),
        layer.name.as_deref().map(Cell::text).unwrap_or(Cell::Null),
        Cell::Int(i64::from(layer.enabled)),
        scalar(layer.opacity),
    ];
    cells.resize(LAYER_COLUMNS.len(), Cell::Null);
    cells
}

fn delta_layer_cells(pk: i64, variant_pk: i64, extras: &LayerExtras) -> Vec<Cell> {
    let mut cells = vec![Cell::Null; LAYER_COLUMNS.len()];
    cells[1] = Cell::Int(0);
    cells[2] = Cell::Int(17);
    cells[3] = Cell::Int(variant_pk);
    let start = LAYER_COLUMNS.len() - 8;
    cells[start..].clone_from_slice(&extras.cells(false));
    cells
}

fn layer_cells(
    _pk: i64,
    ent: i64,
    variant_pk: i64,
    metadata_pk: Option<i64>,
    s: &C1VariantSettings,
    extras: &LayerExtras,
    source_row: Option<&HashMap<String, Option<String>>>,
    defaults: bool,
) -> Vec<Cell> {
    let mut cells = vec![
        Cell::Null,
        Cell::Null,
        Cell::Int(ent),
        Cell::Int(variant_pk),
        metadata_pk.map(Cell::Int).unwrap_or(Cell::Null),
        Cell::Null,
        s.name.as_deref().map(Cell::text).unwrap_or(Cell::Null),
        Cell::Int(i64::from(s.enabled)),
        scalar(s.opacity),
        scalar(s.exposure),
        scalar(s.contrast),
        scalar(s.brightness),
        scalar(s.saturation),
        vector(&s.white_balance_gains),
        vector(&s.levels_shadow),
        vector(&s.levels_highlight),
        vector(&s.levels_midtone),
        vector(&s.levels_target_shadow),
        vector(&s.levels_target_highlight),
        scalar(s.clarity),
        scalar(s.clarity_structure),
        Cell::Text(s.clarity_method.to_string()),
        scalar(s.highlights_recovery),
        scalar(s.shadows_recovery),
        scalar(s.whites_recovery),
        scalar(s.blacks_recovery),
        scalar(s.sharpening.amount),
        scalar(s.sharpening.radius),
        scalar(s.sharpening.threshold),
        scalar(s.sharpening.halo_suppression),
        scalar(s.noise_reduction.luminance),
        scalar(s.noise_reduction.details),
        scalar(s.noise_reduction.color),
        Cell::Int(i64::from(s.noise_reduction.method)),
        scalar(s.dehaze_amount),
        vector(&s.crop_px),
        Cell::Int(i64::from(s.crop_outside_image)),
        s.crop_aspect_ratio.as_deref().map(Cell::text).unwrap_or(Cell::Null),
        scalar(s.rotation),
        s.icc_profile.as_deref().map(Cell::text).unwrap_or(Cell::Null),
        s.film_curve.as_deref().map(Cell::text).unwrap_or(Cell::Null),
        src_real(source_row, "ZGEOMETRYKEYSTONEFOCALLENGTH"),
        src_text(source_row, "ZLENSOPTICCENTER"),
    ];
    cells.extend(extras.cells(defaults));
    cells
}

/// The 41 `ZRETOUCHINGLAYER` columns, in the real catalog's order.
const RETOUCH_COLUMNS: &[&str] = &[
    "Z_ENT INTEGER",
    "Z_PK INTEGER PRIMARY KEY",
    "ZVARIANT INTEGER",
    "ZFACE TEXT",
    "ZBLEMISHREMOVALMASKUUID TEXT",
    "ZBLEMISHREMOVALMASKCROP TEXT",
    "ZBLEMISHREMOVALAMOUNT REAL",
    "ZDARKCIRCLESREDUCTIONMASKUUID TEXT",
    "ZDARKCIRCLESREDUCTIONMASKCROP TEXT",
    "ZDARKCIRCLESREDUCTIONAMOUNT REAL",
    "ZFACESCULPTINGMASKUUID TEXT",
    "ZFACESCULPTINGMASKCROP TEXT",
    "ZFACESCULPTINGCONTOURING REAL",
    "ZFACESCULPTINGHIGHLIGHTING REAL",
    "ZSKINEVENINGMASKUUID TEXT",
    "ZSKINEVENINGMASKCROP TEXT",
    "ZSKINEVENINGAMOUNT REAL",
    "ZSKINEVENINGTEXTURE REAL",
    "ZTEETHWHITENINGMASKUUID TEXT",
    "ZTEETHWHITENINGMASKCROP TEXT",
    "ZTEETHWHITENINGBRIGHTNESS REAL",
    "ZTEETHWHITENINGDESATURATION REAL",
    "ZOPACITY REAL",
    "ZINCLUDENECKINBLEMISHREMOVAL INTEGER",
    "ZINCLUDENECKINSKINEVENING INTEGER",
    "ZEYEBRIGHTLEFTIRISMASKUUID TEXT",
    "ZEYEBRIGHTLEFTIRISMASKCROP TEXT",
    "ZEYEBRIGHTLEFTIRISAMOUNT REAL",
    "ZEYEBRIGHTLEFTSCLERAMASKUUID TEXT",
    "ZEYEBRIGHTLEFTSCLERAMASKCROP TEXT",
    "ZEYEBRIGHTLEFTSCLERAAMOUNT REAL",
    "ZEYEBRIGHTRIGHTIRISMASKUUID TEXT",
    "ZEYEBRIGHTRIGHTIRISMASKCROP TEXT",
    "ZEYEBRIGHTRIGHTIRISAMOUNT REAL",
    "ZEYEBRIGHTRIGHTSCLERAMASKUUID TEXT",
    "ZEYEBRIGHTRIGHTSCLERAMASKCROP TEXT",
    "ZEYEBRIGHTRIGHTSCLERAAMOUNT REAL",
    "ZEYEBRIGHTBOTHIRISAMOUNT REAL",
    "ZEYEBRIGHTBOTHSCLERAAMOUNT REAL",
    "ZSTRAYHAIRREMOVALINNERAMOUNT REAL",
    "ZSTRAYHAIRREMOVALOUTERAMOUNT REAL",
];

fn retouch_cells(
    variant_pk: i64,
    layer: &crate::c1_retouch::RetouchingLayer,
) -> Vec<Cell> {
    let text = |value: &Option<String>| value.as_deref().map(Cell::text).unwrap_or(Cell::Null);
    let number = |value: Option<f32>| value.filter(|v| v.is_finite()).map(real).unwrap_or(Cell::Null);
    let flag = |value: Option<bool>| value.map(|v| Cell::Int(i64::from(v))).unwrap_or(Cell::Null);
    let mask = |slot: &crate::c1_retouch::MaskSlot| {
        (text(&slot.mask_uuid), text(&slot.mask_crop))
    };
    let blemish = mask(&layer.blemish_removal_mask);
    let dark = mask(&layer.dark_circles_mask);
    let sculpt = mask(&layer.face_sculpting_mask);
    let skin = mask(&layer.skin_evening_mask);
    let teeth = mask(&layer.teeth_whitening_mask);
    let left_iris = mask(&layer.eye_bright_left_iris_mask);
    let left_sclera = mask(&layer.eye_bright_left_sclera_mask);
    let right_iris = mask(&layer.eye_bright_right_iris_mask);
    let right_sclera = mask(&layer.eye_bright_right_sclera_mask);
    vec![
        Cell::Int(51),
        Cell::Null,
        Cell::Int(layer.variant.unwrap_or(variant_pk)),
        text(&layer.face_raw),
        blemish.0,
        blemish.1,
        number(layer.blemish_removal_amount),
        dark.0,
        dark.1,
        number(layer.dark_circles_amount),
        sculpt.0,
        sculpt.1,
        number(layer.face_sculpting_contouring),
        number(layer.face_sculpting_highlighting),
        skin.0,
        skin.1,
        number(layer.skin_evening_amount),
        number(layer.skin_evening_texture),
        teeth.0,
        teeth.1,
        number(layer.teeth_whitening_brightness),
        number(layer.teeth_whitening_desaturation),
        number(layer.opacity),
        flag(layer.include_neck_in_blemish_removal),
        flag(layer.include_neck_in_skin_evening),
        left_iris.0,
        left_iris.1,
        number(layer.eye_bright_left_iris_amount),
        left_sclera.0,
        left_sclera.1,
        number(layer.eye_bright_left_sclera_amount),
        right_iris.0,
        right_iris.1,
        number(layer.eye_bright_right_iris_amount),
        right_sclera.0,
        right_sclera.1,
        number(layer.eye_bright_right_sclera_amount),
        number(layer.eye_bright_both_iris_amount),
        number(layer.eye_bright_both_sclera_amount),
        number(layer.stray_hair_removal_inner_amount),
        number(layer.stray_hair_removal_outer_amount),
    ]
}

fn retouch_table(import: &C1Import) -> SynthTable {
    let mut table = SynthTable::new("ZRETOUCHINGLAYER", RETOUCH_COLUMNS);
    let mut pk = 1;
    for variant in import.variants() {
        for layer in &variant.retouching {
            table = table.row_with_id(pk, retouch_cells(variant.pk, layer));
            pk += 1;
        }
    }
    table
}

fn scalar(value: f32) -> Cell {
    Cell::Text(if value.is_finite() { value.to_string() } else { "0".to_string() })
}

fn real(value: f32) -> Cell {
    Cell::Real(if value.is_finite() { f64::from(value) } else { 0.0 })
}

fn vector(values: &[f32]) -> Cell {
    let parts: Vec<String> = values
        .iter()
        .map(|v| if v.is_finite() { v.to_string() } else { "0".to_string() })
        .collect();
    Cell::Text(parts.join(";"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c1_model::C1VariantSettings;

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-sample.cocatalogdb")
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "portrait-session-export-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_exported_session_reads_back_with_the_same_images_and_edits() {
        let source = C1Import::open(&fixture()).unwrap();
        let dir = temp_dir("roundtrip");
        let target = dir.join("Exported.cosessiondb");
        let export = export_session(&source, &target).unwrap();
        assert_eq!(export.session_dir, target);
        assert!(export.database.ends_with("Exported.cosessiondb/Exported.cosessiondb"));
        assert_eq!((export.images, export.variants), (3, 4));
        let originals =
            source.images().iter().filter(|i| Path::new(&i.abs_path).is_file()).count();
        assert_eq!(export.media_missing, source.images().len() - originals);
        assert_eq!(export.media_linked + export.media_copied, originals);
        for folder in SESSION_FOLDERS {
            assert!(target.join(folder).is_dir(), "{folder} missing");
        }

        let back = C1Import::open(&target).unwrap();
        assert!(back.is_session());
        assert_eq!(back.images().len(), source.images().len());
        assert_eq!(back.variants().len(), source.variants().len());
        assert_eq!(back.catalog.collections.len(), 11);

        for (a, b) in source.images().iter().zip(back.images()) {
            assert_eq!(a.pk, b.pk);
            assert_eq!(a.uuid, b.uuid);
            assert_eq!(a.file_name, b.file_name);
            assert!(b.abs_path.contains(MEDIA_DIR), "{}", b.abs_path);
            assert!(b.abs_path.ends_with(&a.file_name), "{}", b.abs_path);
            if Path::new(&a.abs_path).is_file() {
                let media = target.join(&b.abs_path);
                assert!(media.is_file(), "{} was not written", media.display());
                assert_eq!(
                    std::fs::metadata(&media).unwrap().len(),
                    std::fs::metadata(&a.abs_path).unwrap().len()
                );
            }
            assert_eq!((a.width, a.height), (b.width, b.height));
            assert_eq!(a.iso, b.iso);
            assert_eq!(a.camera_model, b.camera_model);
            assert_eq!(a.aperture, b.aperture);
            assert_eq!(a.shutter, b.shutter);
            assert_eq!(a.focal_length, b.focal_length);
            assert_eq!(a.shot_at, b.shot_at);
        }

        for (a, b) in source.variants().iter().zip(back.variants()) {
            assert_eq!(a.pk, b.pk);
            assert_eq!(a.image_pk, b.image_pk);
            assert_eq!(a.index, b.index);
            assert_eq!(a.canvas, b.canvas);
            assert_eq!(b.rating, a.rating, "rating of variant {}", a.pk);
            assert_eq!(b.color_tag, a.color_tag, "color tag of variant {}", a.pk);
            assert_eq!(b.keywords, a.keywords, "keywords of variant {}", a.pk);
            same_edits(&a.settings, &b.settings);
            assert_eq!(back.variants_of(a.image_pk).len(), source.variants_of(a.image_pk).len());
        }

        assert!(export_session(&source, &target).is_err(), "overwrite must be refused");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The catalog fixture has no keywords, so the synthetic session (a keyword
    /// tree Wedding > Bride, rating 6) carries that half of the round trip.
    #[test]
    fn ratings_and_keywords_survive_the_round_trip() {
        let dir = temp_dir("metadata");
        let source_path = dir.join("Synth.cosessiondb");
        std::fs::write(&source_path, crate::c1catalog::test_support::synthetic_session()).unwrap();
        let source = C1Import::open(&source_path).unwrap();
        assert!(source.variants().iter().any(|v| v.rating == 6), "fixture must carry a rating");
        assert!(
            source.variants().iter().any(|v| v.keywords.iter().any(|k| k == "Wedding/Bride")),
            "fixture must carry a keyword tree"
        );

        let target = dir.join("Meta.cosessiondb");
        let export = export_session(&source, &target).unwrap();
        assert_eq!(export.media_missing, source.images().len());
        assert_eq!(export.media_linked + export.media_copied, 0);
        assert_eq!(export.media_selected, 0);
        assert_eq!(export.media_trashed, 0);
        assert_eq!(std::fs::read_dir(target.join("Selects")).unwrap().count(), 0);
        assert_eq!(std::fs::read_dir(target.join("Trash")).unwrap().count(), 0);
        let back = C1Import::open(&target).unwrap();
        assert_eq!(back.variants().len(), source.variants().len());
        for (a, b) in source.variants().iter().zip(back.variants()) {
            assert_eq!(b.rating, a.rating, "rating of variant {}", a.pk);
            assert_eq!(b.color_tag, a.color_tag, "color tag of variant {}", a.pk);
            assert_eq!(b.keywords, a.keywords, "keywords of variant {}", a.pk);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `ZCOLLECTION` / `ZIMAGEINCOLLECTION` were the tables the export used to
    /// drop: the fixture's eleven collections came back as zero.
    #[test]
    fn the_collection_tree_survives_the_round_trip() {
        let source = C1Import::open(&fixture()).unwrap();
        assert_eq!(source.catalog.collections.len(), 11, "fixture must carry 11 collections");

        let dir = temp_dir("collections");
        let target = dir.join("Collections.cosessiondb");
        export_session(&source, &target).unwrap();
        let back = C1Import::open(&target).unwrap();

        assert_eq!(back.catalog.collections.len(), source.catalog.collections.len());
        for a in &source.catalog.collections {
            let b = back
                .catalog
                .collections
                .iter()
                .find(|b| b.pk == a.pk)
                .unwrap_or_else(|| panic!("collection {} missing", a.pk));
            assert_eq!(b.parent, a.parent, "parent of collection {}", a.pk);
            assert_eq!(b.index, a.index, "index of collection {}", a.pk);
            assert_eq!(b.images, a.images, "members of collection {}", a.pk);
            assert_eq!(b.folder_path, a.folder_path, "folder path of collection {}", a.pk);
        }

        // The engine tree, not just the rows: same root count and children.
        let (a, b) = (source.collections(), back.collections());
        assert_eq!(b.len(), a.len(), "root collections");
        assert_eq!(
            b.iter().map(|c| c.children.len()).sum::<usize>(),
            a.iter().map(|c| c.children.len()).sum::<usize>()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_locations_are_winroot_plus_relative() {
        let source = C1Import::open(&fixture()).unwrap();
        let dir = temp_dir("path-shape");
        let target = dir.join("Paths.cosessiondb");
        let export = export_session(&source, &target).unwrap();
        let locations = rows(&export.database, "ZPATHLOCATION");
        assert!(!locations.is_empty());

        let no_drive = |path: &str| !(path.len() >= 2 && path.as_bytes()[1] == b':');
        for row in &locations {
            assert_eq!(row_i64(row, "Z_ENT"), Some(38));
            assert_eq!(row_i64(row, "ZISRELATIVE"), Some(0));
            let win = row_text(row, "ZWINROOT").unwrap_or("");
            let mac = row_text(row, "ZMACROOT").unwrap_or("");
            assert!(win.is_empty() ^ mac.is_empty(), "exactly one root: {win:?} {mac:?}");
            let rel = row_text(row, "ZRELATIVEPATH").unwrap_or("");
            assert!(no_drive(rel), "relative path has a drive: {rel}");
            assert!(!rel.starts_with(['\\', '/']), "relative path starts at a root: {rel}");
            assert!(row_text(row, "ZVOLUME").is_none());
            assert!(row_text(row, "ZWINATTRIBUTE").is_none());
        }

        let target_root = root_prefix(&std::env::current_dir().unwrap().join(&target));
        let media = locations.iter().find(|row| row_i64(row, "Z_PK") == Some(1)).unwrap();
        assert_eq!(row_text(media, "ZWINROOT"), Some(target_root.as_str()));
        assert_eq!(row_text(media, "ZRELATIVEPATH"), Some(MEDIA_DIR));

        for image in source.images() {
            let image_row = rows(&export.database, "ZIMAGE")
                .into_iter()
                .find(|row| row_i64(row, "Z_PK") == Some(image.pk))
                .unwrap();
            let loc = row_i64(&image_row, "ZIMAGELOCATION").unwrap();
            let location = locations.iter().find(|row| row_i64(row, "Z_PK") == Some(loc)).unwrap();
            assert_eq!(row_text(location, "ZRELATIVEPATH"), Some(MEDIA_DIR));
            assert!(no_drive(&image.file_name), "{}", image.file_name);
            assert!(!format!("{MEDIA_DIR}/{}", image.file_name).contains(':'));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rated_images_are_linked_into_selects_only() {
        let source = C1Import::open(&fixture()).unwrap();
        let rated = source
            .variants()
            .iter()
            .find(|variant| variant.rating > 0)
            .expect("fixture must carry a rated variant");
        assert_eq!(rated.rating, 5);
        let dir = temp_dir("selects");
        let target = dir.join("Selects.cosessiondb");
        let export = export_session(&source, &target).unwrap();
        assert_eq!(export.media_selected, 1);
        assert_eq!(export.media_trashed, 0);

        for image in source.images() {
            let media = target.join("Capture").join(&image.file_name);
            assert!(media.is_file(), "{}", media.display());
            let selected = target.join("Selects").join(&image.file_name);
            if image.pk == rated.image_pk {
                assert!(selected.is_file(), "{}", selected.display());
                assert_eq!(
                    std::fs::metadata(&selected).unwrap().len(),
                    std::fs::metadata(&media).unwrap().len()
                );
            } else {
                assert!(!selected.exists(), "unrated {} entered Selects", image.file_name);
            }
            assert!(!target.join("Trash").join(&image.file_name).exists());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn collection_queries_find_preserved_variant_links() {
        let import = C1Import::open(&fixture()).unwrap();
        let mut source = SourceRows::load(&import).unwrap();
        assert!(source.compare_variants.is_empty() && source.variant_members.is_empty());
        let variant = import.variants()[0].pk;
        let collection = import.catalog.collections[0].pk;
        let link = |collection_column: &str, variant_column: &str| {
            [
                ("Z_ENT", 47),
                ("Z_PK", 3),
                (collection_column, collection),
                (variant_column, variant),
            ]
            .into_iter()
            .map(|(key, value)| (key.to_owned(), Some(value.to_string())))
            .collect()
        };
        source.compare_variants.push(link("ZCOMPAREINCOLLECTIONS", "ZCOMPAREVARIANTS"));
        source.variant_members.push(link("ZCOLLECTION", "ZVARIANT"));
        let dir = temp_dir("variant-links");
        let database = dir.join("test.cosessiondb");
        let tables = session_tables(&import, &media_plan(&import, &source), &dir, &source).unwrap();
        write_tables(&database, &tables).unwrap();
        assert_eq!(rows(&database, "ZCOMPAREVARIANTS"), source.compare_variants);
        assert_eq!(rows(&database, "ZVARIANTINCOLLECTION"), source.variant_members);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn entities_and_display_tables_match_the_source() {
        let source = C1Import::open(&fixture()).unwrap();
        let dir = temp_dir("entities-display");
        let target = dir.join("Tables.cosessiondb");
        let export = export_session(&source, &target).unwrap();

        let source_entities = rows(&fixture(), "ZENTITIES");
        let out_entities = rows(&export.database, "ZENTITIES");
        assert_eq!(out_entities.len(), 50);
        assert_eq!(out_entities.len(), source_entities.len());
        for a in &source_entities {
            let ent = row_i64(a, "Z_ENT").unwrap();
            let b = out_entities
                .iter()
                .find(|row| row_i64(row, "Z_ENT") == Some(ent))
                .unwrap_or_else(|| panic!("Z_ENT {ent} missing"));
            assert_eq!(row_text(b, "ZNAME"), row_text(a, "ZNAME"), "Z_ENT {ent}");
        }

        for table in [
            "ZIMAGEINCOLLECTIONPROPERTIES",
            "ZSELECTEDVARIANTS",
            "ZVERSIONINFO",
            "ZDOCUMENTSETTING",
            "ZENABLEDOUTPUTRECIPE",
            "ZPROCESSHISTORY",
        ] {
            let a_rows = rows(&fixture(), table);
            let b_rows = rows(&export.database, table);
            assert_eq!(b_rows.len(), a_rows.len(), "{table}");
            for a in &a_rows {
                let pk = row_i64(a, "Z_PK").unwrap();
                let b = b_rows
                    .iter()
                    .find(|row| row_i64(row, "Z_PK") == Some(pk))
                    .unwrap_or_else(|| panic!("{table} pk {pk} missing"));
                for column in a.keys() {
                    assert!(
                        same_value(row_text(a, column), row_text(b, column)),
                        "{table}.{column} pk {pk}"
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One end-to-end gate for the capability matrix: counts, ratings, colour
    /// tags, crop/rotation, collections, local layers, retouch rows, media and
    /// Selects/Trash. Keystone/29-column gaps are reported by the matrix, not
    /// asserted as passing here.
    #[test]
    fn full_session_matrix_round_trip() {
        let source = C1Import::open(&fixture()).unwrap();
        let dir = temp_dir("matrix");
        let target = dir.join("Matrix.cosessiondb");
        let export = export_session(&source, &target).unwrap();
        assert_eq!(export.images, 3);
        assert_eq!(export.variants, 4);
        assert_eq!(export.media_linked + export.media_copied, 3);
        assert_eq!(export.media_missing, 0);
        assert_eq!(export.media_selected, 1);
        assert_eq!(export.media_trashed, 0);

        let back = C1Import::open(&target).unwrap();
        assert_eq!(back.images().len(), 3);
        assert_eq!(back.variants().len(), 4);
        assert_eq!(back.catalog.collections.len(), 11);
        assert_eq!(
            source.variants()[0].local_layers.len(),
            back.variants()[0].local_layers.len()
        );
        assert_eq!(
            source.variants()[0].retouching.len(),
            back.variants()[0].retouching.len()
        );

        for (a, b) in source.variants().iter().zip(back.variants()) {
            assert_eq!(b.rating, a.rating, "rating {}", a.pk);
            assert_eq!(b.color_tag, a.color_tag, "colour tag {}", a.pk);
            assert_eq!(b.settings.crop_px, a.settings.crop_px, "crop {}", a.pk);
            assert_eq!(b.settings.rotation, a.settings.rotation, "rotation {}", a.pk);
            assert_eq!(b.settings.canvas_size, a.settings.canvas_size, "canvas {}", a.pk);
            assert_eq!(
                b.local_layers.len(),
                a.local_layers.len(),
                "local layers {}",
                a.pk
            );
        }
        for (a, b) in source.catalog.collections.iter().zip(&back.catalog.collections) {
            assert_eq!(b.pk, a.pk);
            assert_eq!(b.images, a.images, "collection members {}", a.pk);
        }

        for image in source.images() {
            assert!(target.join("Capture").join(&image.file_name).is_file());
        }
        assert!(target.join("Selects").join("_DSC5673.NEF").is_file());
        assert_eq!(std::fs::read_dir(target.join("Trash")).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Local mask layers are rebuilt from the `Z_ENT` 23 rows; only their
    /// references travel, never the `.comask` pixels.
    #[test]
    fn local_mask_layers_survive_as_references() {
        let source = C1Import::open(&fixture()).unwrap();
        let source_layers: Vec<_> = source
            .variants()
            .iter()
            .find(|v| v.pk == 1)
            .expect("variant 1")
            .local_layers
            .iter()
            .map(|l| (l.index, l.name.clone(), l.enabled, l.mask_uuid.clone()))
            .collect();
        assert_eq!(source_layers.len(), 5, "variant 1 must carry five local layers");

        let dir = temp_dir("local-layers");
        let target = dir.join("Layers.cosessiondb");
        export_session(&source, &target).unwrap();
        let back = C1Import::open(&target).unwrap();
        let back_layers: Vec<_> = back
            .variants()
            .iter()
            .find(|v| v.pk == 1)
            .expect("variant 1")
            .local_layers
            .iter()
            .map(|l| (l.index, l.name.clone(), l.enabled, l.mask_uuid.clone()))
            .collect();
        assert_eq!(back_layers, source_layers);

        let total = |c: &C1Import| -> usize {
            c.variants().iter().map(|v| v.local_layers.len()).sum()
        };
        assert_eq!(total(&back), total(&source));

        let comask = std::fs::read_dir(target.join(MEDIA_DIR))
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("comask"))
            })
            .count();
        assert_eq!(comask, 0, "mask pixels must not be copied");
        let _ = std::fs::remove_dir_all(&dir);
    }

    const IMAGE_A: &[&str] = &[
        "ZIMPORTDATE",
        "ZEXP_DRIVEMODE",
        "ZEXP_EXPOSUREPROG",
        "ZEXP_FLASHMODE",
        "ZEXP_METERINGMODE",
        "ZEXP_EXPOSURECOMPENSATION",
        "ZEXP_COLORSPACE",
        "ZEXP_EXPOSUREMODE",
        "ZCAMERA_SOFTWARE",
        "ZCAMERA_SERIAL",
        "ZCAMERA_OWNER",
        "ZPIXELWIDTH",
        "ZPIXELHEIGHT",
    ];
    const LAYER_A: &[&str] = &[
        "ZLENSOPTICCENTERENABLED",
        "ZLEVELSTARGETENABLE",
        "ZRETOUCHINGBLEMISHREMOVALAMOUNT",
        "ZRETOUCHINGFACESCULPTINGCONTOURING",
        "ZRETOUCHINGSKINEVENINGAMOUNT",
        "ZRETOUCHINGOPACITY",
        "ZRETOUCHINGINCLUDENECKINBLEMISHREMOVAL",
        "ZRETOUCHINGINCLUDENECKINSKINEVENING",
    ];
    const META_A: &[&str] = &["ZCONTACT_CREATOR", "ZIMAGE_DATECREATED"];

    fn rows(path: &Path, table: &str) -> Vec<HashMap<String, Option<String>>> {
        crate::c1catalog::export_table_rows(path, table).unwrap()
    }

    #[test]
    fn exported_document_points_to_its_root_collection() {
        let mut import = C1Import::open(&fixture()).unwrap();
        let collections = &mut import.catalog.collections;
        let root_pk = collections.iter().map(|collection| collection.pk).max().unwrap() + 1;
        let root = collections.iter_mut().find(|collection| collection.parent.is_none()).unwrap();
        let original_pk = root.pk;
        root.pk = root_pk;
        for collection in collections {
            if collection.parent == Some(original_pk) {
                collection.parent = Some(root_pk);
            }
        }

        let dir = temp_dir("root-collection");
        let export = export_session(&import, &dir.join("Root.cosessiondb")).unwrap();
        let document = rows(&export.database, "ZDOCUMENTCONTENT");
        assert_eq!(document.len(), 1);
        let referenced_pk = row_i64(&document[0], "ZROOTCOLLECTION").unwrap();
        assert_eq!(referenced_pk, root_pk);
        let collections = rows(&export.database, "ZCOLLECTION");
        let root = collections.iter().find(|row| row_i64(row, "Z_PK") == Some(referenced_pk)).unwrap();
        assert_eq!(row_i64(root, "Z_ENT"), Some(7));
        assert_eq!(row_i64(root, "ZPARENT"), Some(-1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exported_variant_references_resolve_to_its_rebuilt_layers() {
        let (dir, _, export) = export_fixture("layer-references");
        let source_variants = rows(&fixture(), "ZVARIANT");
        let exported_variants = rows(&export.database, "ZVARIANT");
        let exported_layers = rows(&export.database, "ZVARIANTLAYER");
        let mut remapped = false;
        assert!(exported_variants.len() > 1);
        for variant in &exported_variants {
            let pk = row_i64(variant, "Z_PK").unwrap();
            let source = source_variants.iter().find(|row| row_i64(row, "Z_PK") == Some(pk)).unwrap();
            for (column, entity) in [
                ("ZADJUSTMENTLAYER", 17),
                ("ZDEFAULTLAYER", 18),
                ("ZCOMBINEDSETTINGS", 20),
            ] {
                let layer_pk = row_i64(variant, column).unwrap();
                remapped |= row_i64(source, column).is_some_and(|source_pk| source_pk != layer_pk);
                let layer = exported_layers.iter()
                    .find(|row| row_i64(row, "Z_PK") == Some(layer_pk)).unwrap();
                assert_eq!(row_i64(layer, "ZVARIANT"), Some(pk), "{column}");
                assert_eq!(row_i64(layer, "Z_ENT"), Some(entity), "{column}");
            }
        }
        assert!(remapped, "fixture must exercise changed layer IDs");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn same_value(a: Option<&str>, b: Option<&str>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a == b
                    || matches!(
                        (a.trim().parse::<f64>(), b.trim().parse::<f64>()),
                        (Ok(a), Ok(b)) if (a - b).abs() < 1e-4
                    )
            }
            _ => false,
        }
    }

    fn row_by<'a>(
        rows: &'a [HashMap<String, Option<String>>],
        variant: i64,
        ent: i64,
    ) -> Option<&'a HashMap<String, Option<String>>> {
        rows.iter().find(|row| {
            row_i64(row, "ZVARIANT") == Some(variant) && row_i64(row, "Z_ENT") == Some(ent)
        })
    }

    fn source_meta<'a>(
        layers: &'a [HashMap<String, Option<String>>],
        metadata: &'a [HashMap<String, Option<String>>],
        variant: i64,
    ) -> Option<&'a HashMap<String, Option<String>>> {
        let meta_pk = row_i64(row_by(layers, variant, 18)?, "ZMETADATA")?;
        metadata.iter().find(|row| row_i64(row, "Z_PK") == Some(meta_pk))
    }

    fn export_fixture(tag: &str) -> (PathBuf, PathBuf, SessionExport) {
        let source = C1Import::open(&fixture()).unwrap();
        let dir = temp_dir(tag);
        let target = dir.join("ACols.cosessiondb");
        let export = export_session(&source, &target).unwrap();
        (dir, target, export)
    }

    #[test]
    fn a_columns_first_eight_round_trip() {
        let source = C1Import::open(&fixture()).unwrap();
        let dir = temp_dir("a-first-eight");
        let target = dir.join("First8.cosessiondb");
        let export = export_session(&source, &target).unwrap();
        let back = C1Import::open(&target).unwrap();
        assert_eq!(back.images().len(), source.images().len());
        assert_eq!(back.variants().len(), source.variants().len());

        let src_layers = rows(&fixture(), "ZVARIANTLAYER");
        let out_layers = rows(&export.database, "ZVARIANTLAYER");
        for variant in source.variants() {
            for ent in [17, 18, 20] {
                let a = row_by(&src_layers, variant.pk, ent).unwrap();
                let b = row_by(&out_layers, variant.pk, ent)
                    .unwrap_or_else(|| panic!("layer {}/{ent} missing", variant.pk));
                for column in [
                    "ZGEOMETRYKEYSTONEFOCALLENGTH",
                    "ZLENSOPTICCENTER",
                    "ZLENSOPTICCENTERENABLED",
                    "ZLEVELSTARGETENABLE",
                    "ZRETOUCHINGBLEMISHREMOVALAMOUNT",
                    "ZRETOUCHINGFACESCULPTINGCONTOURING",
                    "ZRETOUCHINGSKINEVENINGAMOUNT",
                    "ZRETOUCHINGOPACITY",
                ] {
                    assert!(
                        same_value(row_text(a, column), row_text(b, column)),
                        "{column} variant {} ent {ent}",
                        variant.pk
                    );
                }
            }
        }

        for column in ["ZGEOMETRYKEYSTONEFOCALLENGTH", "ZLENSOPTICCENTER"] {
            assert!(src_layers.iter().any(|row| row_text(row, column).is_some()));
            for layer in out_layers.iter().filter(|layer| row_i64(layer, "Z_ENT") == Some(23)) {
                assert_eq!(row_text(layer, column), None);
            }
        }
        let out_variants = rows(&export.database, "ZVARIANT");
        let source_variants = rows(&fixture(), "ZVARIANT");
        assert!(source_variants.iter().any(|row| row_text(row, "ZVARIANTUUID").is_some()));
        for out in &out_variants {
            let pk = row_i64(out, "Z_PK").unwrap();
            let original = source_variants.iter().find(|row| row_i64(row, "Z_PK") == Some(pk)).unwrap();
            assert_eq!(row_text(out, "ZVARIANTUUID"), row_text(original, "ZVARIANTUUID"));
        }
        assert!(export_session(&source, &target).is_err(), "overwrite must be refused");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_columns_all_23_round_trip_and_retouch_rows() {
        let source = C1Import::open(&fixture()).unwrap();
        let dir = temp_dir("a-all");
        let target = dir.join("AllA.cosessiondb");
        let export = export_session(&source, &target).unwrap();
        let _ = C1Import::open(&target).unwrap();

        let src_images = rows(&fixture(), "ZIMAGE");
        let out_images = rows(&export.database, "ZIMAGE");
        for image in source.images() {
            let a = src_images.iter().find(|row| row_i64(row, "Z_PK") == Some(image.pk)).unwrap();
            let b = out_images.iter().find(|row| row_i64(row, "Z_PK") == Some(image.pk)).unwrap();
            for column in IMAGE_A {
                assert!(same_value(row_text(a, column), row_text(b, column)), "{column}");
            }
        }

        let src_layers = rows(&fixture(), "ZVARIANTLAYER");
        let src_meta = rows(&fixture(), "ZVARIANTMETADATA");
        let out_meta = rows(&export.database, "ZVARIANTMETADATA");
        let out_layers = rows(&export.database, "ZVARIANTLAYER");
        for variant in source.variants() {
            for ent in [17, 18, 20] {
                let a = row_by(&src_layers, variant.pk, ent).unwrap();
                let b = row_by(&out_layers, variant.pk, ent).unwrap();
                for column in LAYER_A {
                    assert!(
                        same_value(row_text(a, column), row_text(b, column)),
                        "{column} variant {} ent {ent}",
                        variant.pk
                    );
                }
            }
            let a = source_meta(&src_layers, &src_meta, variant.pk).unwrap();
            let b = out_meta.iter().find(|row| row_i64(row, "Z_PK") == Some(variant.pk)).unwrap();
            for column in META_A {
                assert!(same_value(row_text(a, column), row_text(b, column)), "{column}");
            }
        }

        // Per-face retouch rows: raw ZFACE, every mask slot and NULL amounts.
        let src_retouch = rows(&fixture(), "ZRETOUCHINGLAYER");
        let out_retouch = rows(&export.database, "ZRETOUCHINGLAYER");
        assert_eq!(out_retouch.len(), src_retouch.len());
        let columns: Vec<&str> = RETOUCH_COLUMNS
            .iter()
            .map(|column| column.split_whitespace().next().unwrap())
            .collect();
        for (a, b) in src_retouch.iter().zip(&out_retouch) {
            for column in &columns {
                if *column == "Z_PK" {
                    continue;
                }
                assert!(
                    same_value(row_text(a, column), row_text(b, column)),
                    "{column} in retouch row"
                );
            }
            assert!(row_text(b, "ZBLEMISHREMOVALAMOUNT").is_none() || row_text(a, "ZBLEMISHREMOVALAMOUNT").is_some());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Run with `AURORA_A_EXPORT_PATH=<dir>` to leave one export for the
    /// external Python pointer check; normal test runs skip it.
    #[test]
    fn writes_fixture_export_when_requested() {
        let Some(path) = std::env::var_os("AURORA_A_EXPORT_PATH") else { return };
        let target = PathBuf::from(path);
        if target.exists() {
            std::fs::remove_dir_all(&target).unwrap();
        }
        let source = C1Import::open(&fixture()).unwrap();
        export_session(&source, &target).unwrap();
    }

    fn same_edits(a: &C1VariantSettings, b: &C1VariantSettings) {
        assert_eq!(a.exposure, b.exposure);
        assert_eq!(a.contrast, b.contrast);
        assert_eq!(a.brightness, b.brightness);
        assert_eq!(a.saturation, b.saturation);
        assert_eq!(a.white_balance_gains, b.white_balance_gains);
        assert_eq!(a.levels_shadow, b.levels_shadow);
        assert_eq!(a.levels_highlight, b.levels_highlight);
        assert_eq!(a.levels_midtone, b.levels_midtone);
        assert_eq!(a.levels_target_shadow, b.levels_target_shadow);
        assert_eq!(a.levels_target_highlight, b.levels_target_highlight);
        assert_eq!(a.clarity, b.clarity);
        assert_eq!(a.clarity_structure, b.clarity_structure);
        assert_eq!(a.clarity_method, b.clarity_method);
        assert_eq!(a.highlights_recovery, b.highlights_recovery);
        assert_eq!(a.shadows_recovery, b.shadows_recovery);
        assert_eq!(a.whites_recovery, b.whites_recovery);
        assert_eq!(a.blacks_recovery, b.blacks_recovery);
        assert_eq!(a.sharpening, b.sharpening);
        assert_eq!(a.noise_reduction.luminance, b.noise_reduction.luminance);
        assert_eq!(a.noise_reduction.details, b.noise_reduction.details);
        assert_eq!(a.noise_reduction.color, b.noise_reduction.color);
        assert_eq!(a.noise_reduction.method, b.noise_reduction.method);
        assert_eq!(a.dehaze_amount, b.dehaze_amount);
        assert_eq!(a.crop_px, b.crop_px);
        assert_eq!(a.crop_outside_image, b.crop_outside_image);
        assert_eq!(a.crop_aspect_ratio, b.crop_aspect_ratio);
        assert_eq!(a.rotation, b.rotation);
        assert_eq!(a.icc_profile, b.icc_profile);
        assert_eq!(a.film_curve, b.film_curve);
    }
}

/// `.comask` copy path: byte-identical, and the three counters.
#[cfg(test)]
mod mask_copy_tests {
    use super::copy_masks_from;
    use std::path::Path;

    fn temp(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("comask-copy-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn copies_the_sidecar_under_the_same_relative_path() {
        let root = temp("copy");
        let lam = root.join("source").join("Adjustments").join("LAM").join("2026/9/13/0");
        std::fs::create_dir_all(&lam).unwrap();
        let bytes: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        std::fs::write(lam.join("_DSC5673.NEF.comask"), &bytes).unwrap();

        let out = root.join("session");
        let copy = copy_masks_from(
            &root.join("source").join("Adjustments").join("LAM"),
            &out,
            &["_dsc5673.nef".to_string()],
        )
        .unwrap();
        assert_eq!((copy.copied, copy.missing, copy.absent), (1, 0, 0));
        let copied = out.join("Adjustments/LAM/2026/9/13/0/_DSC5673.NEF.comask");
        assert!(copied.is_file());
        assert_eq!(std::fs::read(&copied).unwrap(), bytes);
    }

    #[test]
    fn missing_and_absent_are_counted_not_faked() {
        let root = temp("missing");
        let lam = root.join("source").join("Adjustments").join("LAM");
        std::fs::create_dir_all(&lam).unwrap();
        let copy = copy_masks_from(&lam, &root.join("session"), &["B.NEF".to_string()]).unwrap();
        assert_eq!((copy.copied, copy.missing, copy.absent), (0, 1, 0));
        assert!(!Path::new(&root.join("session").join("Adjustments/LAM/B.NEF.comask")).exists());

        let copy = copy_masks_from(&root.join("absent"), &root.join("session2"), &["C.NEF".to_string()]).unwrap();
        assert_eq!((copy.copied, copy.missing, copy.absent), (0, 0, 1));
    }
}
