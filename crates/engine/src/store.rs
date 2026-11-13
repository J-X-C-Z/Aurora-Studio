//! On-disk persistence for a Lumen catalog folder.
//!
//! Everything the user changes lives in `<folder>/.lumen`:
//!
//! * `catalog.json` - one entry per image (rating, tag, keywords, variant).
//! * `adjust/<hash>.json` - the [`AdjustmentSet`] of one image, keyed by a
//!   stable hash of the image's path relative to the folder.
//!
//! Nothing outside `.lumen` is written except an XMP sidecar, and only through
//! [`Store::save_xmp`] or [`Store::save_metadata`]. Rating, label and keywords
//! live in both places: `save_metadata` keeps them in step, and loading prefers
//! the sidecar when somebody edited it after Lumen's last write. Adjustments
//! never touch XMP. Every write is atomic (write `.tmp`, then rename) so a
//! crash cannot leave a half-written file.

use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adjust::AdjustmentSet;
use crate::catalog::ColorTag;

/// Version of the on-disk format this build writes.
pub const SCHEMA_VERSION: u32 = 2;

/// Name of the hidden directory inside the catalog folder.
pub const LUMEN_DIR: &str = ".lumen";

/// Everything that can go wrong in the store.
#[derive(Debug)]
pub enum StoreError {
    /// The file on disk was written by a newer version of Lumen. Read-only.
    Newer { found: u32 },
    /// A relative path was absolute, empty, or escaped the catalog folder.
    InvalidPath(String),
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Newer { found } => {
                write!(f, "catalog was written by a newer Lumen (schema {found}); read-only")
            }
            StoreError::InvalidPath(p) => write!(f, "invalid image path: {p}"),
            StoreError::Io(e) => write!(f, "store I/O error: {e}"),
            StoreError::Json(e) => write!(f, "store JSON error: {e}"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StoreError::Io(e) => Some(e),
            StoreError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError::Io(e)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        StoreError::Json(e)
    }
}

/// One image's metadata as stored on disk.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// Path of the image relative to the catalog folder, using `/` separators.
    pub rel_path: String,
    /// 0 ..= 5 stars; -1 means rejected.
    pub rating: i8,
    /// Index into `catalog::ColorTag::ALL` (0 = none, 1 = red, ...).
    pub color_tag: i8,
    pub keywords: Vec<String>,
    /// Variant number within the file; 1 is the original.
    pub variant: u32,
    /// Unix seconds of the last change.
    pub updated_at: i64,
    /// Soft-deleted; the file is still on disk.
    pub trashed: bool,
    /// Full identity and source path, including external and offline references.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<crate::ImageInfo>,
}

/// Which copy of the metadata [`Store::load_metadata`] returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataSource {
    /// `.lumen/catalog.json`, Lumen's own record.
    Index,
    /// The `.xmp` sidecar beside the image.
    Xmp,
}

/// Rating, label and keywords of one image, with where they came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub rating: i8,
    pub label: String,
    pub keywords: Vec<String>,
    pub source: MetadataSource,
}

/// The whole index of one catalog folder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub schema_version: u32,
    pub images: Vec<Entry>,
    #[serde(default)]
    pub(crate) folders: Vec<FolderRecord>,
    #[serde(default)]
    pub(crate) excluded: Vec<PathBuf>,
    #[serde(default)]
    pub(crate) excluded_variants: Vec<ExcludedVariant>,
    #[serde(default)]
    pub(crate) next_id: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct FolderRecord {
    pub id: u64,
    pub path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ExcludedVariant {
    pub path: PathBuf,
    pub variant: u32,
    pub key: String,
}

impl Default for Catalog {
    fn default() -> Self {
        Catalog { schema_version: SCHEMA_VERSION, images: Vec::new(), folders: Vec::new(), excluded: Vec::new(), excluded_variants: Vec::new(), next_id: 1 }
    }
}

impl Catalog {
    pub fn entry(&self, rel_path: &str) -> Option<&Entry> {
        self.images.iter().find(|e| e.rel_path == rel_path)
    }

    pub fn upsert(&mut self, entry: Entry) {
        match self.images.iter_mut().find(|e| e.rel_path == entry.rel_path) {
            Some(slot) => *slot = entry,
            None => self.images.push(entry),
        }
    }

    pub fn remove(&mut self, rel_path: &str) -> bool {
        let before = self.images.len();
        self.images.retain(|e| e.rel_path != rel_path);
        self.images.len() != before
    }
}

/// A folder of images plus everything Lumen remembers about them.
pub struct Store {
    root: PathBuf,
}

/// Shape of one `adjust/<hash>.json` file on disk.
#[derive(Serialize)]
struct AdjustFile<'a> {
    schema_version: u32,
    adjustments: &'a AdjustmentSet,
}

impl Store {
    /// Opens (and creates) the store of one catalog folder.
    pub fn open(folder: &Path) -> Result<Self> {
        let root = folder.join(LUMEN_DIR);
        fs::create_dir_all(root.join("adjust"))
            .with_context(|| format!("creating store directory {}", root.join("adjust").display()))?;
        Ok(Store { root })
    }

    /// The `.lumen` directory of this store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn index_path(&self) -> PathBuf {
        self.root.join("catalog.json")
    }

    fn adjust_path(&self, rel_path: &str) -> Result<PathBuf> {
        validate_rel_path(rel_path)?;
        Ok(self.root.join("adjust").join(format!("{}.json", fnv1a_hex(rel_path))))
    }

    /// Reads the index. A missing file is an empty catalog; unreadable or
    /// half-written records are skipped one by one.
    pub fn load_index(&self) -> Result<Catalog> {
        let path = self.index_path();
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Catalog::default()),
            Err(e) => return Err(StoreError::Io(e).into()),
        };
        let value: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Ok(Catalog::default()),
        };
        let version = value.get("schema_version").and_then(Value::as_u64).unwrap_or(1) as u32;
        if version > SCHEMA_VERSION {
            return Err(StoreError::Newer { found: version }.into());
        }
        let mut images = Vec::new();
        if let Some(items) = value.get("images").and_then(Value::as_array) {
            for item in items {
                if let Some(entry) = parse_entry(item) {
                    images.push(entry);
                }
            }
        }
        let mut catalog = Catalog { images, ..Catalog::default() };
        if let Some(folders) = value.get("folders") {
            catalog.folders = serde_json::from_value(folders.clone())?;
        }
        if let Some(excluded) = value.get("excluded") {
            catalog.excluded = serde_json::from_value(excluded.clone())?;
        }
        if let Some(excluded) = value.get("excluded_variants") {
            catalog.excluded_variants = serde_json::from_value(excluded.clone())?;
        }
        catalog.next_id = value.get("next_id").and_then(Value::as_u64).unwrap_or(1).max(1);
        Ok(catalog)
    }

    /// Writes the index atomically. Refuses to write over a newer catalog.
    pub fn save_index(&self, catalog: &Catalog) -> Result<()> {
        self.guard_newer_index()?;
        self.backup_schema_one_index()?;
        let bytes = serde_json::to_vec_pretty(catalog)?;
        write_atomic(&self.index_path(), &bytes)
    }

    /// Commit library metadata together; restore prior bytes on any I/O error.
    /// shortcut: rollback is process-local; a crash journal is needed for crash-atomic multi-file operations.
    pub(crate) fn commit_library(
        &self,
        catalog: &Catalog,
        adjustments: &[(String, AdjustmentSet)],
        removed: &[String],
        collections: Option<&[crate::UserCollection]>,
    ) -> Result<()> {
        self.guard_newer_index()?;
        self.backup_schema_one_index()?;
        let mut writes = Vec::new();
        for (key, adjustments) in adjustments {
            let bytes = serde_json::to_vec_pretty(&AdjustFile { schema_version: SCHEMA_VERSION, adjustments })?;
            writes.push((self.adjust_path(key)?, Some(bytes)));
        }
        for key in removed {
            writes.push((self.adjust_path(key)?, None));
        }
        if !removed.is_empty() {
            for entry in self.load_index()?.images.iter().filter(|e| removed.contains(&e.rel_path)) {
                if let Some(info) = &entry.info {
                    writes.push((self.preview_path(info.id), None));
                }
            }
        }
        if let Some(collections) = collections {
            self.guard_collections()?;
            writes.push((self.collections_path(), Some(serde_json::to_vec_pretty(collections)?)));
        }
        writes.push((self.index_path(), Some(serde_json::to_vec_pretty(catalog)?)));
        let previous: Vec<_> = writes.iter().map(|(path, _)| match fs::read(path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }).collect::<std::io::Result<_>>()?;
        for (n, (path, bytes)) in writes.iter().enumerate() {
            if let Err(err) = restore_file(path, bytes.as_deref()) {
                let mut rollback = Vec::new();
                for i in (0..n).rev() {
                    if let Err(e) = restore_file(&writes[i].0, previous[i].as_deref()) {
                        rollback.push(format!("{}: {e}", writes[i].0.display()));
                    }
                }
                if !rollback.is_empty() {
                    anyhow::bail!("{err}; metadata rollback failed: {}", rollback.join("; "));
                }
                return Err(err);
            }
        }
        Ok(())
    }

    fn preview_path(&self, id: crate::ImageId) -> PathBuf {
        self.root.join("previews").join(format!("{}.png", id.0))
    }

    pub(crate) fn save_library_preview(&self, id: crate::ImageId, frame: &crate::RgbaImage) -> Result<()> {
        use image::ImageEncoder;
        let mut bytes = Vec::new();
        image::codecs::png::PngEncoder::new(&mut bytes).write_image(
            frame.pixels.as_ref(), frame.width, frame.height, image::ExtendedColorType::Rgba8,
        )?;
        write_atomic(&self.preview_path(id), &bytes)
    }

    pub(crate) fn load_library_preview(&self, id: crate::ImageId) -> Result<image::RgbaImage> {
        let bytes = fs::read(self.preview_path(id)).context("original is offline and no saved preview is available")?;
        Ok(image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)?.to_rgba8())
    }

    /// Reads the adjustments of one image. A missing or damaged file yields
    /// the default set rather than an error.
    pub fn load_adjustments(&self, rel_path: &str) -> Result<AdjustmentSet> {
        let path = self.adjust_path(rel_path)?;
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(AdjustmentSet::default()),
            Err(e) => return Err(StoreError::Io(e).into()),
        };
        let value: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return Ok(AdjustmentSet::default()),
        };
        if let Some(version) = value.get("schema_version").and_then(Value::as_u64) {
            if version as u32 > SCHEMA_VERSION {
                return Err(StoreError::Newer { found: version as u32 }.into());
            }
        }
        let adjustments = value.get("adjustments").unwrap_or(&value);
        Ok(serde_json::from_value::<AdjustmentSet>(adjustments.clone()).unwrap_or_default())
    }

    /// Writes the adjustments of one image atomically.
    pub fn save_adjustments(&self, rel_path: &str, adjustments: &AdjustmentSet) -> Result<()> {
        self.guard_newer_index()?;
        let path = self.adjust_path(rel_path)?;
        let file = AdjustFile { schema_version: SCHEMA_VERSION, adjustments };
        let bytes = serde_json::to_vec_pretty(&file)?;
        write_atomic(&path, &bytes)
    }

    /// `<stem>.xmp` next to the image, never inside `.lumen`.
    pub fn xmp_path(&self, rel_path: &str) -> Result<PathBuf> {
        validate_rel_path(rel_path)?;
        let folder = self.root.parent().ok_or_else(|| StoreError::InvalidPath(rel_path.to_string()))?;
        Ok(crate::XmpMeta::sidecar_path(&folder.join(rel_path)))
    }

    /// Reads the XMP sidecar of one image; `Ok(None)` when there is none.
    pub fn load_xmp(&self, rel_path: &str) -> Result<Option<crate::XmpMeta>> {
        crate::XmpMeta::load(&self.xmp_path(rel_path)?)
    }

    /// Writes rating, label and keywords to the image's XMP sidecar. Callers
    /// opt in; nothing is ever written to XMP implicitly.
    pub fn save_xmp(&self, rel_path: &str, meta: &crate::XmpMeta) -> Result<()> {
        let image = {
            validate_rel_path(rel_path)?;
            let folder = self.root.parent().ok_or_else(|| StoreError::InvalidPath(rel_path.to_string()))?;
            folder.join(rel_path)
        };
        if !image.is_file() {
            return Err(StoreError::InvalidPath(rel_path.to_string()).into());
        }
        meta.save(&image)?;
        Ok(())
    }

    /// Writes rating, label and keywords to the index and the sidecar in one
    /// go. The sidecar goes first and the index last; if either step fails the
    /// sidecar is put back, so a failure never leaves the two disagreeing.
    pub fn save_metadata(
        &self,
        rel_path: &str,
        rating: i8,
        label: &str,
        keywords: &[String],
    ) -> Result<()> {
        self.guard_newer_index()?;
        let rating = rating.clamp(-1, 5);
        let label = label.trim();
        let sidecar = self.xmp_path(rel_path)?;
        let previous = match fs::read(&sidecar) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(StoreError::Io(e).into()),
        };

        let meta =
            crate::XmpMeta { rating, label: label.to_string(), keywords: keywords.to_vec() };
        // Refuses missing images and Capture One bundles before anything is written.
        self.save_xmp(rel_path, &meta)?;

        let index = (|| -> Result<()> {
            let mut catalog = self.load_index()?;
            let mut entry = catalog.entry(rel_path).cloned().unwrap_or_else(|| Entry {
                rel_path: rel_path.to_string(),
                variant: 1,
                ..Entry::default()
            });
            entry.rating = rating;
            entry.color_tag = tag_from_label(label);
            entry.keywords = keywords.to_vec();
            entry.updated_at = now_secs();
            catalog.upsert(entry);
            self.save_index(&catalog)
        })();

        if let Err(err) = index {
            match previous {
                Some(bytes) => {
                    let _ = write_atomic(&sidecar, &bytes);
                }
                None => {
                    let _ = fs::remove_file(&sidecar);
                }
            }
            return Err(err);
        }
        Ok(())
    }

    /// Reads rating, label and keywords. A missing or damaged sidecar leaves
    /// the index untouched and usable.
    ///
    /// Conflict rule: when the two disagree and the sidecar was modified after
    /// Lumen last wrote the index (`Entry::updated_at`), the sidecar wins —
    /// that is the only way an edit made in Capture One or Lightroom reaches
    /// Lumen. Otherwise the index wins, because Lumen's own write is the newer
    /// intent.
    pub fn load_metadata(&self, rel_path: &str) -> Result<Metadata> {
        validate_rel_path(rel_path)?;
        let entry = self.load_index()?.entry(rel_path).cloned().unwrap_or_else(|| Entry {
            rel_path: rel_path.to_string(),
            variant: 1,
            ..Entry::default()
        });
        let index = Metadata {
            rating: entry.rating,
            label: tag_label(entry.color_tag),
            keywords: entry.keywords.clone(),
            source: MetadataSource::Index,
        };
        let xmp = match self.load_xmp(rel_path) {
            Ok(Some(meta)) => meta,
            _ => return Ok(index),
        };
        if xmp.rating == index.rating && xmp.label == index.label && xmp.keywords == index.keywords
        {
            return Ok(index);
        }
        if self.sidecar_newer_than(rel_path, entry.updated_at) {
            return Ok(Metadata {
                rating: xmp.rating,
                label: xmp.label,
                keywords: xmp.keywords,
                source: MetadataSource::Xmp,
            });
        }
        Ok(index)
    }

    fn sidecar_newer_than(&self, rel_path: &str, updated_at: i64) -> bool {
        let Ok(path) = self.xmp_path(rel_path) else { return false };
        let Ok(modified) = fs::metadata(path).and_then(|m| m.modified()) else { return false };
        modified
            .duration_since(UNIX_EPOCH)
            .map(|age| age > Duration::from_secs(updated_at.max(0) as u64))
            .unwrap_or(false)
    }

    /// Drops everything Lumen knows about one image: its index entry and its
    /// adjustment file.
    pub fn forget(&self, rel_path: &str) -> Result<()> {
        self.guard_newer_index()?;
        let path = self.adjust_path(rel_path)?;
        let mut catalog = self.load_index()?;
        let removed = catalog.remove(rel_path);
        if removed {
            self.save_index(&catalog)?;
        }
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(StoreError::Io(e).into()),
        }
        Ok(())
    }

    fn collections_path(&self) -> PathBuf {
        self.root.join("collections.json")
    }

    /// Stores the user's albums, projects and groups as one JSON array.
    pub fn save_user_collections(&self, collections: &[crate::UserCollection]) -> Result<()> {
        self.guard_newer_index()?;
        self.guard_collections()?;
        let bytes = serde_json::to_vec_pretty(collections)?;
        write_atomic(&self.collections_path(), &bytes)
    }

    /// Reads them back; a missing or damaged file yields an empty list.
    pub fn load_user_collections(&self) -> Vec<crate::UserCollection> {
        let Ok(text) = fs::read_to_string(self.collections_path()) else { return Vec::new() };
        serde_json::from_str(&text).unwrap_or_default()
    }

    fn guard_collections(&self) -> Result<()> {
        let bytes = match fs::read(self.collections_path()) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).context("reading existing collections before mutation"),
        };
        serde_json::from_slice::<Vec<crate::UserCollection>>(&bytes).context("existing collections are corrupt; refusing to overwrite")?;
        Ok(())
    }

    /// Preserve the original index, flushed before the first schema-2 replacement.
    /// Adjustment files can upgrade independently; this backup alone is not a downgrade path.
    fn backup_schema_one_index(&self) -> Result<()> {
        use std::io::Write;
        let bytes = match fs::read(self.index_path()) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let value: Value = serde_json::from_slice(&bytes)?;
        if value.get("schema_version").and_then(Value::as_u64).unwrap_or(1) != 1 { return Ok(()); }
        let backup = self.root.join("catalog.schema1.backup.json");
        let mut file = match fs::OpenOptions::new().write(true).create_new(true).open(&backup) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let previous: Value = serde_json::from_slice(&fs::read(&backup)?)
                    .context("existing schema-1 backup is corrupt")?;
                anyhow::ensure!(previous.is_object() && previous.get("images").is_some_and(Value::is_array)
                    && previous.get("schema_version").and_then(Value::as_u64).unwrap_or(1) == 1,
                    "existing schema-1 backup is invalid");
                return Ok(());
            }
            Err(e) => return Err(e).context("creating schema-1 catalog backup"),
        };
        let saved = file.write_all(&bytes).and_then(|_| file.sync_all());
        drop(file);
        if let Err(err) = saved {
            fs::remove_file(&backup).context("removing incomplete schema-1 backup")?;
            return Err(err).context("flushing schema-1 catalog backup");
        }
        Ok(())
    }

    /// Errors when the catalog on disk was written by a newer Lumen.
    pub(crate) fn guard_newer_index(&self) -> Result<()> {
        let path = self.index_path();
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).context("reading existing catalog before mutation"),
        };
        let value: Value = serde_json::from_str(&text).context("existing catalog is corrupt; refusing to overwrite")?;
        if let Some(version) = value.get("schema_version").and_then(Value::as_u64) {
            if version as u32 > SCHEMA_VERSION {
                return Err(StoreError::Newer { found: version as u32 }.into());
            }
        }
        anyhow::ensure!(value.is_object() && value.get("images").is_some_and(Value::is_array), "existing catalog has an invalid index shape");
        anyhow::ensure!(value.get("schema_version").is_none_or(|v| v.as_u64().is_some_and(|n| n >= 1)), "invalid catalog schema version");
        anyhow::ensure!(value["images"].as_array().unwrap().iter().all(|item| parse_entry(item).is_some()), "existing catalog contains corrupt image records");
        Ok(())
    }
}

fn restore_file(path: &Path, bytes: Option<&[u8]>) -> Result<()> {
    match bytes {
        Some(bytes) => write_atomic(path, bytes),
        None => match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        },
    }
}

/// Reads one index record, filling in defaults and clamping numbers.
/// Returns `None` for a record that cannot be used at all.
fn parse_entry(item: &Value) -> Option<Entry> {
    let object = item.as_object()?;
    let rel_path = object.get("rel_path").and_then(Value::as_str)?.to_string();
    validate_rel_path(&rel_path).ok()?;
    let rating = object.get("rating").and_then(Value::as_i64).unwrap_or(0).clamp(-1, 5) as i8;
    let color_tag = object.get("color_tag").and_then(Value::as_i64).unwrap_or(0).clamp(0, 7) as i8;
    let keywords = object
        .get("keywords")
        .and_then(Value::as_array)
        .map(|items| {
            items.iter().filter_map(Value::as_str).map(str::to_string).collect::<Vec<String>>()
        })
        .unwrap_or_default();
    let variant = object.get("variant").and_then(Value::as_u64).unwrap_or(1).max(1) as u32;
    let updated_at = object.get("updated_at").and_then(Value::as_i64).unwrap_or(0);
    let trashed = object.get("trashed").and_then(Value::as_bool).unwrap_or(false);
    let info = object.get("info").filter(|v| !v.is_null())
        .map(|v| serde_json::from_value(v.clone())).transpose().ok()?;
    Some(Entry { rel_path, rating, color_tag, keywords, variant, updated_at, trashed, info })
}

/// The `xmp:Label` name of a stored colour tag; empty for none or a bad index.
fn tag_label(color_tag: i8) -> String {
    ColorTag::ALL
        .get(color_tag.max(0) as usize)
        .map(|tag| tag.label().to_string())
        .unwrap_or_default()
}

/// Index of a colour tag given its XMP label; unknown labels read as none.
fn tag_from_label(label: &str) -> i8 {
    ColorTag::ALL.iter().position(|tag| tag.label().eq_ignore_ascii_case(label)).unwrap_or(0) as i8
}

fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Rejects absolute paths and anything that could escape the catalog folder.
fn validate_rel_path(rel_path: &str) -> Result<()> {
    let path = Path::new(rel_path);
    let bad = if rel_path.is_empty() || path.is_absolute() {
        true
    } else {
        path.components()
            .any(|c| matches!(c, Component::ParentDir | Component::RootDir | Component::Prefix(_)))
    };
    if bad {
        return Err(StoreError::InvalidPath(rel_path.to_string()).into());
    }
    Ok(())
}

/// Stable FNV-1a 64-bit hash of a relative path, as 16 hex digits.
fn fnv1a_hex(value: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Writes bytes to a `.tmp` sibling and renames it over the target, so a
/// reader never sees a partial file.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir)?;
    let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
    let tmp = dir.join(format!("{file_name}.tmp"));
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(StoreError::Io(e).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    /// A fresh temporary folder, removed when the guard is dropped.
    struct TempFolder(PathBuf);

    impl TempFolder {
        fn new() -> Self {
            let id = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir()
                .join(format!("lumen-store-test-{}-{id}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            TempFolder(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempFolder {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn sample_entry() -> Entry {
        Entry {
            rel_path: "shoot/img_001.cr3".into(),
            rating: 4,
            color_tag: 2,
            keywords: vec!["wedding".into(), "hero".into()],
            variant: 1,
            updated_at: 1_700_000_000,
            trashed: false,
            info: None,
        }
    }

    fn sample_adjustments() -> AdjustmentSet {
        let mut set = AdjustmentSet::default();
        set.exposure.exposure = 0.75;
        set.white_balance.kelvin = 6200.0;
        set.style = Some("Portrait".into());
        set
    }

    #[test]
    fn round_trips_index_and_adjustments() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();

        let mut catalog = Catalog::default();
        catalog.upsert(sample_entry());
        store.save_index(&catalog).unwrap();
        store.save_adjustments("shoot/img_001.cr3", &sample_adjustments()).unwrap();

        let loaded = store.load_index().unwrap();
        assert_eq!(loaded.schema_version, SCHEMA_VERSION);
        assert_eq!(loaded.images.len(), 1);
        assert_eq!(loaded.images[0], sample_entry());
        assert_eq!(store.load_adjustments("shoot/img_001.cr3").unwrap(), sample_adjustments());
    }

    #[test]
    fn round_trips_user_collections() {
        use crate::catalog::{CollectionKind, ImageId, SmartRule, UserCollection};
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        let albums = vec![UserCollection {
            id: 1 << 40 | 3,
            parent: None,
            name: "Picks".into(),
            kind: CollectionKind::SmartAlbum,
            images: vec![ImageId(4)],
            rules: vec![SmartRule::RatingAtLeast(4), SmartRule::Edited(true)],
            match_all: true,
        }];
        store.save_user_collections(&albums).unwrap();
        assert_eq!(store.load_user_collections(), albums);
    }

    #[test]
    fn missing_or_damaged_collections_read_as_empty() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        assert!(store.load_user_collections().is_empty());
        fs::write(store.root().join("collections.json"), "{ not json").unwrap();
        assert!(store.load_user_collections().is_empty());
    }

    #[test]
    fn skips_bad_records() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        fs::write(
            store.root().join("catalog.json"),
            r#"{"schema_version":1,"images":[
                {"rel_path":"a.cr3","rating":9,"color_tag":99,"variant":0,"updated_at":"x"},
                {"no_path":true},
                {"rel_path":"../escape.cr3"},
                {"rel_path":"b.cr3","keywords":["keep",7]}
            ]}"#,
        )
        .unwrap();

        let catalog = store.load_index().unwrap();
        assert_eq!(catalog.images.len(), 2);
        assert_eq!(catalog.images[0].rel_path, "a.cr3");
        assert_eq!(catalog.images[0].rating, 5, "rating is clamped to 5 stars");
        assert_eq!(catalog.images[0].color_tag, 7, "color tag is clamped to the tag range");
        assert_eq!(catalog.images[0].variant, 1);
        assert_eq!(catalog.images[0].updated_at, 0);
        assert_eq!(catalog.images[1].keywords, vec!["keep".to_string()]);
    }

    #[test]
    fn missing_fields_get_defaults() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        fs::write(store.root().join("catalog.json"), r#"{"images":[{"rel_path":"a.cr3"}]}"#).unwrap();

        let catalog = store.load_index().unwrap();
        let entry = &catalog.images[0];
        assert_eq!(entry.rating, 0);
        assert_eq!(entry.color_tag, 0);
        assert!(entry.keywords.is_empty());
        assert_eq!(entry.variant, 1);
        assert_eq!(entry.updated_at, 0);
    }

    #[test]
    fn newer_schema_is_read_only() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        let path = store.root().join("catalog.json");
        let newer = r#"{"schema_version":3,"images":[]}"#;
        fs::write(&path, newer).unwrap();

        let err = store.load_index().unwrap_err();
        assert!(matches!(err.downcast_ref::<StoreError>(), Some(StoreError::Newer { found: 3 })));

        let err = store.save_index(&Catalog::default()).unwrap_err();
        assert!(matches!(err.downcast_ref::<StoreError>(), Some(StoreError::Newer { .. })));
        assert_eq!(fs::read_to_string(&path).unwrap(), newer, "newer catalog is left untouched");
    }

    #[test]
    fn atomic_write_leaves_no_tmp_files() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        let mut catalog = Catalog::default();
        catalog.upsert(sample_entry());
        store.save_index(&catalog).unwrap();
        store.save_adjustments("shoot/img_001.cr3", &sample_adjustments()).unwrap();

        for dir in [store.root().to_path_buf(), store.root().join("adjust")] {
            for entry in fs::read_dir(&dir).unwrap() {
                let name = entry.unwrap().file_name().to_string_lossy().to_string();
                assert!(!name.ends_with(".tmp"), "leftover temp file: {name}");
            }
        }
        assert!(store.root().join("catalog.json").exists());
        assert!(store.root().join("adjust").exists());
    }

    #[test]
    fn rejects_paths_that_escape() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        for bad in ["../evil.cr3", "sub/../../evil.cr3", "", "/abs/evil.cr3"] {
            let err = store.save_adjustments(bad, &AdjustmentSet::default()).unwrap_err();
            assert!(err.downcast_ref::<StoreError>().is_some(), "{bad} should be rejected");
            assert!(store.load_adjustments(bad).is_err(), "{bad} should be rejected");
        }
        assert!(fs::read_dir(store.root().join("adjust")).unwrap().next().is_none());
    }

    #[test]
    fn writes_only_inside_lumen_dir() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        let mut catalog = Catalog::default();
        catalog.upsert(sample_entry());
        store.save_index(&catalog).unwrap();
        store.save_adjustments("shoot/img_001.cr3", &sample_adjustments()).unwrap();

        let names: Vec<String> = fs::read_dir(temp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec![LUMEN_DIR.to_string()]);
    }

    #[test]
    fn forget_removes_entry_and_file() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        let mut catalog = Catalog::default();
        catalog.upsert(sample_entry());
        store.save_index(&catalog).unwrap();
        store.save_adjustments("shoot/img_001.cr3", &sample_adjustments()).unwrap();

        store.forget("shoot/img_001.cr3").unwrap();
        assert!(store.load_index().unwrap().images.is_empty());
        assert!(store.load_adjustments("shoot/img_001.cr3").unwrap().is_default());
    }

    #[test]
    fn damaged_adjustments_fall_back_to_default() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        let path = store.root().join("adjust").join(format!("{}.json", fnv1a_hex("a.cr3")));
        fs::write(&path, "{not json").unwrap();
        assert!(store.load_adjustments("a.cr3").unwrap().is_default());
    }

    #[test]
    fn xmp_round_trip_beside_the_image() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        fs::write(temp.path().join("img_001.cr3"), b"pixels").unwrap();
        assert_eq!(store.load_xmp("img_001.cr3").unwrap(), None);

        let meta = crate::XmpMeta { rating: 5, label: "Blue".into(), keywords: vec!["portrait".into()] };
        store.save_xmp("img_001.cr3", &meta).unwrap();
        assert!(temp.path().join("img_001.xmp").exists());
        assert_eq!(store.load_xmp("img_001.cr3").unwrap(), Some(meta));
        assert!(!temp.path().join("img_001.xmp.tmp").exists());
    }

    #[test]
    fn xmp_refuses_unknown_images_and_escaping_paths() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        let meta = crate::XmpMeta::default();
        assert!(store.save_xmp("../evil.cr3", &meta).is_err());
        assert!(store.save_xmp("missing.cr3", &meta).is_err());
        assert!(store.load_xmp("../evil.cr3").is_err());
    }

    fn image(temp: &TempFolder, rel_path: &str) {
        let path = temp.path().join(rel_path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"pixels").unwrap();
    }

    fn keywords(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn metadata_writes_index_and_sidecar_together() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        image(&temp, "shoot/img_001.cr3");

        store.save_metadata("shoot/img_001.cr3", 4, "Red", &keywords(&["wedding", "hero"])).unwrap();

        let entry = store.load_index().unwrap().entry("shoot/img_001.cr3").cloned().unwrap();
        assert_eq!(entry.rating, 4);
        assert_eq!(entry.color_tag, 1, "Red is tag index 1");
        assert_eq!(entry.keywords, keywords(&["wedding", "hero"]));
        assert_eq!(entry.variant, 1, "a new entry is a normal variant");
        assert!(entry.updated_at > 0);

        let sidecar =
            crate::XmpMeta { rating: 4, label: "Red".into(), keywords: keywords(&["wedding", "hero"]) };
        assert_eq!(store.load_xmp("shoot/img_001.cr3").unwrap(), Some(sidecar.clone()));

        let meta = store.load_metadata("shoot/img_001.cr3").unwrap();
        assert_eq!(meta.source, MetadataSource::Index, "equal copies need no conflict");
        assert_eq!(meta.rating, 4);
        assert_eq!(meta.label, "Red");
        assert_eq!(meta.keywords, sidecar.keywords);
    }

    #[test]
    fn sidecar_edited_alone_wins_when_newer() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        image(&temp, "shoot/img_001.cr3");
        store.save_metadata("shoot/img_001.cr3", 2, "Blue", &keywords(&["old"])).unwrap();

        // Pretend the index was written a minute ago, then edit only the XMP.
        let mut catalog = store.load_index().unwrap();
        catalog.images[0].updated_at = now_secs() - 60;
        store.save_index(&catalog).unwrap();
        let edited = crate::XmpMeta { rating: 5, label: "Green".into(), keywords: keywords(&["solo"]) };
        store.save_xmp("shoot/img_001.cr3", &edited).unwrap();

        let meta = store.load_metadata("shoot/img_001.cr3").unwrap();
        assert_eq!(meta.source, MetadataSource::Xmp);
        assert_eq!(meta.rating, 5);
        assert_eq!(meta.label, "Green");
        assert_eq!(meta.keywords, keywords(&["solo"]));
        assert_eq!(
            store.load_index().unwrap().images[0].rating,
            2,
            "loading never rewrites the index"
        );
    }

    #[test]
    fn damaged_sidecar_leaves_the_index_readable() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        image(&temp, "shoot/img_001.cr3");
        store.save_metadata("shoot/img_001.cr3", 3, "Yellow", &keywords(&["keep"])).unwrap();
        fs::write(store.xmp_path("shoot/img_001.cr3").unwrap(), "<not xml").unwrap();

        let meta = store.load_metadata("shoot/img_001.cr3").unwrap();
        assert_eq!(meta.source, MetadataSource::Index);
        assert_eq!(meta.rating, 3);
        assert_eq!(meta.label, "Yellow");
        assert_eq!(meta.keywords, keywords(&["keep"]));
        assert_eq!(store.load_index().unwrap().images[0].rating, 3);
    }

    #[test]
    fn failed_index_write_rolls_the_sidecar_back() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        image(&temp, "shoot/img_001.cr3");
        store.save_metadata("shoot/img_001.cr3", 1, "Pink", &keywords(&["first"])).unwrap();
        let before = fs::read(store.xmp_path("shoot/img_001.cr3").unwrap()).unwrap();

        // A newer catalog makes save_index refuse, after the sidecar was written.
        let newer = r#"{"schema_version":3,"images":[]}"#;
        fs::write(store.root().join("catalog.json"), newer).unwrap();

        let err = store
            .save_metadata("shoot/img_001.cr3", 5, "Purple", &keywords(&["second"]))
            .unwrap_err();
        assert!(matches!(err.downcast_ref::<StoreError>(), Some(StoreError::Newer { found: 3 })));
        assert_eq!(
            fs::read(store.xmp_path("shoot/img_001.cr3").unwrap()).unwrap(),
            before,
            "the previous sidecar is restored"
        );
        assert_eq!(fs::read_to_string(store.root().join("catalog.json")).unwrap(), newer);
        assert!(!store.xmp_path("shoot/img_001.cr3").unwrap().with_extension("xmp.tmp").exists());
    }

    #[test]
    fn failed_index_write_leaves_no_new_sidecar() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        image(&temp, "shoot/img_001.cr3");
        fs::write(store.root().join("catalog.json"), r#"{"schema_version":3,"images":[]}"#).unwrap();

        assert!(store.save_metadata("shoot/img_001.cr3", 5, "Red", &keywords(&["x"])).is_err());
        assert!(!store.xmp_path("shoot/img_001.cr3").unwrap().exists());
    }

    #[test]
    fn save_adjustments_never_touches_xmp() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        image(&temp, "shoot/img_001.cr3");

        store.save_adjustments("shoot/img_001.cr3", &sample_adjustments()).unwrap();
        assert!(!store.xmp_path("shoot/img_001.cr3").unwrap().exists());

        store.save_metadata("shoot/img_001.cr3", 2, "Blue", &keywords(&["meta"])).unwrap();
        let before = fs::read(store.xmp_path("shoot/img_001.cr3").unwrap()).unwrap();
        store.save_adjustments("shoot/img_001.cr3", &sample_adjustments()).unwrap();
        assert_eq!(fs::read(store.xmp_path("shoot/img_001.cr3").unwrap()).unwrap(), before);
    }

    #[test]
    fn metadata_refuses_capture_one_bundles() {
        let temp = TempFolder::new();
        let store = Store::open(temp.path()).unwrap();
        image(&temp, "Shoot.cocatalogdb/a.CR3");

        assert!(store.save_metadata("Shoot.cocatalogdb/a.CR3", 5, "Red", &keywords(&["x"])).is_err());
        assert!(!temp.path().join("Shoot.cocatalogdb").join("a.xmp").exists());
        assert!(store.load_index().unwrap().images.is_empty());
    }
}
