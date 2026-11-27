//! High-performance SQLite (WAL) catalog/session persistence and disk
//! thumbnail proxy cache for Aurora Studio.
//!
//! Directory layout under `<library_root>/.aurora/`:
//! * `aurora.db` (+ `-wal` / `-shm`): SQLite WAL database storing image index,
//!   library profile (Folder / Session / Catalog), user collections, smart
//!   rules, variants metadata and favorite folders.
//! * `previews/t_<hash>_<max_size>.athumb`: lock-free disk thumbnail cache
//!   with 24-byte header (`ATH1` + max_size + mtime_ns + adjust_hash) + JPEG
//!   payload.
//!
//! Capture One-aligned Session layout (`LibraryKind::Session`):
//! * `<session_root>/Capture/`
//! * `<session_root>/Selects/`
//! * `<session_root>/Output/`
//! * `<session_root>/Trash/`
//! * `<session_root>/.aurora/aurora.db` (`library_kind = "session"`)

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageFormat};
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::catalog::{CollectionKind, ImageId, SmartRule, UserCollection};
use crate::engine::RgbaImage;
use crate::store::{Catalog, Entry, write_atomic};

pub const AURORA_META_DIR: &str = ".aurora";
pub const AURORA_DB_FILE: &str = "aurora.db";
pub const AURORA_SESSION_EXT: &str = "aurorasession";
pub const AURORA_CATALOG_EXT: &str = "auroracatalog";
const DB_SCHEMA_VERSION: u32 = 1;
const THUMB_MAGIC: &[u8; 4] = b"ATH1";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LibraryKind {
    #[default]
    Folder,
    Session,
    Catalog,
}

impl LibraryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Folder => "folder",
            Self::Session => "session",
            Self::Catalog => "catalog",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "session" => Self::Session,
            "catalog" => Self::Catalog,
            _ => Self::Folder,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SessionFolderRole {
    Capture,
    Selects,
    Output,
    Trash,
}

impl SessionFolderRole {
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::Capture => "Capture",
            Self::Selects => "Selects",
            Self::Output => "Output",
            Self::Trash => "Trash",
        }
    }

    pub fn meta_key(self) -> &'static str {
        match self {
            Self::Capture => "session_capture_dir",
            Self::Selects => "session_selects_dir",
            Self::Output => "session_output_dir",
            Self::Trash => "session_trash_dir",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FavoriteFolder {
    pub rel_path: String,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibraryProfile {
    pub kind: LibraryKind,
    pub name: String,
    pub root: PathBuf,
    pub capture_dir: Option<PathBuf>,
    pub selects_dir: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub trash_dir: Option<PathBuf>,
    pub favorites: Vec<FavoriteFolder>,
}

impl LibraryProfile {
    pub fn role_path(&self, role: SessionFolderRole) -> Option<&PathBuf> {
        match role {
            SessionFolderRole::Capture => self.capture_dir.as_ref(),
            SessionFolderRole::Selects => self.selects_dir.as_ref(),
            SessionFolderRole::Output => self.output_dir.as_ref(),
            SessionFolderRole::Trash => self.trash_dir.as_ref(),
        }
    }
}

/// Thread-safe handle to `<library_root>/.aurora/aurora.db`.
#[derive(Clone)]
pub struct AuroraDb {
    library_root: PathBuf,
    meta_dir: PathBuf,
    db_path: PathBuf,
    conn: Arc<Mutex<Connection>>,
}

impl std::fmt::Debug for AuroraDb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuroraDb")
            .field("library_root", &self.library_root)
            .field("db_path", &self.db_path)
            .finish()
    }
}

impl AuroraDb {
    /// Opens or initializes `<library_root>/.aurora/aurora.db` in WAL mode.
    pub fn open(library_root: &Path) -> Result<Self> {
        let meta_dir = library_root.join(AURORA_META_DIR);
        fs::create_dir_all(meta_dir.join("previews"))
            .with_context(|| format!("creating {}", meta_dir.join("previews").display()))?;
        let db_path = meta_dir.join(AURORA_DB_FILE);
        let conn = Connection::open(&db_path)
            .with_context(|| format!("opening SQLite db {}", db_path.display()))?;
        configure_connection(&conn)?;
        init_schema(&conn)?;

        let inferred_kind = infer_kind_from_path(library_root);
        let default_name = library_root
            .file_stem()
            .or_else(|| library_root.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Aurora Library".to_string());
        seed_default_meta(&conn, inferred_kind, &default_name)?;

        let db = Self {
            library_root: library_root.to_path_buf(),
            meta_dir,
            db_path,
            conn: Arc::new(Mutex::new(conn)),
        };
        if inferred_kind == LibraryKind::Session {
            db.ensure_session_dirs()?;
        }
        Ok(db)
    }

    pub fn library_root(&self) -> &Path {
        &self.library_root
    }

    pub fn meta_dir(&self) -> &Path {
        &self.meta_dir
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Creates or converts `root` into a Capture One-aligned Session with
    /// `Capture/`, `Selects/`, `Output/`, and `Trash/` subdirectories.
    pub fn create_session(root: &Path, name: Option<&str>) -> Result<Self> {
        fs::create_dir_all(root).with_context(|| format!("creating session root {}", root.display()))?;
        let db = Self::open(root)?;
        let session_name = name
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| root.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "Session".to_string());
        {
            let conn = db.conn.lock();
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('library_kind', 'session')
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [],
            )?;
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('library_name', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![session_name],
            )?;
        }
        db.ensure_session_dirs()?;
        Ok(db)
    }

    /// Creates or converts `root` into an Aurora Catalog.
    pub fn create_catalog(root: &Path, name: Option<&str>) -> Result<Self> {
        fs::create_dir_all(root).with_context(|| format!("creating catalog root {}", root.display()))?;
        let db = Self::open(root)?;
        let catalog_name = name
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| root.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "Catalog".to_string());
        {
            let conn = db.conn.lock();
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('library_kind', 'catalog')
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [],
            )?;
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('library_name', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![catalog_name],
            )?;
        }
        Ok(db)
    }

    pub fn ensure_session_dirs(&self) -> Result<()> {
        for role in [
            SessionFolderRole::Capture,
            SessionFolderRole::Selects,
            SessionFolderRole::Output,
            SessionFolderRole::Trash,
        ] {
            let dir = self.library_root.join(role.dir_name());
            fs::create_dir_all(&dir)
                .with_context(|| format!("creating session role dir {}", dir.display()))?;
            let conn = self.conn.lock();
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?1, ?2)
                 ON CONFLICT(key) DO NOTHING",
                params![role.meta_key(), role.dir_name()],
            )?;
        }
        Ok(())
    }

    pub fn profile(&self) -> Result<LibraryProfile> {
        let conn = self.conn.lock();
        let meta = load_meta_map(&conn)?;
        let kind = meta
            .get("library_kind")
            .map(|s| LibraryKind::from_str(s))
            .unwrap_or_else(|| infer_kind_from_path(&self.library_root));
        let name = meta
            .get("library_name")
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                self.library_root
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "Aurora".to_string())
            });

        let resolve_role = |role: SessionFolderRole| -> Option<PathBuf> {
            if kind != LibraryKind::Session {
                return None;
            }
            let rel = meta
                .get(role.meta_key())
                .map(String::as_str)
                .unwrap_or_else(|| role.dir_name());
            Some(self.library_root.join(rel))
        };

        let mut stmt = conn.prepare_cached(
            "SELECT rel_path, label FROM favorites ORDER BY added_at ASC, rel_path ASC",
        )?;
        let favorites = stmt
            .query_map([], |row| {
                Ok(FavoriteFolder {
                    rel_path: row.get(0)?,
                    label: row.get(1)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(LibraryProfile {
            kind,
            name,
            root: self.library_root.clone(),
            capture_dir: resolve_role(SessionFolderRole::Capture),
            selects_dir: resolve_role(SessionFolderRole::Selects),
            output_dir: resolve_role(SessionFolderRole::Output),
            trash_dir: resolve_role(SessionFolderRole::Trash),
            favorites,
        })
    }

    pub fn add_favorite(&self, rel_path: &str, label: &str) -> Result<()> {
        let conn = self.conn.lock();
        let now = now_unix_secs() as i64;
        conn.execute(
            "INSERT INTO favorites(rel_path, label, added_at) VALUES(?1, ?2, ?3)
             ON CONFLICT(rel_path) DO UPDATE SET label = excluded.label",
            params![rel_path, label, now],
        )?;
        Ok(())
    }

    pub fn remove_favorite(&self, rel_path: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM favorites WHERE rel_path = ?1", params![rel_path])?;
        Ok(())
    }

    pub fn image_count(&self) -> Result<usize> {
        let conn = self.conn.lock();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))?;
        Ok(count.max(0) as usize)
    }

    pub fn load_catalog(&self) -> Result<Catalog> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT rel_path, rating, color_tag_idx, keywords_json, variant_index, updated_at, trashed
             FROM images ORDER BY rel_path ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            let rel_path: String = row.get(0)?;
            let rating: i64 = row.get(1)?;
            let color_tag_idx: i64 = row.get(2)?;
            let keywords_json: String = row.get(3)?;
            let variant_index: i64 = row.get(4)?;
            let updated_at: i64 = row.get(5)?;
            let trashed: i64 = row.get(6)?;
            Ok((rel_path, rating, color_tag_idx, keywords_json, variant_index, updated_at, trashed))
        })?;

        let mut images = Vec::new();
        for row in rows {
            let (rel_path, rating, color_tag_idx, keywords_json, variant_index, updated_at, trashed) = row?;
            let keywords: Vec<String> = serde_json::from_str(&keywords_json).unwrap_or_default();
            images.push(Entry {
                rel_path,
                rating: rating.clamp(-1, 5) as i8,
                color_tag: color_tag_idx.clamp(0, 7) as i8,
                keywords,
                variant: variant_index.max(1) as u32,
                updated_at,
                trashed: trashed != 0,
            });
        }
        Ok(Catalog {
            schema_version: DB_SCHEMA_VERSION,
            images,
        })
    }

    /// Replaces or updates all entries in `images` inside a single SQLite transaction.
    pub fn save_catalog(&self, catalog: &Catalog) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        {
            let mut existing_stmt = tx.prepare_cached("SELECT rel_path FROM images")?;
            let existing: HashSet<String> = existing_stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .filter_map(std::result::Result::ok)
                .collect();

            let incoming: HashSet<&str> = catalog.images.iter().map(|e| e.rel_path.as_str()).collect();
            let mut del_stmt = tx.prepare_cached("DELETE FROM images WHERE rel_path = ?1")?;
            for old in existing {
                if !incoming.contains(old.as_str()) {
                    del_stmt.execute(params![old])?;
                }
            }

            let mut upsert_stmt = tx.prepare_cached(
                "INSERT INTO images(rel_path, rating, color_tag_idx, keywords_json, variant_index, updated_at, trashed)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(rel_path) DO UPDATE SET
                   rating = excluded.rating,
                   color_tag_idx = excluded.color_tag_idx,
                   keywords_json = excluded.keywords_json,
                   variant_index = excluded.variant_index,
                   updated_at = excluded.updated_at,
                   trashed = excluded.trashed",
            )?;
            for entry in &catalog.images {
                let keywords_json = serde_json::to_string(&entry.keywords)?;
                upsert_stmt.execute(params![
                    entry.rel_path,
                    entry.rating as i64,
                    entry.color_tag as i64,
                    keywords_json,
                    entry.variant.max(1) as i64,
                    entry.updated_at,
                    if entry.trashed { 1i64 } else { 0i64 },
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Incremental single-row upsert without rewriting the entire catalog.
    pub fn upsert_entry(&self, entry: &Entry) -> Result<()> {
        let conn = self.conn.lock();
        let keywords_json = serde_json::to_string(&entry.keywords)?;
        conn.execute(
            "INSERT INTO images(rel_path, rating, color_tag_idx, keywords_json, variant_index, updated_at, trashed)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(rel_path) DO UPDATE SET
               rating = excluded.rating,
               color_tag_idx = excluded.color_tag_idx,
               keywords_json = excluded.keywords_json,
               variant_index = excluded.variant_index,
               updated_at = excluded.updated_at,
               trashed = excluded.trashed",
            params![
                entry.rel_path,
                entry.rating as i64,
                entry.color_tag as i64,
                keywords_json,
                entry.variant.max(1) as i64,
                entry.updated_at,
                if entry.trashed { 1i64 } else { 0i64 },
            ],
        )?;
        Ok(())
    }

    pub fn remove_entry(&self, rel_path: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM images WHERE rel_path = ?1", params![rel_path])?;
        conn.execute("DELETE FROM collection_items WHERE rel_path = ?1", params![rel_path])?;
        Ok(())
    }

    pub fn save_user_collections(
        &self,
        collections: &[UserCollection],
        id_to_rel: &HashMap<ImageId, String>,
    ) -> Result<()> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM collection_items", [])?;
        tx.execute("DELETE FROM collections", [])?;
        {
            let mut c_stmt = tx.prepare_cached(
                "INSERT INTO collections(id, parent_id, name, kind, rules_json, match_all, sort_order)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            let mut item_stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO collection_items(collection_id, rel_path, sort_order)
                 VALUES(?1, ?2, ?3)",
            )?;
            for (idx, c) in collections.iter().enumerate() {
                let rules_json = serde_json::to_string(&c.rules)?;
                c_stmt.execute(params![
                    c.id as i64,
                    c.parent.map(|p| p as i64),
                    c.name,
                    collection_kind_str(c.kind),
                    rules_json,
                    if c.match_all { 1i64 } else { 0i64 },
                    idx as i64,
                ])?;
                for (item_idx, img_id) in c.images.iter().enumerate() {
                    if let Some(rel) = id_to_rel.get(img_id) {
                        item_stmt.execute(params![c.id as i64, rel, item_idx as i64])?;
                    }
                }
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn load_user_collections(&self, rel_to_id: &HashMap<String, ImageId>) -> Result<Vec<UserCollection>> {
        let conn = self.conn.lock();
        let mut c_stmt = conn.prepare_cached(
            "SELECT id, parent_id, name, kind, rules_json, match_all
             FROM collections ORDER BY sort_order ASC, id ASC",
        )?;
        let rows = c_stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let parent_id: Option<i64> = row.get(1)?;
            let name: String = row.get(2)?;
            let kind_str: String = row.get(3)?;
            let rules_json: String = row.get(4)?;
            let match_all: i64 = row.get(5)?;
            Ok((id, parent_id, name, kind_str, rules_json, match_all))
        })?;

        let mut item_stmt = conn.prepare_cached(
            "SELECT rel_path FROM collection_items WHERE collection_id = ?1 ORDER BY sort_order ASC",
        )?;

        let mut out = Vec::new();
        for row in rows {
            let (id, parent_id, name, kind_str, rules_json, match_all) = row?;
            let rules: Vec<SmartRule> = serde_json::from_str(&rules_json).unwrap_or_default();
            let mut images = Vec::new();
            let rels = item_stmt.query_map(params![id], |r| r.get::<_, String>(0))?;
            for rel in rels {
                if let Some(img_id) = rel_to_id.get(&rel?) {
                    images.push(*img_id);
                }
            }
            out.push(UserCollection {
                id: id as u64,
                parent: parent_id.map(|p| p as u64),
                name,
                kind: parse_collection_kind(&kind_str),
                images,
                rules,
                match_all: match_all != 0,
            });
        }
        Ok(out)
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        conn.query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| r.get(0))
            .optional()
            .map_err(Into::into)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }
}

/// Lock-free disk thumbnail cache stored under `<meta_dir>/previews/`.
///
/// Format:
/// `[0..4]   = b"ATH1"`
/// `[4..8]   = max_size: u32 LE`
/// `[8..16]  = mtime_ns: u64 LE`
/// `[16..24] = adjust_hash: u64 LE`
/// `[24..]   = JPEG bytes`
#[derive(Clone, Debug)]
pub struct DiskThumbCache {
    dir: PathBuf,
}

impl DiskThumbCache {
    pub fn new(meta_dir: &Path) -> Self {
        Self {
            dir: meta_dir.join("previews"),
        }
    }

    fn entry_path(&self, rel_path: &str, max_size: u32) -> PathBuf {
        self.dir.join(format!("t_{}_{}.athumb", fnv1a_hex(rel_path), max_size))
    }

    pub fn get(&self, rel_path: &str, max_size: u32, mtime_ns: u64, adjust_hash: u64) -> Option<RgbaImage> {
        let path = self.entry_path(rel_path, max_size);
        let bytes = fs::read(path).ok()?;
        if bytes.len() <= 24 || &bytes[0..4] != THUMB_MAGIC {
            return None;
        }
        let cached_size = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let cached_mtime = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
        let cached_adj = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
        if cached_size != max_size || cached_mtime != mtime_ns || cached_adj != adjust_hash {
            return None;
        }
        let decoded = image::load_from_memory_with_format(&bytes[24..], ImageFormat::Jpeg).ok()?;
        let rgba = decoded.to_rgba8();
        let (width, height) = rgba.dimensions();
        Some(RgbaImage::new(width, height, rgba.into_raw()))
    }

    pub fn put(
        &self,
        rel_path: &str,
        max_size: u32,
        mtime_ns: u64,
        adjust_hash: u64,
        image: &RgbaImage,
    ) -> Result<()> {
        if image.width == 0 || image.height == 0 {
            return Ok(());
        }
        let mut rgb = Vec::with_capacity((image.width as usize) * (image.height as usize) * 3);
        for px in image.pixels.chunks_exact(4) {
            rgb.push(px[0]);
            rgb.push(px[1]);
            rgb.push(px[2]);
        }
        let mut jpeg = Vec::with_capacity(24 * 1024);
        {
            let mut enc = JpegEncoder::new_with_quality(&mut jpeg, 85);
            enc.encode(&rgb, image.width, image.height, ColorType::Rgb8.into())?;
        }
        let mut out = Vec::with_capacity(24 + jpeg.len());
        out.extend_from_slice(THUMB_MAGIC);
        out.extend_from_slice(&max_size.to_le_bytes());
        out.extend_from_slice(&mtime_ns.to_le_bytes());
        out.extend_from_slice(&adjust_hash.to_le_bytes());
        out.extend_from_slice(&jpeg);

        fs::create_dir_all(&self.dir)?;
        write_atomic(&self.entry_path(rel_path, max_size), &out)
    }

    pub fn invalidate(&self, rel_path: &str) {
        let prefix = format!("t_{}_", fnv1a_hex(rel_path));
        if let Ok(entries) = fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(&prefix) && name.ends_with(".athumb") {
                        let _ = fs::remove_file(entry.path());
                    }
                }
            }
        }
    }
}

pub fn file_mtime_ns(path: &Path) -> u64 {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

pub fn hash_adjustments(adj: &crate::adjust::AdjustmentSet) -> u64 {
    let bytes = serde_json::to_vec(adj).unwrap_or_default();
    fnv1a_bytes(&bytes)
}

pub fn is_aurora_session_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(AURORA_SESSION_EXT))
        || path
            .join(AURORA_META_DIR)
            .join(AURORA_DB_FILE)
            .is_file()
            && path.join("Capture").is_dir()
            && path.join("Selects").is_dir()
}

pub fn is_aurora_catalog_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(AURORA_CATALOG_EXT))
}

fn infer_kind_from_path(root: &Path) -> LibraryKind {
    if is_aurora_session_path(root) {
        LibraryKind::Session
    } else if is_aurora_catalog_path(root) {
        LibraryKind::Catalog
    } else {
        LibraryKind::Folder
    }
}

fn configure_connection(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA temp_store = MEMORY;
         PRAGMA cache_size = -16000;
         PRAGMA mmap_size = 268435456;
         PRAGMA busy_timeout = 5000;
         PRAGMA foreign_keys = ON;",
    )?;
    Ok(())
}

fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
         );
         CREATE TABLE IF NOT EXISTS images (
            rel_path TEXT PRIMARY KEY,
            rating INTEGER NOT NULL DEFAULT 0,
            color_tag_idx INTEGER NOT NULL DEFAULT 0,
            keywords_json TEXT NOT NULL DEFAULT '[]',
            trashed INTEGER NOT NULL DEFAULT 0,
            variant_group TEXT,
            variant_index INTEGER NOT NULL DEFAULT 1,
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            mtime_ns INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL DEFAULT 0
         );
         CREATE INDEX IF NOT EXISTS idx_images_rating ON images(rating);
         CREATE INDEX IF NOT EXISTS idx_images_trashed ON images(trashed);
         CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            rules_json TEXT NOT NULL DEFAULT '[]',
            match_all INTEGER NOT NULL DEFAULT 1,
            sort_order INTEGER NOT NULL DEFAULT 0
         );
         CREATE TABLE IF NOT EXISTS collection_items (
            collection_id INTEGER NOT NULL,
            rel_path TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(collection_id, rel_path),
            FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE
         );
         CREATE TABLE IF NOT EXISTS favorites (
            rel_path TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            added_at INTEGER NOT NULL DEFAULT 0
         );",
    )?;
    Ok(())
}

fn seed_default_meta(conn: &Connection, kind: LibraryKind, name: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?1) ON CONFLICT(key) DO NOTHING",
        params![DB_SCHEMA_VERSION.to_string()],
    )?;
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('library_kind', ?1) ON CONFLICT(key) DO NOTHING",
        params![kind.as_str()],
    )?;
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('library_name', ?1) ON CONFLICT(key) DO NOTHING",
        params![name],
    )?;
    Ok(())
}

fn load_meta_map(conn: &Connection) -> Result<HashMap<String, String>> {
    let mut stmt = conn.prepare_cached("SELECT key, value FROM meta")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    let mut map = HashMap::new();
    for r in rows {
        let (k, v) = r?;
        map.insert(k, v);
    }
    Ok(map)
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn fnv1a_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fnv1a_hex(s: &str) -> String {
    format!("{:016x}", fnv1a_bytes(s.as_bytes()))
}

fn collection_kind_str(kind: CollectionKind) -> &'static str {
    match kind {
        CollectionKind::AllImages => "AllImages",
        CollectionKind::RecentImports => "RecentImports",
        CollectionKind::RecentCaptures => "RecentCaptures",
        CollectionKind::Trash => "Trash",
        CollectionKind::Folder => "Folder",
        CollectionKind::Album => "Album",
        CollectionKind::SmartAlbum => "SmartAlbum",
        CollectionKind::Project => "Project",
        CollectionKind::Group => "Group",
    }
}

fn parse_collection_kind(s: &str) -> CollectionKind {
    match s {
        "AllImages" => CollectionKind::AllImages,
        "RecentImports" => CollectionKind::RecentImports,
        "RecentCaptures" => CollectionKind::RecentCaptures,
        "Trash" => CollectionKind::Trash,
        "Folder" => CollectionKind::Folder,
        "SmartAlbum" => CollectionKind::SmartAlbum,
        "Project" => CollectionKind::Project,
        "Group" => CollectionKind::Group,
        _ => CollectionKind::Album,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "aurora-store-test-{tag}-{}-{}",
            std::process::id(),
            now_unix_secs()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn session_creates_four_role_directories_and_sqlite_wal() {
        let dir = temp_dir("session").join("Wedding.aurorasession");
        let db = AuroraDb::create_session(&dir, Some("Wedding Shoot")).unwrap();
        let profile = db.profile().unwrap();
        assert_eq!(profile.kind, LibraryKind::Session);
        assert_eq!(profile.name, "Wedding Shoot");
        assert!(profile.capture_dir.as_ref().unwrap().is_dir());
        assert!(profile.selects_dir.as_ref().unwrap().is_dir());
        assert!(profile.output_dir.as_ref().unwrap().is_dir());
        assert!(profile.trash_dir.as_ref().unwrap().is_dir());
        assert!(dir.join(".aurora/aurora.db").is_file());
        let _ = fs::remove_dir_all(dir.parent().unwrap());
    }

    #[test]
    fn sqlite_catalog_roundtrip_and_incremental_upsert() {
        let dir = temp_dir("catalog");
        let db = AuroraDb::open(&dir).unwrap();
        let e1 = Entry {
            rel_path: "Capture/a.nef".into(),
            rating: 5,
            color_tag: 4,
            keywords: vec!["portrait".into()],
            variant: 1,
            updated_at: 100,
            trashed: false,
        };
        db.upsert_entry(&e1).unwrap();
        assert_eq!(db.image_count().unwrap(), 1);
        let loaded = db.load_catalog().unwrap();
        assert_eq!(loaded.images, vec![e1.clone()]);

        let mut e2 = e1.clone();
        e2.rating = 3;
        e2.trashed = true;
        db.upsert_entry(&e2).unwrap();
        let loaded2 = db.load_catalog().unwrap();
        assert_eq!(loaded2.images, vec![e2]);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disk_thumb_cache_roundtrip_and_invalidation() {
        let dir = temp_dir("thumb");
        let db = AuroraDb::open(&dir).unwrap();
        let cache = DiskThumbCache::new(db.meta_dir());
        let img = RgbaImage::new(4, 4, vec![128; 4 * 4 * 4]);
        cache.put("a.nef", 360, 12345, 999, &img).unwrap();

        let hit = cache.get("a.nef", 360, 12345, 999).expect("expected cache hit");
        assert_eq!((hit.width, hit.height), (4, 4));
        assert!(cache.get("a.nef", 360, 12346, 999).is_none());
        assert!(cache.get("a.nef", 360, 12345, 1000).is_none());

        cache.invalidate("a.nef");
        assert!(cache.get("a.nef", 360, 12345, 999).is_none());
        let _ = fs::remove_dir_all(dir);
    }
}
