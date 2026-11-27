//! Persistent on-disk preview pyramid cache backed by SQLite (WAL mode).
//!
//! Stores rendered preview levels (`Thumbnail`, `Standard`, `OneToOne`,
//! `SmartPreview`) as atomic blob files indexed by `(image_uuid,
//! settings_digest, level_kind)` with source-fingerprint validation, byte-budget
//! LRU eviction, and tombstone-based cleanup.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};

/// Preview pyramid tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyramidLevelKind {
    /// Grid / filmstrip thumbnail (typically 320 px long side).
    Thumbnail,
    /// Interactive loupe standard preview (typically 1440 / 2048 px long side).
    Standard,
    /// Full-resolution 1:1 preview.
    OneToOne,
    /// Offline proxy preview (typically 2560 px long side).
    SmartPreview,
}

impl PyramidLevelKind {
    pub const ALL: [Self; 4] = [
        Self::Thumbnail,
        Self::Standard,
        Self::OneToOne,
        Self::SmartPreview,
    ];

    /// Canonical string stored in SQLite `pyramid_levels.level_kind`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Thumbnail => "thumbnail",
            Self::Standard => "standard",
            Self::OneToOne => "one_to_one",
            Self::SmartPreview => "smart_preview",
        }
    }

    /// Parse from the canonical SQLite string.
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "thumbnail" => Some(Self::Thumbnail),
            "standard" => Some(Self::Standard),
            "one_to_one" => Some(Self::OneToOne),
            "smart_preview" => Some(Self::SmartPreview),
            _ => None,
        }
    }

    /// Recommended default long-edge dimension in pixels (`None` for native 1:1).
    pub const fn default_long_side(self) -> Option<u32> {
        match self {
            Self::Thumbnail => Some(320),
            Self::Standard => Some(2048),
            Self::OneToOne => None,
            Self::SmartPreview => Some(2560),
        }
    }
}

/// Configuration for [`DiskPreviewCache`].
#[derive(Clone, Debug)]
pub struct DiskPreviewCacheConfig {
    /// Root directory containing `previews.db` and `blobs/`.
    pub root_dir: PathBuf,
    /// Maximum total bytes across all stored pyramid blobs before LRU eviction.
    pub max_bytes: u64,
    /// Grace period (in milliseconds) before tombstoned files are deleted on sweep.
    pub tombstone_grace_ms: u64,
}

impl DiskPreviewCacheConfig {
    pub fn new(root_dir: impl Into<PathBuf>, max_bytes: u64) -> Self {
        Self {
            root_dir: root_dir.into(),
            max_bytes,
            tombstone_grace_ms: 0,
        }
    }
}

/// Metadata + payload for writing one level into the disk pyramid cache.
#[derive(Clone, Debug)]
pub struct PyramidLevelWrite<'a> {
    pub image_uuid: &'a str,
    pub settings_digest: &'a str,
    pub source_fingerprint: &'a str,
    pub level_kind: PyramidLevelKind,
    pub long_side: u32,
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

/// One cached level retrieved from disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedPyramidLevel {
    pub image_uuid: String,
    pub settings_digest: String,
    pub source_fingerprint: String,
    pub level_kind: PyramidLevelKind,
    pub long_side: u32,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Summary statistics of the on-disk pyramid index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskCacheStats {
    pub pyramid_count: u64,
    pub level_count: u64,
    pub total_bytes: u64,
    pub tombstone_count: u64,
}

struct DbState {
    conn: Connection,
    last_ts_ms: i64,
}

/// Thread-safe persistent preview pyramid cache with SQLite WAL index.
pub struct DiskPreviewCache {
    root_dir: PathBuf,
    blobs_dir: PathBuf,
    max_bytes: u64,
    tombstone_grace_ms: u64,
    state: Mutex<DbState>,
    tmp_seq: AtomicU64,
}

impl DiskPreviewCache {
    /// Open or create the preview cache at `config.root_dir`.
    pub fn open(config: DiskPreviewCacheConfig) -> Result<Self> {
        let root_dir = config.root_dir;
        let blobs_dir = root_dir.join("blobs");
        fs::create_dir_all(&blobs_dir)
            .with_context(|| format!("create blobs dir {}", blobs_dir.display()))?;

        let db_path = root_dir.join("previews.db");
        let conn = Connection::open(&db_path)
            .with_context(|| format!("open sqlite db {}", db_path.display()))?;

        configure_sqlite(&conn)?;
        init_schema(&conn)?;

        let cache = Self {
            root_dir,
            blobs_dir,
            max_bytes: config.max_bytes,
            tombstone_grace_ms: config.tombstone_grace_ms,
            state: Mutex::new(DbState {
                conn,
                last_ts_ms: 0,
            }),
            tmp_seq: AtomicU64::new(1),
        };

        // Best-effort startup sweep of expired tombstones.
        let _ = cache.sweep_tombstones();
        Ok(cache)
    }

    /// Root directory of this cache instance.
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Configured maximum byte budget.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Insert or replace a pyramid level, writing its blob atomically and
    /// enforcing the configured `max_bytes` LRU budget.
    pub fn put(&self, entry: PyramidLevelWrite<'_>) -> Result<()> {
        let rel_path = self.make_rel_path(
            entry.image_uuid,
            entry.settings_digest,
            entry.level_kind,
            entry.long_side,
        );
        let abs_path = self.root_dir.join(&rel_path);
        self.write_blob_atomic(&abs_path, entry.data)?;

        let byte_len = entry.data.len() as i64;
        let level_str = entry.level_kind.as_str();

        let paths_to_delete_now: Vec<String> = {
            let mut guard = self.state.lock();
            let now_ms = next_timestamp_ms(&mut guard.last_ts_ms);
            let tx = guard.conn.transaction()?;

            // 1. If the same image_uuid has pyramids with a different source_fingerprint,
            //    tombstone and drop them to prevent stale-source hits.
            let stale_fp_paths = collect_stale_fingerprint_paths(
                &tx,
                entry.image_uuid,
                entry.source_fingerprint,
            )?;
            for path in &stale_fp_paths {
                insert_tombstone(&tx, path, now_ms)?;
            }
            if !stale_fp_paths.is_empty() {
                tx.execute(
                    "DELETE FROM pyramid_levels
                     WHERE image_uuid = ?1
                       AND settings_digest IN (
                           SELECT settings_digest FROM pyramids
                           WHERE image_uuid = ?1 AND source_fingerprint != ?2
                       )",
                    params![entry.image_uuid, entry.source_fingerprint],
                )?;
                tx.execute(
                    "DELETE FROM pyramids
                     WHERE image_uuid = ?1 AND source_fingerprint != ?2",
                    params![entry.image_uuid, entry.source_fingerprint],
                )?;
            }

            // 2. If this exact level previously pointed to a different file path, tombstone it.
            let prev_rel: Option<String> = tx
                .query_row(
                    "SELECT rel_path FROM pyramid_levels
                     WHERE image_uuid = ?1 AND settings_digest = ?2 AND level_kind = ?3",
                    params![entry.image_uuid, entry.settings_digest, level_str],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(old_path) = prev_rel {
                if old_path != rel_path {
                    insert_tombstone(&tx, &old_path, now_ms)?;
                }
            }

            // 3. Upsert level row.
            tx.execute(
                "INSERT INTO pyramid_levels (
                    image_uuid, settings_digest, level_kind, long_side,
                    width, height, byte_len, rel_path
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(image_uuid, settings_digest, level_kind) DO UPDATE SET
                    long_side = excluded.long_side,
                    width = excluded.width,
                    height = excluded.height,
                    byte_len = excluded.byte_len,
                    rel_path = excluded.rel_path",
                params![
                    entry.image_uuid,
                    entry.settings_digest,
                    level_str,
                    entry.long_side,
                    entry.width,
                    entry.height,
                    byte_len,
                    rel_path,
                ],
            )?;

            // 4. Recompute pyramid total_bytes and upsert parent pyramid row.
            let pyramid_total: i64 = tx.query_row(
                "SELECT COALESCE(SUM(byte_len), 0) FROM pyramid_levels
                 WHERE image_uuid = ?1 AND settings_digest = ?2",
                params![entry.image_uuid, entry.settings_digest],
                |row| row.get(0),
            )?;
            tx.execute(
                "INSERT INTO pyramids (
                    image_uuid, settings_digest, source_fingerprint,
                    last_access_ms, total_bytes
                 ) VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(image_uuid, settings_digest) DO UPDATE SET
                    source_fingerprint = excluded.source_fingerprint,
                    last_access_ms = excluded.last_access_ms,
                    total_bytes = excluded.total_bytes",
                params![
                    entry.image_uuid,
                    entry.settings_digest,
                    entry.source_fingerprint,
                    now_ms,
                    pyramid_total,
                ],
            )?;

            // Remove any tombstone for the path we just wrote (re-materialised).
            tx.execute(
                "DELETE FROM deleted_pyramids WHERE rel_path = ?1",
                params![rel_path],
            )?;

            // 5. Enforce max_bytes via LRU eviction, keeping the entry just written.
            evict_lru_tx(
                &tx,
                self.max_bytes,
                now_ms,
                Some((entry.image_uuid, entry.settings_digest)),
            )?;

            let due = if self.tombstone_grace_ms == 0 {
                drain_due_tombstones_tx(&tx, now_ms, 0)?
            } else {
                Vec::new()
            };

            tx.commit()?;
            due
        };

        for rel in paths_to_delete_now {
            let _ = fs::remove_file(self.root_dir.join(rel));
        }

        Ok(())
    }

    /// Look up one pyramid level.
    ///
    /// Validates `source_fingerprint`: if the stored fingerprint differs, the
    /// mismatched pyramid is invalidated and tombstoned and `Ok(None)` is
    /// returned. On hit, updates `last_access_ms` for LRU tracking.
    pub fn get(
        &self,
        image_uuid: &str,
        settings_digest: &str,
        source_fingerprint: &str,
        level_kind: PyramidLevelKind,
    ) -> Result<Option<CachedPyramidLevel>> {
        let level_str = level_kind.as_str();

        enum Lookup {
            Miss,
            FingerprintMismatch(Vec<String>),
            Hit {
                long_side: u32,
                width: u32,
                height: u32,
                byte_len: usize,
                rel_path: String,
            },
        }

        let lookup = {
            let mut guard = self.state.lock();
            let now_ms = next_timestamp_ms(&mut guard.last_ts_ms);
            let tx = guard.conn.transaction()?;

            let stored_fp: Option<String> = tx
                .query_row(
                    "SELECT source_fingerprint FROM pyramids
                     WHERE image_uuid = ?1 AND settings_digest = ?2",
                    params![image_uuid, settings_digest],
                    |row| row.get(0),
                )
                .optional()?;

            let Some(stored_fp) = stored_fp else {
                tx.commit()?;
                return Ok(None);
            };

            if stored_fp != source_fingerprint {
                let paths = collect_pyramid_paths(&tx, image_uuid, settings_digest)?;
                for path in &paths {
                    insert_tombstone(&tx, path, now_ms)?;
                }
                tx.execute(
                    "DELETE FROM pyramid_levels
                     WHERE image_uuid = ?1 AND settings_digest = ?2",
                    params![image_uuid, settings_digest],
                )?;
                tx.execute(
                    "DELETE FROM pyramids
                     WHERE image_uuid = ?1 AND settings_digest = ?2",
                    params![image_uuid, settings_digest],
                )?;
                let due = if self.tombstone_grace_ms == 0 {
                    drain_due_tombstones_tx(&tx, now_ms, 0)?
                } else {
                    Vec::new()
                };
                tx.commit()?;
                Lookup::FingerprintMismatch(due)
            } else {
                let row: Option<(u32, u32, u32, i64, String)> = tx
                    .query_row(
                        "SELECT long_side, width, height, byte_len, rel_path
                         FROM pyramid_levels
                         WHERE image_uuid = ?1 AND settings_digest = ?2 AND level_kind = ?3",
                        params![image_uuid, settings_digest, level_str],
                        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
                    )
                    .optional()?;

                match row {
                    None => {
                        tx.commit()?;
                        Lookup::Miss
                    }
                    Some((long_side, width, height, byte_len, rel_path)) => {
                        tx.execute(
                            "UPDATE pyramids SET last_access_ms = ?1
                             WHERE image_uuid = ?2 AND settings_digest = ?3",
                            params![now_ms, image_uuid, settings_digest],
                        )?;
                        tx.commit()?;
                        Lookup::Hit {
                            long_side,
                            width,
                            height,
                            byte_len: byte_len.max(0) as usize,
                            rel_path,
                        }
                    }
                }
            }
        };

        match lookup {
            Lookup::Miss => Ok(None),
            Lookup::FingerprintMismatch(due) => {
                for rel in due {
                    let _ = fs::remove_file(self.root_dir.join(rel));
                }
                Ok(None)
            }
            Lookup::Hit {
                long_side,
                width,
                height,
                byte_len,
                rel_path,
            } => {
                let abs_path = self.root_dir.join(&rel_path);
                let data = match fs::read(&abs_path) {
                    Ok(bytes) if bytes.len() == byte_len => bytes,
                    _ => {
                        // Missing or truncated blob on disk: drop stale level row.
                        self.drop_level_row(image_uuid, settings_digest, level_kind)?;
                        return Ok(None);
                    }
                };
                Ok(Some(CachedPyramidLevel {
                    image_uuid: image_uuid.to_owned(),
                    settings_digest: settings_digest.to_owned(),
                    source_fingerprint: source_fingerprint.to_owned(),
                    level_kind,
                    long_side,
                    width,
                    height,
                    data,
                }))
            }
        }
    }

    /// Invalidate all cached pyramids for `image_uuid` (recording tombstones).
    pub fn invalidate_image(&self, image_uuid: &str) -> Result<usize> {
        let (removed, due) = {
            let mut guard = self.state.lock();
            let now_ms = next_timestamp_ms(&mut guard.last_ts_ms);
            let tx = guard.conn.transaction()?;

            let mut stmt = tx.prepare(
                "SELECT rel_path FROM pyramid_levels WHERE image_uuid = ?1",
            )?;
            let paths: Vec<String> = stmt
                .query_map(params![image_uuid], |r| r.get(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            drop(stmt);

            for path in &paths {
                insert_tombstone(&tx, path, now_ms)?;
            }
            tx.execute(
                "DELETE FROM pyramid_levels WHERE image_uuid = ?1",
                params![image_uuid],
            )?;
            tx.execute(
                "DELETE FROM pyramids WHERE image_uuid = ?1",
                params![image_uuid],
            )?;

            let due = if self.tombstone_grace_ms == 0 {
                drain_due_tombstones_tx(&tx, now_ms, 0)?
            } else {
                Vec::new()
            };
            tx.commit()?;
            (paths.len(), due)
        };

        for rel in due {
            let _ = fs::remove_file(self.root_dir.join(rel));
        }
        Ok(removed)
    }

    /// Delete files whose tombstone age has passed `tombstone_grace_ms`.
    pub fn sweep_tombstones(&self) -> Result<usize> {
        let due = {
            let mut guard = self.state.lock();
            let now_ms = next_timestamp_ms(&mut guard.last_ts_ms);
            let tx = guard.conn.transaction()?;
            let due = drain_due_tombstones_tx(&tx, now_ms, self.tombstone_grace_ms)?;
            tx.commit()?;
            due
        };

        let mut deleted = 0usize;
        for rel in due {
            let abs = self.root_dir.join(&rel);
            if abs.exists() && fs::remove_file(&abs).is_ok() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Remove any files under `blobs/` that are not referenced by `pyramid_levels`.
    pub fn cleanup_orphans(&self) -> Result<usize> {
        let live_paths: HashSet<PathBuf> = {
            let guard = self.state.lock();
            let mut stmt = guard.conn.prepare("SELECT rel_path FROM pyramid_levels")?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows.into_iter().map(|rel| self.root_dir.join(rel)).collect()
        };

        let mut removed = 0usize;
        let mut stack = vec![self.blobs_dir.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.is_file() && !live_paths.contains(&path) {
                    if fs::remove_file(&path).is_ok() {
                        removed += 1;
                    }
                }
            }
        }
        Ok(removed)
    }

    /// Return aggregate statistics from the SQLite index.
    pub fn stats(&self) -> Result<DiskCacheStats> {
        let guard = self.state.lock();
        let pyramid_count: i64 = guard
            .conn
            .query_row("SELECT COUNT(*) FROM pyramids", [], |r| r.get(0))?;
        let level_count: i64 = guard
            .conn
            .query_row("SELECT COUNT(*) FROM pyramid_levels", [], |r| r.get(0))?;
        let total_bytes: i64 = guard.conn.query_row(
            "SELECT COALESCE(SUM(total_bytes), 0) FROM pyramids",
            [],
            |r| r.get(0),
        )?;
        let tombstone_count: i64 = guard
            .conn
            .query_row("SELECT COUNT(*) FROM deleted_pyramids", [], |r| r.get(0))?;
        Ok(DiskCacheStats {
            pyramid_count: pyramid_count.max(0) as u64,
            level_count: level_count.max(0) as u64,
            total_bytes: total_bytes.max(0) as u64,
            tombstone_count: tombstone_count.max(0) as u64,
        })
    }

    /// Return the SQLite journal mode string (expected `"wal"`).
    pub fn journal_mode(&self) -> Result<String> {
        let guard = self.state.lock();
        let mode: String = guard
            .conn
            .query_row("PRAGMA journal_mode", [], |r| r.get(0))?;
        Ok(mode)
    }

    fn drop_level_row(
        &self,
        image_uuid: &str,
        settings_digest: &str,
        level_kind: PyramidLevelKind,
    ) -> Result<()> {
        let mut guard = self.state.lock();
        let tx = guard.conn.transaction()?;
        tx.execute(
            "DELETE FROM pyramid_levels
             WHERE image_uuid = ?1 AND settings_digest = ?2 AND level_kind = ?3",
            params![image_uuid, settings_digest, level_kind.as_str()],
        )?;
        let remaining: i64 = tx.query_row(
            "SELECT COALESCE(SUM(byte_len), 0) FROM pyramid_levels
             WHERE image_uuid = ?1 AND settings_digest = ?2",
            params![image_uuid, settings_digest],
            |r| r.get(0),
        )?;
        if remaining == 0 {
            tx.execute(
                "DELETE FROM pyramids
                 WHERE image_uuid = ?1 AND settings_digest = ?2",
                params![image_uuid, settings_digest],
            )?;
        } else {
            tx.execute(
                "UPDATE pyramids SET total_bytes = ?1
                 WHERE image_uuid = ?2 AND settings_digest = ?3",
                params![remaining, image_uuid, settings_digest],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    fn make_rel_path(
        &self,
        image_uuid: &str,
        settings_digest: &str,
        level_kind: PyramidLevelKind,
        long_side: u32,
    ) -> String {
        let hash = fnv1a_64(&[
            image_uuid.as_bytes(),
            b"|",
            settings_digest.as_bytes(),
            b"|",
            level_kind.as_str().as_bytes(),
        ]);
        let bucket = format!("{:02x}", (hash & 0xff) as u8);
        format!(
            "blobs/{bucket}/{hash:016x}_{}_{long_side}.bin",
            level_kind.as_str()
        )
    }

    fn write_blob_atomic(&self, target: &Path, data: &[u8]) -> Result<()> {
        let parent = target
            .parent()
            .context("blob target path must have a parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("create bucket dir {}", parent.display()))?;

        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let file_name = target
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("blob");
        let tmp_path = parent.join(format!(".{file_name}.{pid}.{seq}.tmp"));

        {
            let mut file = fs::File::create(&tmp_path)
                .with_context(|| format!("create temp blob {}", tmp_path.display()))?;
            file.write_all(data)
                .with_context(|| format!("write temp blob {}", tmp_path.display()))?;
            file.flush()?;
        }

        if let Err(err) = fs::rename(&tmp_path, target) {
            let _ = fs::remove_file(&tmp_path);
            return Err(err).with_context(|| {
                format!(
                    "atomic rename {} -> {}",
                    tmp_path.display(),
                    target.display()
                )
            });
        }
        Ok(())
    }
}

fn configure_sqlite(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "busy_timeout", 5000)?;
    Ok(())
}

fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS pyramids (
            image_uuid         TEXT NOT NULL,
            settings_digest    TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            last_access_ms     INTEGER NOT NULL,
            total_bytes        INTEGER NOT NULL,
            PRIMARY KEY (image_uuid, settings_digest)
        );
        CREATE INDEX IF NOT EXISTS idx_pyramids_lru
            ON pyramids (last_access_ms ASC);

        CREATE TABLE IF NOT EXISTS pyramid_levels (
            image_uuid      TEXT NOT NULL,
            settings_digest TEXT NOT NULL,
            level_kind      TEXT NOT NULL,
            long_side       INTEGER NOT NULL,
            width           INTEGER NOT NULL,
            height          INTEGER NOT NULL,
            byte_len        INTEGER NOT NULL,
            rel_path        TEXT NOT NULL,
            PRIMARY KEY (image_uuid, settings_digest, level_kind)
        );
        CREATE INDEX IF NOT EXISTS idx_pyramid_levels_rel_path
            ON pyramid_levels (rel_path);

        CREATE TABLE IF NOT EXISTS deleted_pyramids (
            rel_path      TEXT PRIMARY KEY,
            deleted_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_deleted_pyramids_ts
            ON deleted_pyramids (deleted_at_ms ASC);",
    )?;
    Ok(())
}

fn next_timestamp_ms(last: &mut i64) -> i64 {
    let sys = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let ts = if sys > *last { sys } else { *last + 1 };
    *last = ts;
    ts
}

fn fnv1a_64(parts: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for &b in *part {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn insert_tombstone(tx: &rusqlite::Transaction<'_>, rel_path: &str, now_ms: i64) -> Result<()> {
    tx.execute(
        "INSERT INTO deleted_pyramids (rel_path, deleted_at_ms)
         VALUES (?1, ?2)
         ON CONFLICT(rel_path) DO UPDATE SET deleted_at_ms = excluded.deleted_at_ms",
        params![rel_path, now_ms],
    )?;
    Ok(())
}

fn collect_pyramid_paths(
    tx: &rusqlite::Transaction<'_>,
    image_uuid: &str,
    settings_digest: &str,
) -> Result<Vec<String>> {
    let mut stmt = tx.prepare(
        "SELECT rel_path FROM pyramid_levels
         WHERE image_uuid = ?1 AND settings_digest = ?2",
    )?;
    let paths = stmt
        .query_map(params![image_uuid, settings_digest], |r| r.get(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(paths)
}

fn collect_stale_fingerprint_paths(
    tx: &rusqlite::Transaction<'_>,
    image_uuid: &str,
    expected_fp: &str,
) -> Result<Vec<String>> {
    let mut stmt = tx.prepare(
        "SELECT l.rel_path
         FROM pyramid_levels l
         INNER JOIN pyramids p
            ON p.image_uuid = l.image_uuid AND p.settings_digest = l.settings_digest
         WHERE p.image_uuid = ?1 AND p.source_fingerprint != ?2",
    )?;
    let paths = stmt
        .query_map(params![image_uuid, expected_fp], |r| r.get(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(paths)
}

fn evict_lru_tx(
    tx: &rusqlite::Transaction<'_>,
    max_bytes: u64,
    now_ms: i64,
    protect: Option<(&str, &str)>,
) -> Result<()> {
    let mut total: i64 = tx.query_row(
        "SELECT COALESCE(SUM(total_bytes), 0) FROM pyramids",
        [],
        |r| r.get(0),
    )?;
    let limit = max_bytes as i64;
    if total <= limit {
        return Ok(());
    }

    let mut stmt = tx.prepare(
        "SELECT image_uuid, settings_digest, total_bytes
         FROM pyramids
         ORDER BY last_access_ms ASC",
    )?;
    let victims: Vec<(String, String, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    for (uuid, digest, bytes) in victims {
        if total <= limit {
            break;
        }
        if let Some((p_uuid, p_digest)) = protect {
            if uuid == p_uuid && digest == p_digest {
                continue;
            }
        }
        let paths = collect_pyramid_paths(tx, &uuid, &digest)?;
        for path in &paths {
            insert_tombstone(tx, path, now_ms)?;
        }
        tx.execute(
            "DELETE FROM pyramid_levels
             WHERE image_uuid = ?1 AND settings_digest = ?2",
            params![uuid, digest],
        )?;
        tx.execute(
            "DELETE FROM pyramids
             WHERE image_uuid = ?1 AND settings_digest = ?2",
            params![uuid, digest],
        )?;
        total -= bytes;
    }

    Ok(())
}

fn drain_due_tombstones_tx(
    tx: &rusqlite::Transaction<'_>,
    now_ms: i64,
    grace_ms: u64,
) -> Result<Vec<String>> {
    let cutoff = now_ms.saturating_sub(grace_ms as i64);
    let mut stmt = tx.prepare(
        "SELECT rel_path FROM deleted_pyramids WHERE deleted_at_ms <= ?1",
    )?;
    let paths: Vec<String> = stmt
        .query_map(params![cutoff], |r| r.get(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    if !paths.is_empty() {
        tx.execute(
            "DELETE FROM deleted_pyramids WHERE deleted_at_ms <= ?1",
            params![cutoff],
        )?;
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    struct TempDirGuard {
        path: PathBuf,
    }

    impl TempDirGuard {
        fn new(name: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path = std::env::temp_dir()
                .join(format!("aurora_disk_preview_test_{name}_{pid}_{n}"));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn cold_start_reopen_db_hits_all_pyramid_levels() {
        let dir = TempDirGuard::new("cold_reopen");
        let cfg = DiskPreviewCacheConfig::new(&dir.path, 16 << 20);

        {
            let cache = DiskPreviewCache::open(cfg.clone()).expect("open cache");
            assert_eq!(cache.journal_mode().expect("journal").to_lowercase(), "wal");

            for (idx, level) in PyramidLevelKind::ALL.iter().copied().enumerate() {
                let long_side = level.default_long_side().unwrap_or(6000);
                let payload = vec![(idx as u8).wrapping_add(17); 256 + idx * 64];
                cache
                    .put(PyramidLevelWrite {
                        image_uuid: "img-001",
                        settings_digest: "digest-v1",
                        source_fingerprint: "fp-src-a",
                        level_kind: level,
                        long_side,
                        width: long_side,
                        height: (long_side * 2) / 3,
                        data: &payload,
                    })
                    .expect("put level");
            }
        }

        // Cold start: reopen from disk and verify every level hits with exact bytes.
        let reopened = DiskPreviewCache::open(cfg).expect("reopen cache");
        let stats = reopened.stats().expect("stats");
        assert_eq!(stats.pyramid_count, 1);
        assert_eq!(stats.level_count, 4);

        for (idx, level) in PyramidLevelKind::ALL.iter().copied().enumerate() {
            let long_side = level.default_long_side().unwrap_or(6000);
            let expected = vec![(idx as u8).wrapping_add(17); 256 + idx * 64];
            let hit = reopened
                .get("img-001", "digest-v1", "fp-src-a", level)
                .expect("get level")
                .expect("level must hit after reopen");
            assert_eq!(hit.level_kind, level);
            assert_eq!(hit.long_side, long_side);
            assert_eq!(hit.width, long_side);
            assert_eq!(hit.height, (long_side * 2) / 3);
            assert_eq!(hit.data, expected);
        }
    }

    #[test]
    fn digest_change_and_source_fingerprint_mismatch_invalidate() {
        let dir = TempDirGuard::new("digest_fp");
        let cache =
            DiskPreviewCache::open(DiskPreviewCacheConfig::new(&dir.path, 16 << 20))
                .expect("open cache");

        let data = vec![0xABu8; 512];
        cache
            .put(PyramidLevelWrite {
                image_uuid: "img-002",
                settings_digest: "digest-A",
                source_fingerprint: "fp-1",
                level_kind: PyramidLevelKind::Standard,
                long_side: 2048,
                width: 2048,
                height: 1365,
                data: &data,
            })
            .expect("put");

        // Digest change misses.
        assert!(cache
            .get("img-002", "digest-B", "fp-1", PyramidLevelKind::Standard)
            .expect("get")
            .is_none());

        // Same digest & fingerprint hits.
        assert!(cache
            .get("img-002", "digest-A", "fp-1", PyramidLevelKind::Standard)
            .expect("get")
            .is_some());

        // Source file replaced -> fingerprint mismatch invalidates old record and misses.
        assert!(cache
            .get("img-002", "digest-A", "fp-2", PyramidLevelKind::Standard)
            .expect("get")
            .is_none());

        // Even querying with old fingerprint now misses because entry was purged.
        assert!(cache
            .get("img-002", "digest-A", "fp-1", PyramidLevelKind::Standard)
            .expect("get")
            .is_none());
        assert_eq!(cache.stats().expect("stats").pyramid_count, 0);
    }

    #[test]
    fn lru_eviction_by_max_bytes_and_orphan_cleanup() {
        let dir = TempDirGuard::new("lru_evict");
        // Budget fits 2 entries of 100 bytes each, not 3.
        let cache = DiskPreviewCache::open(DiskPreviewCacheConfig::new(&dir.path, 220))
            .expect("open cache");

        for (uuid, byte) in [("img-1", 1u8), ("img-2", 2u8)] {
            cache
                .put(PyramidLevelWrite {
                    image_uuid: uuid,
                    settings_digest: "d1",
                    source_fingerprint: "fp",
                    level_kind: PyramidLevelKind::Thumbnail,
                    long_side: 320,
                    width: 320,
                    height: 240,
                    data: &vec![byte; 100],
                })
                .expect("put");
        }

        // Touch img-1 so img-2 becomes the least-recently-used entry.
        assert!(cache
            .get("img-1", "d1", "fp", PyramidLevelKind::Thumbnail)
            .expect("touch img-1")
            .is_some());

        // Insert img-3 (100 bytes): total would be 300 > 220 -> img-2 must be evicted.
        cache
            .put(PyramidLevelWrite {
                image_uuid: "img-3",
                settings_digest: "d1",
                source_fingerprint: "fp",
                level_kind: PyramidLevelKind::Thumbnail,
                long_side: 320,
                width: 320,
                height: 240,
                data: &vec![3u8; 100],
            })
            .expect("put img-3");

        let stats = cache.stats().expect("stats");
        assert!(stats.total_bytes <= 220, "total_bytes={}", stats.total_bytes);
        assert_eq!(stats.pyramid_count, 2);

        assert!(cache
            .get("img-2", "d1", "fp", PyramidLevelKind::Thumbnail)
            .expect("get img-2")
            .is_none());
        assert!(cache
            .get("img-1", "d1", "fp", PyramidLevelKind::Thumbnail)
            .expect("get img-1")
            .is_some());
        assert!(cache
            .get("img-3", "d1", "fp", PyramidLevelKind::Thumbnail)
            .expect("get img-3")
            .is_some());

        // Place an orphan file under blobs/ and verify cleanup_orphans removes only it.
        let orphan = dir.path.join("blobs").join("orphan_garbage.bin");
        fs::write(&orphan, b"stale orphan").expect("write orphan");
        assert!(orphan.exists());

        let cleaned = cache.cleanup_orphans().expect("cleanup_orphans");
        assert_eq!(cleaned, 1);
        assert!(!orphan.exists());

        // Live entries remain intact.
        assert!(cache
            .get("img-1", "d1", "fp", PyramidLevelKind::Thumbnail)
            .expect("get img-1 after orphan cleanup")
            .is_some());
        assert!(cache
            .get("img-3", "d1", "fp", PyramidLevelKind::Thumbnail)
            .expect("get img-3 after orphan cleanup")
            .is_some());
    }

    #[test]
    fn concurrent_read_write_does_not_corrupt() {
        let dir = TempDirGuard::new("concurrent");
        let cache = Arc::new(
            DiskPreviewCache::open(DiskPreviewCacheConfig::new(&dir.path, 32 << 20))
                .expect("open cache"),
        );

        let mut handles = Vec::new();
        for tid in 0..8u8 {
            let c = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for iter in 0..20u8 {
                    let uuid = format!("img-t{tid}-i{}", iter % 5);
                    let digest = format!("dig-{}", iter % 3);
                    let fp = "fp-stable";
                    let mut payload = vec![0u8; 384];
                    for (idx, b) in payload.iter_mut().enumerate() {
                        *b = tid
                            .wrapping_mul(31)
                            .wrapping_add(iter)
                            .wrapping_add(idx as u8);
                    }
                    c.put(PyramidLevelWrite {
                        image_uuid: &uuid,
                        settings_digest: &digest,
                        source_fingerprint: fp,
                        level_kind: PyramidLevelKind::Thumbnail,
                        long_side: 320,
                        width: 320,
                        height: 213,
                        data: &payload,
                    })
                    .expect("concurrent put");

                    if let Some(hit) = c
                        .get(&uuid, &digest, fp, PyramidLevelKind::Thumbnail)
                        .expect("concurrent get")
                    {
                        assert_eq!(hit.data.len(), 384);
                        assert_eq!(hit.width, 320);
                        assert_eq!(hit.height, 213);
                        // Verify payload internal consistency (every byte follows same stride).
                        let base = hit.data[0];
                        for (idx, &b) in hit.data.iter().enumerate() {
                            assert_eq!(b, base.wrapping_add(idx as u8));
                        }
                    }
                }
            }));
        }

        for h in handles {
            h.join().expect("worker thread panicked");
        }

        // Verify database integrity after concurrent stress by reopening.
        drop(cache);
        let reopened =
            DiskPreviewCache::open(DiskPreviewCacheConfig::new(&dir.path, 32 << 20))
                .expect("reopen after concurrent stress");
        assert!(reopened.stats().expect("stats").level_count > 0);
    }
}
