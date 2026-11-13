//! Read-only access to a Capture One `.cocatalogdb` or `.cosessiondb`.
//!
//! The file is plain SQLite; the API is a minimal read-only b-tree reader so the
//! engine keeps zero new dependencies and works offline. Only table lookups are
//! implemented: no indexes, no writes, no WAL replay.
//!
//! A catalog and a session share one schema; only `ZDOCUMENTCONTENT.ZDOCUMENTTYPE`
//! says which one a file is.
//!
//! C1 never has exclusive access to its catalog, so the file is opened read-only
//! and a `.readlock` sibling only produces a warning, never a block.

use crate::c1_masks::mask_layers_from_rows;
use crate::c1_model::{from_layer_row, to_adjustment_set, C1VariantSettings};
use crate::c1_retouch::{group_retouching_rows, RetouchingLayer};
use crate::catalog::CollectionKind;
use anyhow::{anyhow, bail, Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// One SQLite column value, in the storage classes the catalog actually uses.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Value {
    /// Text form used by [`from_layer_row`], which takes C1's stored strings.
    fn as_string(&self) -> Option<String> {
        match self {
            Value::Null | Value::Blob(_) => None,
            Value::Int(i) => Some(i.to_string()),
            Value::Real(r) => Some(r.to_string()),
            Value::Text(t) => Some(t.clone()),
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Real(r) => Some(*r),
            _ => None,
        }
    }

    fn as_u32(&self) -> Option<u32> {
        self.as_f64().filter(|v| *v >= 0.0).map(|v| v as u32)
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Value::Text(t) => Some(t),
            _ => None,
        }
    }
}

type Row = HashMap<String, Value>;

/// What a Capture One document database is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum C1DocumentType {
    #[default]
    Catalog,
    Session,
}

/// Minimal read-only SQLite reader (schema + full table scans).
pub struct SqliteReader {
    file: File,
    page_size: usize,
    usable: usize,
    encoding: u32,
    /// Pages taken from a replayed `-wal`; they shadow the database file.
    wal_pages: HashMap<u32, Vec<u8>>,
}

impl SqliteReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("opening {} read-only", path.display()))?;
        let mut header = [0u8; 100];
        file.read_exact(&mut header).context("reading SQLite header")?;
        if &header[..16] != b"SQLite format 3\0" {
            bail!("{} is not a SQLite database", path.display());
        }
        let raw_page = u16::from_be_bytes([header[16], header[17]]);
        let page_size = if raw_page == 1 { 65536 } else { raw_page as usize };
        if page_size < 512 || !page_size.is_power_of_two() {
            bail!("unsupported SQLite page size {page_size}");
        }
        let reserved = header[20] as usize;
        let encoding = u32::from_be_bytes([header[56], header[57], header[58], header[59]]);
        Ok(Self { file, page_size, usable: page_size - reserved, encoding, wal_pages: HashMap::new() })
    }

    fn read_page(&mut self, page: u32) -> Result<Vec<u8>> {
        if let Some(buf) = self.wal_pages.get(&page) {
            return Ok(buf.clone());
        }
        let offset = u64::from(page.checked_sub(1).ok_or_else(|| anyhow!("page 0"))?)
            * self.page_size as u64;
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; self.page_size];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Replays a `-wal` file into memory. The database file is never written.
    ///
    /// Only frames up to and including the last commit frame are applied; a
    /// trailing uncommitted transaction is reported, not applied. A frame whose
    /// salt or checksum does not match makes the replay fail, so a broken WAL
    /// is reported instead of silently ignored.
    pub fn apply_wal(&mut self, wal_path: &Path) -> Result<WalReplay> {
        let data = std::fs::read(wal_path)
            .with_context(|| format!("reading {}", wal_path.display()))?;
        if data.len() < 32 {
            bail!("WAL header is truncated ({} bytes)", data.len());
        }
        let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        // 0x377f0682 checksums its words little-endian, 0x377f0683 big-endian.
        let big_endian = match magic {
            0x377f_0682 => false,
            0x377f_0683 => true,
            _ => bail!("not a SQLite WAL file (magic {magic:#x})"),
        };
        let page_size = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
        if page_size != self.page_size {
            bail!("WAL page size {page_size} does not match the database page size {}", self.page_size);
        }
        let salt1 = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let salt2 = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        let stored = (
            u32::from_be_bytes([data[24], data[25], data[26], data[27]]),
            u32::from_be_bytes([data[28], data[29], data[30], data[31]]),
        );
        if wal_checksum(&data[..24], (0, 0), big_endian) != stored {
            bail!("WAL header checksum does not match");
        }

        let frame_size = 24 + page_size;
        let slots = (data.len() - 32) / frame_size;
        let mut frames: Vec<(u32, Vec<u8>)> = Vec::new();
        let mut commit_at: Option<usize> = None;
        let mut offset = 32usize;
        let mut index = 0usize;
        let mut sum = stored;
        while offset + frame_size <= data.len() {
            let head = &data[offset..offset + 24];
            let pgno = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
            let db_size = u32::from_be_bytes([head[4], head[5], head[6], head[7]]);
            let frame_salt = (
                u32::from_be_bytes([head[8], head[9], head[10], head[11]]),
                u32::from_be_bytes([head[12], head[13], head[14], head[15]]),
            );
            let frame_sum = (
                u32::from_be_bytes([head[16], head[17], head[18], head[19]]),
                u32::from_be_bytes([head[20], head[21], head[22], head[23]]),
            );
            if frame_salt != (salt1, salt2) {
                bail!(
                    "WAL frame {index} has a stale salt; {} frames from there on cannot be replayed",
                    slots - index
                );
            }
            let page = &data[offset + 24..offset + frame_size];
            let next = wal_checksum(page, wal_checksum(&head[..8], sum, big_endian), big_endian);
            if next != frame_sum {
                bail!(
                    "WAL frame {index} fails its checksum; {} frames from there on cannot be replayed",
                    slots - index
                );
            }
            sum = next;
            frames.push((pgno, page.to_vec()));
            if db_size != 0 {
                commit_at = Some(frames.len());
            }
            offset += frame_size;
            index += 1;
        }
        if offset != data.len() {
            bail!(
                "WAL ends in a partial frame after {index} frames; at least one frame is missing"
            );
        }

        let total = frames.len();
        let applied = commit_at.unwrap_or(0);
        for (pgno, page) in frames.into_iter().take(applied) {
            self.wal_pages.insert(pgno, page);
        }
        Ok(WalReplay {
            frames_total: total,
            frames_applied: applied,
            frames_uncommitted: total - applied,
            pages_overridden: self.wal_pages.len(),
        })
    }

    fn table_root(&mut self, table: &str) -> Result<Option<u32>> {
        Ok(self
            .master_rows()?
            .iter()
            .find(|r| {
                r.get("name").and_then(Value::as_str) == Some(table)
                    && r.get("type").and_then(Value::as_str) == Some("table")
            })
            .and_then(|r| r.get("rootpage"))
            .and_then(Value::as_u32))
    }

    /// Whether `sqlite_master` names a table, without reading its rows.
    pub fn has_table(&mut self, table: &str) -> Result<bool> {
        Ok(self.table_root(table)?.is_some())
    }

    /// `sqlite_master` has a fixed column order, so it can be read without a schema.
    fn master_rows(&mut self) -> Result<Vec<Row>> {
        let mut raw = Vec::new();
        self.walk(1, &mut raw)?;
        Ok(raw
            .into_iter()
            .map(|(_, cols)| {
                ["type", "name", "tbl_name", "rootpage", "sql"]
                    .into_iter()
                    .zip(cols)
                    .map(|(name, v)| (name.to_string(), v))
                    .collect()
            })
            .collect())
    }

    /// Full table scan. Column order comes from the `CREATE TABLE` text.
    pub fn select(&mut self, table: &str) -> Result<Vec<Row>> {
        let root = self
            .table_root(table)?
            .ok_or_else(|| anyhow!("table {table} not found"))?;
        let columns = self.columns(table)?;
        let pk_alias = self.primary_key_alias(table)?;
        let mut rows = Vec::new();
        self.walk(root, &mut rows)?;
        Ok(rows
            .into_iter()
            .map(|(rowid, raw)| {
                let mut map: Row = columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), raw.get(i).cloned().unwrap_or(Value::Null)))
                    .collect();
                // `Z_PK` is declared INTEGER PRIMARY KEY, so SQLite stores NULL in
                // the record and takes the value from the rowid.
                if matches!(map.get("Z_PK"), None | Some(Value::Null)) {
                    map.insert("Z_PK".into(), Value::Int(rowid));
                }
                // The same holds for any other INTEGER PRIMARY KEY column, e.g.
                // `ZENTITIES.Z_ENT`.
                if let Some(alias) = &pk_alias {
                    if matches!(map.get(alias.as_str()), None | Some(Value::Null)) {
                        map.insert(alias.clone(), Value::Int(rowid));
                    }
                }
                map
            })
            .collect())
    }

    /// The column that is an `INTEGER PRIMARY KEY`, if the table has one.
    /// SQLite stores such a column as NULL in the record and uses the rowid.
    fn primary_key_alias(&mut self, table: &str) -> Result<Option<String>> {
        let rows = self.master_rows()?;
        let sql = rows
            .iter()
            .find(|r| r.get("name").and_then(Value::as_str) == Some(table))
            .and_then(|r| r.get("sql"))
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("no schema for table {table}"))?;
        Ok(parse_columns(sql).into_iter().find(|name| {
            let upper = sql.to_ascii_uppercase();
            upper.contains(&format!("{} INTEGER PRIMARY KEY", name.to_ascii_uppercase()))
                && !name.eq_ignore_ascii_case("Z_PK")
        }))
    }

    /// Column names in declaration order, parsed from `sqlite_master.sql`.
    pub fn columns(&mut self, table: &str) -> Result<Vec<String>> {
        let rows = self.master_rows()?;
        let sql = rows
            .iter()
            .find(|r| r.get("name").and_then(Value::as_str) == Some(table))
            .and_then(|r| r.get("sql"))
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("no schema for table {table}"))?;
        Ok(parse_columns(sql))
    }

    fn walk(&mut self, page: u32, out: &mut Vec<(i64, Vec<Value>)>) -> Result<()> {
        let buf = self.read_page(page)?;
        let hdr = if page == 1 { 100 } else { 0 };
        let kind = buf[hdr];
        let cells = u16::from_be_bytes([buf[hdr + 3], buf[hdr + 4]]) as usize;
        let interior = kind == 2 || kind == 5;
        let ptr_base = hdr + if interior { 12 } else { 8 };

        if interior {
            let right = u32::from_be_bytes([
                buf[hdr + 8],
                buf[hdr + 9],
                buf[hdr + 10],
                buf[hdr + 11],
            ]);
            for i in 0..cells {
                let at = ptr_base + i * 2;
                let off = u16::from_be_bytes([buf[at], buf[at + 1]]) as usize;
                let child = u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
                self.walk(child, out)?;
            }
            return self.walk(right, out);
        }

        for i in 0..cells {
            let at = ptr_base + i * 2;
            let off = u16::from_be_bytes([buf[at], buf[at + 1]]) as usize;
            let (len, n1) = varint(&buf[off..])?;
            let (rowid, n2) = varint(&buf[off + n1..])?;
            let payload = self.payload(&buf[off + n1 + n2..], len as usize)?;
            out.push((rowid as i64, record(&payload, self.encoding)?));
        }
        Ok(())
    }

    /// Reassembles a record payload, following the overflow chain when needed.
    fn payload(&mut self, cell: &[u8], len: usize) -> Result<Vec<u8>> {
        let max_local = self.usable - 35;
        if len <= max_local {
            if cell.len() < len {
                bail!("truncated cell");
            }
            return Ok(cell[..len].to_vec());
        }
        let min_local = ((self.usable - 12) * 32 / 255) - 23;
        let k = min_local + ((len - min_local) % (self.usable - 4));
        let local = if k <= max_local { k } else { min_local };
        if cell.len() < local + 4 {
            bail!("truncated overflow cell");
        }
        let mut out = cell[..local].to_vec();
        let mut next = u32::from_be_bytes([cell[local], cell[local + 1], cell[local + 2], cell[local + 3]]);
        while out.len() < len {
            if next == 0 {
                bail!("overflow chain ended early");
            }
            let buf = self.read_page(next)?;
            next = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
            let take = (len - out.len()).min(self.usable - 4);
            out.extend_from_slice(&buf[4..4 + take]);
        }
        Ok(out)
    }
}

fn varint(bytes: &[u8]) -> Result<(u64, usize)> {
    let mut value = 0u64;
    for i in 0..9 {
        let b = *bytes.get(i).ok_or_else(|| anyhow!("truncated varint"))?;
        if i == 8 {
            return Ok(((value << 8) | u64::from(b), 9));
        }
        value = (value << 7) | u64::from(b & 0x7f);
        if b & 0x80 == 0 {
            return Ok((value, i + 1));
        }
    }
    unreachable!()
}

/// SQLite's WAL checksum: a rolling sum over 8-byte word pairs.
fn wal_checksum(data: &[u8], seed: (u32, u32), big_endian: bool) -> (u32, u32) {
    let (mut s0, mut s1) = seed;
    for chunk in data.chunks_exact(8) {
        let (x0, x1) = if big_endian {
            (
                u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                u32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            )
        } else {
            (
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            )
        };
        s0 = s0.wrapping_add(x0).wrapping_add(s1);
        s1 = s1.wrapping_add(x1).wrapping_add(s0);
    }
    (s0, s1)
}

fn record(bytes: &[u8], encoding: u32) -> Result<Vec<Value>> {
    let (header_len, n) = varint(bytes)?;
    let header_len = header_len as usize;
    if header_len > bytes.len() {
        bail!("record header longer than payload");
    }
    let mut types = Vec::new();
    let mut at = n;
    while at < header_len {
        let (t, len) = varint(&bytes[at..])?;
        types.push(t);
        at += len;
    }

    let mut out = Vec::with_capacity(types.len());
    let mut body = header_len;
    for t in types {
        let size = match t {
            0 | 8 | 9 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 6,
            6 | 7 => 8,
            10 | 11 => bail!("reserved serial type {t}"),
            n => ((n - if n % 2 == 0 { 12 } else { 13 }) / 2) as usize,
        };
        let chunk = bytes
            .get(body..body + size)
            .ok_or_else(|| anyhow!("record body truncated"))?;
        out.push(match t {
            0 => Value::Null,
            7 => Value::Real(f64::from_be_bytes(chunk.try_into().unwrap())),
            8 => Value::Int(0),
            9 => Value::Int(1),
            1..=6 => Value::Int(int_of(chunk)),
            n if n % 2 == 0 => Value::Blob(chunk.to_vec()),
            _ => Value::Text(decode_text(chunk, encoding)),
        });
        body += size;
    }
    Ok(out)
}

fn int_of(bytes: &[u8]) -> i64 {
    match bytes.len() {
        8 => i64::from_be_bytes(bytes.try_into().unwrap()),
        n => {
            let mut v: i64 = if bytes[0] & 0x80 != 0 { -1 } else { 0 };
            for b in &bytes[..n] {
                v = (v << 8) | i64::from(*b);
            }
            v
        }
    }
}

fn decode_text(bytes: &[u8], encoding: u32) -> String {
    match encoding {
        2 | 3 => {
            let units: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| {
                    if encoding == 2 {
                        u16::from_le_bytes([c[0], c[1]])
                    } else {
                        u16::from_be_bytes([c[0], c[1]])
                    }
                })
                .collect();
            String::from_utf16_lossy(&units)
        }
        _ => String::from_utf8_lossy(bytes).into_owned(),
    }
}

/// Column names from a `CREATE TABLE` statement, skipping table constraints.
fn parse_columns(sql: &str) -> Vec<String> {
    let Some(open) = sql.find('(') else { return Vec::new() };
    let mut depth = 0usize;
    let mut end = open;
    for (i, c) in sql.char_indices().skip_while(|(i, _)| *i < open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = &sql[open + 1..end.max(open + 1)];

    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(inner[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(inner[start..].trim());

    parts
        .iter()
        .filter_map(|part| {
            let first = part.split_whitespace().next()?.trim_matches(|c| c == '"' || c == '\'' || c == '`' || c == '[' || c == ']');
            let upper = first.to_ascii_uppercase();
            if matches!(
                upper.as_str(),
                "PRIMARY" | "FOREIGN" | "UNIQUE" | "CHECK" | "CONSTRAINT"
            ) {
                None
            } else if first.is_empty() {
                None
            } else {
                Some(first.to_string())
            }
        })
        .collect()
}

/// One image row plus its location.
#[derive(Debug, Clone)]
pub struct C1Image {
    pub pk: i64,
    pub uuid: String,
    pub file_name: String,
    pub abs_path: String,
    pub width: u32,
    pub height: u32,
    pub iso: Option<i64>,
    pub camera_make: String,
    pub camera_model: String,
    pub camera_lens: String,
    pub aperture: Option<f32>,
    pub shutter: Option<f32>,
    pub focal_length: Option<f32>,
    pub shot_at: Option<i64>,
}

/// One variant with the adjustments C1 stored for it.
#[derive(Debug, Clone)]
pub struct C1Variant {
    pub pk: i64,
    pub image_pk: i64,
    pub index: i64,
    pub canvas: Option<(f32, f32)>,
    pub settings: C1VariantSettings,
    /// Read-only `ZRETOUCHINGLAYER` rows. Not applied to edits or rendering.
    pub retouching: Vec<RetouchingLayer>,
    /// Parsed `Z_ENT=23` rows. They are not applied until their `.comask`
    /// geometry is available.
    pub local_layers: Vec<crate::c1_masks::C1MaskLayer>,
    pub adjustments: crate::AdjustmentSet,
    pub rating: i64,
    pub color_tag: i64,
    pub keywords: Vec<String>,
    pub mask_layers: usize,
    pub mask_names: Vec<String>,
}

/// One node of C1's collection tree (`ZCOLLECTION`).
#[derive(Debug, Clone)]
pub struct C1Collection {
    pub pk: i64,
    pub uuid: String,
    pub name: String,
    /// The collection this one hangs under, if any.
    pub parent: Option<i64>,
    /// C1's entity id (`Z_ENT`), which says what kind of collection it is.
    pub ent: i64,
    /// The entity name from `ZENTITIES`, e.g. `AlbumCollection`.
    pub ent_name: String,
    /// What this collection is in the interface's terms.
    pub kind: CollectionKind,
    /// The catalog path a folder collection points at.
    pub folder_path: Option<String>,
    pub index: i64,
    /// Images in this collection (`ZIMAGEINCOLLECTION`).
    pub images: Vec<i64>,
}

/// One node of C1's keyword tree (`ZKEYWORD`).
#[derive(Debug, Clone)]
pub struct C1Keyword {
    pub pk: i64,
    pub name: String,
    pub parent: Option<i64>,
}

/// What a `-wal` replay did.
#[derive(Debug, Clone, Default)]
pub struct WalReplay {
    pub frames_total: usize,
    pub frames_applied: usize,
    /// Frames past the last commit; they belong to an unfinished transaction.
    pub frames_uncommitted: usize,
    pub pages_overridden: usize,
}

#[derive(Debug, Clone, Default)]
pub struct C1Catalog {
    pub document_type: C1DocumentType,
    pub images: Vec<C1Image>,
    pub variants: Vec<C1Variant>,
    pub collections: Vec<C1Collection>,
    pub keywords: Vec<C1Keyword>,
    pub warnings: Vec<String>,
    /// Set when a `-wal` was found; `None` when the catalog has no pending WAL.
    pub wal: Option<WalReplay>,
}

impl C1Catalog {
    /// Reads a catalog without ever writing to it.
    pub fn open_ro(path: &Path) -> Result<Self> {
        let mut db = SqliteReader::open(path)?;
        let mut warnings = Vec::new();

        if path.parent().is_some_and(|p| p.join(".readlock").exists()) {
            warnings.push("Capture One holds a .readlock; reading anyway".into());
        }
        // A pending `-wal` holds edits C1 has not checkpointed into the
        // database file. Replay it into memory; the file itself is never
        // written. If the WAL cannot be replayed we say so with the number of
        // missing frames instead of pretending the catalog is complete.
        let wal_path = PathBuf::from(format!("{}-wal", path.display()));
        let mut wal = None;
        if wal_path.exists() {
            match db.apply_wal(&wal_path) {
                Ok(report) => {
                    if report.frames_uncommitted > 0 {
                        warnings.push(format!(
                            "-wal replayed {} of {} frames; {} uncommitted frames ignored",
                            report.frames_applied, report.frames_total, report.frames_uncommitted
                        ));
                    }
                    wal = Some(report);
                }
                Err(err) => warnings.push(format!("-wal could not be replayed: {err}")),
            }
        }

        // A catalog and a session share this schema; a stray SQLite file does not.
        // Only the three tables that hold the images themselves are required:
        // leaner and older sessions drop metadata, keyword and collection tables.
        for table in ["ZIMAGE", "ZVARIANT", "ZVARIANTLAYER"] {
            if !db.has_table(table)? {
                bail!(
                    "{} is not a Capture One catalog or session: it has no {table} table",
                    path.display()
                );
            }
        }
        let document_type = Self::read_document_type(&mut db, path, &mut warnings)?;
        let images = Self::read_images(&mut db, &mut warnings)?;
        let collections = Self::read_collections(&mut db, &mut warnings)?;
        let keywords = Self::read_keywords(&mut db, &mut warnings)?;
        let variants = Self::read_variants(&mut db, &keywords, &mut warnings)?;
        Ok(Self { document_type, images, variants, collections, keywords, warnings, wal })
    }

    /// `ZDOCUMENTCONTENT.ZDOCUMENTTYPE`: 1 is a catalog, 2 a session.
    ///
    /// Those are the only values Capture One is known to write. Anything else
    /// is reported rather than guessed at, so a wrong reader never runs silently.
    fn read_document_type(
        db: &mut SqliteReader,
        path: &Path,
        warnings: &mut Vec<String>,
    ) -> Result<C1DocumentType> {
        if !db.has_table("ZDOCUMENTCONTENT")? {
            // The extension still says what the user opened, so a session never
            // gets silently downgraded to a catalog.
            let by_extension = path
                .extension()
                .and_then(|e| e.to_str())
                .and_then(|e| {
                    if e.eq_ignore_ascii_case("cosessiondb") {
                        Some(C1DocumentType::Session)
                    } else if e.eq_ignore_ascii_case("cocatalogdb") {
                        Some(C1DocumentType::Catalog)
                    } else {
                        None
                    }
                });
            let Some(kind) = by_extension else {
                bail!(
                    "{} has no ZDOCUMENTCONTENT table and its name does not say whether it is a catalog or a session",
                    path.display()
                );
            };
            warnings.push(format!(
                "no ZDOCUMENTCONTENT table; reading {} as a {}",
                path.file_name().unwrap_or_default().to_string_lossy(),
                match kind {
                    C1DocumentType::Catalog => "catalog",
                    C1DocumentType::Session => "session",
                }
            ));
            return Ok(kind);
        }
        let value = db
            .select("ZDOCUMENTCONTENT")?
            .first()
            .and_then(|row| row.get("ZDOCUMENTTYPE"))
            // C1 mixes storage classes; accept the type as text too.
            .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())));
        match value.map(|v| v.round() as i64) {
            Some(1) => Ok(C1DocumentType::Catalog),
            Some(2) => Ok(C1DocumentType::Session),
            Some(other) => bail!("unknown Capture One document type {other}"),
            None => bail!("ZDOCUMENTCONTENT has no ZDOCUMENTTYPE"),
        }
    }

    /// Reads `ZCOLLECTION` plus the `ZIMAGEINCOLLECTION` membership rows.
    ///
    /// C1 has no `ZTYPE`/`ZKIND` columns: what a collection *is* comes from its
    /// `Z_ENT`, which `ZENTITIES` names. Folder collections carry no name
    /// either; their name is the catalog path they point at.
    fn read_collections(
        db: &mut SqliteReader,
        warnings: &mut Vec<String>,
    ) -> Result<Vec<C1Collection>> {
        let mut paths: HashMap<i64, String> = HashMap::new();
        for row in optional_rows(db, "ZPATHLOCATION", warnings)? {
            let Some(pk) = row.get("Z_PK").and_then(Value::as_f64) else { continue };
            paths.insert(pk as i64, location_path(&row));
        }

        let mut entities: HashMap<i64, String> = HashMap::new();
        for row in optional_rows(db, "ZENTITIES", warnings)? {
            let (Some(ent), Some(name)) = (
                row.get("Z_ENT").and_then(Value::as_f64),
                row.get("ZNAME").and_then(Value::as_str),
            ) else {
                continue;
            };
            entities.insert(ent as i64, name.to_string());
        }

        let mut members: HashMap<i64, Vec<i64>> = HashMap::new();
        for row in optional_rows(db, "ZIMAGEINCOLLECTION", warnings)? {
            let (Some(coll), Some(image)) = (
                row.get("ZCOLLECTION").and_then(Value::as_f64),
                row.get("ZIMAGE").and_then(Value::as_f64),
            ) else {
                continue;
            };
            members.entry(coll as i64).or_default().push(image as i64);
        }

        let rows = optional_rows(db, "ZCOLLECTION", warnings)?;
        // The catalog root is the one collection that belongs to no project;
        // it also names the four system collections C1 keeps for itself.
        let root_pk = rows
            .iter()
            .find_map(|r| match (r.get("ZPROJECT").and_then(Value::as_f64), r.get("Z_PK").and_then(Value::as_f64)) {
                (Some(project), Some(pk)) if project < 0.0 => Some(pk as i64),
                _ => None,
            })
            .unwrap_or(0);
        let root = rows.iter().find(|r| r.get("Z_PK").and_then(Value::as_f64) == Some(root_pk as f64));
        let role = |column: &str| {
            root.and_then(|r| r.get(column).and_then(Value::as_f64))
                .map(|v| v as i64)
                .filter(|v| *v > 0)
        };
        let all_images = role("ZALLIMAGESCOLLECTION");
        let recent_captures = role("ZLASTCAPTURESCOLLECTION");
        let recent_imports = role("ZLASTIMPORTSCOLLECTION");
        let trash = role("ZTRASHCOLLECTION");

        let mut out = Vec::new();
        for row in rows {
            let pk = row.get("Z_PK").and_then(Value::as_f64).unwrap_or(0.0) as i64;
            let ent = row.get("Z_ENT").and_then(Value::as_f64).unwrap_or(0.0) as i64;
            let parent = row
                .get("ZPARENT")
                .and_then(Value::as_f64)
                .map(|p| p as i64)
                .filter(|p| *p > 0)
                .or_else(|| {
                    row.get("ZPROJECT")
                        .and_then(Value::as_f64)
                        .map(|p| p as i64)
                        .filter(|p| *p > 0)
                        .filter(|p| *p == root_pk)
                });
            let name = row.get("ZNAME").and_then(Value::as_str).unwrap_or("").to_string();
            let folder_path = row
                .get("ZFOLDERLOCATION")
                .and_then(Value::as_f64)
                .and_then(|p| paths.get(&(p as i64)).cloned());
            let ent_name = entities.get(&ent).cloned().unwrap_or_default();
            let kind = if Some(pk) == all_images {
                CollectionKind::AllImages
            } else if Some(pk) == recent_captures {
                CollectionKind::RecentCaptures
            } else if Some(pk) == recent_imports {
                CollectionKind::RecentImports
            } else if Some(pk) == trash {
                CollectionKind::Trash
            } else {
                match ent_name.as_str() {
                    "AlbumCollection" | "CloudCollection" | "VariantCollection" => CollectionKind::Album,
                    "FolderCollection" | "CatalogFolderCollection" | "TempFolderCollection" | "SessionFolderCollection" => CollectionKind::Folder,
                    "SmartCollection" => CollectionKind::SmartAlbum,
                    "ProjectCollection" | "Collection" => CollectionKind::Project,
                    _ => CollectionKind::Group,
                }
            };
            out.push(C1Collection {
                pk,
                uuid: row.get("ZUUID").and_then(Value::as_str).unwrap_or("").to_string(),
                name,
                parent,
                ent,
                ent_name,
                kind,
                folder_path,
                index: row.get("ZCOLLECTIONINDEX").and_then(Value::as_f64).unwrap_or(0.0) as i64,
                images: members.remove(&pk).unwrap_or_default(),
            });
        }
        // Folder collections carry no name; they are named by the catalog path
        // they point at, relative to the folder they sit under.
        let folder_paths: HashMap<i64, String> = out
            .iter()
            .filter_map(|c| c.folder_path.as_ref().map(|p| (c.pk, p.clone())))
            .collect();
        for c in out.iter_mut() {
            if !c.name.is_empty() {
                continue;
            }
            let Some(path) = c.folder_path.clone() else { continue };
            let parent_path = c
                .parent
                .and_then(|p| folder_paths.get(&p).cloned())
                .unwrap_or_default();
            let relative = path
                .strip_prefix(parent_path.as_str())
                .unwrap_or(&path)
                .trim_start_matches(['\\', '/']);
            c.name = if relative.is_empty() {
                path.clone()
            } else {
                relative.to_string()
            };
        }
        Ok(out)
    }

    /// Reads the keyword tree (`ZKEYWORD`): C1 nests keywords with `ZPARENT`.
    fn read_keywords(
        db: &mut SqliteReader,
        warnings: &mut Vec<String>,
    ) -> Result<Vec<C1Keyword>> {
        Ok(optional_rows(db, "ZKEYWORD", warnings)?
            .iter()
            .filter_map(|row| {
                Some(C1Keyword {
                    pk: row.get("Z_PK")?.as_f64()? as i64,
                    name: row.get("ZNAME").and_then(Value::as_str).unwrap_or("").to_string(),
                    parent: row
                        .get("ZPARENT")
                        .and_then(Value::as_f64)
                        .map(|p| p as i64)
                        .filter(|p| *p > 0),
                })
            })
            .collect())
    }

    fn read_images(
        db: &mut SqliteReader,
        warnings: &mut Vec<String>,
    ) -> Result<Vec<C1Image>> {
        let locations: HashMap<i64, String> = optional_rows(db, "ZPATHLOCATION", warnings)?
            .iter()
            .filter_map(|r| {
                let pk = r.get("Z_PK")?.as_f64()? as i64;
                Some((pk, location_path(r)))
            })
            .collect();

        Ok(db
            .select("ZIMAGE")?
            .iter()
            .map(|r| {
                let loc = r
                    .get("ZIMAGELOCATION")
                    .and_then(Value::as_f64)
                    .map(|pk| pk as i64)
                    .and_then(|pk| locations.get(&pk).cloned())
                    .unwrap_or_default();
                let file_name = r.get("ZIMAGEFILENAME").and_then(Value::as_str).unwrap_or("");
                C1Image {
                    pk: r.get("Z_PK").and_then(Value::as_f64).map(|v| v as i64).unwrap_or(0),
                    uuid: r.get("ZIMAGEUUID").and_then(Value::as_str).unwrap_or("").to_string(),
                    file_name: file_name.to_string(),
                    abs_path: join_file(&loc, file_name),
                    width: r.get("ZWIDTH").and_then(Value::as_u32).unwrap_or(0),
                    height: r.get("ZHEIGHT").and_then(Value::as_u32).unwrap_or(0),
                    iso: r.get("ZISO").and_then(Value::as_f64).map(|v| v as i64),
                    camera_make: r.get("ZCAMERA_MAKE").and_then(Value::as_str).unwrap_or("").to_string(),
                    camera_model: r.get("ZCAMERA_MODEL").and_then(Value::as_str).unwrap_or("").to_string(),
                    camera_lens: r.get("ZCAMERA_LENS").and_then(Value::as_str).unwrap_or("").to_string(),
                    aperture: r.get("ZEXP_APERTURE").and_then(Value::as_f64).map(|v| v as f32),
                    shutter: r.get("ZEXP_SHUTTERSPEED").and_then(Value::as_f64).map(|v| v as f32),
                    focal_length: r.get("ZEXP_FOCALLENGTH").and_then(Value::as_f64).map(|v| v as f32),
                    shot_at: r.get("ZEXP_DATE").and_then(Value::as_f64).map(|v| v as i64),
                }
            })
            .collect())
    }

    fn read_variants(
        db: &mut SqliteReader,
        keywords: &[C1Keyword],
        warnings: &mut Vec<String>,
    ) -> Result<Vec<C1Variant>> {
        let layers = db.select("ZVARIANTLAYER")?;
        // Source pixel size per image, to tell whether a variant's canvas is turned.
        let dims: HashMap<i64, (u32, u32)> = db
            .select("ZIMAGE")?
            .iter()
            .map(|r| {
                let pk = r.get("Z_PK").and_then(Value::as_f64).unwrap_or(0.0) as i64;
                let w = r.get("ZWIDTH").and_then(Value::as_f64).unwrap_or(0.0) as u32;
                let h = r.get("ZHEIGHT").and_then(Value::as_f64).unwrap_or(0.0) as u32;
                (pk, (w, h))
            })
            .collect();
        let metadata: HashMap<i64, Row> = optional_rows(db, "ZVARIANTMETADATA", warnings)?
            .into_iter()
            .filter_map(|r| Some((r.get("Z_PK")?.as_f64()? as i64, r)))
            .collect();

        let variant_rows = db.select("ZVARIANT")?;
        let variant_pks: HashSet<i64> = variant_rows
            .iter()
            .filter_map(|v| v.get("Z_PK").and_then(Value::as_f64).map(|pk| pk as i64))
            .collect();
        // shortcut: catalogs older than the ZFACE schema may lack this table,
        // which is treated as "no retouching rows" rather than a read failure.
        let retouch_rows: Vec<HashMap<String, Option<String>>> = db
            .select("ZRETOUCHINGLAYER")
            .unwrap_or_default()
            .iter()
            .map(|r| r.iter().map(|(k, v)| (k.clone(), v.as_string())).collect())
            .collect();
        let mut retouching = group_retouching_rows(&retouch_rows, &variant_pks);

        let mut out = Vec::new();
        for v in variant_rows {
            let pk = v.get("Z_PK").and_then(Value::as_f64).unwrap_or(0.0) as i64;
            let mut mine: Vec<&Row> = layers
                .iter()
                .filter(|l| l.get("ZVARIANT").and_then(Value::as_f64) == Some(pk as f64))
                .collect();
            mine.sort_by_key(|l| l.get("Z_PK").and_then(Value::as_f64).unwrap_or(0.0) as i64);

            // Z_ENT 18 is the full snapshot, 17 the delta on top of it; 20 is the
            // composite result and 23 are per-mask deltas that must not go global.
            let mut merged: HashMap<String, Option<String>> = HashMap::new();
            let mut mask_layers = 0;
            let mut mask_names = Vec::new();
            let mut local_rows = Vec::new();
            for layer in &mine {
                match layer.get("Z_ENT").and_then(Value::as_f64).unwrap_or(0.0) as i64 {
                    18 | 17 => {
                        for (k, v) in layer.iter() {
                            if let Some(text) = v.as_string() {
                                merged.insert(k.clone(), Some(text));
                            }
                        }
                    }
                    23 => {
                        mask_layers += 1;
                        if let Some(name) = layer.get("ZNAME").and_then(Value::as_str) {
                            mask_names.push(name.to_string());
                        }
                        let row: HashMap<&str, Option<String>> =
                            layer.iter().map(|(k, v)| (k.as_str(), v.as_string())).collect();
                        local_rows.push(row);
                    }
                    _ => {}
                }
            }
            let borrowed: HashMap<&str, Option<String>> =
                merged.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            let local_layers = mask_layers_from_rows(&local_rows);
            let mut settings = from_layer_row(&borrowed);
            // ZCROP is in canvas pixels and ZCANVASSIZE is only stored on ZVARIANT,
            // never on the layer row, so the canvas has to come from the variant.
            let canvas = v.get("ZCANVASSIZE").and_then(Value::as_str).and_then(parse_pair);
            if let Some((w, h)) = canvas {
                settings.canvas_size = [w, h];
            }
            let mut adjustments = to_adjustment_set(&settings);
            // C1 keeps the crop in the variant's canvas space. A portrait canvas on a
            // landscape file (or the reverse) means the variant is turned a quarter:
            // without that turn the crop lands on the wrong pixels.
            if let Some((cw, ch)) = canvas {
                let source = v
                    .get("ZIMAGE")
                    .and_then(Value::as_f64)
                    .map(|pk| pk as i64)
                    .and_then(|pk| dims.get(&pk).copied())
                    .unwrap_or((0, 0));
                let portrait_canvas = ch > cw;
                let portrait_source = source.1 > source.0;
                if source.0 > 0 && portrait_canvas != portrait_source {
                    adjustments.geometry.orientation = 90;
                }
            }

            let mut rating = 0i64;
            let mut color_tag = 0i64;
            for layer in &mine {
                let Some(meta_pk) = layer.get("ZMETADATA").and_then(Value::as_f64) else { continue };
                let Some(meta) = metadata.get(&(meta_pk as i64)) else { continue };
                rating = rating.max(meta.get("ZBASIC_RATING").and_then(Value::as_f64).unwrap_or(0.0) as i64);
                color_tag = color_tag.max(meta.get("ZCOLOR_TAG_INDEX").and_then(Value::as_f64).unwrap_or(0.0) as i64);
            }

            let paths = keyword_paths(keywords);
            let keyword_list = mine
                .iter()
                .filter_map(|l| l.get("ZMETADATA").and_then(Value::as_f64))
                .filter_map(|pk| metadata.get(&(pk as i64)))
                .filter_map(|m| m.get("ZCONTENT_KEYWORDS").and_then(Value::as_str))
                .flat_map(|s| s.split(','))
                .map(str::trim)
                .filter(|s| !s.is_empty())
                // A keyword that lives under a parent keyword is qualified
                // with it, so the interface can show the hierarchy.
                .map(|name| paths.get(name).cloned().unwrap_or_else(|| name.to_string()))
                .collect();

            out.push(C1Variant {
                pk,
                image_pk: v.get("ZIMAGE").and_then(Value::as_f64).map(|v| v as i64).unwrap_or(0),
                index: v.get("ZINDEX").and_then(Value::as_f64).map(|v| v as i64).unwrap_or(0),
                canvas,
                settings,
                local_layers,
                adjustments,
                rating,
                color_tag,
                keywords: keyword_list,
                mask_layers,
                mask_names,
                retouching: retouching.by_variant.remove(&pk).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    pub fn variants_of(&self, image_uuid: &str) -> Vec<&C1Variant> {
        let Some(image) = self.images.iter().find(|i| i.uuid == image_uuid) else {
            return Vec::new();
        };
        self.variants.iter().filter(|v| v.image_pk == image.pk).collect()
    }
}

fn parse_pair(text: &str) -> Option<(f32, f32)> {
    let mut it = text.split(';');
    let w = it.next()?.trim().parse().ok()?;
    let h = it.next()?.trim().parse().ok()?;
    Some((w, h))
}

/// Rows of a table that leaner or older sessions may not carry.
///
/// A missing table reads as empty and is reported once; it never turns the file
/// into an invalid one. Only `ZIMAGE`, `ZVARIANT` and `ZVARIANTLAYER` are
/// required.
fn optional_rows(
    db: &mut SqliteReader,
    table: &str,
    warnings: &mut Vec<String>,
) -> Result<Vec<Row>> {
    if !db.has_table(table)? {
        let warning = format!("no {table} table; reading it as empty");
        if !warnings.contains(&warning) {
            warnings.push(warning);
        }
        return Ok(Vec::new());
    }
    db.select(table)
}

/// The folder a `ZPATHLOCATION` row points at: `ZWINROOT + ZRELATIVEPATH`, or
/// the mac root when the Windows one is empty (a session made on macOS).
fn location_path(row: &Row) -> String {
    let win = row.get("ZWINROOT").and_then(Value::as_str).unwrap_or("");
    let mac = row.get("ZMACROOT").and_then(Value::as_str).unwrap_or("");
    let rel = row.get("ZRELATIVEPATH").and_then(Value::as_str).unwrap_or("");
    let root = if win.is_empty() { mac } else { win };
    if root.is_empty() || root.ends_with(['/', '\\']) {
        format!("{root}{rel}")
    } else {
        let sep = if root.contains('\\') { '\\' } else { '/' };
        format!("{root}{sep}{rel}")
    }
}

/// Appends a file name, keeping the separator style of the folder it joins.
fn join_file(dir: &str, name: &str) -> String {
    let dir = dir.trim_end_matches(['/', '\\']);
    if dir.is_empty() {
        return name.to_string();
    }
    let sep = if dir.contains('\\') { '\\' } else { '/' };
    format!("{dir}{sep}{name}")
}

/// Maps every keyword name to its `parent/child` path.
pub fn keyword_paths(keywords: &[C1Keyword]) -> HashMap<String, String> {
    let by_pk: HashMap<i64, &C1Keyword> = keywords.iter().map(|k| (k.pk, k)).collect();
    let mut out = HashMap::new();
    for keyword in keywords {
        let mut names = vec![keyword.name.clone()];
        let mut parent = keyword.parent;
        let mut guard = 0;
        while let Some(pk) = parent {
            let Some(node) = by_pk.get(&pk) else { break };
            names.push(node.name.clone());
            parent = node.parent;
            guard += 1;
            if guard > 64 {
                break;
            }
        }
        names.reverse();
        out.insert(keyword.name.clone(), names.join("/"));
    }
    out
}

/// WAL fixtures for the tests. A real `-wal` cannot be shipped in testdata,
/// because opening a catalog with SQLite would rewrite it; this one was
/// produced by SQLite from a copy of the fixture instead (`pragma
/// journal_mode=WAL`, then two `ZKEYWORD` inserts and one
/// `ZVARIANTMETADATA.ZCONTENT_KEYWORDS` update, committed) and stored as
/// (byte, u16 length) runs.
#[cfg(all(test, feature = "external-fixtures"))]
#[path = "../local-fixtures/catalog_support.rs"]
pub(crate) mod test_support;

#[cfg(test)]
mod tests {
    #[cfg(feature = "external-fixtures")]
    use super::test_support::{
        fixture_path, fixture_with_wal, session_tables, sqlite_file, synthetic_session, wal_bytes,
    };
    use super::*;

    /// A `-wal` holding the given `(page, db_size, image)` frames; a non-zero
    /// `db_size` marks a commit, exactly like SQLite writes it.
    #[cfg(feature = "external-fixtures")]
    fn wal_with_frames(frames: &[(u32, u32, Vec<u8>)]) -> Vec<u8> {
        let salt = (0x1122_3344u32, 0x5566_7788u32);
        let mut header = Vec::new();
        header.extend(0x377f_0682u32.to_be_bytes());
        header.extend(3_007_000u32.to_be_bytes());
        header.extend(4096u32.to_be_bytes());
        header.extend(0u32.to_be_bytes());
        header.extend(salt.0.to_be_bytes());
        header.extend(salt.1.to_be_bytes());
        let mut sum = wal_checksum(&header, (0, 0), false);
        header.extend(sum.0.to_be_bytes());
        header.extend(sum.1.to_be_bytes());

        let mut out = header;
        for (page, db_size, image) in frames {
            assert_eq!(image.len(), 4096);
            let mut head = Vec::new();
            head.extend(page.to_be_bytes());
            head.extend(db_size.to_be_bytes());
            head.extend(salt.0.to_be_bytes());
            head.extend(salt.1.to_be_bytes());
            sum = wal_checksum(image, wal_checksum(&head[..8], sum, false), false);
            head.extend(sum.0.to_be_bytes());
            head.extend(sum.1.to_be_bytes());
            out.extend(head);
            out.extend(image);
        }
        out
    }


    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lumen-c1-{tag}-{}", std::process::id()))
    }

    /// The synthetic session bytes without one table, written as `name`.
    #[cfg(feature = "external-fixtures")]
    fn session_without(dir: &Path, table: &str, name: &str) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let tables: Vec<_> = session_tables()
            .into_iter()
            .filter(|t| t.name() != table)
            .collect();
        let path = dir.join(name);
        std::fs::write(&path, sqlite_file(&tables)).unwrap();
        path
    }





    #[cfg(feature = "external-fixtures")]
    #[test]
    fn a_session_without_variant_metadata_still_opens() {
        let dir = temp_dir("session-no-metadata");
        let db = session_without(&dir, "ZVARIANTMETADATA", "Lean.cosessiondb");
        let before = std::fs::read(&db).unwrap();

        let c = C1Catalog::open_ro(&db).expect("a lean session is not a broken file");
        assert_eq!(c.document_type, C1DocumentType::Session);
        assert_eq!((c.images.len(), c.variants.len()), (2, 4));
        assert!(
            c.warnings.iter().any(|w| w.contains("ZVARIANTMETADATA") && w.contains("as empty")),
            "warnings: {:?}",
            c.warnings
        );
        // No ratings or keywords without the table, and no error either.
        assert!(c.variants.iter().all(|v| v.rating == 0 && v.keywords.is_empty()));
        // The adjustments and mask layers are still read.
        assert_eq!(c.variants.iter().map(|v| v.mask_layers).sum::<usize>(), 1);
        assert!(c.variants.iter().any(|v| v.settings.exposure == 0.75));
        assert_eq!(std::fs::read(&db).unwrap(), before, "the database file changed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "external-fixtures")]
    #[test]
    fn a_missing_document_content_table_follows_the_file_name() {
        let dir = temp_dir("session-no-doctype");
        for (name, want) in [
            ("Lean.cosessiondb", C1DocumentType::Session),
            ("Lean.cocatalogdb", C1DocumentType::Catalog),
        ] {
            let db = session_without(&dir, "ZDOCUMENTCONTENT", name);
            let before = std::fs::read(&db).unwrap();
            let c = C1Catalog::open_ro(&db).expect("the document type comes from the name");
            assert_eq!(c.document_type, want, "{name}");
            assert!(
                c.warnings.iter().any(|w| w.contains("no ZDOCUMENTCONTENT") && w.contains(name)),
                "{name} warnings: {:?}",
                c.warnings
            );
            assert_eq!(std::fs::read(&db).unwrap(), before, "{name} changed");
        }
        // With neither the table nor a telling extension, guessing is refused.
        let odd = session_without(&dir, "ZDOCUMENTCONTENT", "Lean.sqlite");
        let err = format!("{:#}", C1Catalog::open_ro(&odd).unwrap_err());
        assert!(err.contains("does not say whether it is a catalog or a session"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

















    #[cfg(feature = "external-fixtures")]
    #[test]
    fn a_wal_replays_only_the_frames_up_to_its_last_commit() {
        let dir = temp_dir("wal-commit");
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("Session.cosessiondb");
        let bytes = synthetic_session();
        std::fs::write(&db_path, &bytes).unwrap();

        let page = |n: usize| bytes[n * 4096..(n + 1) * 4096].to_vec();
        // Frames 1-2 rewrite ZIMAGE unchanged, frame 2 carries the commit marker.
        // Frame 3 zeroes ZDOCUMENTCONTENT and was never committed, so applying it
        // would make the database unreadable.
        std::fs::write(
            dir.join("Session.cosessiondb-wal"),
            wal_with_frames(&[(5, 0, page(4)), (5, 7, page(4)), (3, 0, vec![0u8; 4096])]),
        )
        .unwrap();

        let c = C1Catalog::open_ro(&db_path).expect("the committed frames replay");
        let report = c.wal.expect("the wal was found");
        assert_eq!((report.frames_total, report.frames_applied), (3, 2));
        assert_eq!(report.frames_uncommitted, 1);
        assert_eq!(report.pages_overridden, 1);
        assert!(
            c.warnings.iter().any(|w| w.contains("2 of 3 frames") && w.contains("1 uncommitted")),
            "warnings: {:?}",
            c.warnings
        );
        // The uncommitted frame never reached the metadata page.
        assert_eq!(c.document_type, C1DocumentType::Session);
        assert_eq!(c.images.len(), 2);
        assert_eq!(std::fs::read(&db_path).unwrap(), bytes, "the database file changed");
        let _ = std::fs::remove_dir_all(&dir);
    }








    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        assert!(C1Catalog::open_ro(Path::new("no-such.cocatalogdb")).is_err());
    }


    #[test]
    fn parses_column_list_from_create_table() {
        let cols = parse_columns(
            "CREATE TABLE \"t\" (\"Z_PK\" INTEGER PRIMARY KEY, ZNAME VARCHAR, amount REAL, PRIMARY KEY (Z_PK))",
        );
        assert_eq!(cols, vec!["Z_PK", "ZNAME", "amount"]);
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/c1catalog-tests.rs");
}
