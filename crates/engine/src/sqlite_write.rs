//! Writing the small SQLite files the session export needs.
//!
//! There is no SQLite crate in this tree: the catalog reader is a hand-written
//! b-tree reader, and this is the matching writer. It writes what that reader
//! consumes — `sqlite_master` plus ordinary tables — and nothing else: no
//! indexes, triggers, free lists or overflow pages.
//!
//! shortcut: a row that would need an overflow page is rejected instead of
//! chained, and keys live in at most 32-way interior pages; both limits are far
//! above a session's image rows, raise them if an export ever hits the error.

use std::path::Path;

use anyhow::{anyhow, bail, Result};

/// Page size; the reader and every Capture One database we have seen use 4096.
const PAGE: usize = 4096;
/// Page 1's b-tree header starts after the 100-byte file header.
const PAGE1_HEADER: usize = 100;
/// A table-leaf cell larger than this would need an overflow page.
const MAX_LEAF_CELL: usize = PAGE - 35;
/// Children per interior page; 32 keeps every interior cell well under a page.
const MAX_INTERIOR_CHILDREN: usize = 32;

/// One value of a table row.
#[derive(Clone, Debug)]
pub enum Cell {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
}

impl Cell {
    pub fn text(s: &str) -> Self {
        Cell::Text(s.to_string())
    }
}

/// A table to write; `columns` are full definitions, so
/// `"Z_ENT INTEGER PRIMARY KEY"` behaves the same as in a real file.
pub struct SynthTable {
    name: String,
    columns: Vec<String>,
    rows: Vec<(Option<i64>, Vec<Cell>)>,
}

impl SynthTable {
    pub fn new(name: impl Into<String>, columns: &[&str]) -> Self {
        Self {
            name: name.into(),
            columns: columns.iter().map(|c| (*c).to_string()).collect(),
            rows: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Appends a row whose rowid is its position, starting at 1.
    pub fn row(mut self, cells: Vec<Cell>) -> Self {
        self.rows.push((None, cells));
        self
    }

    /// Appends a row with an explicit rowid, which is what an `INTEGER PRIMARY
    /// KEY` column stores: SQLite keeps the value in the b-tree key, not in the
    /// record.
    pub fn row_with_id(mut self, rowid: i64, cells: Vec<Cell>) -> Self {
        self.rows.push((Some(rowid), cells));
        self
    }

    fn records(&self) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut out = Vec::with_capacity(self.rows.len());
        for (index, (rowid, cells)) in self.rows.iter().enumerate() {
            if cells.len() != self.columns.len() {
                bail!(
                    "{} row {} has {} cells for {} columns",
                    self.name,
                    index,
                    cells.len(),
                    self.columns.len()
                );
            }
            out.push((rowid.unwrap_or(index as i64 + 1), record(cells)));
        }
        out.sort_by_key(|(rowid, _)| *rowid);
        Ok(out)
    }
}

/// Writes `tables` to `path` as a new SQLite database.
pub fn write_tables(path: &Path, tables: &[SynthTable]) -> Result<()> {
    std::fs::write(path, sqlite_bytes(tables)?)?;
    Ok(())
}

/// A whole SQLite database file holding `tables`.
pub fn sqlite_bytes(tables: &[SynthTable]) -> Result<Vec<u8>> {
    // SQLite looks tables up by name, so `sqlite_master` is stored that way.
    let mut sorted: Vec<&SynthTable> = tables.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut master: Vec<(i64, Vec<u8>)> = Vec::new();
    let mut body: Vec<Vec<u8>> = Vec::new();
    for (index, table) in sorted.iter().enumerate() {
        let (pages, root) = table_pages(&table.records()?, 2 + body.len() as u32)?;
        body.extend(pages);
        let sql = format!("CREATE TABLE \"{}\" ({})", table.name, table.columns.join(", "));
        master.push((
            index as i64 + 1,
            record(&[
                Cell::text("table"),
                Cell::text(&table.name),
                Cell::text(&table.name),
                Cell::Int(i64::from(root)),
                Cell::Text(sql),
            ]),
        ));
    }

    // A real session's DDL is far wider than one page, so page 1 is an
    // interior page over leaves written after the table pages.
    let (mut file, master_leaves) = master_page(&master, 2 + body.len() as u32)?;
    let header = &mut file[..100];
    header[..16].copy_from_slice(b"SQLite format 3\0");
    header[16..18].copy_from_slice(&(PAGE as u16).to_be_bytes());
    header[18] = 1;
    header[19] = 1;
    header[21] = 64;
    header[22] = 32;
    header[23] = 32;
    header[24..28].copy_from_slice(&1u32.to_be_bytes());
    header[28..32]
        .copy_from_slice(&((body.len() + 1 + master_leaves.len()) as u32).to_be_bytes());
    header[40..44].copy_from_slice(&1u32.to_be_bytes());
    header[44..48].copy_from_slice(&4u32.to_be_bytes());
    header[56..60].copy_from_slice(&1u32.to_be_bytes());
    header[92..96].copy_from_slice(&1u32.to_be_bytes());
    header[96..100].copy_from_slice(&3_047_000u32.to_be_bytes());

    for page in body {
        file.extend(page);
    }
    for page in master_leaves {
        file.extend(page);
    }
    Ok(file)
}

/// Page 1, plus the `sqlite_master` leaves when its rows do not fit one page.
fn master_page(rows: &[(i64, Vec<u8>)], first_leaf: u32) -> Result<(Vec<u8>, Vec<Vec<u8>>)> {
    let cells: Vec<Vec<u8>> = rows
        .iter()
        .map(|(rowid, payload)| table_cell(*rowid, payload))
        .collect();
    if fits(PAGE1_HEADER + 8, &cells) {
        return Ok((page(0x0d, PAGE1_HEADER, &cells, 0)?, Vec::new()));
    }
    let (leaves, level) = leaf_level(rows, first_leaf)?;
    if level.len() < 2 {
        bail!("sqlite_master has one row too large for page 1");
    }
    Ok((interior_page(&level, PAGE1_HEADER)?, leaves))
}

/// Whether `cells` plus their 2-byte pointers fit behind `prefix` bytes.
fn fits(prefix: usize, cells: &[Vec<u8>]) -> bool {
    prefix + cells.iter().map(|cell| 2 + cell.len()).sum::<usize>() <= PAGE
}

/// The pages of one table's b-tree, in page order, plus its root page number.
fn table_pages(rows: &[(i64, Vec<u8>)], first_page: u32) -> Result<(Vec<Vec<u8>>, u32)> {
    let (mut pages, mut level) = leaf_level(rows, first_page)?;
    let mut next_page = first_page + level.len() as u32;
    while level.len() > 1 {
        let mut parents = Vec::new();
        for (index, chunk) in level.chunks(MAX_INTERIOR_CHILDREN).enumerate() {
            let page = next_page + index as u32;
            pages.push(interior_page(chunk, 0)?);
            parents.push((page, chunk[chunk.len() - 1].1));
        }
        next_page += parents.len() as u32;
        level = parents;
    }
    Ok((pages, level[0].0))
}

/// Splits rows into leaf pages and returns the pages plus `(page, last rowid)`
/// per leaf, which is what an interior level keys on.
fn leaf_level(
    rows: &[(i64, Vec<u8>)],
    first_page: u32,
) -> Result<(Vec<Vec<u8>>, Vec<(u32, i64)>)> {
    // Split the rows into leaf pages that each fit inside one 4096-byte page.
    let mut groups: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut cells: Vec<Vec<u8>> = Vec::new();
    let mut used = 8usize;
    for (rowid, payload) in rows {
        if *rowid < 1 {
            bail!("rowid {rowid} is not a positive integer");
        }
        let cell = table_cell(*rowid, payload);
        if cell.len() > MAX_LEAF_CELL {
            bail!("row {rowid} is too large for a session table");
        }
        if !cells.is_empty() && used + 2 + cell.len() > PAGE {
            groups.push(std::mem::take(&mut cells));
            used = 8;
        }
        used += 2 + cell.len();
        cells.push(cell);
    }
    groups.push(cells);

    let mut pages = Vec::with_capacity(groups.len());
    // A table-interior key is the largest rowid below it.
    let mut level = Vec::with_capacity(groups.len());
    let mut consumed = 0usize;
    for (i, group) in groups.iter().enumerate() {
        pages.push(leaf_page(group)?);
        consumed += group.len();
        let key = consumed
            .checked_sub(1)
            .and_then(|last| rows.get(last))
            .map(|(rowid, _)| *rowid)
            .unwrap_or(0);
        level.push((first_page + i as u32, key));
    }
    Ok((pages, level))
}

/// A table-leaf cell: payload length, rowid, payload.
fn table_cell(rowid: i64, payload: &[u8]) -> Vec<u8> {
    let mut cell = varint(payload.len() as u64);
    cell.extend(varint(rowid as u64));
    cell.extend_from_slice(payload);
    cell
}

fn record(cells: &[Cell]) -> Vec<u8> {
    let parts: Vec<(u64, Vec<u8>)> = cells.iter().map(serial).collect();
    let types: Vec<u8> = parts.iter().flat_map(|(t, _)| varint(*t)).collect();
    let header_len = varint_len(1 + types.len() as u64) + types.len();
    let mut out = varint(header_len as u64);
    out.extend(types);
    out.extend(parts.into_iter().flat_map(|(_, body)| body));
    out
}

fn serial(cell: &Cell) -> (u64, Vec<u8>) {
    match cell {
        Cell::Null => (0, Vec::new()),
        Cell::Int(i) if (-128..=127).contains(i) => (1, vec![*i as i8 as u8]),
        Cell::Int(i) if (-32768..=32767).contains(i) => (2, (*i as i16).to_be_bytes().to_vec()),
        Cell::Int(i) if (i32::MIN as i64..=i32::MAX as i64).contains(i) => {
            (4, (*i as i32).to_be_bytes().to_vec())
        }
        Cell::Int(i) => (6, i.to_be_bytes().to_vec()),
        Cell::Real(r) => (7, r.to_be_bytes().to_vec()),
        Cell::Text(t) => (13 + 2 * t.len() as u64, t.as_bytes().to_vec()),
    }
}

fn varint(mut v: u64) -> Vec<u8> {
    let mut bytes = vec![(v & 0x7f) as u8];
    v >>= 7;
    while v > 0 {
        bytes.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    bytes.reverse();
    bytes
}

fn varint_len(mut v: u64) -> usize {
    let mut len = 1;
    while v >= 0x80 {
        v >>= 7;
        len += 1;
    }
    len
}

fn leaf_page(cells: &[Vec<u8>]) -> Result<Vec<u8>> {
    page(0x0d, 0, cells, 0)
}

/// One table-interior page: a 4-byte child pointer per cell and the rightmost
/// child after the header.
fn interior_page(children: &[(u32, i64)], header: usize) -> Result<Vec<u8>> {
    let cells: Vec<Vec<u8>> = children[..children.len() - 1]
        .iter()
        .map(|(page, key)| {
            let mut cell = page.to_be_bytes().to_vec();
            cell.extend(varint(*key as u64));
            cell
        })
        .collect();
    page(0x05, header, &cells, children[children.len() - 1].0)
}

/// One page; `header` is where its b-tree header starts (100 on page 1).
fn page(kind: u8, header: usize, cells: &[Vec<u8>], right: u32) -> Result<Vec<u8>> {
    let pointers_at = header + if kind == 0x05 { 12 } else { 8 };
    let mut buf = vec![0u8; PAGE];
    let mut free = PAGE;
    let mut pointers = Vec::with_capacity(cells.len());
    for cell in cells {
        free = free
            .checked_sub(cell.len())
            .ok_or_else(|| anyhow!("{} cells do not fit one {PAGE}-byte page", cells.len()))?;
        buf[free..free + cell.len()].copy_from_slice(cell);
        pointers.push(free as u16);
    }
    if free < pointers_at + cells.len() * 2 {
        bail!("{} cells and their pointers do not fit one {PAGE}-byte page", cells.len());
    }
    buf[header] = kind;
    buf[header + 3..header + 5].copy_from_slice(&(cells.len() as u16).to_be_bytes());
    buf[header + 5..header + 7].copy_from_slice(&(free as u16).to_be_bytes());
    if kind == 0x05 {
        buf[header + 8..header + 12].copy_from_slice(&right.to_be_bytes());
    }
    for (i, at) in pointers.into_iter().enumerate() {
        buf[pointers_at + i * 2..pointers_at + i * 2 + 2].copy_from_slice(&at.to_be_bytes());
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c1catalog::Value;

    #[test]
    fn a_table_that_needs_several_pages_keeps_every_row() {
        let mut table = SynthTable::new("T", &["Z_PK INTEGER PRIMARY KEY", "ZVALUE TEXT"]);
        let mut expected = Vec::new();
        for i in 1..=600i64 {
            let value = format!("{i:0>120}");
            expected.push((i, value.clone()));
            table = table.row_with_id(i, vec![Cell::Null, Cell::text(&value)]);
        }
        let bytes = sqlite_bytes(&[table]).unwrap();
        assert!(bytes.len() > 2 * PAGE, "600 rows need more than one leaf page");

        let path = std::env::temp_dir().join(format!("sqlite-write-{}.db", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let mut db = crate::c1catalog::SqliteReader::open(&path).unwrap();
        let rows = db.select("T").unwrap();
        assert_eq!(rows.len(), expected.len());
        for (row, (pk, value)) in rows.iter().zip(&expected) {
            assert_eq!(row.get("Z_PK"), Some(&Value::Int(*pk)));
            assert_eq!(row.get("ZVALUE"), Some(&Value::Text(value.clone())));
        }
        let _ = std::fs::remove_file(&path);
    }

    /// A session's `CREATE TABLE` rows together are wider than page 1; the
    /// writer used to put them all there and panicked on the subtract.
    #[test]
    fn a_session_schema_wider_than_page_one_round_trips() {
        // The table names and column counts `c1session_export` writes.
        let schema: [(&str, usize); 12] = [
            ("ZDOCUMENTCONTENT", 2),
            ("ZPATHLOCATION", 8),
            ("ZCOLLECTION", 45),
            ("ZIMAGE", 27),
            ("ZVARIANT", 20),
            ("ZVARIANTLAYER", 49),
            ("ZVARIANTMETADATA", 8),
            ("ZKEYWORD", 14),
            ("ZIMAGEINCOLLECTION", 6),
            ("ZRETOUCHINGLAYER", 42),
            ("ZENTITIES", 2),
            ("ZDOCUMENTSETTING", 4),
        ];
        let mut ddl = 0usize;
        let mut tables = Vec::new();
        for (index, (name, columns)) in schema.iter().enumerate() {
            let columns: Vec<String> =
                (0..*columns).map(|c| format!("ZCAPTURESETTINGC{c} TEXT")).collect();
            let mut cells = vec![Cell::Null; columns.len()];
            cells[0] = Cell::text(name);
            let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
            tables.push(SynthTable::new(*name, &refs).row_with_id(index as i64 + 1, cells));
            ddl += name.len() + columns.iter().map(|c| c.len() + 4).sum::<usize>() + 40;
        }
        assert!(ddl > PAGE - PAGE1_HEADER, "the schema must not fit page 1: {ddl} bytes");

        let bytes = sqlite_bytes(&tables).unwrap();
        let path = std::env::temp_dir().join(format!("sqlite-master-{}.db", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let mut db = crate::c1catalog::SqliteReader::open(&path).unwrap();
        for (name, columns) in &schema {
            assert!(db.has_table(name).unwrap(), "{name} missing from sqlite_master");
            assert_eq!(db.columns(name).unwrap().len(), *columns, "{name} column count");
            let rows = db.select(name).unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].get("ZCAPTURESETTINGC0"), Some(&Value::Text((*name).to_string())));
        }
        let _ = std::fs::remove_file(&path);
    }
}
