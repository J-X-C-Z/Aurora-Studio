//! Read-only Adobe Lightroom Classic catalog (`.lrcat`) importer and Lua develop settings parser.
//!
//! Opens `.lrcat` SQLite databases strictly with `SQLITE_OPEN_READ_ONLY` so the
//! source catalog is never modified, reads core library tables (`Adobe_images`,
//! `AgLibraryFile`, `AgLibraryFolder`, `AgLibraryRootFolder`,
//! `Adobe_imageDevelopSettings`), and parses Lightroom's Lua `s = { ... }`
//! develop-settings text into structured key-value maps.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};

use crate::adjust::{AdjustmentSet, Curve, CurvePoint, WhiteBalanceMode};

/// Lightroom pick/flag state (`Adobe_images.pick`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LrcatPickStatus {
    #[default]
    Unflagged,
    Picked,
    Rejected,
}

impl LrcatPickStatus {
    pub fn from_raw(value: Option<f64>) -> Self {
        match value {
            Some(v) if v > 0.5 => Self::Picked,
            Some(v) if v < -0.5 => Self::Rejected,
            _ => Self::Unflagged,
        }
    }
}

/// One imported image record from a Lightroom `.lrcat` catalog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LrcatImageRecord {
    pub image_id: i64,
    pub file_path: PathBuf,
    pub file_name: String,
    /// 0 ..= 5 star rating.
    pub rating: i8,
    /// Color label text (e.g. `"Red"`, `"Yellow"`, `"Green"`, `"Blue"`, `"Purple"`, or empty).
    pub color_label: String,
    pub pick: LrcatPickStatus,
    pub capture_time: Option<String>,
    pub file_format: Option<String>,
    pub develop_settings_raw: Option<String>,
    pub develop_settings: Option<LuaDevelopSettings>,
}

/// Dynamic Lua value parsed from `Adobe_imageDevelopSettings.text`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LuaValue {
    Nil,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<LuaValue>),
    Table(BTreeMap<String, LuaValue>),
}

impl LuaValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            LuaValue::Number(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            LuaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            LuaValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_table(&self) -> Option<&BTreeMap<String, LuaValue>> {
        match self {
            LuaValue::Table(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[LuaValue]> {
        match self {
            LuaValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// Parsed Lightroom Lua develop settings table.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LuaDevelopSettings {
    /// Flattened view of the develop `settings` map (or top-level table if already flat).
    pub settings: BTreeMap<String, LuaValue>,
}

impl LuaDevelopSettings {
    /// Parses a Lightroom `Adobe_imageDevelopSettings.text` Lua snippet (`s = { ... }`).
    pub fn parse(text: &str) -> Result<Self> {
        let root = parse_lua_develop_table(text)?;
        let settings = extract_settings_map(&root).cloned().unwrap_or(root);
        Ok(Self { settings })
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.settings.get(key)?.as_f64().map(|v| v as f32)
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.settings.get(key)?.as_bool()
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.settings.get(key)?.as_str()
    }

    pub fn get_number_array(&self, key: &str) -> Option<Vec<f64>> {
        let arr = self.settings.get(key)?.as_array()?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            out.push(item.as_f64()?);
        }
        Some(out)
    }

    /// Best-effort read-only mapping of parsed Lua develop settings to Aurora's [`AdjustmentSet`].
    pub fn to_adjustment_set(&self) -> AdjustmentSet {
        let mut adj = AdjustmentSet::default();

        if let Some(wb) = self.get_str("WhiteBalance") {
            if let Some(mode) = WhiteBalanceMode::from_name(wb) {
                adj.white_balance.mode = mode;
            }
        }
        if let Some(temp) = self.get_f32("Temperature") {
            adj.white_balance.kelvin = temp.clamp(2000.0, 50000.0);
        }
        if let Some(tint) = self.get_f32("Tint") {
            adj.white_balance.tint = (tint / 3.0).clamp(-50.0, 50.0);
        }
        if let Some(v) = self.get_f32("Exposure2012").or_else(|| self.get_f32("Exposure")) {
            adj.exposure.exposure = v.clamp(-4.0, 4.0);
        }
        if let Some(v) = self.get_f32("Contrast2012").or_else(|| self.get_f32("Contrast")) {
            adj.exposure.contrast = (v * 0.5).clamp(-50.0, 50.0);
        }
        if let Some(v) = self.get_f32("Saturation") {
            adj.exposure.saturation = v.clamp(-100.0, 100.0);
        }
        if let Some(v) = self.get_f32("Vibrance") {
            adj.extra.insert("crs:Vibrance".to_string(), v.clamp(-100.0, 100.0));
        }
        if let Some(v) = self.get_f32("Highlights2012") {
            adj.hdr.highlights = v.clamp(-100.0, 100.0);
        }
        if let Some(v) = self.get_f32("Shadows2012") {
            adj.hdr.shadows = v.clamp(-100.0, 100.0);
        }
        if let Some(v) = self.get_f32("Whites2012") {
            adj.hdr.whites = v.clamp(-100.0, 100.0);
        }
        if let Some(v) = self.get_f32("Blacks2012") {
            adj.hdr.blacks = v.clamp(-100.0, 100.0);
        }
        if let Some(v) = self.get_f32("Clarity2012").or_else(|| self.get_f32("Clarity")) {
            adj.clarity.clarity = v.clamp(-100.0, 100.0);
        }
        if let Some(v) = self.get_f32("Texture") {
            adj.clarity.structure = v.clamp(-100.0, 100.0);
        }
        if let Some(v) = self.get_f32("Dehaze") {
            adj.dehaze.amount = v.clamp(0.0, 100.0);
        }
        if let Some(bw) = self.get_bool("ConvertToGrayscale") {
            adj.black_white.enabled = bw;
        }
        if let Some(curve) = self.parse_curve_pairs("ToneCurvePV2012") {
            adj.curves.rgb = curve;
        }
        if let Some(curve) = self.parse_curve_pairs("ToneCurvePV2012Red") {
            adj.curves.red = curve;
        }
        if let Some(curve) = self.parse_curve_pairs("ToneCurvePV2012Green") {
            adj.curves.green = curve;
        }
        if let Some(curve) = self.parse_curve_pairs("ToneCurvePV2012Blue") {
            adj.curves.blue = curve;
        }

        adj
    }

    fn parse_curve_pairs(&self, key: &str) -> Option<Curve> {
        let nums = self.get_number_array(key)?;
        if nums.len() < 4 || nums.len() % 2 != 0 {
            return None;
        }
        let points = nums
            .chunks_exact(2)
            .map(|c| {
                CurvePoint::new(
                    (c[0] as f32 / 255.0).clamp(0.0, 1.0),
                    (c[1] as f32 / 255.0).clamp(0.0, 1.0),
                )
            })
            .collect();
        Some(Curve { points })
    }
}

/// Read-only handle to a Lightroom Classic `.lrcat` SQLite catalog.
pub struct LrcatCatalog {
    conn: Connection,
}

impl LrcatCatalog {
    /// Opens an `.lrcat` file strictly read-only (`SQLITE_OPEN_READ_ONLY`).
    pub fn open_read_only(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening Lightroom catalog read-only: {}", path.display()))?;
        Ok(Self { conn })
    }

    /// Reads all image records joined across `Adobe_images`, `AgLibraryFile`,
    /// `AgLibraryFolder`, `AgLibraryRootFolder`, and `Adobe_imageDevelopSettings`.
    pub fn read_images(&self) -> Result<Vec<LrcatImageRecord>> {
        let sql = r#"
            SELECT
                i.id_local,
                r.absolutePath,
                fo.pathFromRoot,
                f.baseName,
                f.extension,
                f.idx_filename,
                i.rating,
                i.colorLabels,
                i.pick,
                i.captureTime,
                i.fileFormat,
                d.text
            FROM Adobe_images i
            INNER JOIN AgLibraryFile f ON f.id_local = i.rootFile
            INNER JOIN AgLibraryFolder fo ON fo.id_local = f.folder
            INNER JOIN AgLibraryRootFolder r ON r.id_local = fo.rootFolder
            LEFT JOIN Adobe_imageDevelopSettings d ON d.image = i.id_local
            ORDER BY i.id_local ASC
        "#;

        let mut stmt = self
            .conn
            .prepare(sql)
            .context("preparing Lightroom catalog query")?;

        let rows = stmt
            .query_map([], |row| {
                let image_id: i64 = row.get(0)?;
                let root_abs: String = row.get::<_, Option<String>>(1)?.unwrap_or_default();
                let folder_rel: String = row.get::<_, Option<String>>(2)?.unwrap_or_default();
                let base_name: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
                let extension: String = row.get::<_, Option<String>>(4)?.unwrap_or_default();
                let idx_filename: Option<String> = row.get(5)?;
                let rating_raw: Option<f64> = row.get(6)?;
                let color_label: String = row.get::<_, Option<String>>(7)?.unwrap_or_default();
                let pick_raw: Option<f64> = row.get(8)?;
                let capture_time: Option<String> = row.get(9)?;
                let file_format: Option<String> = row.get(10)?;
                let develop_raw: Option<String> = row.get(11)?;

                let file_name = if let Some(idx) = idx_filename.filter(|s| !s.is_empty()) {
                    idx
                } else if extension.is_empty() {
                    base_name
                } else {
                    format!("{base_name}.{extension}")
                };

                let file_path = build_catalog_path(&root_abs, &folder_rel, &file_name);
                let rating = rating_raw
                    .map(|v| v.round().clamp(0.0, 5.0) as i8)
                    .unwrap_or(0);
                let pick = LrcatPickStatus::from_raw(pick_raw);
                let develop_settings = develop_raw
                    .as_deref()
                    .and_then(|txt| LuaDevelopSettings::parse(txt).ok());

                Ok(LrcatImageRecord {
                    image_id,
                    file_path,
                    file_name,
                    rating,
                    color_label: color_label.trim().to_string(),
                    pick,
                    capture_time,
                    file_format,
                    develop_settings_raw: develop_raw,
                    develop_settings,
                })
            })
            .context("executing Lightroom catalog query")?;

        let mut records = Vec::new();
        for row in rows {
            records.push(row?);
        }
        Ok(records)
    }
}

fn build_catalog_path(root_abs: &str, folder_rel: &str, file_name: &str) -> PathBuf {
    let mut path = PathBuf::from(root_abs.trim_end_matches(['/', '\\']));
    let rel = folder_rel.trim_matches(['/', '\\']);
    if !rel.is_empty() {
        for seg in rel.split(['/', '\\']).filter(|s| !s.is_empty()) {
            path.push(seg);
        }
    }
    path.push(file_name);
    path
}

fn extract_settings_map(root: &BTreeMap<String, LuaValue>) -> Option<&BTreeMap<String, LuaValue>> {
    if let Some(settings) = root.get("settings").and_then(LuaValue::as_table) {
        return Some(settings);
    }
    if let Some(value) = root.get("value").and_then(LuaValue::as_table) {
        if let Some(settings) = value.get("settings").and_then(LuaValue::as_table) {
            return Some(settings);
        }
        return Some(value);
    }
    None
}

// ───────────────────────────── Lua table parser ─────────────────────────────

fn parse_lua_develop_table(input: &str) -> Result<BTreeMap<String, LuaValue>> {
    let mut parser = LuaParser::new(input);
    parser.skip_ws_and_comments();

    // Optional assignment prefix: `s = { ... }` or `return { ... }`
    if parser.peek() != Some('{') {
        let ident = parser.read_ident()?;
        parser.skip_ws_and_comments();
        if ident != "return" {
            parser.expect_char('=')?;
            parser.skip_ws_and_comments();
        }
    }

    match parser.parse_table()? {
        LuaValue::Table(map) => Ok(map),
        LuaValue::Array(arr) if arr.is_empty() => Ok(BTreeMap::new()),
        _ => Err(anyhow!("expected Lua key-value table at top level")),
    }
}

struct LuaParser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> LuaParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            src: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).map(|b| *b as char)
    }

    fn peek_next(&self) -> Option<char> {
        self.src.get(self.pos + 1).map(|b| *b as char)
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while let Some(c) = self.peek() {
                if c.is_ascii_whitespace() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.peek() == Some('-') && self.peek_next() == Some('-') {
                self.pos += 2;
                while let Some(c) = self.bump() {
                    if c == '\n' {
                        break;
                    }
                }
                continue;
            }
            break;
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<()> {
        self.skip_ws_and_comments();
        match self.bump() {
            Some(c) if c == expected => Ok(()),
            other => Err(anyhow!(
                "expected '{expected}' at byte {}, found {other:?}",
                self.pos
            )),
        }
    }

    fn read_ident(&mut self) -> Result<String> {
        self.skip_ws_and_comments();
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(anyhow!("expected identifier at byte {}", self.pos));
        }
        Ok(std::str::from_utf8(&self.src[start..self.pos])?.to_string())
    }

    fn parse_value(&mut self) -> Result<LuaValue> {
        self.skip_ws_and_comments();
        let ch = self
            .peek()
            .ok_or_else(|| anyhow!("unexpected EOF while parsing Lua value"))?;
        match ch {
            '{' => self.parse_table(),
            '"' | '\'' => Ok(LuaValue::String(self.parse_string()?)),
            '-' | '+' | '0'..='9' | '.' => Ok(LuaValue::Number(self.parse_number()?)),
            _ => {
                let ident = self.read_ident()?;
                match ident.as_str() {
                    "true" => Ok(LuaValue::Bool(true)),
                    "false" => Ok(LuaValue::Bool(false)),
                    "nil" => Ok(LuaValue::Nil),
                    _ => Err(anyhow!("unexpected Lua token '{ident}' at byte {}", self.pos)),
                }
            }
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        let quote = self
            .bump()
            .ok_or_else(|| anyhow!("expected string quote"))?;
        let mut out = String::new();
        while let Some(c) = self.bump() {
            if c == quote {
                return Ok(out);
            }
            if c == '\\' {
                let esc = self
                    .bump()
                    .ok_or_else(|| anyhow!("unterminated string escape"))?;
                match esc {
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    '\\' => out.push('\\'),
                    '"' => out.push('"'),
                    '\'' => out.push('\''),
                    other => out.push(other),
                }
            } else {
                out.push(c);
            }
        }
        Err(anyhow!("unterminated Lua string literal"))
    }

    fn parse_number(&mut self) -> Result<f64> {
        let start = self.pos;
        if matches!(self.peek(), Some('-') | Some('+')) {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-') {
                if matches!(c, '+' | '-') {
                    let prev = self.src.get(self.pos.saturating_sub(1)).copied();
                    if !matches!(prev, Some(b'e') | Some(b'E')) {
                        break;
                    }
                }
                self.pos += 1;
            } else {
                break;
            }
        }
        let slice = std::str::from_utf8(&self.src[start..self.pos])?;
        slice
            .parse::<f64>()
            .with_context(|| format!("invalid Lua number '{slice}'"))
    }

    fn parse_table(&mut self) -> Result<LuaValue> {
        self.expect_char('{')?;
        let mut map = BTreeMap::new();
        let mut arr = Vec::new();
        let mut saw_named = false;

        loop {
            self.skip_ws_and_comments();
            if self.peek() == Some('}') {
                self.pos += 1;
                break;
            }

            // Check for `[key] = value`
            if self.peek() == Some('[') {
                self.pos += 1;
                self.skip_ws_and_comments();
                let key = match self.peek() {
                    Some('"') | Some('\'') => self.parse_string()?,
                    _ => self.parse_number()?.to_string(),
                };
                self.expect_char(']')?;
                self.expect_char('=')?;
                let val = self.parse_value()?;
                saw_named = true;
                map.insert(key, val);
            } else if self.looks_like_named_field() {
                let key = self.read_ident()?;
                self.expect_char('=')?;
                let val = self.parse_value()?;
                saw_named = true;
                map.insert(key, val);
            } else {
                let val = self.parse_value()?;
                arr.push(val);
            }

            self.skip_ws_and_comments();
            if matches!(self.peek(), Some(',') | Some(';')) {
                self.pos += 1;
            }
        }

        if saw_named {
            for (idx, val) in arr.into_iter().enumerate() {
                map.insert((idx + 1).to_string(), val);
            }
            Ok(LuaValue::Table(map))
        } else {
            Ok(LuaValue::Array(arr))
        }
    }

    fn looks_like_named_field(&self) -> bool {
        let mut p = self.pos;
        if p >= self.src.len() {
            return false;
        }
        let first = self.src[p] as char;
        if !(first.is_ascii_alphabetic() || first == '_') {
            return false;
        }
        while p < self.src.len() {
            let c = self.src[p] as char;
            if c.is_ascii_alphanumeric() || c == '_' {
                p += 1;
            } else {
                break;
            }
        }
        while p < self.src.len() && (self.src[p] as char).is_ascii_whitespace() {
            p += 1;
        }
        p < self.src.len() && self.src[p] == b'='
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    const SAMPLE_LUA_DEVELOP: &str = r#"
-- Lightroom Classic develop settings
s = {
    id = "9A1B2C3D-0001",
    internalName = "Custom",
    type = "Develop",
    value = {
        settings = {
            WhiteBalance = "Custom",
            Temperature = 5900,
            Tint = 9,
            Exposure2012 = 0.65,
            Contrast2012 = 16,
            Highlights2012 = -35,
            Shadows2012 = 40,
            Whites2012 = 12,
            Blacks2012 = -10,
            Texture = 8,
            Clarity2012 = 14,
            Dehaze = 20,
            Vibrance = 15,
            Saturation = -5,
            ConvertToGrayscale = false,
            ["Custom Key"] = "hello\nworld",
            ToneCurvePV2012 = {
                0, 10,
                128, 134,
                255, 250,
            },
        },
        uuid = "1111-2222",
    },
    version = 0,
}
"#;

    fn temp_lrcat_path(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("aurora_test_{tag}_{nanos}.lrcat"))
    }

    fn create_sample_lrcat(path: &Path) {
        let conn = Connection::open(path).expect("create sqlite");
        conn.execute_batch(
            r#"
            CREATE TABLE AgLibraryRootFolder (
                id_local INTEGER PRIMARY KEY,
                absolutePath TEXT,
                relativePathFromCatalog TEXT,
                name TEXT
            );
            CREATE TABLE AgLibraryFolder (
                id_local INTEGER PRIMARY KEY,
                rootFolder INTEGER,
                pathFromRoot TEXT
            );
            CREATE TABLE AgLibraryFile (
                id_local INTEGER PRIMARY KEY,
                folder INTEGER,
                baseName TEXT,
                extension TEXT,
                idx_filename TEXT
            );
            CREATE TABLE Adobe_images (
                id_local INTEGER PRIMARY KEY,
                rootFile INTEGER,
                rating REAL,
                colorLabels TEXT,
                pick REAL,
                captureTime TEXT,
                fileFormat TEXT
            );
            CREATE TABLE Adobe_imageDevelopSettings (
                id_local INTEGER PRIMARY KEY,
                image INTEGER,
                text TEXT
            );

            INSERT INTO AgLibraryRootFolder (id_local, absolutePath, name)
                VALUES (1, 'D:/Photos/2026', '2026');
            INSERT INTO AgLibraryFolder (id_local, rootFolder, pathFromRoot)
                VALUES (10, 1, 'SessionA/');
            INSERT INTO AgLibraryFile (id_local, folder, baseName, extension, idx_filename)
                VALUES
                    (100, 10, 'DSC_0001', 'NEF', 'DSC_0001.NEF'),
                    (101, 10, 'DSC_0002', 'NEF', 'DSC_0002.NEF'),
                    (102, 10, 'DSC_0003', 'DNG', '');
            INSERT INTO Adobe_images (id_local, rootFile, rating, colorLabels, pick, captureTime, fileFormat)
                VALUES
                    (1000, 100, 5.0, 'Red', 1.0, '2026-05-01T10:00:00', 'RAW'),
                    (1001, 101, 2.0, 'Blue', -1.0, '2026-05-01T10:01:00', 'RAW'),
                    (1002, 102, NULL, '', 0.0, '2026-05-01T10:02:00', 'DNG');
            "#,
        )
        .expect("init schema");
        conn.execute(
            "INSERT INTO Adobe_imageDevelopSettings (id_local, image, text) VALUES (500, 1000, ?1)",
            [SAMPLE_LUA_DEVELOP],
        )
        .expect("insert develop settings");
    }

    #[test]
    fn parses_lua_develop_settings_and_maps_to_aurora() {
        let parsed = LuaDevelopSettings::parse(SAMPLE_LUA_DEVELOP).expect("parse lua");
        assert_eq!(parsed.get_str("WhiteBalance"), Some("Custom"));
        assert_eq!(parsed.get_f32("Exposure2012"), Some(0.65));
        assert_eq!(parsed.get_f32("Highlights2012"), Some(-35.0));
        assert_eq!(parsed.get_bool("ConvertToGrayscale"), Some(false));
        assert_eq!(parsed.get_str("Custom Key"), Some("hello\nworld"));
        assert_eq!(
            parsed.get_number_array("ToneCurvePV2012"),
            Some(vec![0.0, 10.0, 128.0, 134.0, 255.0, 250.0])
        );

        let adj = parsed.to_adjustment_set();
        assert_eq!(adj.white_balance.mode, WhiteBalanceMode::Custom);
        assert!((adj.white_balance.kelvin - 5900.0).abs() < 1e-4);
        assert!((adj.white_balance.tint - 3.0).abs() < 1e-4);
        assert!((adj.exposure.exposure - 0.65).abs() < 1e-4);
        assert!((adj.exposure.contrast - 8.0).abs() < 1e-4);
        assert!((adj.hdr.shadows - 40.0).abs() < 1e-4);
        assert_eq!(adj.curves.rgb.points.len(), 3);
    }

    #[test]
    fn opens_lrcat_read_only_and_imports_images() {
        let path = temp_lrcat_path("readonly");
        create_sample_lrcat(&path);

        let catalog = LrcatCatalog::open_read_only(&path).expect("open read-only");
        let images = catalog.read_images().expect("read images");
        assert_eq!(images.len(), 3);

        assert_eq!(images[0].image_id, 1000);
        assert_eq!(images[0].file_name, "DSC_0001.NEF");
        assert_eq!(
            images[0].file_path,
            PathBuf::from("D:/Photos/2026")
                .join("SessionA")
                .join("DSC_0001.NEF")
        );
        assert_eq!(images[0].rating, 5);
        assert_eq!(images[0].color_label, "Red");
        assert_eq!(images[0].pick, LrcatPickStatus::Picked);
        let dev = images[0]
            .develop_settings
            .as_ref()
            .expect("parsed develop settings");
        assert_eq!(dev.get_f32("Exposure2012"), Some(0.65));

        assert_eq!(images[1].rating, 2);
        assert_eq!(images[1].color_label, "Blue");
        assert_eq!(images[1].pick, LrcatPickStatus::Rejected);
        assert!(images[1].develop_settings.is_none());

        assert_eq!(images[2].file_name, "DSC_0003.DNG");
        assert_eq!(images[2].rating, 0);
        assert_eq!(images[2].pick, LrcatPickStatus::Unflagged);

        drop(catalog);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn refuses_to_create_missing_lrcat_in_read_only_mode() {
        let missing = temp_lrcat_path("missing");
        assert!(LrcatCatalog::open_read_only(&missing).is_err());
        assert!(!missing.exists());
    }

    #[test]
    fn rejects_malformed_lua_table() {
        assert!(LuaDevelopSettings::parse("s = { Exposure2012 = ").is_err());
    }
}
