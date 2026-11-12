//! Opening a Capture One catalog or session: resolving its SQLite file and
//! handing the stored variants to the engine.
//!
//! The C1 database is **read-only for us**: nothing here writes to it, and the
//! engine keeps everything it saves in its own `.lumen` directory. A `.readlock`
//! or an uncommitted `-wal` only produces a warning (see [`C1Catalog::open_ro`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::c1catalog::{C1Catalog, C1Collection, C1DocumentType, C1Image, C1Variant};
use crate::catalog::{Collection, CollectionKind};

/// C1 collection ids are offset so they never collide with the interface's
/// own collection ids (1..=10 and 1000.. for folders).
pub const C1_COLLECTION_ID_BASE: u64 = 1_000_000;

/// A catalog or session read once, with the images and variants C1 stored.
#[derive(Debug)]
pub struct C1Import {
    /// The SQLite file that was actually opened.
    pub catalog_path: PathBuf,
    pub catalog: C1Catalog,
    variants: HashMap<i64, Vec<usize>>,
}

impl C1Import {
    /// Opens a `.cocatalogdb` / `.cosessiondb` file, its bundle directory, a
    /// `.cos` session folder, or the folder that contains one of them.
    pub fn open(path: &Path) -> Result<Self> {
        let catalog_path = resolve_database(path)?;
        let catalog = C1Catalog::open_ro(&catalog_path)
            .with_context(|| format!("reading {}", catalog_path.display()))?;
        let mut variants: HashMap<i64, Vec<usize>> = HashMap::new();
        for (index, variant) in catalog.variants.iter().enumerate() {
            variants.entry(variant.image_pk).or_default().push(index);
        }
        for list in variants.values_mut() {
            list.sort_by_key(|i| catalog.variants[*i].index);
        }
        Ok(Self { catalog_path, catalog, variants })
    }

    pub fn images(&self) -> &[C1Image] {
        &self.catalog.images
    }

    pub fn variants(&self) -> &[C1Variant] {
        &self.catalog.variants
    }

    /// Whether this database is a catalog or a session.
    pub fn document_type(&self) -> C1DocumentType {
        self.catalog.document_type
    }

    pub fn is_session(&self) -> bool {
        self.catalog.document_type == C1DocumentType::Session
    }

    /// The variants of one image, lowest `ZINDEX` first.
    pub fn variants_of(&self, image_pk: i64) -> Vec<&C1Variant> {
        self.variants
            .get(&image_pk)
            .map(|list| list.iter().map(|i| &self.catalog.variants[*i]).collect())
            .unwrap_or_default()
    }

    /// C1's collection tree, mapped onto the engine's [`Collection`] type.
    ///
    /// The tree C1 stores is a forest of project roots; the engine wants one
    /// tree, so the project roots become the top level.
    pub fn collections(&self) -> Vec<Collection> {
        let all_images = self.catalog.images.len();
        fn build(
            collections: &[C1Collection],
            pk: i64,
            all_images: usize,
        ) -> Collection {
            let node = collections.iter().find(|c| c.pk == pk).expect("collection exists");
            let mut kids: Vec<&C1Collection> =
                collections.iter().filter(|c| c.parent == Some(pk)).collect();
            kids.sort_by_key(|c| c.index);
            Collection {
                id: C1_COLLECTION_ID_BASE + node.pk as u64,
                name: node.name.clone(),
                kind: node.kind,
                count: if node.kind == CollectionKind::AllImages {
                    all_images
                } else {
                    node.images.len()
                },
                children: kids
                    .iter()
                    .map(|c| build(collections, c.pk, all_images))
                    .collect(),
            }
        }
        let mut roots: Vec<&C1Collection> =
            self.catalog.collections.iter().filter(|c| c.parent.is_none()).collect();
        roots.sort_by_key(|c| c.index);
        roots
            .iter()
            .map(|c| build(&self.catalog.collections, c.pk, all_images))
            .collect()
    }

    /// The keywords C1 stores for one image, qualified by the keyword tree.
    ///
    /// C1 keeps the keyword hierarchy in `ZKEYWORD` and the keywords of a
    /// variant as a comma separated list of names; a name that lives under a
    /// parent keyword becomes `parent/child`.
    pub fn keywords_of(&self, image_pk: i64) -> Vec<String> {
        let mut out = Vec::new();
        for variant in self.variants_of(image_pk) {
            for keyword in &variant.keywords {
                if !out.contains(keyword) {
                    out.push(keyword.clone());
                }
            }
        }
        out
    }

    /// C1's keyword tree, as `name -> parent/name` paths.
    pub fn keyword_paths(&self) -> HashMap<String, String> {
        crate::c1catalog::keyword_paths(&self.catalog.keywords)
    }
}

const DATABASE_EXTENSIONS: [&str; 2] = ["cocatalogdb", "cosessiondb"];

fn has_database_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| DATABASE_EXTENSIONS.iter().any(|want| e.eq_ignore_ascii_case(want)))
}

/// Finds the SQLite file behind a path the user picked: the `.cocatalogdb` /
/// `.cosessiondb` file itself, its bundle directory, a `.cos` session folder,
/// or a folder containing one of those.
pub fn resolve_database(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        if has_database_extension(path) {
            return Ok(path.to_path_buf());
        }
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("cos")) {
            return Err(anyhow!(
                "{} is a per-image `.cos` settings file, not a session database",
                path.display()
            ));
        }
        return Err(anyhow!(
            "{} is not a Capture One catalog or session database",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(anyhow!("{} does not exist", path.display()));
    }
    search_directory(path, 0).ok_or_else(|| {
        anyhow!("no .cocatalogdb or .cosessiondb file inside {}", path.display())
    })
}

/// A bundle is a directory named `X.cocatalogdb` holding a file `X.cocatalogdb`;
/// in a session the same bundle can sit inside the session folder.
fn search_directory(dir: &Path, depth: usize) -> Option<PathBuf> {
    let same_name = dir.join(dir.file_name()?);
    if same_name.is_file() && has_database_extension(&same_name) {
        return Some(same_name);
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    entries
        .iter()
        .find(|p| p.is_file() && has_database_extension(p))
        .cloned()
        .or_else(|| {
            if depth >= 3 {
                return None;
            }
            entries
                .iter()
                .filter(|p| p.is_dir())
                .find_map(|p| search_directory(p, depth + 1))
        })
}

/// The store key for an image outside the catalog folder.
pub(crate) fn store_key(path: &Path, variant: u32) -> String {
    let flat = path.to_string_lossy().replace('\\', "/").replace(':', "_");
    let flat = flat.trim_start_matches('/').to_string();
    if variant > 1 { format!("{flat}#v{variant}") } else { flat }
}

#[cfg(test)]
mod tests {
    use super::*;


    const PAGE: usize = 4096;

    /// Page 2 and page 19 of the fixture after SQLite rewrote
    /// `ZDOCUMENTCONTENT.ZDOCUMENTTYPE` to 2 and one `ZPATHLOCATION` row to a
    /// mac root (`/Users/jxcz/Pictures` + `Shoot`); page 2 of a type-3 copy for
    /// the unknown-type test. Stored as (byte, u16 length) runs, like the WAL
    /// fixture in `c1catalog`.
    const SESSION_DOC_PAGE: &str = "0d01000003000101000f0100af01000001000f0100af0100009d0f5c01000001003302000801000902005501004f01000101001001000002003301000201000801000901000101005501000901000102000001000103003201003001003201003601002d01003001003901002d01003101003301002001003202003a01003201003001003a0100300100350100070100ea01000201006601003201006401003802006401003501003701002d01006401003901003701003401002d01003401003001006401006601002d01003801003201006601003901002d0100610100350100300100360100330100390100630100330100620100390100330100310100020200ff0300";
    const SESSION_PATH_PAGE: &str = "0d01000f0100de01000001000401000f01007c01000001000f01007c01000f0100cf01000f0100bb01000f0100a10100006c0f2301000101000901000101000001000d01003501001701000801000002002601002f01005501007301006501007201007301002f01006a01007801006301007a01002f01005001006901006301007401007501007201006501007301005301006801006f02007401001801000401000901000101000001001301000d01002301000801000002002601004301003a01005c01005501007301006501007201007301005c01004a01007801006301007a01003201001201000301000901000101000001001301000d01001701000801000002002601004301003a01005c01005501007301006501007201007301000d01000201000901000101000001001301000d02000801000002002601004301003a01005c01000003002201000001001301000d01003301000801000002002601004301003a01005c01005501007301006501007201007301005c01004a01007801006301007a01003201005c01004401006501007301006b01007401006f0100700100";
    const UNKNOWN_DOC_PAGE: &str = "0d01000003000101000f0100af01000001000f0100af0100009d0f5c01000001003302000801000902005501004f01000101001001000002003301000201000801000901000101005501000901000102000001000103003201003001003201003601002d01003001003901002d01003101003301002001003202003a01003201003001003a0100300100350100070100ea01000301006601003201006401003802006401003501003701002d01006401003901003701003401002d01003401003001006401006601002d01003801003201006601003901002d0100610100350100300100360100330100390100630100330100620100390100330100310100020200ff0300";

    /// A two-page SQLite file with a single `ZIMAGE` table; no variant tables.
    const MINIMAL_DB: &str = concat!(
        "5301005101004c01006901007401006501002001006601006f01007201006d01006101007401002001003301000001001001",
        "00000100010200000100400100200200000300020100000300020100000b00010100000300040100000b0001010000230002",
        "01000001002e01008601002901000d01000003000101000f0100a201000001000f0100a2010000340f5c0100010100070100",
        "1701001902000101008101001301007401006101006201006c01006501005a01004901004d01004101004701004501005a01",
        "004901004d01004101004701004501000201004301005201004501004101005401004501002001005401004101004201004c",
        "01004501002001005a01004901004d01004101004701004501002001002801005a01005f01005001004b0100200100690100",
        "6e01007401006501006701006501007201002001007001007201006901006d01006101007201007901002001006b01006501",
        "007901002c01002001005a01004901004d01004101004701004501004601004901004c01004501004e01004101004d010045",
        "01002001007401006501007801007401002901000d01000003000101000f0100f601000001000f0100f6010000ec0f080100",
        "0101000301000001001701006101002e01006e0100650100660100",
    );

    fn decode_rle(hex: &str) -> Vec<u8> {
        let mut runs = Vec::with_capacity(hex.len() / 2);
        for pair in hex.as_bytes().chunks(2) {
            runs.push(u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap());
        }
        let mut out = Vec::new();
        for run in runs.chunks_exact(3) {
            let count = usize::from(u16::from_le_bytes([run[1], run[2]]));
            out.extend(std::iter::repeat_n(run[0], count));
        }
        out
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lumen-c1-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }


    #[test]
    fn a_missing_path_is_an_error_not_a_panic() {
        assert!(C1Import::open(Path::new("no-such.cocatalogdb")).is_err());
    }

    #[test]
    fn store_keys_are_relative_and_variant_aware() {
        let key = store_key(Path::new("C:\\Users\\me\\shot.NEF"), 1);
        assert_eq!(key, "C_/Users/me/shot.NEF");
        assert_eq!(store_key(Path::new("C:\\Users\\me\\shot.NEF"), 2), "C_/Users/me/shot.NEF#v2");
    }


    #[test]
    fn a_database_without_the_variant_tables_is_rejected() {
        let dir = temp_dir("session-min");
        let db = dir.join("Tiny.cocatalogdb");
        std::fs::write(&db, decode_rle(MINIMAL_DB)).unwrap();

        let err = format!("{:#}", C1Import::open(&db).unwrap_err());
        assert!(
            err.contains("is not a Capture One catalog or session") && err.contains("ZVARIANT"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cos_settings_file_is_not_mistaken_for_a_session() {
        let dir = temp_dir("session-cosfile");
        let file = dir.join("IMG_0001.cos");
        std::fs::write(&file, b"<IMG/>").unwrap();

        let err = format!("{:#}", C1Import::open(&file).unwrap_err());
        assert!(err.contains("settings file"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/c1import-tests.rs");
}
