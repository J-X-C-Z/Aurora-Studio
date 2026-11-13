//! Where the open catalog comes from: a folder of photos, a Capture One catalog
//! or session, or the built-in demo images shown behind the welcome screen.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lumen_engine::{FileEngine, ImageEngine, MockEngine};

use crate::i18n::t;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Procedural images; nothing on disk.
    Demo,
    Folder(PathBuf),
    Catalog(PathBuf),
    Session(PathBuf),
}

impl Source {
    pub fn path(&self) -> Option<&Path> {
        match self {
            Source::Demo => None,
            Source::Folder(p) | Source::Catalog(p) | Source::Session(p) => Some(p),
        }
    }

    /// Short name for the title bar: the folder or catalog name.
    pub fn title(&self) -> String {
        match self {
            Source::Demo => t("Demo images", "演示图像").into(),
            Source::Folder(p) | Source::Catalog(p) | Source::Session(p) => display_name(p),
        }
    }
}

pub fn display_name(path: &Path) -> String {
    path.file_stem().or_else(|| path.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.display().to_string())
}

/// Capture One catalogs and sessions open through the C1 importer.
pub fn is_catalog(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| ["cocatalog", "cocatalogdb", "cosessiondb", "cos", "auroracatalog", "aurorasession"].contains(&e.to_ascii_lowercase().as_str()))
}

/// Opens `path` with the engine that reads it. Runs off the main thread: a
/// folder is scanned before this returns.
pub fn open(path: &Path) -> anyhow::Result<(Arc<dyn ImageEngine>, Source)> {
    let native_file = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("auroracatalog") || e.eq_ignore_ascii_case("aurorasession"));
    let root = if native_file { path.parent().ok_or_else(|| anyhow::anyhow!("Library has no parent folder"))? } else { path };
    if root.is_dir() && (native_file || !is_catalog(path)) {
        let document = super::LibraryDocument::load(root)?;
        if native_file { anyhow::ensure!(document.is_some(), "Aurora library document was not found"); }
        let source = match document.map(|d| d.kind) {
            Some(super::DocumentKind::Session) => Source::Session(root.to_owned()),
            Some(super::DocumentKind::Catalog) => Source::Catalog(root.to_owned()),
            None => Source::Folder(root.to_owned()),
        };
        return Ok((Arc::new(FileEngine::open(root)?), source));
    }
    if is_catalog(path) {
        Ok((Arc::new(FileEngine::open_c1_catalog(path)?), Source::Catalog(path.to_path_buf())))
    } else if path.is_dir() {
        Ok((Arc::new(FileEngine::open(path)?), Source::Folder(path.to_path_buf())))
    } else {
        anyhow::bail!("{}", t("not a folder or a Capture One catalog", "不是文件夹或 Capture One 目录"))
    }
}

pub fn demo() -> (Arc<dyn ImageEngine>, Source) {
    (Arc::new(MockEngine::new()), Source::Demo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{DocumentKind, LibraryDocument};
    #[test]
    fn native_session_and_catalog_open_from_both_folder_and_descriptor() {
        let parent = std::env::temp_dir().join(format!("aurora-library-open-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&parent).unwrap();
        for (name, kind) in [("Catalog", DocumentKind::Catalog), ("Session", DocumentKind::Session)] {
            let root = LibraryDocument::create(&parent, name, kind).unwrap();
            let descriptor = LibraryDocument::descriptor(&root, kind);
            let (engine, folder_source) = open(&root).unwrap();
            assert_eq!(engine.library_root().as_deref(), Some(root.as_path()));
            let (_, file_source) = open(&descriptor).unwrap();
            assert_eq!(folder_source, file_source);
            assert!(matches!((kind, folder_source), (DocumentKind::Catalog, Source::Catalog(_)) | (DocumentKind::Session, Source::Session(_))));
            assert!(is_catalog(&descriptor));
            drop(engine);
        }
        std::fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    #[ignore = "synthetic library throughput probe; creates and removes its own files"]
    fn library_sync_throughput() {
        use lumen_engine::LibraryAction;
        let parent = std::env::temp_dir().join(format!("aurora-library-throughput-{}", std::process::id()));
        std::fs::create_dir(&parent).unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([20u8, 40, 60, 255])).save(parent.join("seed.png")).unwrap();
        let library = LibraryDocument::create(&parent, "Library", DocumentKind::Catalog).unwrap();
        let external = parent.join("external");
        std::fs::create_dir(&external).unwrap();
        let engine = FileEngine::open(&library).unwrap();
        let mut written = 0;
        for count in [1000, 5000] {
            for n in written..count { std::fs::copy(parent.join("seed.png"), external.join(format!("photo-{n:05}.png"))).unwrap(); }
            written = count;
            let start = std::time::Instant::now();
            let files = engine.manage_files(LibraryAction::Synchronize { folder: external.clone(), recursive: false }).unwrap();
            let first = start.elapsed().as_millis();
            assert_eq!(files.len(), count);
            let start = std::time::Instant::now();
            let warm = engine.manage_files(LibraryAction::Synchronize { folder: external.clone(), recursive: false }).unwrap();
            let repeat = start.elapsed().as_millis();
            assert_eq!(files.iter().map(|i| i.id).collect::<Vec<_>>(), warm.iter().map(|i| i.id).collect::<Vec<_>>());
            eprintln!("LIBRARY_THROUGHPUT count={count} incremental_ms={first} warm_ms={repeat}");
        }
        drop(engine);
        let start = std::time::Instant::now();
        let engine = FileEngine::open(&library).unwrap();
        assert_eq!(engine.images().len(), 5000);
        eprintln!("LIBRARY_THROUGHPUT reopen_5000_ms={}", start.elapsed().as_millis());
        drop(engine);
        std::fs::remove_dir_all(parent).unwrap();
    }
}
