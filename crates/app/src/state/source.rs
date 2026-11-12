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
}

impl Source {
    pub fn path(&self) -> Option<&Path> {
        match self {
            Source::Demo => None,
            Source::Folder(p) | Source::Catalog(p) => Some(p),
        }
    }

    /// Short name for the title bar: the folder or catalog name.
    pub fn title(&self) -> String {
        match self {
            Source::Demo => t("Demo images", "演示图像").into(),
            Source::Folder(p) | Source::Catalog(p) => display_name(p),
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
        .is_some_and(|e| ["cocatalog", "cocatalogdb", "cosessiondb", "cos"].contains(&e.to_ascii_lowercase().as_str()))
}

/// Opens `path` with the engine that reads it. Runs off the main thread: a
/// folder is scanned before this returns.
pub fn open(path: &Path) -> anyhow::Result<(Arc<dyn ImageEngine>, Source)> {
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
