//! Where the open catalog comes from: a folder of photos, a Capture One catalog
//! or session, or the built-in demo images shown behind the welcome screen.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use aurora_engine::{FileEngine, ImageEngine, MockEngine};

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

/// A CLI/Open-With request. A loose photo belongs to a persistent folder
/// library, never the procedural demo or the last unrelated catalog.
#[derive(Debug, PartialEq, Eq)]
pub struct OpenRequest {
    pub root: PathBuf,
    pub photos: Vec<PathBuf>,
}

pub fn requests(paths: &[PathBuf]) -> anyhow::Result<Vec<OpenRequest>> {
    use anyhow::Context as _;
    let mut out = Vec::<OpenRequest>::new();
    let mut photos = Vec::new();
    for path in paths {
        let path = std::path::absolute(path)?;
        let meta = path.metadata().with_context(|| format!("{}: {}", t("Cannot open", "打开失败"), path.display()))?;
        if meta.is_dir() || is_catalog(&path) {
            if !out.iter().any(|r| r.root == path) { out.push(OpenRequest { root: path, photos: vec![] }); }
        } else if meta.is_file() && aurora_engine::supports_photo_path(&path) {
            photos.push(path);
        } else {
            anyhow::bail!("{}: {}", t("Unsupported image format", "不支持的图像格式"), path.display());
        }
    }
    for path in photos {
        let parent = path.parent().context("photo has no parent directory")?;
        let request = if let Some(i) = out.iter().position(|r| !is_catalog(&r.root) && path.starts_with(&r.root)) {
            &mut out[i]
        } else {
            out.push(OpenRequest { root: parent.to_owned(), photos: vec![] });
            out.last_mut().unwrap()
        };
        if !request.photos.contains(&path) { request.photos.push(path); }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_photos_use_a_real_library_and_keep_selection_order() {
        let dir = std::env::temp_dir().join(format!("aurora-open-request-{}", std::process::id()));
        let other = dir.join("two");
        std::fs::create_dir_all(&other).unwrap();
        let a = dir.join("z.png"); let b = dir.join("a.nef"); let c = other.join("x.jpg");
        for p in [&a, &b, &c] { std::fs::write(p, []).unwrap(); }
        let single = requests(std::slice::from_ref(&a)).unwrap();
        assert_eq!(single, vec![OpenRequest { root: dir.clone(), photos: vec![a.clone()] }]);
        let list = requests(&[a.clone(), b.clone(), a.clone(), c.clone(), dir.clone()]).unwrap();
        assert_eq!(list, vec![OpenRequest { root: dir.clone(), photos: vec![a, b, c] }]);
        assert!(requests(&[dir.join("missing.png")]).is_err());
        let unsupported = dir.join("notes.txt"); std::fs::write(&unsupported, []).unwrap();
        assert!(requests(&[unsupported]).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

pub fn demo() -> (Arc<dyn ImageEngine>, Source) {
    (Arc::new(MockEngine::new()), Source::Demo)
}
