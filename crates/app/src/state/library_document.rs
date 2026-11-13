//! Portable library preferences; paths inside the library stay relative.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DocumentKind { Catalog, Session }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LibraryDocument {
    pub version: u32,
    pub kind: DocumentKind,
    pub name: String,
    #[serde(default)]
    pub shortcuts: BTreeMap<String, PathBuf>,
    #[serde(default)]
    pub favorites: Vec<PathBuf>,
    #[serde(default = "yes")]
    pub include_subfolders: bool,
}
fn yes() -> bool { true }

pub fn valid_name(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    !name.trim().is_empty() && name.trim() == name && !name.ends_with('.')
        && !name.chars().any(|c| c.is_control() || "<>:\"/\\|?*".contains(c))
        && !["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"].contains(&stem.as_str())
        && name != "." && name != ".."
}

impl LibraryDocument {
    pub fn descriptor(root: &Path, kind: DocumentKind) -> PathBuf {
        root.join(match kind { DocumentKind::Catalog => "library.auroracatalog", DocumentKind::Session => "session.aurorasession" })
    }

    pub fn load(root: &Path) -> anyhow::Result<Option<Self>> {
        for kind in [DocumentKind::Catalog, DocumentKind::Session] {
            let path = Self::descriptor(root, kind);
            match std::fs::read(&path) {
                Ok(bytes) => {
                    let doc: Self = serde_json::from_slice(&bytes)?;
                    anyhow::ensure!(doc.version == 1 && doc.kind == kind, "Unsupported Aurora library document: {}", path.display());
                    return Ok(Some(doc));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
                Err(e) => return Err(e.into()),
            }
        }
        Ok(None)
    }

    pub fn save(&self, root: &Path) -> anyhow::Result<()> {
        use std::io::Write;
        let path = Self::descriptor(root, self.kind);
        let temp = path.with_extension(format!("tmp-{}", std::process::id()));
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&temp)?;
        let result = (|| -> anyhow::Result<()> {
            file.write_all(&serde_json::to_vec_pretty(self)?)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temp, &path)?;
            Ok(())
        })();
        if result.is_err() { let _ = std::fs::remove_file(&temp); }
        result
    }

    pub fn create(parent: &Path, name: &str, kind: DocumentKind) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(valid_name(name), "Invalid library name");
        let root = parent.join(name);
        std::fs::create_dir(&root)?;
        let mut doc = Self { version: 1, kind, name: name.into(), shortcuts: BTreeMap::new(), favorites: Vec::new(), include_subfolders: true };
        let result = (|| -> anyhow::Result<()> {
            if kind == DocumentKind::Session {
                for role in ["Capture", "Selects", "Output", "Trash"] {
                    std::fs::create_dir(root.join(role))?;
                    doc.shortcuts.insert(role.into(), role.into());
                }
            } else {
                std::fs::create_dir(root.join("Originals"))?;
            }
            doc.save(&root)
        })();
        if let Err(e) = result {
            // Only this new, exclusively-created library is rolled back.
            for name in ["Capture", "Selects", "Output", "Trash", "Originals"] { let _ = std::fs::remove_dir(root.join(name)); }
            let _ = std::fs::remove_dir(&root);
            return Err(e);
        }
        Ok(root)
    }

    pub fn resolve(root: &Path, path: &Path) -> PathBuf {
        if path.is_absolute() { path.to_owned() } else { root.join(path) }
    }

    pub fn portable(root: &Path, path: &Path) -> PathBuf {
        path.strip_prefix(root).unwrap_or(path).to_owned()
    }

    pub fn relocate(&mut self, root: &Path, from: &Path, to: &Path) {
        for path in self.shortcuts.values_mut().chain(self.favorites.iter_mut()) {
            let absolute = Self::resolve(root, path);
            if let Ok(tail) = absolute.strip_prefix(from) { *path = Self::portable(root, &to.join(tail)); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn session_is_portable_and_creation_never_replaces_a_folder() {
        let root = std::env::temp_dir().join(format!("aurora-document-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let name = format!("session-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos());
        let session = LibraryDocument::create(&root, &name, DocumentKind::Session).unwrap();
        let doc = LibraryDocument::load(&session).unwrap().unwrap();
        assert_eq!(doc.shortcuts.len(), 4);
        assert!(session.join("Capture").is_dir());
        assert_eq!(doc.shortcuts["Capture"], Path::new("Capture"));
        assert!(LibraryDocument::create(&root, &name, DocumentKind::Session).is_err());
        assert!(LibraryDocument::load(&session).unwrap().is_some());
        std::fs::remove_dir_all(&session).unwrap();
    }
    #[test]
    fn library_names_cannot_escape_or_use_windows_devices() {
        for name in ["", "..", "../outside", "a/b", "a\\b", "CON", "nul.txt", "a:", "folder.", "COM1", " trailing "] { assert!(!valid_name(name), "{name}"); }
        assert!(valid_name("旅行照片 2026"));
    }
    #[test]
    fn moving_folders_updates_shortcuts_and_favorites_by_components() {
        let root = std::env::current_dir().unwrap();
        let mut doc = LibraryDocument { version: 1, kind: DocumentKind::Session, name: "Test".into(), shortcuts: BTreeMap::from([("Capture".into(), PathBuf::from("a/Capture"))]), favorites: vec!["a/sub".into(), "ab/sub".into()], include_subfolders: true };
        doc.relocate(&root, &root.join("a"), &root.join("renamed"));
        assert_eq!(doc.shortcuts["Capture"], Path::new("renamed/Capture"));
        assert_eq!(doc.favorites, vec![PathBuf::from("renamed/sub"), PathBuf::from("ab/sub")]);
    }
}
