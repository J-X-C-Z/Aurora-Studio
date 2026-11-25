//! Library filesystem helpers over the Windows `FileEngine`.
//!
//! The operation bodies moved to [`super::services::LibraryService`]; what
//! remains here are the path-normalization, sidecar and no-clobber helpers they
//! share, a one-line `FileEngine::manage_library` delegation with the original
//! signature, and the library test suite.
//!
//! All mutations preflight the whole batch before touching a file. A collision,
//! a missing source or an I/O failure therefore leaves the original batch in
//! place (or restores the files already moved).

use super::*;
use crate::LibraryAction;
use anyhow::{bail, ensure, Context, Result};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

pub(super) fn canonical_root(path: &Path) -> Result<PathBuf> {
    Ok(plain_path(&fs::canonicalize(path).with_context(|| format!("library root {}", path.display()))?))
}

fn plain_path(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    #[cfg(windows)]
    {
        if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
            return PathBuf::from(format!(r"\\{rest}"));
        }
        if let Some(rest) = text.strip_prefix(r"\\?\") {
            return PathBuf::from(rest);
        }
    }
    let _ = text;
    path.to_path_buf()
}

pub(super) fn path_key(path: &Path) -> String {
    let key = path.to_string_lossy().replace('\\', "/");
    if cfg!(windows) { key.to_lowercase() } else { key }
}

pub(super) fn is_private(path: &Path) -> bool {
    path.components().any(|component| {
        let name = component.as_os_str().to_string_lossy().to_ascii_lowercase();
        name == ".portrait"
            || name == ".lumen"
            || ["cocatalogdb", "cosessiondb", "cos"]
                .iter()
                .any(|extension| name.ends_with(&format!(".{extension}")))
    })
}

pub(super) fn resolve_inside(root: &Path, path: &Path) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "empty path");
    ensure!(!path.components().any(|component| matches!(component, Component::ParentDir)), "parent traversal is not allowed");
    let path = if path.is_absolute() {
        plain_path(path)
    } else {
        ensure!(
            !path.components().any(|component| matches!(component, Component::RootDir | Component::Prefix(_))),
            "ambiguous path"
        );
        root.join(path)
    };
    let path = path.components().filter(|component| !matches!(component, Component::CurDir)).collect::<PathBuf>();
    ensure!(path.starts_with(root), "path must be inside the library root");
    Ok(path)
}

pub(super) fn existing_directory(root: &Path, path: &Path) -> Result<PathBuf> {
    let path = resolve_inside(root, path)?;
    ensure!(path.is_dir(), "directory unavailable: {}", path.display());
    let path = canonical_root(&path)?;
    ensure!(!is_private(&path), "private catalog directories cannot be used");
    Ok(path)
}

pub(super) fn validate_name(name: &str) -> Result<()> {
    ensure!(!name.is_empty() && name != "." && name != ".." && name.encode_utf16().count() <= 255, "invalid Windows filename");
    ensure!(!name.ends_with(['.', ' ']) && !name.chars().any(|c| c < ' ' || "<>:\"/\\|?*".contains(c)), "invalid Windows filename");
    let stem = name.split('.').next().unwrap_or("").trim_end_matches(' ').to_ascii_uppercase();
    let device = ["CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"].contains(&stem.as_str())
        || ["COM", "LPT"].iter().any(|prefix| {
            stem.strip_prefix(prefix)
                .is_some_and(|number| ["1", "2", "3", "4", "5", "6", "7", "8", "9", "¹", "²", "³"].contains(&number))
        });
    ensure!(!device, "reserved Windows filename");
    Ok(())
}

pub(super) fn ensure_absent(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
        Ok(_) => bail!("destination already exists: {}", path.display()),
    }
}

pub(super) fn rel_key(root: &Path, path: &Path) -> Result<String> {
    let relative = path.strip_prefix(root).with_context(|| format!("{} is outside {}", path.display(), root.display()))?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

pub(super) fn sidecars(path: &Path) -> Result<Vec<PathBuf>> {
    let Some(parent) = path.parent() else { return Ok(Vec::new()) };
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else { return Ok(Vec::new()) };
    let stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("");
    let mut found = Vec::new();
    for entry in fs::read_dir(parent)? {
        let entry = entry?;
        let candidate = entry.path();
        let Some(candidate_name) = candidate.file_name().and_then(|name| name.to_str()) else { continue };
        if !candidate.is_file() || !candidate.extension().is_some_and(|ext| ext.to_string_lossy().eq_ignore_ascii_case("xmp")) {
            continue;
        }
        if candidate.file_stem().and_then(|stem| stem.to_str()) == Some(stem)
            || candidate_name.eq_ignore_ascii_case(&format!("{name}.xmp"))
        {
            found.push(candidate);
        }
    }
    found.sort();
    ensure!(found.len() <= 2, "ambiguous XMP sidecars for {}", path.display());
    Ok(found)
}

pub(super) fn selected_paths(entries: &[FileEntry], ids: &[ImageId]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for id in ids {
        let entry = entries.iter().find(|entry| entry.info.id == *id).with_context(|| format!("unknown image identity {}", id.0))?;
        if !paths.contains(&entry.path) {
            paths.push(entry.path.clone());
        }
    }
    Ok(paths)
}

/// A folder is offline when its path is gone or every recorded image under it
/// is missing. An existing folder whose images are still present is online.
pub(super) fn folder_is_offline(entries: &[FileEntry], folder: &Path) -> bool {
    if !folder.exists() {
        return true;
    }
    let mut recorded = entries.iter().filter(|entry| entry.path.starts_with(folder)).peekable();
    recorded.peek().is_some() && recorded.all(|entry| !entry.path.exists())
}

fn transfer_target(from: &Path, to: &Path, sidecar: &Path) -> PathBuf {
    let from_name = from.file_name().and_then(|name| name.to_str()).unwrap_or("");
    let sidecar_name = sidecar.file_name().and_then(|name| name.to_str()).unwrap_or("");
    if sidecar_name.eq_ignore_ascii_case(&format!("{from_name}.xmp")) {
        to.with_file_name(format!("{}.xmp", to.file_name().and_then(|name| name.to_str()).unwrap_or("image")))
    } else {
        to.with_extension("xmp")
    }
}

pub(super) fn expand_sidecars(pairs: Vec<(PathBuf, PathBuf)>) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut all = pairs;
    let originals = all.clone();
    for (from, to) in originals {
        for sidecar in sidecars(&from)? {
            let target = transfer_target(&from, &to, &sidecar);
            if !all.iter().any(|(source, _)| path_key(source) == path_key(&sidecar)) {
                all.push((sidecar, target));
            }
        }
    }
    Ok(all)
}

pub(super) fn entry_record(info: &ImageInfo, rel_path: String) -> Entry {
    Entry {
        rel_path,
        rating: info.rating,
        color_tag: ColorTag::ALL.iter().position(|tag| *tag == info.color_tag).unwrap_or(0) as i8,
        keywords: info.keywords.clone(),
        variant: info.variant,
        updated_at: super::now_secs(),
        trashed: info.trashed,
    }
}

#[cfg(windows)]
pub(super) fn move_no_replace(from: &Path, to: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32;
    }
    let from: Vec<_> = from.as_os_str().encode_wide().chain(Some(0)).collect();
    let to: Vec<_> = to.as_os_str().encode_wide().chain(Some(0)).collect();
    // MOVEFILE_COPY_ALLOWED, no MOVEFILE_REPLACE_EXISTING: no-clobber across volumes.
    if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 2) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(not(windows))]
pub(super) fn move_no_replace(from: &Path, to: &Path) -> Result<()> {
    copy_no_replace(from, to)?;
    fs::remove_file(from)?;
    Ok(())
}

pub(super) fn copy_no_replace(from: &Path, to: &Path) -> Result<()> {
    let mut source = fs::File::open(from).with_context(|| format!("reading {}", from.display()))?;
    let mut target = fs::OpenOptions::new().write(true).create_new(true).open(to).with_context(|| format!("creating {}", to.display()))?;
    let copied = (|| -> Result<()> {
        io::copy(&mut source, &mut target)?;
        target.sync_all()?;
        Ok(())
    })();
    if let Err(error) = copied {
        let _ = fs::remove_file(to);
        return Err(error);
    }
    Ok(())
}

pub(super) fn rollback(moves: &[(PathBuf, PathBuf)], copy: bool, error: anyhow::Error) -> anyhow::Error {
    let mut failures = Vec::new();
    for (from, to) in moves.iter().rev() {
        let result = if copy { fs::remove_file(to).map_err(Into::into) } else { move_no_replace(to, from) };
        if let Err(failure) = result {
            failures.push(format!("{} -> {}: {failure}", to.display(), from.display()));
        }
    }
    if failures.is_empty() { error } else { anyhow::anyhow!("{error}; rollback failed: {}", failures.join("; ")) }
}

impl FileEngine {
    /// Library operation entry point; the bodies live in
    /// [`super::services::LibraryService`] and the signature is unchanged.
    pub(super) fn manage_library(&self, action: LibraryAction) -> Result<Vec<ImageInfo>> {
        self.library().manage_library(action)
    }
}

#[cfg(windows)]
pub(super) fn recycle_paths(paths: &[PathBuf]) -> Result<()> {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;

    #[repr(C)]
    struct ShFileOpStructW {
        hwnd: *mut c_void,
        w_func: u32,
        p_from: *const u16,
        p_to: *const u16,
        f_flags: u16,
        f_any_operations_aborted: i32,
        h_name_mappings: *mut c_void,
        lpsz_progress_title: *const u16,
    }

    #[link(name = "shell32")]
    unsafe extern "system" {
        fn SHFileOperationW(operation: *mut ShFileOpStructW) -> i32;
    }

    let mut from = Vec::new();
    for path in paths {
        from.extend(path.as_os_str().encode_wide());
        from.push(0);
    }
    from.push(0);
    let mut operation = ShFileOpStructW {
        hwnd: std::ptr::null_mut(),
        w_func: 3, // FO_DELETE
        p_from: from.as_ptr(),
        p_to: std::ptr::null(),
        // ALLOWUNDO | NOCONFIRMATION | SILENT | NOERRORUI
        f_flags: 0x0040 | 0x0010 | 0x0004 | 0x0400,
        f_any_operations_aborted: 0,
        h_name_mappings: std::ptr::null_mut(),
        lpsz_progress_title: std::ptr::null(),
    };
    let result = unsafe { SHFileOperationW(&mut operation) };
    if result != 0 {
        bail!("Recycle Bin operation failed with code {result}");
    }
    ensure!(operation.f_any_operations_aborted == 0, "Recycle Bin operation was aborted");
    Ok(())
}

#[cfg(not(windows))]
pub(super) fn recycle_paths(_paths: &[PathBuf]) -> Result<()> {
    bail!("Recycle Bin is Windows-only; pass permanent = true to delete explicitly")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ImageEngine;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "aurora-library-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn folder(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            fs::create_dir_all(&path).unwrap();
            path
        }

        fn image(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            image::RgbaImage::from_pixel(4, 3, image::Rgba([20, 40, 60, 255])).save(&path).unwrap();
            path
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn edits(exposure: f32) -> AdjustmentSet {
        let mut adjustments = AdjustmentSet::default();
        adjustments.exposure.exposure = exposure;
        adjustments
    }

    #[test]
    fn library_root_and_collection_path_resolve_canonical_folders() {
        let fixture = Fixture::new("root");
        fixture.image("shoot/a.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let root = canonical_root(&fixture.0).unwrap();
        assert_eq!(engine.library_root(), Some(root.clone()));
        let folder = engine.collections().into_iter().find(|node| node.id == 10).unwrap();
        let shoot = folder.children.iter().find(|node| node.name == "shoot").unwrap();
        assert_eq!(engine.collection_path(shoot.id), Some(root.join("shoot")));
        assert_eq!(engine.collection_path(u64::MAX), None);
    }

    #[test]
    fn mock_engine_has_no_library_root_and_rejects_file_actions() {
        let engine = crate::MockEngine::new();
        assert!(engine.library_root().is_none());
        assert!(engine.manage_files(LibraryAction::CreateFolder { path: "new".into() }).is_err());
    }

    #[test]
    fn library_service_delegates_manage_files() {
        let fixture = Fixture::new("service");
        fixture.image("a.png");
        fixture.folder("dest");
        let engine = FileEngine::open(&fixture.0).unwrap();
        // Use the engine's canonical root so the macOS `/var` -> `/private/var`
        // symlink does not trip the inside-root check.
        let root = engine.library_root().unwrap();
        let id = engine.images()[0].id;
        let moved = engine
            .library()
            .manage_library(LibraryAction::Move { images: vec![id], destination: root.join("dest") })
            .unwrap();
        assert_eq!(moved.len(), 1);
        assert_eq!(engine.images()[0].path, root.join("dest/a.png"));
        assert!(root.join("dest/a.png").is_file());
    }

    #[test]
    fn synchronize_adds_images_from_a_subfolder_and_rejects_outside_paths() {
        let fixture = Fixture::new("sync");
        let engine = FileEngine::open(&fixture.0).unwrap();
        fixture.image("external/a.png");
        assert!(engine.manage_files(LibraryAction::Synchronize { folder: fixture.0.join("external"), recursive: false }).unwrap().len() == 1);
        assert!(engine.manage_files(LibraryAction::Synchronize { folder: fixture.0.join(".."), recursive: false }).is_err());
    }

    #[test]
    fn move_rename_preserves_edits_and_reopens() {
        let fixture = Fixture::new("move");
        fixture.image("shoot/a.png");
        fixture.folder("selects");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let id = engine.images()[0].id;
        engine.save_adjustments(id, &edits(0.75)).unwrap();
        engine.manage_files(LibraryAction::Move { images: vec![id], destination: fixture.0.join("selects") }).unwrap();
        engine.manage_files(LibraryAction::Rename { image: id, name: "hero.png".into() }).unwrap();
        assert_eq!(engine.images()[0].path, fixture.0.join("selects/hero.png"));
        drop(engine);
        let reopened = FileEngine::open(&fixture.0).unwrap();
        assert_eq!(reopened.images()[0].path, fixture.0.join("selects/hero.png"));
        assert_eq!(reopened.load_adjustments(reopened.images()[0].id).exposure.exposure, 0.75);
    }

    #[test]
    fn move_collision_rejects_the_whole_batch() {
        let fixture = Fixture::new("collision");
        fixture.image("a.png");
        fixture.image("b.png");
        fixture.image("dest/a.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let ids: Vec<_> = engine.images().iter().filter(|image| image.path != fixture.0.join("dest/a.png")).map(|image| image.id).collect();
        let error = engine.manage_files(LibraryAction::Move { images: ids, destination: fixture.0.join("dest") }).unwrap_err();
        assert!(error.to_string().contains("already exists"));
        assert!(fixture.0.join("a.png").is_file());
        assert!(fixture.0.join("b.png").is_file());
        assert!(!fixture.0.join("dest/b.png").exists());
    }

    #[test]
    fn forget_hides_original_from_reopen_but_keeps_the_file() {
        let fixture = Fixture::new("forget");
        let path = fixture.image("a.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let id = engine.images()[0].id;
        engine.manage_files(LibraryAction::Forget { images: vec![id] }).unwrap();
        assert!(path.is_file());
        assert!(engine.images().is_empty());
        drop(engine);
        assert!(FileEngine::open(&fixture.0).unwrap().images().is_empty());
    }

    #[test]
    fn locate_relinks_an_offline_source_without_changing_identity() {
        let fixture = Fixture::new("locate");
        let source = fixture.image("a.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let id = engine.images()[0].id;
        let moved = fixture.0.join("moved.png");
        fs::rename(&source, &moved).unwrap();
        engine.manage_files(LibraryAction::LocateImage { image: id, to: moved.clone() }).unwrap();
        assert_eq!(engine.images()[0].id, id);
        assert_eq!(engine.images()[0].path, moved);
    }

    #[test]
    fn create_folder_rejects_escape_and_reserved_names() {
        let fixture = Fixture::new("create");
        let engine = FileEngine::open(&fixture.0).unwrap();
        assert!(engine.manage_files(LibraryAction::CreateFolder { path: "../escape".into() }).is_err());
        assert!(engine.manage_files(LibraryAction::CreateFolder { path: "CON".into() }).is_err());
        engine.manage_files(LibraryAction::CreateFolder { path: "new".into() }).unwrap();
        assert!(fixture.0.join("new").is_dir());
    }

    #[test]
    fn copy_never_overwrites_and_adds_new_images() {
        let fixture = Fixture::new("copy");
        let source = fixture.image("a.png");
        fixture.image("dest/a.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let id = engine.images()[0].id;
        let error = engine.manage_files(LibraryAction::Copy { images: vec![id], destination: fixture.0.join("dest") }).unwrap_err();
        assert!(error.to_string().contains("already exists"));
        assert!(source.is_file());
        assert_eq!(engine.images().len(), 2);
        fixture.folder("fresh");
        let copied = engine.manage_files(LibraryAction::Copy { images: vec![id], destination: fixture.0.join("fresh") }).unwrap();
        assert_eq!(copied.len(), 1);
        assert!(fixture.0.join("fresh/a.png").is_file());
    }

    #[test]
    fn delete_requires_explicit_permanent_and_removes_the_record() {
        let fixture = Fixture::new("delete");
        let path = fixture.image("a.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let id = engine.images()[0].id;
        engine.manage_files(LibraryAction::Delete { images: vec![id], permanent: true }).unwrap();
        assert!(!path.exists());
        assert!(engine.images().is_empty());
        drop(engine);
        assert!(FileEngine::open(&fixture.0).unwrap().images().is_empty());
    }

    #[test]
    fn delete_rejects_unknown_and_private_paths() {
        let fixture = Fixture::new("delete-edge");
        let path = fixture.image("a.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        assert!(engine.manage_files(LibraryAction::Delete { images: vec![ImageId(999)], permanent: true }).is_err());
        let id = engine.images()[0].id;
        fs::remove_file(&path).unwrap();
        assert!(engine.manage_files(LibraryAction::Delete { images: vec![id], permanent: true }).is_err());
    }

    #[test]
    fn rename_many_rejects_conflicting_targets() {
        let fixture = Fixture::new("rename-many");
        fixture.image("a.png");
        fixture.image("b.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let ids: Vec<_> = engine.images().iter().map(|image| image.id).collect();
        let error = engine.manage_files(LibraryAction::RenameMany { names: vec![(ids[0], "same.png".into()), (ids[1], "same.png".into())] }).unwrap_err();
        assert!(error.to_string().contains("duplicate destination") || error.to_string().contains("conflicting"));
    }

    #[test]
    fn rename_many_moves_all_images_and_keeps_edits() {
        let fixture = Fixture::new("rename-many-ok");
        fixture.image("a.png");
        fixture.image("b.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let ids: Vec<_> = engine.images().iter().map(|image| image.id).collect();
        engine.save_adjustments(ids[0], &edits(0.4)).unwrap();
        let images = engine.manage_files(LibraryAction::RenameMany {
            names: vec![(ids[0], "x.png".into()), (ids[1], "y.png".into())],
        }).unwrap();
        assert_eq!(images.len(), 2);
        assert!(fixture.0.join("x.png").is_file());
        assert!(fixture.0.join("y.png").is_file());
        assert_eq!(engine.load_adjustments(ids[0]).exposure.exposure, 0.4);
    }

    #[test]
    fn locate_folder_relinks_every_image_in_the_offline_folder() {
        let fixture = Fixture::new("locate-folder");
        let offline = fixture.image("offline/a.png");
        let online = fixture.folder("online");
        let engine = FileEngine::open(&fixture.0).unwrap();
        let id = engine.images()[0].id;
        let moved = online.join("a.png");
        fs::rename(&offline, &moved).unwrap();
        engine.manage_files(LibraryAction::Locate { from: fixture.0.join("offline"), to: online.clone() }).unwrap();
        assert_eq!(engine.images()[0].id, id);
        assert_eq!(engine.images()[0].path, moved);
    }

    #[test]
    fn locate_rejects_a_folder_whose_images_are_still_online() {
        let fixture = Fixture::new("locate-online");
        fixture.image("online/a.png");
        fixture.folder("target");
        let engine = FileEngine::open(&fixture.0).unwrap();
        assert!(engine
            .manage_files(LibraryAction::Locate {
                from: fixture.0.join("online"),
                to: fixture.0.join("target"),
            })
            .is_err());
    }

    #[test]
    fn move_folder_preserves_relative_paths() {
        let fixture = Fixture::new("folder");
        fixture.image("shoot/a.png");
        let engine = FileEngine::open(&fixture.0).unwrap();
        assert!(engine.manage_files(LibraryAction::MoveFolder { from: fixture.0.clone(), to: fixture.0.join("root-move") }).is_err());
        engine.manage_files(LibraryAction::MoveFolder { from: fixture.0.join("shoot"), to: fixture.0.join("renamed") }).unwrap();
        assert_eq!(engine.images()[0].path, fixture.0.join("renamed/a.png"));
        assert!(fixture.0.join("renamed/a.png").is_file());
    }
}
