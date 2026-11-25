//! Library filesystem commands over the Windows `FileEngine`.
//!
//! The names and return types follow Levin's `FileEngine` implementation:
//! `library_root() -> Option<PathBuf>`, `collection_path(id) -> Option<PathBuf>`,
//! and `manage_files(LibraryAction) -> Result<Vec<ImageInfo>>`.
//!
//! All mutations preflight the whole batch before touching a file. A collision,
//! a missing source or an I/O failure therefore leaves the original batch in
//! place (or restores the files already moved).

use super::*;
use crate::{LibraryAction, UserCollection};
use crate::engine::ImageEngine;
use anyhow::{anyhow, bail, ensure, Context, Result};
use std::collections::{HashMap, HashSet};
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

fn path_key(path: &Path) -> String {
    let key = path.to_string_lossy().replace('\\', "/");
    if cfg!(windows) { key.to_lowercase() } else { key }
}

fn is_private(path: &Path) -> bool {
    path.components().any(|component| {
        let name = component.as_os_str().to_string_lossy().to_ascii_lowercase();
        name == ".portrait"
            || name == ".lumen"
            || ["cocatalogdb", "cosessiondb", "cos"]
                .iter()
                .any(|extension| name.ends_with(&format!(".{extension}")))
    })
}

/// Canonicalize the deepest existing ancestor and rejoin the components that do
/// not exist yet. macOS reports `/var/...` from `temp_dir()` while the physical
/// path is `/private/var/...`; comparing a canonical root against a
/// non-canonical request would falsely reject a path inside the library.
fn physical_path(path: &Path) -> Option<PathBuf> {
    if let Ok(path) = fs::canonicalize(path) {
        return Some(path);
    }
    let mut existing = path;
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        let parent = existing.parent()?;
        tail.push(existing.file_name()?.to_os_string());
        if let Ok(base) = fs::canonicalize(parent) {
            let mut out = base;
            for part in tail.iter().rev() {
                out.push(part);
            }
            return Some(out);
        }
        existing = parent;
    }
}

fn resolve_inside(root: &Path, path: &Path) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "empty path");
    ensure!(!path.components().any(|component| matches!(component, Component::ParentDir)), "parent traversal is not allowed");
    let path = if path.is_absolute() {
        plain_path(&physical_path(path).ok_or_else(|| anyhow!("path is unavailable: {}", path.display()))?)
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

fn existing_directory(root: &Path, path: &Path) -> Result<PathBuf> {
    let path = resolve_inside(root, path)?;
    ensure!(path.is_dir(), "directory unavailable: {}", path.display());
    let path = canonical_root(&path)?;
    ensure!(!is_private(&path), "private catalog directories cannot be used");
    Ok(path)
}

fn validate_name(name: &str) -> Result<()> {
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

fn ensure_absent(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
        Ok(_) => bail!("destination already exists: {}", path.display()),
    }
}

fn rel_key(root: &Path, path: &Path) -> Result<String> {
    let relative = path.strip_prefix(root).with_context(|| format!("{} is outside {}", path.display(), root.display()))?;
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn sidecars(path: &Path) -> Result<Vec<PathBuf>> {
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

fn selected_paths(entries: &[FileEntry], ids: &[ImageId]) -> Result<Vec<PathBuf>> {
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
fn folder_is_offline(entries: &[FileEntry], folder: &Path) -> bool {
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

fn expand_sidecars(pairs: Vec<(PathBuf, PathBuf)>) -> Result<Vec<(PathBuf, PathBuf)>> {
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

fn entry_record(info: &ImageInfo, rel_path: String) -> Entry {
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
fn move_no_replace(from: &Path, to: &Path) -> Result<()> {
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
fn move_no_replace(from: &Path, to: &Path) -> Result<()> {
    copy_no_replace(from, to)?;
    fs::remove_file(from)?;
    Ok(())
}

fn copy_no_replace(from: &Path, to: &Path) -> Result<()> {
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

fn rollback(moves: &[(PathBuf, PathBuf)], copy: bool, error: anyhow::Error) -> anyhow::Error {
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
    pub(super) fn manage_library(&self, action: LibraryAction) -> Result<Vec<ImageInfo>> {
        ensure!(!self.c1_catalog, "library filesystem operations are read-only for a Capture One catalog");
        if let LibraryAction::Synchronize { folder, recursive } = &action {
            let folder = existing_directory(&self.root, folder)?;
            self.rescan()?;
            let _ = recursive;
            return Ok(self.images().into_iter().filter(|image| image.path.starts_with(&folder)).collect());
        }
        match action {
            LibraryAction::Move { images, destination } => self.move_images(&images, &destination),
            LibraryAction::MoveFolder { from, to } => self.move_folder(&from, &to),
            LibraryAction::Rename { image, name } => {
                validate_name(&name)?;
                let source = selected_paths(&self.state.lock().entries, &[image])?.remove(0);
                let target = source.with_file_name(name);
                self.move_paths(vec![(source, target)])
            }
            LibraryAction::RenameMany { names } => self.rename_many(&names),
            LibraryAction::Locate { from, to } => self.relink_folder(&from, &to),
            LibraryAction::LocateImage { image, to } => self.relink_image(image, &to),
            LibraryAction::Forget { images } => self.forget(&images),
            LibraryAction::CreateFolder { path } => self.create_folder(&path),
            LibraryAction::Copy { images, destination } => self.copy_images(&images, &destination),
            LibraryAction::Delete { images, permanent } => self.delete_images(&images, permanent),
            LibraryAction::Synchronize { .. } => unreachable!(),
        }
    }

    fn move_images(&self, images: &[ImageId], destination: &Path) -> Result<Vec<ImageInfo>> {
        let destination = existing_directory(&self.root, destination)?;
        let entries = self.state.lock().entries.clone();
        let mut pairs = Vec::new();
        for source in selected_paths(&entries, images)? {
            let target = destination.join(source.file_name().context("source has no filename")?);
            if source != target {
                pairs.push((source, target));
            }
        }
        self.move_paths(pairs)
    }

    fn rename_many(&self, names: &[(ImageId, String)]) -> Result<Vec<ImageInfo>> {
        let entries = self.state.lock().entries.clone();
        let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
        for (image, name) in names {
            validate_name(name)?;
            let source = selected_paths(&entries, &[*image])?.remove(0);
            let target = source.with_file_name(name);
            if let Some((_, previous)) = pairs.iter().find(|(from, _)| path_key(from) == path_key(&source)) {
                ensure!(path_key(previous) == path_key(&target), "variants of one file have conflicting rename targets");
            } else if source != target {
                pairs.push((source, target));
            }
        }
        self.move_paths(pairs)
    }

    fn move_paths(&self, pairs: Vec<(PathBuf, PathBuf)>) -> Result<Vec<ImageInfo>> {
        if pairs.is_empty() {
            return Ok(self.images());
        }
        let pairs = expand_sidecars(pairs)?;
        self.apply_pairs(&pairs, false)?;
        let map: HashMap<PathBuf, PathBuf> = pairs
            .iter()
            .filter(|(from, _)| !from.extension().is_some_and(|extension| extension.to_string_lossy().eq_ignore_ascii_case("xmp")))
            .cloned()
            .collect();
        self.relink_entries(&map)
    }

    fn move_folder(&self, from: &Path, to: &Path) -> Result<Vec<ImageInfo>> {
        let from = existing_directory(&self.root, from)?;
        ensure!(from != self.root && !self.root.starts_with(&from), "cannot move the library root or its ancestor");
        let to = resolve_inside(&self.root, to)?;
        validate_name(to.file_name().and_then(|name| name.to_str()).context("invalid folder name")?)?;
        let parent = existing_directory(&self.root, to.parent().context("missing destination parent")?)?;
        let to = parent.join(to.file_name().context("missing destination name")?);
        ensure!(!to.starts_with(&from), "cannot move a folder into itself");
        ensure_absent(&to)?;
        let entries = self.state.lock().entries.clone();
        let map: HashMap<PathBuf, PathBuf> = entries
            .iter()
            .filter(|entry| entry.path.starts_with(&from))
            .map(|entry| (entry.path.clone(), to.join(entry.path.strip_prefix(&from).unwrap_or(Path::new("")))))
            .collect();
        fs::rename(&from, &to).with_context(|| format!("moving folder {} to {}", from.display(), to.display()))?;
        match self.relink_entries(&map) {
            Ok(images) => Ok(images),
            Err(error) => {
                let _ = fs::rename(&to, &from);
                Err(error)
            }
        }
    }

    fn relink_folder(&self, from: &Path, to: &Path) -> Result<Vec<ImageInfo>> {
        let from = resolve_inside(&self.root, from)?;
        ensure!(from != self.root && !is_private(&from), "cannot relink the library root or private data");
        let to = existing_directory(&self.root, to)?;
        let entries = self.state.lock().entries.clone();
        ensure!(folder_is_offline(&entries, &from), "Locate requires an offline folder");
        let map: HashMap<PathBuf, PathBuf> = entries
            .iter()
            .filter(|entry| entry.path.starts_with(&from))
            .map(|entry| (entry.path.clone(), to.join(entry.path.strip_prefix(&from).unwrap_or(Path::new("")))))
            .collect();
        for target in map.values() {
            ensure!(target.is_file(), "relink target unavailable: {}", target.display());
        }
        self.relink_entries(&map)
    }

    fn relink_image(&self, image: ImageId, to: &Path) -> Result<Vec<ImageInfo>> {
        let entries = self.state.lock().entries.clone();
        let from = selected_paths(&entries, &[image])?.remove(0);
        ensure!(!from.exists(), "LocateImage requires an offline source");
        let to = canonical_root(&resolve_inside(&self.root, to)?)?;
        ensure!(to.is_file() && !is_private(&to), "relink target must be an existing image");
        ensure!(
            !entries.iter().any(|entry| entry.path != from && path_key(&entry.path) == path_key(&to)),
            "relink target already belongs to another source"
        );
        self.relink_entries(&HashMap::from([(from, to)]))
    }

    fn create_folder(&self, path: &Path) -> Result<Vec<ImageInfo>> {
        let path = resolve_inside(&self.root, path)?;
        validate_name(path.file_name().and_then(|name| name.to_str()).context("invalid folder name")?)?;
        let parent = existing_directory(&self.root, path.parent().context("folder has no parent")?)?;
        let path = parent.join(path.file_name().context("missing folder name")?);
        ensure!(!is_private(&path), "folder name is reserved for private data");
        ensure_absent(&path)?;
        fs::create_dir(&path)?;
        Ok(self.images())
    }

    fn copy_images(&self, images: &[ImageId], destination: &Path) -> Result<Vec<ImageInfo>> {
        let destination = existing_directory(&self.root, destination)?;
        let entries = self.state.lock().entries.clone();
        let sources = selected_paths(&entries, images)?;
        let mut pairs = Vec::new();
        for source in &sources {
            let target = destination.join(source.file_name().context("source has no filename")?);
            ensure!(path_key(&target) != path_key(source), "cannot copy a file onto itself");
            pairs.push((source.clone(), target));
        }
        let pairs = expand_sidecars(pairs)?;
        self.apply_pairs(&pairs, true)?;
        let mut copied = Vec::new();
        for source in sources {
            let source_entry = entries.iter().find(|entry| entry.path == source).cloned().context("source disappeared")?;
            let target = destination.join(source.file_name().context("source has no filename")?);
            let info = self.add_file(&target)?;
            if let Some(entry) = self.state.lock().entries.iter_mut().find(|entry| entry.info.id == info.id) {
                entry.adjustments = source_entry.adjustments.clone();
            }
            if let Some(store) = &self.store {
                store.save_adjustments(&rel_key(&self.root, &target)?, &source_entry.adjustments)?;
            }
            let record = {
                let state = self.state.lock();
                let entry = state.entries.iter().find(|entry| entry.info.id == info.id).unwrap();
                entry_record(&entry.info, entry.rel_path.clone())
            };
            let catalog = {
                let mut state = self.state.lock();
                state.catalog.upsert(record);
                state.catalog.clone()
            };
            if let Some(store) = &self.store {
                store.save_index(&catalog)?;
            }
            copied.push(self.images().into_iter().find(|image| image.id == info.id).unwrap());
        }
        Ok(copied)
    }

    fn delete_images(&self, images: &[ImageId], permanent: bool) -> Result<Vec<ImageInfo>> {
        let entries = self.state.lock().entries.clone();
        let paths = selected_paths(&entries, images)?;
        let mut delete = Vec::new();
        for path in &paths {
            ensure!(path.is_file() && !is_private(path), "source unavailable or protected: {}", path.display());
            delete.push(path.clone());
            for sidecar in sidecars(path)? {
                delete.push(sidecar);
            }
        }
        if permanent {
            for path in &delete {
                fs::remove_file(path)?;
            }
        } else {
            recycle_paths(&delete)?;
        }
        let mut state = self.state.lock();
        state.entries.retain(|entry| !images.contains(&entry.info.id));
        for path in &paths {
            if !state.entries.iter().any(|entry| entry.path == *path) {
                if let Some(store) = &self.store {
                    let mut excluded = store.load_excluded();
                    excluded.push(path.clone());
                    store.save_excluded(&excluded)?;
                }
            }
        }
        for image in images {
            if let Some(entry) = entries.iter().find(|entry| entry.info.id == *image) {
                state.catalog.remove(&entry.rel_path);
                if let Some(store) = &self.store {
                    store.remove_adjustments(&entry.rel_path)?;
                }
            }
        }
        if let Some(store) = &self.store {
            store.save_index(&state.catalog)?;
        }
        Ok(state.entries.iter().map(|entry| entry.info.clone()).collect())
    }

    fn forget(&self, images: &[ImageId]) -> Result<Vec<ImageInfo>> {
        let mut state = self.state.lock();
        let removed: Vec<FileEntry> = state.entries.iter().filter(|entry| images.contains(&entry.info.id)).cloned().collect();
        ensure!(!removed.is_empty(), "no matching images");
        state.entries.retain(|entry| !images.contains(&entry.info.id));
        for entry in &removed {
            state.catalog.remove(&entry.rel_path);
            if let Some(store) = &self.store {
                store.remove_adjustments(&entry.rel_path)?;
            }
        }
        if let Some(store) = &self.store {
            let mut excluded = store.load_excluded();
            for entry in &removed {
                if !state.entries.iter().any(|remaining| remaining.path == entry.path) {
                    excluded.push(entry.path.clone());
                }
            }
            store.save_excluded(&excluded)?;
            let mut albums: Vec<UserCollection> = store.load_user_collections();
            for album in &mut albums {
                album.images.retain(|id| !images.contains(id));
            }
            store.save_user_collections(&albums)?;
            store.save_index(&state.catalog)?;
        }
        Ok(state.entries.iter().map(|entry| entry.info.clone()).collect())
    }

    fn apply_pairs(&self, pairs: &[(PathBuf, PathBuf)], copy: bool) -> Result<()> {
        let mut sources = HashSet::new();
        let mut targets = HashSet::new();
        for (from, to) in pairs {
            ensure!(from.is_file() && !is_private(from), "source unavailable or protected: {}", from.display());
            ensure!(!is_private(to), "destination is private data");
            ensure!(to.parent().is_some_and(Path::is_dir), "destination folder unavailable: {}", to.display());
            validate_name(to.file_name().and_then(|name| name.to_str()).context("invalid destination filename")?)?;
            ensure!(sources.insert(path_key(from)), "one source cannot move to multiple destinations");
            ensure!(targets.insert(path_key(to)), "duplicate destination: {}", to.display());
            ensure_absent(to)?;
            ensure!(
                !self.state.lock().entries.iter().any(|entry| {
                    path_key(&entry.path) == path_key(to)
                        && !pairs.iter().any(|(source, _)| path_key(source) == path_key(&entry.path))
                }),
                "destination already cataloged: {}",
                to.display()
            );
        }
        let mut completed = Vec::new();
        for (from, to) in pairs {
            let result = if copy { copy_no_replace(from, to) } else { move_no_replace(from, to) };
            if let Err(error) = result {
                return Err(rollback(&completed, copy, error));
            }
            completed.push((from.clone(), to.clone()));
        }
        Ok(())
    }

    fn relink_entries(&self, map: &HashMap<PathBuf, PathBuf>) -> Result<Vec<ImageInfo>> {
        if map.is_empty() {
            return Ok(self.images());
        }
        let mut state = self.state.lock();
        let mut entries = std::mem::take(&mut state.entries);
        for entry in &mut entries {
            let Some(target) = map.get(&entry.path) else { continue };
            let old_path = entry.path.clone();
            let old_rel = entry.rel_path.clone();
            let old_base = rel_key(&self.root, &old_path)?;
            let suffix = old_rel.strip_prefix(&old_base).unwrap_or("");
            let new_rel = format!("{}{suffix}", rel_key(&self.root, target)?);
            entry.path = target.clone();
            entry.info.path = target.clone();
            entry.info.file_name = target.file_name().unwrap_or_default().to_string_lossy().into_owned();
            entry.rel_path = new_rel.clone();
            state.catalog.remove(&old_rel);
            state.catalog.upsert(entry_record(&entry.info, new_rel.clone()));
            if let Some(store) = &self.store {
                store.save_adjustments(&new_rel, &entry.adjustments)?;
                store.remove_adjustments(&old_rel)?;
            }
        }
        state.entries = entries;
        state.sources.clear();
        state.ai_inputs.clear();
        state.native_raw = None;
        // Ids were reassigned, so every pin would point at the wrong image.
        state.source_cache.pins.clear();
        state.source_cache.recent.clear();
        state.source_cache.current_bytes = 0;
        state.preview = PreviewCache::default();
        if let Some(store) = &self.store {
            store.save_index(&state.catalog)?;
        }
        Ok(state.entries.iter().map(|entry| entry.info.clone()).collect())
    }
}

#[cfg(windows)]
fn recycle_paths(paths: &[PathBuf]) -> Result<()> {
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
fn recycle_paths(_paths: &[PathBuf]) -> Result<()> {
    bail!("Recycle Bin is Windows-only; pass permanent = true to delete explicitly")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "aurora-library-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
            ));
            fs::create_dir_all(&path).unwrap();
            // `temp_dir()` is `/var/...` on macOS while the physical path is
            // `/private/var/...`. The engine stores a canonical root, so the
            // fixture must use the same physical form or path equality is
            // platform-dependent.
            Self(canonical_root(&path).unwrap())
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
