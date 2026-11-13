//! Library operations share the catalog's identities and adjustment keys.

use super::*;
use anyhow::{Context, Result, bail, ensure};
use crate::{LibraryAction, UserCollection};
use crate::store::{FolderRecord, ExcludedVariant};
use std::collections::HashSet;
use std::fs;
use std::path::Component;

impl FileEngine {
    /// Offline viewing uses the last saved render, without reapplying edits or upscaling.
    pub(super) fn offline_library_preview(&self, image: ImageId, max_size: u32) -> Result<Option<crate::RgbaImage>> {
        let st = self.state.lock();
        let entry = st.entries.iter().find(|e| e.info.id == image).context("unknown image identity")?;
        if entry.path.is_file() { return Ok(None); }
        let store = self.store.as_ref().context("offline preview storage unavailable")?;
        let mut frame = store.load_library_preview(image)?;
        let long = frame.width().max(frame.height());
        let max = max_size.max(1);
        if long > max {
            let w = ((frame.width() as u64 * max as u64) / long as u64).max(1) as u32;
            let h = ((frame.height() as u64 * max as u64) / long as u64).max(1) as u32;
            frame = downscale_rgba(&frame, w, h);
        }
        Ok(Some(crate::RgbaImage::new(frame.width(), frame.height(), frame.into_raw())))
    }

    pub(super) fn writable_store(&self) -> Result<&Store> {
        let store = self.store.as_ref().context("library persistence is unavailable")?;
        store.guard_newer_index()?;
        Ok(store)
    }

    pub(super) fn restore_entries(root: &Path, store: Option<&Store>, catalog: &Catalog) -> Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        let mut used = HashSet::new();
        let mut next = catalog.images.iter().filter_map(|e| e.info.as_ref()).map(|i| i.id.0).max().unwrap_or(0) + 1;
        for record in &catalog.images {
            let mut info = match &record.info {
                Some(info) => info.clone(),
                None => {
                    // Legacy C1 keys encode an absolute Windows path as C_/...;
                    // the C1 importer supplies its actual path and metadata.
                    if record.rel_path.as_bytes().get(1..3) == Some(b"_/") { continue; }
                    let path = root.join(&record.rel_path);
                    let (w, h) = dimensions(&path);
                    let (exif, iptc) = crate::exif::read_metadata(&path);
                    let info = ImageInfo {
                        id: ImageId(next), path: path.clone(),
                        file_name: path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                        kind: file_kind(&path), width: w, height: h,
                        file_size: fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
                        exif, iptc, keywords: record.keywords.clone(), rating: record.rating,
                        color_tag: ColorTag::None, variant: record.variant,
                        has_adjustments: false, trashed: record.trashed,
                     heif: None, };
                    next += 1;
                    info
                }
            };
            ensure!(used.insert(info.id), "duplicate image identity in catalog");
            if catalog.excluded.contains(&info.path) { continue; }
            info.rating = record.rating;
            info.color_tag = ColorTag::ALL[record.color_tag.clamp(0, 7) as usize];
            info.keywords = record.keywords.clone();
            info.variant = record.variant;
            info.trashed = record.trashed;
            let adjustments = match store {
                Some(store) => store.load_adjustments(&record.rel_path)?,
                None => AdjustmentSet::default(),
            };
            info.has_adjustments = !adjustments.is_default();
            entries.push(FileEntry { path: info.path.clone(), info, rel_path: record.rel_path.clone(), adjustments });
        }
        Ok(entries)
    }

    pub(super) fn unique_store_key(&self, path: &Path, variant: u32, id: ImageId, entries: &[FileEntry]) -> String {
        let key = match path.strip_prefix(&self.root) {
            Ok(rel) => {
                let key = rel.to_string_lossy().replace('\\', "/");
                if variant > 1 { format!("{key}#v{variant}") } else { key }
            }
            Err(_) => crate::c1import::store_key(path, variant),
        };
        // Keys stay put on moves so no adjustment or mask reference needs rekeying.
        if entries.iter().any(|e| e.rel_path == key) { format!("{key}#id{}", id.0) } else { key }
    }

    pub(super) fn discovered_entry(&self, path: &Path, id: ImageId, rel_path: String, legacy: Option<&Entry>) -> Result<FileEntry> {
        let adjustments = match &self.store {
            Some(store) => store.load_adjustments(&rel_path)?,
            None => AdjustmentSet::default(),
        };
        let (width, height) = dimensions(path);
        let (exif, iptc) = crate::exif::read_metadata(path);
        let info = ImageInfo {
            id, path: path.to_path_buf(),
            file_name: path.file_name().context("image has no filename")?.to_string_lossy().into_owned(),
            kind: file_kind(path), width, height, file_size: fs::metadata(path)?.len(), exif,
            keywords: legacy.map(|e| e.keywords.clone()).unwrap_or_else(|| iptc.keywords.clone()), iptc,
            rating: legacy.map(|e| e.rating).unwrap_or(0),
            color_tag: legacy.map(|e| ColorTag::ALL[e.color_tag.clamp(0, 7) as usize]).unwrap_or(ColorTag::None),
            variant: legacy.map(|e| e.variant).unwrap_or(1),
            has_adjustments: !adjustments.is_default(), trashed: legacy.is_some_and(|e| e.trashed),
         heif: None, };
        Ok(FileEntry { path: path.to_path_buf(), info, rel_path, adjustments })
    }

    pub(super) fn commit_library(
        &self, st: &mut State, entries: Vec<FileEntry>, mut catalog: Catalog,
        removed: &[String], collections: Option<&[UserCollection]>,
    ) -> Result<()> {
        let store = self.writable_store()?;
        let mut adjustments = Vec::new();
        let previous: HashMap<_, _> = st.entries.iter().map(|e| (e.rel_path.as_str(), &e.adjustments)).collect();
        let mut parents = HashSet::from([self.root.clone()]);
        catalog.images.clear();
        for e in &entries {
            if previous.get(e.rel_path.as_str()).is_none_or(|old| **old != e.adjustments) {
                adjustments.push((e.rel_path.clone(), e.adjustments.clone()));
            }
            catalog.images.push(Entry {
                rel_path: e.rel_path.clone(), rating: e.info.rating,
                color_tag: ColorTag::ALL.iter().position(|c| *c == e.info.color_tag).unwrap_or(0) as i8,
                keywords: e.info.keywords.clone(), variant: e.info.variant,
                updated_at: now_secs(), trashed: e.info.trashed, info: Some(e.info.clone()),
            });
            parents.insert(e.path.parent().context("image has no parent")?.to_path_buf());
        }
        let mut parents: Vec<_> = parents.into_iter().collect();
        parents.sort();
        for parent in parents { add_folder(&mut catalog, &self.root, &parent); }
        catalog.next_id = catalog.next_id.max(st.next_id).max(entries.iter().map(|e| e.info.id.0 + 1).max().unwrap_or(1));
        store.commit_library(&catalog, &adjustments, removed, collections)?;
        st.next_id = catalog.next_id;
        st.catalog = catalog;
        st.entries = entries;
        st.sources.clear();
        st.lru.clear();
        st.preview = PreviewCache::default();
        self.eyes.lock().clear();
        Ok(())
    }

    pub(super) fn synchronize_folder(&self, folder: &Path, recursive: bool, reimport: bool) -> Result<Vec<ImageInfo>> {
        self.writable_store()?;
        let folder = resolve_path(&self.root, folder)?;
        ensure!(folder.is_dir(), "folder unavailable: {}", folder.display());
        let folder = canonical_path(&folder)?;
        ensure!(!protected(&folder), "cannot synchronize a catalog's private data");
        let (files, folders) = scan(&folder, recursive)?;
        let mut st = self.state.lock();
        let mut entries = st.entries.clone();
        let mut catalog = st.catalog.clone();
        for folder in folders { add_folder(&mut catalog, &self.root, &folder); }
        let mut known_paths: HashSet<_> = entries.iter().map(|e| e.path.clone()).collect();
        let mut keys: HashSet<_> = entries.iter().map(|e| e.rel_path.clone()).collect();
        let excluded: HashSet<_> = catalog.excluded.iter().cloned().collect();
        let legacy: HashMap<_, _> = st.catalog.images.iter().map(|e| (e.rel_path.as_str(), e)).collect();
        let mut next_id = st.next_id.max(entries.iter().map(|e| e.info.id.0 + 1).max().unwrap_or(1));
        for path in files {
            if !known_paths.insert(path.clone()) { continue; }
            if excluded.contains(&path) {
                if !reimport { continue; }
                catalog.excluded.retain(|p| p != &path);
                catalog.excluded_variants.retain(|v| v.path != path);
            }
            let id = ImageId(next_id);
            next_id += 1;
            let mut key = self.unique_store_key(&path, 1, id, &[]);
            while !keys.insert(key.clone()) { key.push_str(&format!("#id{}", id.0)); }
            let record = legacy.get(key.as_str()).copied();
            entries.push(self.discovered_entry(&path, id, key, record)?);
        }
        catalog.next_id = next_id;
        self.commit_library(&mut st, entries, catalog, &[], None)?;
        Ok(st.entries.iter().map(|e| e.info.clone()).collect())
    }

    pub(super) fn collection_tree(&self) -> Vec<Collection> {
        fn folder_node(folder: &FolderRecord, folders: &[FolderRecord], entries: &[FileEntry]) -> Collection {
            Collection {
                id: folder.id,
                name: folder.path.file_name().unwrap_or(folder.path.as_os_str()).to_string_lossy().into_owned(),
                kind: CollectionKind::Folder,
                count: entries.iter().filter(|e| e.path.parent() == Some(folder.path.as_path())).count(),
                children: folders.iter().filter(|f| f.path.parent() == Some(folder.path.as_path()))
                    .map(|f| folder_node(f, folders, entries)).collect(),
            }
        }
        let st = self.state.lock();
        let folders = &st.catalog.folders;
        let roots = folders.iter().filter(|f| !folders.iter().any(|parent| f.path.parent() == Some(parent.path.as_path())))
            .map(|f| folder_node(f, folders, &st.entries)).collect::<Vec<_>>();
        let count = st.entries.len();
        let node = |id, name: &str, kind, count, children| Collection { id, name: name.into(), kind, count, children };
        vec![
            node(1, "All Images", CollectionKind::AllImages, count, vec![]),
            node(2, "Recent Imports", CollectionKind::RecentImports, count.min(12), vec![]),
            node(3, "Recent Captures", CollectionKind::RecentCaptures, 0, vec![]),
            node(4, "Trash", CollectionKind::Trash, st.entries.iter().filter(|e| e.info.trashed).count(), vec![]),
            node(10, "Folders", CollectionKind::Group, folders.len(), roots),
        ]
    }

    pub(super) fn manage_library(&self, action: LibraryAction) -> Result<Vec<ImageInfo>> {
        self.writable_store()?;
        if let LibraryAction::Synchronize { folder, recursive } = action {
            return self.synchronize_folder(&folder, recursive, false);
        }
        let mut st = self.state.lock();
        let mut entries = st.entries.clone();
        let mut catalog = st.catalog.clone();
        let mut transfers = Vec::new();
        let mut removed = Vec::new();
        let mut collections = None;
        let mut created = None;
        let mut folder_move = None;
        match action {
            LibraryAction::Move { images, destination } => {
                let destination = existing_directory(&self.root, &destination)?;
                let paths = selected_paths(&entries, &images)?;
                for path in paths {
                    let to = destination.join(path.file_name().context("source has no filename")?);
                    if path != to { relink(&mut entries, &path, &to); transfers.push((path, to)); }
                }
            }
            LibraryAction::Rename { image, name } => {
                validate_filename(&name)?;
                let path = selected_paths(&entries, &[image])?.remove(0);
                let to = path.with_file_name(name);
                if path != to { relink(&mut entries, &path, &to); transfers.push((path, to)); }
            }
            LibraryAction::RenameMany { names } => {
                for (image, name) in names {
                    validate_filename(&name)?;
                    let path = selected_paths(&st.entries, &[image])?.remove(0);
                    let to = path.with_file_name(name);
                    if let Some((_, previous)) = transfers.iter().find(|(from, _)| from == &path) {
                        ensure!(previous == &to, "variants of one file have conflicting rename targets");
                    } else {
                        transfers.push((path, to));
                    }
                }
                transfers.retain(|(from, to)| from != to);
                for (from, to) in &transfers { relink(&mut entries, from, to); }
            }
            LibraryAction::Locate { from, to } => {
                let from = resolve_path(&self.root, &from)?;
                let to = existing_directory(&self.root, &to)?;
                ensure!(!from.exists(), "Locate requires an offline folder");
                ensure!(from != self.root && !protected(&from), "cannot relink the library root or private data");
                remap_folder(&mut entries, &mut catalog, &from, &to, true)?;
            }
            LibraryAction::LocateImage { image, to } => {
                let from = selected_paths(&entries, &[image])?.remove(0);
                ensure!(!from.exists(), "LocateImage requires an offline source");
                let to = canonical_path(&resolve_path(&self.root, &to)?)?;
                ensure!(to.is_file() && is_supported(&to) && !protected(&to), "relink target must be an existing supported image");
                ensure!(!entries.iter().any(|e| e.path != from && path_key(&e.path) == path_key(&to)), "relink target already belongs to another source");
                relink(&mut entries, &from, &to);
                for variant in &mut catalog.excluded_variants {
                    if variant.path == from { variant.path = to.clone(); }
                }
            }
            LibraryAction::MoveFolder { from, to } => {
                let from = existing_directory(&self.root, &from)?;
                ensure!(!self.root.starts_with(&from), "cannot move the library root or its ancestor");
                let to = resolve_path(&self.root, &to)?;
                validate_filename(to.file_name().and_then(|n| n.to_str()).context("invalid folder name")?)?;
                let parent = existing_directory(&self.root, to.parent().context("missing destination parent")?)?;
                let to = parent.join(to.file_name().unwrap());
                ensure!(!protected(&to) && !to.starts_with(&from), "cannot move a folder into itself or private data");
                ensure_absent(&to)?;
                ensure!(!catalog.folders.iter().any(|f| f.path.starts_with(&to) && !f.path.starts_with(&from)), "destination folder already cataloged");
                remap_folder(&mut entries, &mut catalog, &from, &to, false)?;
                folder_move = Some((from, to));
            }
            LibraryAction::Forget { images } => {
                let paths = selected_paths(&entries, &images)?;
                let forgotten: HashSet<_> = images.into_iter().collect();
                entries.retain(|e| {
                    if forgotten.contains(&e.info.id) {
                        removed.push(e.rel_path.clone());
                        catalog.excluded_variants.push(ExcludedVariant { path: e.path.clone(), variant: e.info.variant, key: e.rel_path.clone() });
                        false
                    } else { true }
                });
                for path in paths {
                    if !entries.iter().any(|e| e.path == path) && !catalog.excluded.contains(&path) { catalog.excluded.push(path); }
                }
                let store = self.writable_store()?;
                // Unlike the UI's tolerant reader, a mutation must not silently discard a damaged album file.
                let path = store.root().join("collections.json");
                let mut albums: Vec<UserCollection> = match fs::read(&path) {
                    Ok(bytes) => serde_json::from_slice(&bytes)?,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
                    Err(e) => return Err(e.into()),
                };
                for album in &mut albums { album.images.retain(|id| !forgotten.contains(id)); }
                collections = Some(albums);
            }
            LibraryAction::CreateFolder { path } => {
                let path = resolve_path(&self.root, &path)?;
                validate_filename(path.file_name().and_then(|s| s.to_str()).context("invalid folder name")?)?;
                let parent = existing_directory(&self.root, path.parent().context("folder has no parent")?)?;
                ensure!(parent.starts_with(&self.root), "folder must be inside the library root");
                let path = parent.join(path.file_name().unwrap());
                ensure!(!protected(&path), "folder name is reserved for private data");
                ensure_absent(&path)?;
                fs::create_dir(&path)?;
                add_folder(&mut catalog, &self.root, &path);
                created = Some(path);
            }
            LibraryAction::Synchronize { .. } => unreachable!(),
        }
        for (from, to) in &transfers {
            for variant in &mut catalog.excluded_variants {
                if variant.path == *from { variant.path = to.clone(); }
            }
        }
        let moves = transfer_plan(&st.entries, &transfers)?;
        if let Some((from, to)) = &folder_move {
            fs::rename(from, to).with_context(|| format!("moving folder {} to {}; cross-volume folder moves are unsupported", from.display(), to.display()))?;
        }
        let mut completed = Vec::new();
        for (from, to) in &moves {
            if let Err(err) = move_no_replace(from, to) {
                return Err(rollback_moves(&completed, err));
            }
            completed.push((from.clone(), to.clone()));
        }
        if let Err(err) = self.commit_library(&mut st, entries, catalog, &removed, collections.as_deref()) {
            let err = rollback_moves(&completed, err);
            if let Some((from, to)) = folder_move {
                let rollback = ensure_absent(&from).and_then(|_| fs::rename(&to, &from).map_err(Into::into));
                if let Err(e) = rollback { bail!("{err}; folder rollback {} -> {} failed: {e}", to.display(), from.display()); }
            }
            if let Some(path) = created {
                if let Err(e) = fs::remove_dir(&path) { bail!("{err}; removing new folder {} failed: {e}", path.display()); }
            }
            return Err(err);
        }
        Ok(st.entries.iter().map(|e| e.info.clone()).collect())
    }

    pub(super) fn capture_directory(&self) -> Result<PathBuf> {
        let descriptor = self.root.join("session.aurorasession");
        let bytes = match fs::read(&descriptor) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let path = self.root.join("Capture");
                fs::create_dir_all(&path)?;
                return Ok(path);
            }
            Err(e) => return Err(e).context("reading session descriptor"),
        };
        let value: serde_json::Value = serde_json::from_slice(&bytes).context("invalid session descriptor")?;
        ensure!(value.get("version").and_then(|v| v.as_u64()) == Some(1) && value.get("kind").and_then(|v| v.as_str()) == Some("Session"), "unsupported session descriptor");
        let capture = value.pointer("/shortcuts/Capture").and_then(|v| v.as_str()).context("session descriptor has no Capture shortcut")?;
        ensure!(!capture.is_empty(), "Capture shortcut is empty");
        existing_directory(&self.root, Path::new(capture)).context("configured Capture folder unavailable")
    }
}

fn selected_paths(entries: &[FileEntry], ids: &[ImageId]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for id in ids {
        let e = entries.iter().find(|e| e.info.id == *id).context("unknown image identity")?;
        if !paths.contains(&e.path) { paths.push(e.path.clone()); }
    }
    Ok(paths)
}

fn relink(entries: &mut [FileEntry], from: &Path, to: &Path) {
    for e in entries.iter_mut().filter(|e| e.path == from) {
        e.path = to.to_path_buf();
        e.info.path = to.to_path_buf();
        e.info.file_name = to.file_name().unwrap_or_default().to_string_lossy().into_owned();
    }
}

fn remap_folder(entries: &mut [FileEntry], catalog: &mut Catalog, from: &Path, to: &Path, validate_targets: bool) -> Result<()> {
    let paths: HashSet<_> = entries.iter().filter(|e| e.path.starts_with(from)).map(|e| e.path.clone()).collect();
    ensure!(!paths.is_empty() || catalog.folders.iter().any(|f| f.path.starts_with(from)), "folder has no catalog records");
    let mut targets = HashSet::new();
    for path in &paths {
        let target = to.join(path.strip_prefix(from)?);
        if validate_targets { ensure!(target.is_file(), "relink target unavailable: {}", target.display()); }
        ensure!(!protected(&target), "relink target is protected");
        ensure!(targets.insert(path_key(&target)), "duplicate relink target");
        ensure!(!entries.iter().any(|e| !paths.contains(&e.path) && path_key(&e.path) == path_key(&target)), "relink target already cataloged");
        relink(entries, path, &target);
    }
    for f in &mut catalog.folders {
        if let Ok(suffix) = f.path.strip_prefix(from) { f.path = to.join(suffix); }
    }
    let mut unique = HashSet::new();
    ensure!(catalog.folders.iter().all(|f| unique.insert(path_key(&f.path))), "relink folder already cataloged");
    for p in &mut catalog.excluded {
        if let Ok(suffix) = p.strip_prefix(from) { *p = to.join(suffix); }
    }
    for v in &mut catalog.excluded_variants {
        if let Ok(suffix) = v.path.strip_prefix(from) { v.path = to.join(suffix); }
    }
    Ok(())
}

fn add_folder(catalog: &mut Catalog, root: &Path, path: &Path) {
    let mut paths = vec![path.to_path_buf()];
    if path.starts_with(root) {
        let mut parent = path;
        while parent != root {
            let Some(p) = parent.parent() else { break };
            paths.push(p.to_path_buf());
            parent = p;
        }
    }
    for path in paths.into_iter().rev() {
        if !catalog.folders.iter().any(|f| f.path == path) {
            let id = catalog.folders.iter().map(|f| f.id + 1).max().unwrap_or(FOLDER_BASE);
            catalog.folders.push(FolderRecord { id, path });
        }
    }
}

fn scan(folder: &Path, recursive: bool) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut files = Vec::new();
    let mut folders = vec![folder.to_path_buf()];
    let mut pending = vec![folder.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_symlink() || protected(&path) { continue; }
            if kind.is_dir() {
                if entry.file_name() == "lumen-output" { continue; }
                folders.push(path.clone());
                if recursive { pending.push(path); }
            } else if kind.is_file() && is_supported(&path) { files.push(path); }
        }
    }
    files.sort();
    folders.sort();
    Ok((files, folders))
}

fn resolve_path(root: &Path, path: &Path) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "empty path");
    ensure!(!path.components().any(|c| matches!(c, Component::ParentDir)), "parent traversal is not allowed");
    let path = if path.is_absolute() { plain_path(path) } else {
        ensure!(!path.components().any(|c| matches!(c, Component::RootDir | Component::Prefix(_))), "ambiguous path");
        root.join(path)
    };
    Ok(path.components().filter(|c| !matches!(c, Component::CurDir)).collect())
}

fn existing_directory(root: &Path, path: &Path) -> Result<PathBuf> {
    let path = resolve_path(root, path)?;
    ensure!(path.is_dir(), "directory unavailable: {}", path.display());
    let path = canonical_path(&path)?;
    ensure!(!protected(&path), "private catalog directories cannot be used");
    Ok(path)
}

pub(super) fn canonical_path(path: &Path) -> Result<PathBuf> {
    Ok(plain_path(&fs::canonicalize(path)?))
}

fn plain_path(path: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        use std::path::Prefix;
        let mut parts = path.components();
        if let Some(Component::Prefix(prefix)) = parts.next() {
            let base = match prefix.kind() {
                Prefix::VerbatimDisk(drive) => Some(PathBuf::from(format!("{}:\\", drive as char))),
                Prefix::VerbatimUNC(server, share) => Some(PathBuf::from(r"\\").join(server).join(share)),
                _ => None,
            };
            if let Some(mut base) = base {
                for c in parts { if !matches!(c, Component::RootDir) { base.push(c.as_os_str()); } }
                return base;
            }
        }
    }
    path.to_path_buf()
}

fn protected(path: &Path) -> bool {
    path.components().any(|c| {
        let name = c.as_os_str().to_string_lossy().to_ascii_lowercase();
        name == ".lumen" || ["cocatalogdb", "cosessiondb", "cos"].iter().any(|ext| name.ends_with(&format!(".{ext}")))
    })
}

fn path_key(path: &Path) -> String {
    let value = path.to_string_lossy().replace('\\', "/");
    if cfg!(windows) { value.to_lowercase() } else { value }
}

fn validate_filename(name: &str) -> Result<()> {
    ensure!(!name.is_empty() && name != "." && name != ".." && name.encode_utf16().count() <= 255, "invalid Windows filename");
    ensure!(!name.ends_with(['.', ' ']) && !name.chars().any(|c| c < ' ' || "<>:\"/\\|?*".contains(c)), "invalid Windows filename");
    let stem = name.split('.').next().unwrap().trim_end_matches(' ').to_uppercase();
    let device = ["CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"].contains(&stem.as_str())
        || ["COM", "LPT"].iter().any(|prefix| stem.strip_prefix(prefix).is_some_and(|n| ["1", "2", "3", "4", "5", "6", "7", "8", "9", "¹", "²", "³"].contains(&n)));
    ensure!(!device, "reserved Windows filename");
    Ok(())
}

fn ensure_absent(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
        Ok(_) => bail!("destination already exists: {}", path.display()),
    }
}

fn transfer_plan(entries: &[FileEntry], transfers: &[(PathBuf, PathBuf)]) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut moves = transfers.to_vec();
    for (from, to) in transfers {
        ensure!(from.is_file() && !protected(from), "source unavailable or protected: {}", from.display());
        validate_filename(to.file_name().and_then(|s| s.to_str()).context("invalid filename")?)?;
        existing_directory(Path::new("."), to.parent().context("missing destination parent")?)?;
        // Preserve sidecar bytes, including uppercase .XMP files.
        let sidecars: Vec<_> = fs::read_dir(from.parent().context("missing source parent")?)?
            .map(|entry| entry.map(|e| e.path())).collect::<std::io::Result<Vec<_>>>()?
            .into_iter().filter(|p| p.file_stem() == from.file_stem() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("xmp"))).collect();
        ensure!(sidecars.len() <= 1, "ambiguous XMP sidecars for {}", from.display());
        for sidecar in sidecars {
            ensure!(!entries.iter().any(|e| e.path != *from && e.path.with_extension("xmp") == from.with_extension("xmp") && !transfers.iter().any(|(p, _)| p == &e.path)), "XMP is shared by an unselected image");
            let target = to.with_extension(sidecar.extension().unwrap());
            if sidecar != target && !moves.contains(&(sidecar.clone(), target.clone())) { moves.push((sidecar, target)); }
        }
        // An orphan destination sidecar must not be attached to the moved image.
        let destination_sidecar = to.with_extension("xmp");
        for e in fs::read_dir(to.parent().unwrap())? {
            let path = e?.path();
            ensure!(!(path.file_stem() == destination_sidecar.file_stem() && path.extension().is_some_and(|e| e.eq_ignore_ascii_case("xmp")) && path != from.with_extension(path.extension().unwrap())), "destination XMP already exists: {}", path.display());
        }
        ensure!(!entries.iter().any(|e| e.path != *from && path_key(&e.path) == path_key(to)), "destination already cataloged (possibly offline)");
    }
    let mut targets = HashSet::new();
    let mut sources = HashSet::new();
    for (from, to) in &moves {
        ensure!(sources.insert(path_key(from)), "one source cannot move to multiple destinations");
        ensure!(targets.insert(path_key(to)), "duplicate destination: {}", to.display());
        ensure_absent(to)?;
    }
    Ok(moves)
}

/// MoveFileEx without REPLACE_EXISTING provides no-clobber moves across volumes.
#[cfg(windows)]
fn move_no_replace(from: &Path, to: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" { fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32; }
    let from: Vec<_> = from.as_os_str().encode_wide().chain(Some(0)).collect();
    let to: Vec<_> = to.as_os_str().encode_wide().chain(Some(0)).collect();
    if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 2) } == 0 { return Err(std::io::Error::last_os_error().into()); }
    Ok(())
}

#[cfg(not(windows))]
fn move_no_replace(from: &Path, to: &Path) -> Result<()> {
    use std::io::Write;
    if fs::hard_link(from, to).is_err() {
        let mut src = fs::File::open(from)?;
        let mut dst = fs::OpenOptions::new().write(true).create_new(true).open(to)?;
        let copied = (|| -> Result<()> {
            std::io::copy(&mut src, &mut dst)?;
            dst.flush()?;
            dst.sync_all()?;
            fs::set_permissions(to, src.metadata()?.permissions())?;
            Ok(())
        })();
        if let Err(err) = copied { fs::remove_file(to)?; return Err(err); }
    }
    if let Err(err) = fs::remove_file(from) { fs::remove_file(to)?; return Err(err.into()); }
    Ok(())
}

fn rollback_moves(moves: &[(PathBuf, PathBuf)], err: anyhow::Error) -> anyhow::Error {
    let mut failures = Vec::new();
    for (from, to) in moves.iter().rev() {
        if let Err(e) = move_no_replace(to, from) { failures.push(format!("{} -> {}: {e}", to.display(), from.display())); }
    }
    if failures.is_empty() { err } else { anyhow::anyhow!("{err}; file rollback failed: {}", failures.join("; ")) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
            let workspace = manifest.ancestors().find(|p| p.join("90_work/build-cache").is_dir())
                .unwrap_or_else(|| manifest.ancestors().nth(2).unwrap());
            let base = workspace.join("90_work/build-cache/library-fs/fixtures");
            fs::create_dir_all(&base).unwrap();
            let path = base.join(format!("{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::create_dir(&path).unwrap();
            Self(canonical_path(&path).unwrap())
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
        fn open(&self) -> FileEngine { FileEngine::open(&self.0).unwrap() }
    }
    impl Drop for Fixture {
        fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); }
    }
    fn edits(exposure: f32) -> AdjustmentSet {
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = exposure;
        adj
    }
    fn folder_id(engine: &FileEngine, path: &Path) -> u64 {
        engine.state.lock().catalog.folders.iter().find(|f| f.path == path).unwrap().id
    }
    fn album(ids: Vec<ImageId>) -> UserCollection {
        UserCollection { id: 1 << 40, parent: None, name: "Picks".into(), kind: CollectionKind::Album, images: ids, rules: vec![], match_all: true }
    }

    #[test]
    fn move_rename_xmp_variants_edits_membership_and_reopen() {
        let f = Fixture::new();
        let src = f.image("shoot/a.png");
        let bytes = fs::read(&src).unwrap();
        let xmp = b"<sidecar>preserve these exact bytes</sidecar>";
        fs::write(src.with_extension("XMP"), xmp).unwrap();
        let destination = f.folder("selects");
        let engine = f.open();
        let first = engine.images()[0].id;
        engine.save_adjustments(first, &edits(0.75)).unwrap();
        let second = engine.new_variant(first, &edits(-1.25)).unwrap().id;
        let third = engine.new_variant(first, &edits(1.5)).unwrap().id;
        let mut info = engine.images().into_iter().find(|i| i.id == second).unwrap();
        info.rating = 5;
        info.keywords = vec!["hero".into()];
        engine.update_info(&info).unwrap();
        let albums = vec![album(vec![first, second, third])];
        engine.save_user_collections(&albums).unwrap();
        let dest_id = folder_id(&engine, &destination);
        let moved = engine.manage_files(LibraryAction::Move { images: vec![second], destination: destination.clone() }).unwrap();
        assert_eq!(moved.len(), 3);
        assert!(moved.iter().all(|i| i.path == destination.join("a.png")));
        assert!(!src.exists());
        assert_eq!(fs::read(destination.join("a.XMP")).unwrap(), xmp);
        let renamed = engine.manage_files(LibraryAction::Rename { image: first, name: "hero.png".into() }).unwrap();
        assert!(renamed.iter().all(|i| i.path == destination.join("hero.png")));
        assert_eq!(fs::read(destination.join("hero.png")).unwrap(), bytes);
        assert_eq!(fs::read(destination.join("hero.XMP")).unwrap(), xmp);
        let members: HashSet<_> = engine.collection_images(dest_id).unwrap().into_iter().collect();
        assert_eq!(members, HashSet::from([first, second, third]));
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.images(), renamed);
        assert_eq!(engine.load_adjustments(first), edits(0.75));
        assert_eq!(engine.load_adjustments(second), edits(-1.25));
        assert_eq!(engine.load_adjustments(third), edits(1.5));
        assert_eq!(engine.user_collections(), albums);
        assert_eq!(engine.collection_path(dest_id), Some(destination));
    }

    #[test]
    fn collision_preflights_entire_batch_and_sidecars_without_data_loss() {
        let f = Fixture::new();
        let a = f.image("source/a.png");
        let b = f.image("source/b.png");
        let collision = f.image("dest/b.png");
        fs::write(collision, b"existing destination").unwrap();
        fs::write(a.with_extension("xmp"), b"original XMP").unwrap();
        let engine = f.open();
        let before = engine.images();
        let ids = before.iter().filter(|i| i.path.parent() == a.parent()).map(|i| i.id).collect();
        let json = fs::read(engine.store.as_ref().unwrap().root().join("catalog.json")).unwrap();
        assert!(engine.manage_files(LibraryAction::Move { images: ids, destination: f.0.join("dest") }).is_err());
        assert!(a.exists() && b.exists());
        assert_eq!(fs::read(f.0.join("dest/b.png")).unwrap(), b"existing destination");
        assert!(!f.0.join("dest/a.png").exists());
        assert_eq!(engine.images(), before);
        assert_eq!(fs::read(engine.store.as_ref().unwrap().root().join("catalog.json")).unwrap(), json);
        fs::write(f.0.join("dest/a.XMP"), b"orphan XMP").unwrap();
        let id = before.iter().find(|i| i.path == a).unwrap().id;
        assert!(engine.manage_files(LibraryAction::Move { images: vec![id], destination: f.0.join("dest") }).is_err());
        assert_eq!(fs::read(a.with_extension("xmp")).unwrap(), b"original XMP");
        assert_eq!(fs::read(f.0.join("dest/a.XMP")).unwrap(), b"orphan XMP");
    }

    #[test]
    fn persistence_failure_rolls_back_image_and_xmp_moves() {
        let f = Fixture::new();
        let src = f.image("a.png");
        fs::write(src.with_extension("xmp"), b"XMP").unwrap();
        let destination = f.folder("dest");
        let engine = f.open();
        let before = engine.images();
        let store = engine.store.as_ref().unwrap();
        let json = fs::read(store.root().join("catalog.json")).unwrap();
        let obstruction = store.root().join("catalog.json.tmp");
        fs::create_dir(&obstruction).unwrap();
        assert!(engine.manage_files(LibraryAction::Move { images: vec![before[0].id], destination: destination.clone() }).is_err());
        assert!(src.is_file());
        assert_eq!(fs::read(src.with_extension("xmp")).unwrap(), b"XMP");
        assert!(!destination.join("a.png").exists() && !destination.join("a.xmp").exists());
        assert_eq!(engine.images(), before);
        assert_eq!(fs::read(store.root().join("catalog.json")).unwrap(), json);
        fs::remove_dir(obstruction).unwrap();
        drop(engine);
        assert_eq!(f.open().images(), before);
    }

    #[test]
    fn locate_validates_every_offline_target_and_preserves_identity_and_edits() {
        let f = Fixture::new();
        let a = f.image("offline/a.png");
        let b = f.image("offline/nested/b.png");
        let external = Fixture::new();
        let engine = f.open();
        let before = engine.images();
        let first = before.iter().find(|i| i.path == a).unwrap().id;
        let variant = engine.new_variant(first, &edits(1.0)).unwrap().id;
        let old_folder_id = folder_id(&engine, &f.0.join("offline"));
        fs::rename(f.0.join("offline"), external.0.join("online")).unwrap();
        fs::rename(external.0.join("online/nested/b.png"), external.0.join("missing.png")).unwrap();
        assert!(engine.manage_files(LibraryAction::Locate { from: f.0.join("offline"), to: external.0.join("online") }).is_err());
        assert!(engine.images().iter().any(|i| i.path == b));
        fs::rename(external.0.join("missing.png"), external.0.join("online/nested/b.png")).unwrap();
        let bytes = fs::read(external.0.join("online/a.png")).unwrap();
        let infos = engine.manage_files(LibraryAction::Locate { from: f.0.join("offline"), to: external.0.join("online") }).unwrap();
        assert_eq!(infos.len(), 3);
        assert!(infos.iter().all(|i| i.path.starts_with(external.0.join("online"))));
        assert_eq!(engine.collection_path(old_folder_id), Some(external.0.join("online")));
        assert_eq!(fs::read(external.0.join("online/a.png")).unwrap(), bytes);
        assert!(!f.0.join("offline").exists());
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.images(), infos);
        assert_eq!(engine.load_adjustments(variant), edits(1.0));
    }

    #[test]
    fn synchronize_preserves_ids_offline_variants_and_empty_folder_tree() {
        let f = Fixture::new();
        let original = f.image("z.png");
        let empty = f.folder("empty/deep");
        let engine = f.open();
        let id = engine.images()[0].id;
        engine.save_adjustments(id, &edits(0.5)).unwrap();
        let variant = engine.new_variant(id, &edits(-0.5)).unwrap().id;
        let empty_id = folder_id(&engine, &empty);
        let root_id = folder_id(&engine, &f.0);
        assert_eq!(engine.library_root(), engine.collection_path(root_id));
        assert!(engine.collection_images(empty_id).unwrap().is_empty());
        fs::rename(original, f.0.join("offline.bin")).unwrap();
        f.image("a.png");
        f.image("new/child/b.png");
        let partial = engine.manage_files(LibraryAction::Synchronize { folder: f.0.clone(), recursive: false }).unwrap();
        assert_eq!(partial.len(), 3);
        let all = engine.manage_files(LibraryAction::Synchronize { folder: f.0.clone(), recursive: true }).unwrap();
        assert_eq!(all.len(), 4);
        assert!(all.iter().any(|i| i.id == id && i.file_name == "z.png"));
        assert!(all.iter().any(|i| i.id == variant));
        assert_eq!(folder_id(&engine, &empty), empty_id);
        engine.rescan().unwrap();
        assert_eq!(engine.images(), all);
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.images(), all);
        assert_eq!(engine.load_adjustments(id), edits(0.5));
        assert_eq!(engine.load_adjustments(variant), edits(-0.5));
        assert_eq!(engine.collection_path(empty_id), Some(empty));
        let root = engine.collections().into_iter().find(|c| c.id == 10).unwrap().children.into_iter().find(|c| c.id == root_id).unwrap();
        assert!(root.children.iter().any(|c| c.name == "empty" && c.children.iter().any(|c| c.id == empty_id)));
    }

    #[test]
    fn external_import_reference_survives_reopen_even_offline() {
        let library = Fixture::new();
        let source = Fixture::new();
        let path = source.image("shoot/external.png");
        let engine = library.open();
        let imported = engine.import(&[path.clone()]).unwrap();
        let id = imported[0].id;
        engine.save_adjustments(id, &edits(0.8)).unwrap();
        assert_eq!(engine.import(&[path.clone()]).unwrap()[0].id, id);
        fs::rename(&path, path.with_extension("offline")).unwrap();
        drop(engine);
        let engine = library.open();
        assert_eq!(engine.images()[0].id, id);
        assert_eq!(engine.images()[0].path, path);
        assert_eq!(engine.load_adjustments(id), edits(0.8));
    }

    #[test]
    fn forget_removes_only_selected_variant_then_last_source_stays_excluded_on_reopen() {
        let f = Fixture::new();
        let path = f.image("a.png");
        fs::write(path.with_extension("xmp"), b"keep originals").unwrap();
        let engine = f.open();
        let first = engine.images()[0].id;
        engine.save_adjustments(first, &edits(1.0)).unwrap();
        let second = engine.new_variant(first, &edits(2.0)).unwrap().id;
        engine.save_user_collections(&[album(vec![first, second])]).unwrap();
        let key = engine.state.lock().entries.iter().find(|e| e.info.id == first).unwrap().rel_path.clone();
        let infos = engine.manage_files(LibraryAction::Forget { images: vec![first] }).unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].id, second);
        assert!(engine.store.as_ref().unwrap().load_adjustments(&key).unwrap().is_default());
        assert_eq!(engine.store.as_ref().unwrap().load_user_collections()[0].images, vec![second]);
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.images(), infos);
        assert_eq!(engine.load_adjustments(second), edits(2.0));
        assert!(engine.manage_files(LibraryAction::Forget { images: vec![second] }).unwrap().is_empty());
        assert!(engine.store.as_ref().unwrap().load_index().unwrap().images.is_empty());
        assert_eq!(fs::read(path.with_extension("xmp")).unwrap(), b"keep originals");
        drop(engine);
        let engine = f.open();
        assert!(engine.images().is_empty());
        engine.rescan().unwrap();
        assert!(engine.images().is_empty());
        let fresh = engine.import(&[path]).unwrap()[0].id;
        assert!(fresh.0 > second.0);
        assert!(engine.load_adjustments(fresh).is_default());
    }

    #[test]
    fn forget_failure_restores_adjustments_albums_and_catalog() {
        let f = Fixture::new();
        f.image("a.png");
        let engine = f.open();
        let id = engine.images()[0].id;
        engine.save_adjustments(id, &edits(1.0)).unwrap();
        let albums = vec![album(vec![id])];
        engine.save_user_collections(&albums).unwrap();
        let obstruction = engine.store.as_ref().unwrap().root().join("catalog.json.tmp");
        fs::create_dir(&obstruction).unwrap();
        assert!(engine.manage_files(LibraryAction::Forget { images: vec![id] }).is_err());
        fs::remove_dir(obstruction).unwrap();
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.images()[0].id, id);
        assert_eq!(engine.load_adjustments(id), edits(1.0));
        assert_eq!(engine.store.as_ref().unwrap().load_user_collections(), albums);
    }

    #[test]
    fn folder_move_preserves_whole_directory_ids_variants_and_reopen() {
        let f = Fixture::new();
        f.image("shoot/nested/a.png");
        fs::write(f.0.join("shoot/unreferenced.txt"), b"keep me too").unwrap();
        let engine = f.open();
        let id = engine.images()[0].id;
        let variant = engine.new_variant(id, &edits(1.0)).unwrap().id;
        let folder = folder_id(&engine, &f.0.join("shoot"));
        let nested = folder_id(&engine, &f.0.join("shoot/nested"));
        let infos = engine.manage_files(LibraryAction::MoveFolder { from: f.0.join("shoot"), to: f.0.join("renamed") }).unwrap();
        assert!(infos.iter().all(|i| i.path == f.0.join("renamed/nested/a.png")));
        assert_eq!(engine.collection_path(folder), Some(f.0.join("renamed")));
        assert_eq!(engine.collection_path(nested), Some(f.0.join("renamed/nested")));
        assert_eq!(fs::read(f.0.join("renamed/unreferenced.txt")).unwrap(), b"keep me too");
        assert!(!f.0.join("shoot").exists());
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.images(), infos);
        assert_eq!(engine.load_adjustments(variant), edits(1.0));
        assert_eq!(engine.collection_path(folder), Some(f.0.join("renamed")));
    }

    #[test]
    fn folder_move_collisions_root_descendants_and_persistence_failure_are_safe() {
        let f = Fixture::new();
        let path = f.image("shoot/a.png");
        f.folder("exists");
        let engine = f.open();
        let before = engine.images();
        for (from, to) in [
            (f.0.join("shoot"), f.0.join("exists")),
            (f.0.clone(), f.0.join("elsewhere")),
            (f.0.join("shoot"), f.0.join("shoot/child")),
        ] {
            assert!(engine.manage_files(LibraryAction::MoveFolder { from, to }).is_err());
            assert!(path.exists());
        }
        let obstruction = engine.store.as_ref().unwrap().root().join("catalog.json.tmp");
        fs::create_dir(&obstruction).unwrap();
        assert!(engine.manage_files(LibraryAction::MoveFolder { from: f.0.join("shoot"), to: f.0.join("new") }).is_err());
        assert!(path.exists() && !f.0.join("new").exists());
        assert_eq!(engine.images(), before);
        fs::remove_dir(obstruction).unwrap();
        drop(engine);
        assert_eq!(f.open().images(), before);
    }

    #[test]
    fn create_folder_validates_names_containment_and_rollback() {
        let f = Fixture::new();
        let external = Fixture::new();
        let engine = f.open();
        for name in ["CON", "nul.png", "LPT9", "a.", "a ", "a:b", "../escape", ".lumen"] {
            assert!(engine.manage_files(LibraryAction::CreateFolder { path: PathBuf::from(name) }).is_err(), "{name}");
        }
        assert!(engine.manage_files(LibraryAction::CreateFolder { path: external.0.join("escape") }).is_err());
        engine.manage_files(LibraryAction::CreateFolder { path: "new".into() }).unwrap();
        let id = folder_id(&engine, &f.0.join("new"));
        assert!(engine.manage_files(LibraryAction::CreateFolder { path: "new".into() }).is_err());
        assert!(engine.manage_files(LibraryAction::Rename { image: ImageId(999), name: "bad?.png".into() }).is_err());
        let obstruction = engine.store.as_ref().unwrap().root().join("catalog.json.tmp");
        fs::create_dir(&obstruction).unwrap();
        assert!(engine.manage_files(LibraryAction::CreateFolder { path: "failed".into() }).is_err());
        assert!(!f.0.join("failed").exists());
        fs::remove_dir(obstruction).unwrap();
        drop(engine);
        assert_eq!(f.open().collection_path(id), Some(f.0.join("new")));
    }

    #[test]
    fn capture_shortcut_is_used_and_invalid_configuration_never_falls_back() {
        let f = Fixture::new();
        let external = Fixture::new();
        let configured = f.folder("CustomCapture");
        let engine = f.open();
        let descriptor = f.0.join("session.aurorasession");
        let write = |path: &str| fs::write(&descriptor, serde_json::to_vec(&serde_json::json!({"version":1,"kind":"Session","shortcuts":{"Capture":path}})).unwrap()).unwrap();
        write("CustomCapture");
        assert_eq!(engine.capture_directory().unwrap(), configured);
        write(external.0.to_str().unwrap());
        assert_eq!(engine.capture_directory().unwrap(), external.0);
        write("missing");
        assert!(engine.capture_directory().is_err());
        assert!(engine.capture().is_err());
        assert!(!f.0.join("Capture").exists());
        fs::write(&descriptor, b"not json").unwrap();
        assert!(engine.capture_directory().is_err());
        assert!(!f.0.join("Capture").exists());
        fs::remove_file(descriptor).unwrap();
        assert_eq!(engine.capture_directory().unwrap(), f.0.join("Capture"));
    }

    #[test]
    fn default_library_api_reports_unsupported() {
        let engine = crate::MockEngine::default();
        assert_eq!(engine.library_root(), None);
        assert_eq!(engine.collection_path(1000), None);
        assert!(engine.manage_files(LibraryAction::Forget { images: vec![] }).unwrap_err().to_string().contains("unsupported"));
    }

    #[test]
    fn batch_rename_deduplicates_variants_preflights_collisions_and_reopens() {
        let f = Fixture::new();
        let a = f.image("a.png");
        let b = f.image("b.png");
        fs::write(a.with_extension("xmp"), b"a-XMP").unwrap();
        let engine = f.open();
        let infos = engine.images();
        let first = infos.iter().find(|i| i.path == a).unwrap().id;
        let second = infos.iter().find(|i| i.path == b).unwrap().id;
        let variant = engine.new_variant(first, &edits(1.0)).unwrap().id;
        for names in [
            vec![(first, "same.png".into()), (second, "same.png".into())],
            vec![(first, "b.png".into()), (second, "a.png".into())],
            vec![(first, "good.png".into()), (second, "NUL.png".into())],
            vec![(first, "one.png".into()), (variant, "two.png".into())],
        ] {
            assert!(engine.manage_files(LibraryAction::RenameMany { names }).is_err());
            assert!(a.exists() && b.exists());
            assert!(!f.0.join("good.png").exists());
        }
        let names = vec![(first, "frame-01.png".into()), (variant, "frame-01.png".into()), (second, "frame-02.png".into())];
        let obstruction = engine.store.as_ref().unwrap().root().join("catalog.json.tmp");
        fs::create_dir(&obstruction).unwrap();
        assert!(engine.manage_files(LibraryAction::RenameMany { names: names.clone() }).is_err());
        assert!(a.exists() && b.exists() && a.with_extension("xmp").exists());
        fs::remove_dir(obstruction).unwrap();
        let renamed = engine.manage_files(LibraryAction::RenameMany { names }).unwrap();
        assert!(!a.exists() && !b.exists());
        assert_eq!(fs::read(f.0.join("frame-01.xmp")).unwrap(), b"a-XMP");
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.images(), renamed);
        assert_eq!(engine.load_adjustments(variant), edits(1.0));
    }

    #[cfg(windows)]
    #[test]
    fn locked_sidecar_failure_rolls_back_already_moved_image() {
        use std::os::windows::fs::OpenOptionsExt;
        let f = Fixture::new();
        let a = f.image("a.png");
        fs::write(a.with_extension("xmp"), b"locked XMP").unwrap();
        let destination = f.folder("dest");
        let engine = f.open();
        let before = engine.images();
        let lock = fs::OpenOptions::new().read(true).share_mode(0).open(a.with_extension("xmp")).unwrap();
        assert!(engine.manage_files(LibraryAction::Move { images: vec![before[0].id], destination: destination.clone() }).is_err());
        drop(lock);
        assert!(a.exists() && a.with_extension("xmp").exists());
        assert!(!destination.join("a.png").exists());
        assert_eq!(engine.images(), before);
    }

    #[test]
    fn schema_one_metadata_and_edits_migrate_to_stable_schema_two() {
        let f = Fixture::new();
        f.image("a.png");
        let store = Store::open(&f.0).unwrap();
        fs::write(store.root().join("catalog.json"), br#"{"schema_version":1,"images":[{"rel_path":"a.png","rating":4,"color_tag":2,"keywords":["legacy"],"variant":1,"updated_at":1,"trashed":false}]}"#).unwrap();
        let original_index = fs::read(store.root().join("catalog.json")).unwrap();
        store.save_adjustments("a.png", &edits(1.0)).unwrap();
        let engine = f.open();
        let infos = engine.images();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].rating, 4);
        assert_eq!(infos[0].keywords, vec!["legacy"]);
        assert_eq!(engine.load_adjustments(infos[0].id), edits(1.0));
        let json: serde_json::Value = serde_json::from_slice(&fs::read(store.root().join("catalog.json")).unwrap()).unwrap();
        assert_eq!(json["schema_version"], 2);
        assert_eq!(fs::read(store.root().join("catalog.schema1.backup.json")).unwrap(), original_index);
        assert!(json["schema_version"].as_u64().unwrap() > 1, "legacy Store::guard_newer_index must refuse writes");
        drop(engine);
        assert_eq!(f.open().images(), infos);
        assert_eq!(fs::read(store.root().join("catalog.schema1.backup.json")).unwrap(), original_index);
    }

    #[test]
    fn offline_thumbnail_and_preview_survive_restart_without_upscale_or_cache_rewrite() {
        let f = Fixture::new();
        let path = f.image("a.png");
        let engine = f.open();
        let id = engine.images()[0].id;
        let online = engine.thumbnail(id, &AdjustmentSet::default(), 4).unwrap();
        let cache = engine.store.as_ref().unwrap().root().join(format!("previews/{}.png", id.0));
        let cached_bytes = fs::read(&cache).unwrap();
        let cached_modified = fs::metadata(&cache).unwrap().modified().unwrap();
        fs::rename(&path, path.with_extension("offline")).unwrap();
        drop(engine);
        let engine = f.open();
        let offline = engine.thumbnail(id, &edits(2.0), 1000).unwrap();
        assert_eq!((offline.width, offline.height), (online.width, online.height));
        assert_eq!(offline.pixels.as_ref(), online.pixels.as_ref());
        let request = PreviewRequest { image: id, adjustments: edits(2.0), region: NormRect::FULL, max_size: 2,
            quality: RenderQuality::Final, bypass: false, mask_overlay: None, generation: 17 , display_profile: None, proof_profile: None, proof_paper_white: 1.0, proof_black_point: 0.0, };
        let preview = engine.render_preview(&request).unwrap();
        assert!(preview.image.width <= 2 && preview.image.height <= 2);
        assert_eq!(preview.generation, 17);
        assert_eq!(preview.histogram.red.iter().sum::<u32>(), preview.image.width * preview.image.height);
        assert_eq!(fs::read(&cache).unwrap(), cached_bytes);
        assert_eq!(fs::metadata(&cache).unwrap().modified().unwrap(), cached_modified);
        engine.manage_files(LibraryAction::Forget { images: vec![id] }).unwrap();
        assert!(!cache.exists());
        assert!(engine.thumbnail(id, &AdjustmentSet::default(), 4).is_err());
    }

    #[test]
    fn corrupt_or_unreadable_catalog_blocks_move_forget_and_open_without_data_loss() {
        let f = Fixture::new();
        let source = f.image("a.png");
        let destination = f.folder("dest");
        let engine = f.open();
        let before = engine.images();
        let index = engine.store.as_ref().unwrap().root().join("catalog.json");
        let original = fs::read(&index).unwrap();
        for corrupt in [b"{broken".as_slice(), b"[]".as_slice(), br#"{"images":[{"no_path":true}]}"#.as_slice()] {
            fs::write(&index, corrupt).unwrap();
            assert!(engine.manage_files(LibraryAction::Move { images: vec![before[0].id], destination: destination.clone() }).is_err());
            assert!(engine.manage_files(LibraryAction::Forget { images: vec![before[0].id] }).is_err());
            assert!(source.exists() && !destination.join("a.png").exists());
            assert_eq!(engine.images(), before);
            assert_eq!(fs::read(&index).unwrap(), corrupt);
            assert!(FileEngine::open(&f.0).is_err());
        }
        fs::remove_file(&index).unwrap();
        fs::create_dir(&index).unwrap();
        assert!(engine.manage_files(LibraryAction::Move { images: vec![before[0].id], destination }).is_err());
        assert!(engine.manage_files(LibraryAction::Forget { images: vec![before[0].id] }).is_err());
        assert!(source.exists());
        fs::remove_dir(&index).unwrap();
        fs::write(index, original).unwrap();
    }

    #[test]
    fn user_collection_api_round_trips_parents_rules_members_and_refuses_corrupt_overwrite() {
        let f = Fixture::new();
        f.image("a.png");
        let engine = f.open();
        let id = engine.images()[0].id;
        let mut picks = album(vec![id]);
        picks.parent = Some(123);
        picks.kind = CollectionKind::SmartAlbum;
        picks.rules = vec![crate::SmartRule::RatingAtLeast(4), crate::SmartRule::Edited(true)];
        engine.save_user_collections(&[picks.clone()]).unwrap();
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.user_collections(), vec![picks]);
        let path = engine.store.as_ref().unwrap().root().join("collections.json");
        fs::write(&path, b"{damaged").unwrap();
        assert!(engine.save_user_collections(&[]).is_err());
        assert!(engine.manage_files(LibraryAction::Forget { images: vec![id] }).is_err());
        assert_eq!(fs::read(path).unwrap(), b"{damaged");
        assert_eq!(engine.images()[0].id, id);
    }

    #[test]
    fn locate_single_image_while_parent_exists_preserves_all_variants_and_reopens() {
        let f = Fixture::new();
        let external = Fixture::new();
        let source = f.image("shoot/a.png");
        let bound = f.image("shoot/b.png");
        let engine = f.open();
        let id = engine.images().iter().find(|i| i.path == source).unwrap().id;
        engine.save_adjustments(id, &edits(0.5)).unwrap();
        let variant = engine.new_variant(id, &edits(1.0)).unwrap().id;
        let to = external.0.join("renamed.png");
        fs::rename(&source, &to).unwrap();
        assert!(source.parent().unwrap().is_dir());
        let before = engine.images();
        assert!(engine.manage_files(LibraryAction::LocateImage { image: id, to: bound }).is_err());
        assert_eq!(engine.images(), before);
        let bytes = fs::read(&to).unwrap();
        let infos = engine.manage_files(LibraryAction::LocateImage { image: variant, to: to.clone() }).unwrap();
        assert!(infos.iter().filter(|i| [id, variant].contains(&i.id)).all(|i| i.path == to));
        assert_eq!(fs::read(&to).unwrap(), bytes);
        assert!(!source.exists());
        drop(engine);
        let engine = f.open();
        assert_eq!(engine.images(), infos);
        assert_eq!(engine.load_adjustments(id), edits(0.5));
        assert_eq!(engine.load_adjustments(variant), edits(1.0));
        assert!(engine.manage_files(LibraryAction::LocateImage { image: id, to }).is_err());
    }

    #[test]
    #[ignore = "manual synthetic release scaling measurement"]
    fn synchronize_scaling_1k_5k() {
        let mut timings = Vec::new();
        for count in [1000, 5000] {
            let f = Fixture::new();
            let template = f.image("template.png");
            let png = fs::read(&template).unwrap();
            fs::remove_file(template).unwrap();
            for i in 0..count { fs::write(f.0.join(format!("{i:05}.png")), &png).unwrap(); }
            let start = Instant::now();
            let engine = f.open();
            let cold = start.elapsed().as_secs_f64() * 1000.0;
            let before = engine.images();
            assert_eq!(before.len(), count);
            let start = Instant::now();
            engine.rescan().unwrap();
            let warm = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(engine.images(), before);
            eprintln!("synthetic sync {count}: cold={cold:.2}ms warm={warm:.2}ms");
            timings.push((cold, warm));
        }
        eprintln!("5k/1k ratios: cold={:.2} warm={:.2}", timings[1].0 / timings[0].0, timings[1].1 / timings[0].1);
    }
}
