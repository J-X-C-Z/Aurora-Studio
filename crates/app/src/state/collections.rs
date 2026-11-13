//! Collections as the library shows them: the engine's tree flattened, plus
//! albums, smart albums, projects and groups the user creates, the trash, and
//! the session's recent imports and captures.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use gpui_kit::Context;
use lumen_engine::*;

use super::{AppState, StateEvent, ViewZoom};
use crate::i18n::t;

#[derive(Clone, Debug)]
pub struct CollNode {
    pub id: u64,
    pub parent: Option<u64>,
    pub name: String,
    pub kind: CollectionKind,
    /// Created in the interface, so it can be renamed, deleted and filled.
    pub user: bool,
    /// Full path for folders.
    pub path: Option<PathBuf>,
    /// Explicit members; `None` derives membership from the kind.
    pub members: Option<BTreeSet<ImageId>>,
    pub rules: Vec<SmartRule>,
    pub match_all: bool,
}

/// The first id handed to user collections, above anything an engine uses.
const USER_BASE: u64 = 1 << 40;

fn fallback_folder_path(name: &str, parent: Option<&Path>) -> Option<PathBuf> {
    let path = Path::new(name);
    if path.is_absolute() {
        Some(path.to_path_buf())
    } else {
        parent.map(|p| p.join(path))
    }
}

fn folder_contains(folder: &Path, image: &Path, recursive: bool) -> bool {
    !folder.as_os_str().is_empty() && image.parent().is_some_and(|p| if recursive { p.starts_with(folder) } else { p == folder })
}

fn descendants(nodes: &[CollNode], id: u64) -> BTreeSet<u64> {
    let mut found = BTreeSet::new();
    let mut pending = vec![id];
    while let Some(id) = pending.pop() {
        if found.insert(id) {
            pending.extend(nodes.iter().filter(|n| n.parent == Some(id)).map(|n| n.id));
        }
    }
    found
}

/// Only manual albums contribute to a project's search universe.
fn manual_album_members(nodes: &[CollNode], project: u64) -> BTreeSet<ImageId> {
    let mut members = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut pending = vec![project];
    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }
        for child in nodes.iter().filter(|n| n.parent == Some(id)) {
            match child.kind {
                CollectionKind::Album => members.extend(child.members.iter().flatten().copied()),
                CollectionKind::Group => pending.push(child.id),
                _ => {}
            }
        }
    }
    members
}

fn project_scope(nodes: &[CollNode], id: u64) -> Option<BTreeSet<ImageId>> {
    let mut parent = nodes.iter().find(|n| n.id == id)?.parent;
    let mut seen = BTreeSet::new();
    while let Some(id) = parent {
        if !seen.insert(id) {
            return Some(BTreeSet::new());
        }
        let node = nodes.iter().find(|n| n.id == id)?;
        if node.kind == CollectionKind::Project {
            return Some(manual_album_members(nodes, id));
        }
        parent = node.parent;
    }
    None
}

fn valid_parent(nodes: &[CollNode], id: Option<u64>, kind: CollectionKind, mut parent: Option<u64>) -> bool {
    if !matches!(kind, CollectionKind::Album | CollectionKind::SmartAlbum | CollectionKind::Project | CollectionKind::Group) {
        return false;
    }
    let subtree = id.map(|id| descendants(nodes, id)).unwrap_or_default();
    let contains_project = kind == CollectionKind::Project || nodes.iter().any(|n| subtree.contains(&n.id) && n.kind == CollectionKind::Project);
    let mut seen = BTreeSet::new();
    while let Some(id) = parent {
        if subtree.contains(&id) || !seen.insert(id) {
            return false;
        }
        let Some(node) = nodes.iter().find(|n| n.id == id) else { return false };
        if !matches!(node.kind, CollectionKind::Group | CollectionKind::Project) || (contains_project && node.kind == CollectionKind::Project) {
            return false;
        }
        parent = node.parent;
    }
    true
}

pub(crate) fn flatten(engine: &dyn ImageEngine, tree: &[Collection]) -> Vec<CollNode> {
    fn walk(engine: &dyn ImageEngine, nodes: &mut Vec<CollNode>, c: &Collection, parent: Option<u64>, parent_path: Option<&Path>) {
        let path = (c.kind == CollectionKind::Folder)
            .then(|| engine.collection_path(c.id).or_else(|| fallback_folder_path(&c.name, parent_path)))
            .flatten();
        nodes.push(CollNode {
            id: c.id,
            parent,
            name: c.name.clone(),
            kind: c.kind,
            user: false,
            path: path.clone(),
            members: engine.collection_images(c.id).map(|v| v.into_iter().collect()),
            rules: Vec::new(),
            match_all: true,
        });
        for child in &c.children {
            walk(engine, nodes, child, Some(c.id), path.as_deref());
        }
    }
    let mut nodes = Vec::new();
    for c in tree {
        walk(engine, &mut nodes, c, None, None);
    }
    for user in engine.user_collections() {
        nodes.retain(|n| n.id != user.id);
        nodes.push(CollNode { id: user.id, parent: user.parent, name: user.name, kind: user.kind, user: true, path: None,
            members: (user.kind == CollectionKind::Album).then(|| user.images.into_iter().collect()), rules: user.rules, match_all: user.match_all });
    }
    nodes
}

pub fn rule_matches(rule: &SmartRule, info: &ImageInfo, edited: bool) -> bool {
    let has = |hay: &Option<String>, needle: &str| hay.as_deref().is_some_and(|h| h.to_lowercase().contains(&needle.to_lowercase()));
    match rule {
        SmartRule::RatingAtLeast(r) => info.rating >= *r,
        SmartRule::ColorTag(c) => info.color_tag == *c,
        SmartRule::KeywordContains(k) => info.keywords.iter().any(|w| w.to_lowercase().contains(&k.to_lowercase())),
        SmartRule::NameContains(n) => info.file_name.to_lowercase().contains(&n.to_lowercase()),
        SmartRule::CameraContains(c) => has(&info.exif.camera_model, c) || has(&info.exif.camera_make, c),
        SmartRule::LensContains(l) => has(&info.exif.lens, l),
        SmartRule::IsoAtLeast(i) => info.exif.iso.is_some_and(|v| v >= *i),
        SmartRule::Edited(e) => edited == *e,
        SmartRule::Kind(k) => info.kind == *k,
    }
}

impl AppState {
    pub fn node(&self, id: u64) -> Option<&CollNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn child_nodes(&self, parent: Option<u64>) -> Vec<&CollNode> {
        self.nodes.iter().filter(|n| n.parent == parent).collect()
    }

    fn edited(&self, id: ImageId) -> bool {
        self.edits.get(&id).is_some_and(|e| !e.adj.is_default())
    }

    /// Images in a collection; `None` means every image outside the trash.
    pub fn members(&self, id: u64) -> Option<BTreeSet<ImageId>> {
        let node = self.node(id)?;
        if let Some(m) = &node.members
            && !matches!(node.kind, CollectionKind::Folder | CollectionKind::Project | CollectionKind::Group)
        {
            if node.kind == CollectionKind::SmartAlbum
                && let Some(scope) = project_scope(&self.nodes, id)
            {
                return Some(m.intersection(&scope).copied().collect());
            }
            return Some(m.clone());
        }
        let all = || self.images.iter().map(|i| i.id);
        match node.kind {
            CollectionKind::AllImages => None,
            CollectionKind::RecentImports => (!self.recent_imports.is_empty()).then(|| self.recent_imports.clone()),
            CollectionKind::RecentCaptures => Some(self.recent_captures.clone()),
            CollectionKind::Trash => Some(self.trash.clone()),
            CollectionKind::Album => Some(BTreeSet::new()),
            CollectionKind::Folder => {
                Some(if let Some(path) = &node.path {
                    self.images
                        .iter()
                        .filter(|i| folder_contains(path, &i.path, self.library_include_subfolders))
                        .map(|i| i.id)
                        .collect()
                } else {
                    // A display-only folder root has no filesystem prefix.
                    let mut out = node.members.clone().unwrap_or_default();
                    for child in self.child_nodes(Some(id)) {
                        out.extend(self.members(child.id).unwrap_or_default());
                    }
                    out
                })
            }
            CollectionKind::SmartAlbum => {
                if node.rules.is_empty() {
                    return Some(BTreeSet::new());
                }
                let scope = project_scope(&self.nodes, id);
                Some(
                    self.images
                        .iter()
                        .filter(|i| scope.as_ref().is_none_or(|m| m.contains(&i.id)))
                        .filter(|i| {
                            let edited = self.edited(i.id);
                            if node.match_all {
                                node.rules.iter().all(|r| rule_matches(r, i, edited))
                            } else {
                                node.rules.iter().any(|r| rule_matches(r, i, edited))
                            }
                        })
                        .map(|i| i.id)
                        .collect(),
                )
            }
            CollectionKind::Project => Some(manual_album_members(&self.nodes, id)),
            CollectionKind::Group => {
                let mut out = BTreeSet::new();
                for child in self.child_nodes(Some(id)) {
                    match self.members(child.id) {
                        Some(m) => out.extend(m),
                        None => return Some(all().collect()),
                    }
                }
                Some(out)
            }
        }
    }

    pub fn collection_count(&self, id: u64) -> usize {
        if let Some(node) = self.node(id) {
            if node.kind == CollectionKind::Folder {
                if let Some(path) = &node.path { return self.images.iter().filter(|i| !self.trash.contains(&i.id) && folder_contains(path, &i.path, self.library_include_subfolders)).count(); }
            }
        }
        match self.members(id) {
            Some(m) => {
                let trash = self.node(id).is_some_and(|n| n.kind == CollectionKind::Trash);
                m.iter().filter(|i| trash || !self.trash.contains(i)).count()
            }
            None => self.images.iter().filter(|i| !self.trash.contains(&i.id)).count(),
        }
    }

    fn persist_collections(&mut self, cx: &mut Context<Self>) -> bool {
        let user: Vec<UserCollection> = self
            .nodes
            .iter()
            .filter(|n| n.user)
            .map(|n| UserCollection {
                id: n.id,
                parent: n.parent,
                name: n.name.clone(),
                kind: n.kind,
                images: n.members.iter().flatten().copied().collect(),
                rules: n.rules.clone(),
                match_all: n.match_all,
            })
            .collect();
        if let Err(err) = self.engine.save_user_collections(&user) {
            self.notify(format!("{}: {err}", t("Saving collections failed; edits kept in this window", "保存集合失败；编辑已保留在当前窗口")), cx);
            return false;
        }
        cx.notify();
        true
    }

    pub fn create_collection(&mut self, parent: Option<u64>, kind: CollectionKind, name: Option<String>, cx: &mut Context<Self>) -> u64 {
        if !valid_parent(&self.nodes, None, kind, parent) {
            self.notify(t("This collection cannot be nested here", "无法在此处嵌套此集合"), cx);
            return self.active_collection;
        }
        let id = self.nodes.iter().map(|n| n.id).max().unwrap_or(0).max(USER_BASE) + 1;
        let name = name.unwrap_or_else(|| {
            match kind {
                CollectionKind::SmartAlbum => t("New Smart Album", "新建智能相册"),
                CollectionKind::Project => t("New Project", "新建项目"),
                CollectionKind::Group => t("New Group", "新建组"),
                _ => t("New Album", "新建相册"),
            }
            .to_string()
        });
        self.nodes.push(CollNode {
            id,
            parent,
            name,
            kind,
            user: true,
            path: None,
            members: (kind == CollectionKind::Album).then(BTreeSet::new),
            rules: if kind == CollectionKind::SmartAlbum { vec![SmartRule::RatingAtLeast(4)] } else { Vec::new() },
            match_all: true,
        });
        self.set_active_collection(id, cx);
        self.persist_collections(cx);
        id
    }

    /// Shows a collection in the browser and keeps the viewer on one of its images.
    pub fn set_active_collection(&mut self, id: u64, cx: &mut Context<Self>) {
        self.active_collection = id;
        self.active_collection_name = self.node(id).map(|n| if n.user { n.name.clone() } else { crate::i18n::builtin(&n.name) }).unwrap_or_default().into();
        let visible: Vec<ImageId> = self.visible_images().iter().map(|i| i.id).collect();
        self.selection.retain(|id| visible.contains(id));
        if visible.is_empty() {
            self.clear_collection_selection(cx);
        }
        if let Some(first) = visible.first()
            && !self.current.is_some_and(|c| visible.contains(&c))
        {
            self.select(*first, false, false, cx);
        }
        cx.notify();
    }

    fn clear_collection_selection(&mut self, cx: &mut Context<Self>) {
        self.end_gesture(cx);
        let changed = self.current.take().is_some();
        self.selection.clear();
        self.active_layer = None;
        self.active_node = None;
        self.zoom = ViewZoom::default();
        self.hover_look = None;
        self.sampled_color = None;
        self.picker_point = None;
        self.source_note = None;
        self.preview.pending = None;
        self.preview.rendering = false;
        self.preview.before_dirty = true;
        self.preview.error = None;
        self.preview.histogram = Default::default();
        self.preview.elapsed_ms = 0.0;
        for frame in [self.preview.frame.take(), self.preview.before.take()].into_iter().flatten() {
            frame.release(cx);
        }
        if let Some(scope) = self.preview.scope.take() {
            cx.drop_image(scope, None);
        }
        if changed {
            cx.emit(StateEvent::CurrentChanged);
        }
    }

    pub fn rename_collection(&mut self, id: u64, name: String, cx: &mut Context<Self>) -> bool {
        let Some(n) = self.nodes.iter_mut().find(|n| n.id == id && n.user) else { return false };
        n.name = name;
        self.set_active_collection(self.active_collection, cx);
        self.persist_collections(cx)
    }

    pub fn can_move_collection(&self, id: u64, parent: Option<u64>) -> bool {
        self.node(id).is_some_and(|n| n.user && valid_parent(&self.nodes, Some(id), n.kind, parent))
    }

    pub fn move_collection(&mut self, id: u64, parent: Option<u64>, cx: &mut Context<Self>) -> bool {
        if !self.can_move_collection(id, parent) {
            self.notify(t("Invalid collection destination", "集合目标位置无效"), cx);
            return false;
        }
        self.nodes.iter_mut().find(|n| n.id == id).unwrap().parent = parent;
        self.set_active_collection(self.active_collection, cx);
        self.persist_collections(cx)
    }

    pub fn delete_collection(&mut self, id: u64, cx: &mut Context<Self>) {
        if !self.node(id).is_some_and(|n| n.user) {
            return;
        }
        let doomed = descendants(&self.nodes, id);
        self.nodes.retain(|n| !(n.user && doomed.contains(&n.id)));
        let active = if doomed.contains(&self.active_collection) { 1 } else { self.active_collection };
        self.set_active_collection(active, cx);
        self.persist_collections(cx);
    }

    pub fn add_to_album(&mut self, id: u64, images: &[ImageId], cx: &mut Context<Self>) {
        let Some(n) = self.nodes.iter_mut().find(|n| n.id == id && n.kind == CollectionKind::Album) else { return };
        let members = n.members.get_or_insert_with(BTreeSet::new);
        let before = members.len();
        members.extend(images.iter().copied());
        let added = members.len() - before;
        let name = n.name.clone();
        self.set_active_collection(self.active_collection, cx);
        if self.persist_collections(cx) {
            self.notify(format!("{added} {} “{name}”", t("images added to", "张图像已加入")), cx);
        }
    }

    pub fn remove_from_album(&mut self, id: u64, images: &[ImageId], cx: &mut Context<Self>) {
        if let Some(m) = self.nodes.iter_mut().find(|n| n.id == id).and_then(|n| n.members.as_mut()) {
            for i in images {
                m.remove(i);
            }
        }
        self.set_active_collection(self.active_collection, cx);
        self.persist_collections(cx);
    }

    pub fn set_smart_rules(&mut self, id: u64, rules: Vec<SmartRule>, match_all: bool, cx: &mut Context<Self>) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
            n.rules = rules;
            n.match_all = match_all;
        }
        self.set_active_collection(self.active_collection, cx);
        self.persist_collections(cx);
    }

    /// Moves the selection to the trash, or restores it when viewing the trash.
    pub fn toggle_trash_selection(&mut self, cx: &mut Context<Self>) {
        let ids: Vec<ImageId> = self.selection.iter().copied().collect();
        if ids.is_empty() {
            return;
        }
        let restoring = ids.iter().all(|i| self.trash.contains(i));
        if let Err(err) = self.engine.set_trashed(&ids, !restoring) {
            self.notify(err.to_string(), cx);
            return;
        }
        for id in &ids {
            if restoring {
                self.trash.remove(id);
            } else {
                self.trash.insert(*id);
            }
        }
        let msg = if restoring { t("restored from trash", "已从废纸篓恢复") } else { t("moved to trash", "已移到废纸篓") };
        self.notify(format!("{} {msg}", ids.len()), cx);
        self.set_active_collection(self.active_collection, cx);
        cx.notify();
    }

    pub fn empty_trash(&mut self, cx: &mut Context<Self>) {
        let ids: Vec<ImageId> = self.trash.iter().copied().collect();
        if ids.is_empty() {
            return;
        }
        self.end_gesture(cx);
        // The shared executor removes UI records only after Forget succeeds; originals stay on disk.
        self.run_library_action(LibraryAction::Forget { images: ids }, cx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64, parent: Option<u64>, kind: CollectionKind, members: &[u64]) -> CollNode {
        CollNode {
            id, parent, kind, name: id.to_string(), user: true, path: None,
            members: (!members.is_empty() || kind == CollectionKind::Album).then(|| members.iter().map(|id| ImageId(*id)).collect()),
            rules: Vec::new(), match_all: true,
        }
    }

    #[test]
    fn folder_membership_matches_components_and_subfolder_setting() {
        let root = std::env::temp_dir().join("aurora-collection-test");
        let folder = root.join("Session");
        assert!(folder_contains(&folder, &folder.join("photo.nef"), true));
        assert!(folder_contains(&folder, &folder.join("photo.nef"), false));
        assert!(folder_contains(&folder, &folder.join("nested/photo.nef"), true));
        assert!(!folder_contains(&folder, &folder.join("nested/photo.nef"), false));
        assert!(!folder_contains(&folder, &root.join("Session-backup/photo.nef"), true));
        assert!(!folder_contains(Path::new(""), &folder.join("photo.nef"), true));
    }

    #[test]
    fn fallback_paths_never_use_display_only_roots() {
        let root = std::env::temp_dir().join("aurora-collection-test");
        assert_eq!(fallback_folder_path("Folders", None), None);
        assert_eq!(fallback_folder_path("Session", None), None);
        assert_eq!(fallback_folder_path(root.to_str().unwrap(), None), Some(root.clone()));
        assert_eq!(fallback_folder_path("2026/Session", Some(&root)), Some(root.join("2026/Session")));
    }

    #[test]
    #[cfg(windows)]
    fn windows_drive_and_unc_roots_are_real_paths() {
        assert!(folder_contains(Path::new("C:/"), Path::new(r"C:\Photos\photo.nef"), true));
        assert!(!folder_contains(Path::new("C:/"), Path::new("D:/Photos/photo.nef"), true));
        let unc = Path::new(r"\\server\photos");
        assert!(folder_contains(unc, Path::new(r"\\server\photos\nested\photo.nef"), true));
        assert!(!folder_contains(unc, Path::new(r"\\server\photos-backup\photo.nef"), true));
        assert_eq!(fallback_folder_path("C:/Photos", None), Some(PathBuf::from("C:/Photos")));
    }

    #[test]
    fn project_scope_uses_manual_albums_in_nested_groups_only() {
        let nodes = vec![
            node(1, None, CollectionKind::Project, &[]),
            node(2, Some(1), CollectionKind::Album, &[10, 11]),
            node(3, Some(1), CollectionKind::Group, &[]),
            node(4, Some(3), CollectionKind::Group, &[]),
            node(5, Some(4), CollectionKind::Album, &[11, 12]),
            node(6, Some(4), CollectionKind::SmartAlbum, &[99]),
            node(7, None, CollectionKind::Album, &[98]),
            node(8, Some(1), CollectionKind::Project, &[]),
            node(9, Some(8), CollectionKind::Album, &[97]),
        ];
        let expected = BTreeSet::from([ImageId(10), ImageId(11), ImageId(12)]);
        assert_eq!(manual_album_members(&nodes, 1), expected);
        assert_eq!(project_scope(&nodes, 6), Some(expected));
    }

    #[test]
    fn smart_album_alone_cannot_expand_an_empty_project() {
        let nodes = vec![
            node(1, None, CollectionKind::Project, &[]),
            node(2, Some(1), CollectionKind::SmartAlbum, &[99]),
            node(3, Some(1), CollectionKind::Group, &[]),
            node(4, Some(3), CollectionKind::SmartAlbum, &[98]),
        ];
        assert_eq!(project_scope(&nodes, 2), Some(BTreeSet::new()));
        assert_eq!(project_scope(&nodes, 4), Some(BTreeSet::new()));
    }

    #[test]
    fn ordinary_group_smart_albums_are_global_until_moved_into_a_project() {
        let mut nodes = vec![
            node(1, None, CollectionKind::Group, &[]),
            node(2, Some(1), CollectionKind::SmartAlbum, &[]),
            node(3, None, CollectionKind::Project, &[]),
            node(4, Some(3), CollectionKind::Album, &[10]),
        ];
        assert_eq!(project_scope(&nodes, 2), None);
        assert!(valid_parent(&nodes, Some(1), CollectionKind::Group, Some(3)));
        nodes[0].parent = Some(3);
        assert_eq!(project_scope(&nodes, 2), Some(BTreeSet::from([ImageId(10)])));
        nodes[0].parent = None;
        assert_eq!(project_scope(&nodes, 2), None);
    }

    #[test]
    fn moves_reject_cycles_noncontainers_and_unknown_parents() {
        let nodes = vec![
            node(1, None, CollectionKind::Group, &[]),
            node(2, Some(1), CollectionKind::Group, &[]),
            node(3, Some(2), CollectionKind::Album, &[]),
            node(4, None, CollectionKind::Folder, &[]),
        ];
        assert!(!valid_parent(&nodes, Some(1), CollectionKind::Group, Some(1)));
        assert!(!valid_parent(&nodes, Some(1), CollectionKind::Group, Some(2)));
        assert!(!valid_parent(&nodes, Some(2), CollectionKind::Group, Some(3)));
        assert!(!valid_parent(&nodes, Some(2), CollectionKind::Group, Some(4)));
        assert!(!valid_parent(&nodes, Some(2), CollectionKind::Group, Some(999)));
        assert!(!valid_parent(&nodes, None, CollectionKind::Folder, None));
        assert!(valid_parent(&nodes, Some(3), CollectionKind::Album, None));
    }

    #[test]
    fn project_nesting_cannot_be_hidden_inside_groups() {
        let nodes = vec![
            node(1, None, CollectionKind::Project, &[]),
            node(2, Some(1), CollectionKind::Group, &[]),
            node(3, None, CollectionKind::Group, &[]),
            node(4, Some(3), CollectionKind::Project, &[]),
            node(5, None, CollectionKind::Group, &[]),
        ];
        assert!(!valid_parent(&nodes, None, CollectionKind::Project, Some(1)));
        assert!(!valid_parent(&nodes, None, CollectionKind::Project, Some(2)));
        assert!(!valid_parent(&nodes, Some(3), CollectionKind::Group, Some(1)));
        assert!(!valid_parent(&nodes, Some(3), CollectionKind::Group, Some(2)));
        assert!(valid_parent(&nodes, None, CollectionKind::Project, Some(5)));
        assert!(valid_parent(&nodes, Some(5), CollectionKind::Group, Some(2)));
        assert!(valid_parent(&nodes, None, CollectionKind::Album, Some(2)));
        assert!(valid_parent(&nodes, None, CollectionKind::SmartAlbum, Some(1)));
    }

    #[test]
    fn malformed_group_cycles_terminate() {
        let nodes = vec![
            node(1, Some(2), CollectionKind::Group, &[]),
            node(2, Some(1), CollectionKind::Group, &[]),
            node(3, Some(1), CollectionKind::Album, &[10]),
            node(4, Some(2), CollectionKind::SmartAlbum, &[]),
        ];
        assert_eq!(descendants(&nodes, 1), BTreeSet::from([1, 2, 3, 4]));
        assert_eq!(manual_album_members(&nodes, 1), BTreeSet::from([ImageId(10)]));
        assert_eq!(project_scope(&nodes, 4), Some(BTreeSet::new()));
        assert!(!valid_parent(&nodes, None, CollectionKind::Album, Some(1)));
    }

    #[test]
    fn reopened_collections_keep_editability_rules_and_project_membership() {
        let root = std::env::temp_dir().join(format!("aurora-album-reopen-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap();
        image::RgbaImage::from_pixel(4, 3, image::Rgba([12u8, 20, 30, 255])).save(root.join("image.png")).unwrap();
        let engine = FileEngine::open(&root).unwrap();
        let id = engine.images()[0].id;
        let project = USER_BASE + 1;
        let album = UserCollection { id: project + 1, parent: Some(project), name: "Picks".into(), kind: CollectionKind::Album, images: vec![id], rules: vec![], match_all: true };
        let smart = UserCollection { id: project + 2, parent: Some(project), name: "Stars".into(), kind: CollectionKind::SmartAlbum, images: vec![], rules: vec![SmartRule::RatingAtLeast(4)], match_all: true };
        engine.save_user_collections(&[UserCollection { id: project, parent: None, name: "Project".into(), kind: CollectionKind::Project, images: vec![], rules: vec![], match_all: true }, album, smart]).unwrap();
        drop(engine);
        let engine = FileEngine::open(&root).unwrap();
        let nodes = flatten(&engine, &engine.collections());
        assert_eq!(engine.images()[0].id, id);
        assert!(nodes.iter().find(|n| n.id == project + 1).unwrap().user);
        assert_eq!(nodes.iter().find(|n| n.id == project + 2).unwrap().rules, vec![SmartRule::RatingAtLeast(4)]);
        assert_eq!(project_scope(&nodes, project + 2), Some(BTreeSet::from([id])));
        drop(engine);
        std::fs::remove_dir_all(root).unwrap();
    }
}
