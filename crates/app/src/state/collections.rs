//! Collections as the library shows them: the engine's tree flattened, plus
//! albums, smart albums, projects and groups the user creates, the trash, and
//! the session's recent imports and captures.

use std::collections::BTreeSet;

use gpui_kit::Context;
use lumen_engine::*;

use super::AppState;
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
    pub path: Option<String>,
    /// Explicit members; `None` derives membership from the kind.
    pub members: Option<BTreeSet<ImageId>>,
    pub rules: Vec<SmartRule>,
    pub match_all: bool,
}

/// The first id handed to user collections, above anything an engine uses.
const USER_BASE: u64 = 1 << 40;

pub(crate) fn flatten(engine: &dyn ImageEngine, tree: &[Collection]) -> Vec<CollNode> {
    fn walk(engine: &dyn ImageEngine, nodes: &mut Vec<CollNode>, c: &Collection, parent: Option<u64>, parent_path: Option<&str>) {
        let path = (c.kind == CollectionKind::Folder).then(|| match parent_path {
            Some(p) => format!("{}/{}", p.trim_end_matches('/'), c.name),
            None => c.name.clone(),
        });
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
        if let Some(m) = &node.members {
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
                let path = node.path.clone().unwrap_or_default().replace('\\', "/");
                Some(
                    self.images
                        .iter()
                        .filter(|i| i.path.to_string_lossy().replace('\\', "/").starts_with(&path))
                        .map(|i| i.id)
                        .collect(),
                )
            }
            CollectionKind::SmartAlbum => {
                if node.rules.is_empty() {
                    return Some(BTreeSet::new());
                }
                Some(
                    self.images
                        .iter()
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
            CollectionKind::Project | CollectionKind::Group => {
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
        match self.members(id) {
            Some(m) => {
                let trash = self.node(id).is_some_and(|n| n.kind == CollectionKind::Trash);
                m.iter().filter(|i| trash || !self.trash.contains(i)).count()
            }
            None => self.images.iter().filter(|i| !self.trash.contains(&i.id)).count(),
        }
    }

    fn persist_collections(&mut self, cx: &mut Context<Self>) {
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
            self.notify(format!("{}: {err}", t("Saving collections failed", "保存集合失败")), cx);
        }
        cx.notify();
    }

    pub fn create_collection(&mut self, parent: Option<u64>, kind: CollectionKind, name: Option<String>, cx: &mut Context<Self>) -> u64 {
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
        // Albums and smart albums may live in projects and groups; projects only in groups.
        let parent = parent.filter(|p| self.node(*p).is_some_and(|n| matches!(n.kind, CollectionKind::Group | CollectionKind::Project) && !(kind == CollectionKind::Project && n.kind == CollectionKind::Project)));
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
        self.active_collection = id;
        self.persist_collections(cx);
        id
    }

    /// Shows a collection in the browser and keeps the viewer on one of its images.
    pub fn set_active_collection(&mut self, id: u64, cx: &mut Context<Self>) {
        self.active_collection = id;
        self.active_collection_name = self.node(id).map(|n| if n.user { n.name.clone() } else { crate::i18n::builtin(&n.name) }).unwrap_or_default().into();
        let visible: Vec<ImageId> = self.visible_images().iter().map(|i| i.id).collect();
        if let Some(first) = visible.first()
            && !self.current.is_some_and(|c| visible.contains(&c))
        {
            self.select(*first, false, false, cx);
        }
        cx.notify();
    }

    pub fn rename_collection(&mut self, id: u64, name: String, cx: &mut Context<Self>) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id && n.user) {
            n.name = name;
        }
        self.persist_collections(cx);
    }

    pub fn delete_collection(&mut self, id: u64, cx: &mut Context<Self>) {
        let mut doomed = vec![id];
        let mut i = 0;
        while i < doomed.len() {
            let p = doomed[i];
            doomed.extend(self.nodes.iter().filter(|n| n.parent == Some(p)).map(|n| n.id));
            i += 1;
        }
        self.nodes.retain(|n| !(n.user && doomed.contains(&n.id)));
        if doomed.contains(&self.active_collection) {
            self.active_collection = 1;
        }
        self.persist_collections(cx);
    }

    pub fn add_to_album(&mut self, id: u64, images: &[ImageId], cx: &mut Context<Self>) {
        let Some(n) = self.nodes.iter_mut().find(|n| n.id == id && n.kind == CollectionKind::Album) else { return };
        let members = n.members.get_or_insert_with(BTreeSet::new);
        let before = members.len();
        members.extend(images.iter().copied());
        let added = members.len() - before;
        let name = n.name.clone();
        self.persist_collections(cx);
        self.notify(format!("{added} {} “{name}”", t("images added to", "张图像已加入")), cx);
    }

    pub fn remove_from_album(&mut self, id: u64, images: &[ImageId], cx: &mut Context<Self>) {
        if let Some(m) = self.nodes.iter_mut().find(|n| n.id == id).and_then(|n| n.members.as_mut()) {
            for i in images {
                m.remove(i);
            }
        }
        self.persist_collections(cx);
    }

    pub fn set_smart_rules(&mut self, id: u64, rules: Vec<SmartRule>, match_all: bool, cx: &mut Context<Self>) {
        if let Some(n) = self.nodes.iter_mut().find(|n| n.id == id) {
            n.rules = rules;
            n.match_all = match_all;
        }
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
        if !restoring {
            self.selection.clear();
            if let Some(next) = self.visible_images().first().map(|i| i.id) {
                self.select(next, false, false, cx);
            }
        }
        cx.notify();
    }

    pub fn empty_trash(&mut self, cx: &mut Context<Self>) {
        let ids: Vec<ImageId> = self.trash.iter().copied().collect();
        self.images.retain(|i| !self.trash.contains(&i.id));
        for id in &ids {
            self.edits.remove(id);
            if let Some(img) = self.thumbs.remove(id) {
                cx.drop_image(img, None);
            }
        }
        self.trash.clear();
        if self.current.is_some_and(|c| ids.contains(&c)) {
            self.current = None;
            if let Some(next) = self.visible_images().first().map(|i| i.id) {
                self.select(next, false, false, cx);
            }
        }
        self.notify(format!("{} {}", ids.len(), t("images removed from the catalog", "张图像已从目录中移除")), cx);
    }
}
