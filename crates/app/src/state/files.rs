//! File operations run off the UI thread; image identities and edits stay intact.
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use gpui_kit::{AppContext as _, Context};
use lumen_engine::*;
use super::{AppState, DocumentKind, Edit, LibraryDocument};
use crate::i18n::t;

impl AppState {
    pub fn refresh_file_status(&mut self, cx: &mut Context<Self>) {
        self.file_status_generation += 1;
        let generation = self.file_status_generation;
        let images = self.images.clone();
        cx.spawn(async move |this, cx| {
            let offline = cx.background_spawn(async move {
                images.iter().filter(|i| !i.path.is_file()).map(|i| i.id).collect()
            }).await;
            this.update(cx, |s, cx| { if s.file_status_generation == generation { s.library_offline = offline; cx.notify(); } }).ok();
        }).detach();
    }

    pub fn run_library_action(&mut self, action: LibraryAction, cx: &mut Context<Self>) {
        if self.library_busy { self.notify(t("A file operation is already running", "正在处理文件，请稍候"), cx); return; }
        self.end_gesture(cx);
        let known_edits: HashSet<_> = self.edits.keys().copied().collect();
        let select_folder = match &action { LibraryAction::Synchronize { folder, .. } | LibraryAction::CreateFolder { path: folder } => Some(folder.clone()), _ => None };
        let remap = match &action { LibraryAction::MoveFolder { from, to } | LibraryAction::Locate { from, to } => Some((from.clone(), to.clone())), _ => None };
        self.library_busy = true;
        self.file_status_generation += 1;
        cx.notify();
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async move {
                let images = engine.manage_files(action)?;
                let nodes = super::collections::flatten(engine.as_ref(), &engine.collections());
                let offline: HashSet<_> = images.iter().filter(|i| !i.path.is_file()).map(|i| i.id).collect();
                let edits = images.iter().filter(|i| !known_edits.contains(&i.id)).map(|i| (i.id, engine.load_adjustments(i.id))).collect::<Vec<_>>();
                Ok::<_, anyhow::Error>((images, nodes, offline, edits))
            }).await;
            this.update(cx, |s, cx| {
                s.library_busy = false;
                match result {
                    Err(e) => s.notify(format!("{}: {e:#}", t("File operation failed", "文件操作失败")), cx),
                    Ok((images, mut nodes, offline, edits)) => {
                        let ids: HashSet<_> = images.iter().map(|i| i.id).collect();
                        let new_ids = images.iter().filter(|i| !s.edits.contains_key(&i.id) && !offline.contains(&i.id)).map(|i| i.id).collect();
                        s.edits.retain(|id, _| ids.contains(id));
                        for (id, adj) in edits { s.edits.entry(id).or_insert_with(|| Edit { adj, ..Default::default() }); }
                        let removed: Vec<_> = s.thumbs.keys().filter(|id| !ids.contains(id)).copied().collect();
                        for id in removed { if let Some(thumb) = s.thumbs.remove(&id) { cx.drop_image(thumb, None); } }
                        s.selection.retain(|id| ids.contains(id));
                        s.recent_imports.retain(|id| ids.contains(id));
                        s.recent_captures.retain(|id| ids.contains(id));
                        s.trash = images.iter().filter(|i| i.trashed).map(|i| i.id).collect();
                        for user in s.nodes.iter().filter(|n| n.user) {
                            let mut user = user.clone();
                            if let Some(members) = &mut user.members { members.retain(|id| ids.contains(id)); }
                            nodes.retain(|n| n.id != user.id);
                            nodes.push(user);
                        }
                        s.nodes = nodes;
                        s.images = images;
                        s.library_offline = offline;
                        if let Some(path) = select_folder {
                            if let Some(node) = s.nodes.iter().find(|n| n.path.as_deref().is_some_and(|p| Path::new(p) == path)) { s.active_collection = node.id; }
                        }
                        if s.node(s.active_collection).is_none() { s.active_collection = 1; }
                        let visible: Vec<_> = s.visible_images().iter().map(|i| i.id).collect();
                        if !s.current.is_some_and(|id| visible.contains(&id)) {
                            s.current = None;
                            s.preview.generation += 1;
                            if let Some(frame) = s.preview.frame.take() { frame.release(cx); }
                            if let Some(frame) = s.preview.before.take() { frame.release(cx); }
                            if let Some(id) = visible.first() { s.select(*id, false, false, cx); }
                        }
                        s.active_collection_name = s.node(s.active_collection).map(|n| n.name.clone()).unwrap_or_default().into();
                        s.queue_thumbs(new_ids, cx);
                        s.notify(t("Library updated", "图库已更新"), cx);
                        if let (Some((from, to)), Some(root)) = (remap, s.engine.library_root()) {
                            if let Some(mut doc) = s.library_document.clone() {
                                doc.relocate(&root, &from, &to);
                                s.save_library_document(doc, cx);
                            }
                            for recipe in &mut s.recipes {
                                if let Some(destination) = &recipe.destination { if let Ok(tail) = destination.strip_prefix(&from) { recipe.destination = Some(to.join(tail)); } }
                            }
                        }
                    }
                }
                cx.notify();
            }).ok();
        }).detach();
    }

    pub fn browse_library_folder(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        if let Some(id) = self.nodes.iter().find(|n| n.path.as_deref().is_some_and(|p| Path::new(p) == path)).map(|n| n.id) {
            self.set_active_collection(id, cx);
        } else { self.run_library_action(LibraryAction::Synchronize { folder: path, recursive: self.library_include_subfolders }, cx); }
    }

    pub fn save_library_document(&mut self, doc: LibraryDocument, cx: &mut Context<Self>) {
        let Some(root) = self.engine.library_root() else { return };
        if let Err(e) = doc.save(&root) { self.notify(format!("{}: {e}", t("Saving library failed", "保存图库失败")), cx); return; }
        self.library_include_subfolders = doc.include_subfolders;
        self.library_document = Some(doc);
        cx.notify();
    }

    pub fn toggle_library_favorite(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let Some(root) = self.engine.library_root() else { return };
        let mut doc = self.library_document.clone().unwrap_or_else(|| LibraryDocument { version: 1, kind: DocumentKind::Catalog, name: self.source.title(), shortcuts: Default::default(), favorites: Vec::new(), include_subfolders: self.library_include_subfolders });
        let key = LibraryDocument::portable(&root, &path);
        if doc.favorites.contains(&key) { doc.favorites.retain(|p| *p != key); } else { doc.favorites.push(key); }
        self.save_library_document(doc, cx);
    }

    pub fn set_library_role(&mut self, role: &str, path: PathBuf, cx: &mut Context<Self>) {
        let Some(root) = self.engine.library_root() else { return };
        let Some(mut doc) = self.library_document.clone().filter(|d| d.kind == DocumentKind::Session) else { return };
        doc.shortcuts.insert(role.into(), LibraryDocument::portable(&root, &path));
        self.save_library_document(doc, cx);
        if role == "Output" {
            if let Some(doc) = &self.library_document {
                if doc.shortcuts.get(role).is_some_and(|p| LibraryDocument::resolve(&root, p) == path) {
                    for recipe in &mut self.recipes { recipe.destination = Some(path.clone()); }
                }
            }
        }
    }
}
