//! Library commands shared by the folder tree, thumbnails, and File menu.
use std::path::PathBuf;
use std::rc::Rc;
use gpui_kit::component::{WindowExt as _, button::{Button, ButtonVariants as _}, input::{Input, InputState}, menu::{PopupMenu, PopupMenuItem}, h_flex, v_flex};
use gpui_kit::{App, AppContext as _, Entity, Focusable as _, IntoElement, ParentElement, PathPromptOptions, Styled, Window, div, px};
use lumen_engine::{ImageId, LibraryAction};
use crate::i18n::t;
use crate::state::{AppState, DocumentKind, LibraryDocument};

pub fn choose_folder(cx: &mut App, done: impl FnOnce(PathBuf, &mut App) + 'static) {
    let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: Some(t("Choose Folder", "选择文件夹").into()) });
    cx.spawn(async move |cx| {
        if let Ok(Ok(Some(paths))) = rx.await {
            if let Some(path) = paths.into_iter().next() { cx.update(|cx| done(path, cx)); }
        }
    }).detach();
}

fn locate_original(state: &Entity<AppState>, image: ImageId, cx: &mut App) {
    let state = state.clone();
    let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some(t("Locate Original", "定位原始文件").into()) });
    cx.spawn(async move |cx| {
        if let Ok(Ok(Some(paths))) = rx.await {
            if let Some(to) = paths.into_iter().next() { cx.update(|cx| state.update(cx, |s, cx| s.run_library_action(LibraryAction::LocateImage { image, to }, cx))); }
        }
    }).detach();
}

fn prompt_text(title: &'static str, value: String, window: &mut Window, cx: &mut App, done: impl Fn(String, &mut Window, &mut App) + 'static) {
    let input = cx.new(|cx| InputState::new(window, cx).default_value(value));
    let done = Rc::new(done);
    window.open_dialog(cx, move |dialog, window, cx| {
        let input2 = input.clone();
        let done = done.clone();
        input.read(cx).focus_handle(cx).focus(window, cx);
        dialog.title(title).w(px(430.0)).child(Input::new(&input)).footer(
            h_flex().justify_end().child(Button::new("file-name-ok").primary().label(t("Continue", "继续")).on_click(move |_, window, cx| {
                let name = input2.read(cx).value().trim().to_string();
                if !name.is_empty() { window.close_dialog(cx); done(name, window, cx); }
            }))
        )
    });
}

fn confirm(title: &'static str, detail: String, window: &mut Window, cx: &mut App, done: impl Fn(&mut App) + 'static) {
    let done = Rc::new(done);
    window.open_dialog(cx, move |dialog, _, _| {
        let done = done.clone();
        dialog.title(title).w(px(460.0)).child(div().text_sm().child(detail.clone())).footer(
            h_flex().justify_end().child(Button::new("file-confirm").primary().label(t("Confirm", "确认")).on_click(move |_, window, cx| { window.close_dialog(cx); done(cx); }))
        )
    });
}

pub fn new_document(state: &Entity<AppState>, kind: DocumentKind, window: &mut Window, cx: &mut App) {
    let st = state.clone();
    let title = if kind == DocumentKind::Session { t("New Session", "新建会话") } else { t("New Catalog", "新建目录") };
    prompt_text(title, String::new(), window, cx, move |name, window, cx| {
        if !crate::state::library_document::valid_name(&name) { st.update(cx, |s, cx| s.notify(t("Invalid folder name", "文件夹名称无效"), cx)); return; }
        let state = st.clone();
        let replace = window.window_handle();
        choose_folder(cx, move |parent, cx| {
            cx.spawn(async move |cx| {
                let result = cx.background_spawn(async move { LibraryDocument::create(&parent, &name, kind) }).await;
                cx.update(|cx| match result {
                    Ok(root) => {
                        if state.read(cx).source == crate::state::Source::Demo { super::welcome::open_path(state, root, replace, cx); }
                        else {
                            let opened = std::env::current_exe().and_then(|exe| std::process::Command::new(exe).arg(&root).spawn());
                            state.update(cx, |s, cx| match opened {
                                Ok(_) => { s.settings.remember(&root); s.notify(t("Library opened in a new window", "图库已在新窗口打开"), cx); },
                                Err(e) => s.notify(format!("{}: {e}", t("Cannot open library", "无法打开图库")), cx),
                            });
                        }
                    },
                    Err(e) => state.update(cx, |s, cx| s.notify(format!("{}: {e}", t("Cannot create library", "无法创建图库")), cx)),
                });
            }).detach();
        });
    });
}

pub fn browse(state: &Entity<AppState>, cx: &mut App) {
    let st = state.clone();
    choose_folder(cx, move |path, cx| st.update(cx, |s, cx| s.browse_library_folder(path, cx)));
}

pub fn synchronize(state: &Entity<AppState>, folder: PathBuf, cx: &mut App) {
    state.update(cx, |s, cx| s.run_library_action(LibraryAction::Synchronize { folder, recursive: s.library_include_subfolders }, cx));
}

pub fn folder_menu(mut menu: PopupMenu, state: &Entity<AppState>, path: PathBuf, cx: &mut App) -> PopupMenu {
    let busy = state.read(cx).library_busy;
    let (a, b, c, d) = (state.clone(), state.clone(), state.clone(), state.clone());
    let (reveal, sync, locate, create, favorite) = (path.clone(), path.clone(), path.clone(), path.clone(), path.clone());
    let root = state.read(cx).engine.library_root();
    let is_favorite = root.as_ref().zip(state.read(cx).library_document.as_ref()).is_some_and(|(root, doc)| doc.favorites.contains(&LibraryDocument::portable(root, &path)));
    menu = menu
        .item(PopupMenuItem::new(t("Show in Explorer", "在资源管理器中显示")).on_click(move |_, _, cx| cx.reveal_path(&reveal)))
        .item(PopupMenuItem::new(t("Synchronize Folder…", "同步文件夹…")).disabled(busy).on_click(move |_, _, cx| synchronize(&a, sync.clone(), cx)))
        .item(PopupMenuItem::new(t("Locate Folder…", "定位文件夹…")).disabled(busy).on_click(move |_, _, cx| {
            let state = b.clone(); let from = locate.clone();
            choose_folder(cx, move |to, cx| state.update(cx, |s, cx| s.run_library_action(LibraryAction::Locate { from, to }, cx)));
        }))
        .item(PopupMenuItem::new(t("New Folder Inside…", "在此新建文件夹…")).disabled(busy).on_click(move |_, window, cx| {
            let state = c.clone(); let parent = create.clone();
            prompt_text(t("New Folder", "新建文件夹"), String::new(), window, cx, move |name, _, cx| {
                if !crate::state::library_document::valid_name(&name) { state.update(cx, |s, cx| s.notify(t("Invalid folder name", "文件夹名称无效"), cx)); return; }
                state.update(cx, |s, cx| s.run_library_action(LibraryAction::CreateFolder { path: parent.join(name) }, cx));
            });
        }))
        .item(PopupMenuItem::new(if is_favorite { t("Remove from Favorites", "从收藏夹移除") } else { t("Add to Favorites", "添加到收藏夹") }).on_click(move |_, _, cx| d.update(cx, |s, cx| s.toggle_library_favorite(favorite.clone(), cx))));
    if root.as_ref().is_some_and(|root| *root != path) {
        let (st_rename, st_move) = (state.clone(), state.clone());
        let (rename_path, move_path) = (path.clone(), path.clone());
        menu = menu
            .item(PopupMenuItem::new(t("Rename Folder…", "重命名文件夹…")).disabled(busy).on_click(move |_, window, cx| {
                let st = st_rename.clone(); let from = rename_path.clone();
                let name = from.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                prompt_text(t("Rename Folder", "重命名文件夹"), name, window, cx, move |name, _, cx| {
                    if !crate::state::library_document::valid_name(&name) { st.update(cx, |s, cx| s.notify(t("Invalid folder name", "文件夹名称无效"), cx)); return; }
                    if let Some(parent) = from.parent() { st.update(cx, |s, cx| s.run_library_action(LibraryAction::MoveFolder { from: from.clone(), to: parent.join(name) }, cx)); }
                });
            }))
            .item(PopupMenuItem::new(t("Move Folder…", "移动文件夹…")).disabled(busy).on_click(move |_, _, cx| {
                let st = st_move.clone(); let from = move_path.clone();
                choose_folder(cx, move |parent, cx| {
                    if let Some(name) = from.file_name() { let to = parent.join(name); st.update(cx, |s, cx| s.run_library_action(LibraryAction::MoveFolder { from, to }, cx)); }
                });
            }));
    }
    if state.read(cx).library_document.as_ref().is_some_and(|doc| doc.kind == DocumentKind::Session) {
        menu = menu.separator();
        for (role, label) in [("Capture", t("Set as Capture Folder", "设为拍摄文件夹")), ("Selects", t("Set as Selects Folder", "设为精选文件夹")), ("Output", t("Set as Output Folder", "设为输出文件夹")), ("Trash", t("Set as Trash Folder", "设为废纸篓文件夹"))] {
            let st = state.clone(); let path = path.clone();
            menu = menu.item(PopupMenuItem::new(label).disabled(busy).on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_library_role(role, path.clone(), cx))));
        }
    }
    menu
}

pub fn image_menu(mut menu: PopupMenu, state: &Entity<AppState>, id: ImageId, cx: &mut App) -> PopupMenu {
    let s = state.read(cx);
    let Some(info) = s.info(id) else { return menu };
    let ids = if s.selection.contains(&id) { s.selection.iter().copied().collect::<Vec<_>>() } else { vec![id] };
    let path = info.path.clone();
    let name = info.path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    let extension = info.path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
    let busy = s.library_busy;
    let trash = s.trash.contains(&id);
    let (a, b, c, d, e) = (state.clone(), state.clone(), state.clone(), state.clone(), state.clone());
    let (move_ids, forget_ids, trash_ids) = (ids.clone(), ids.clone(), ids.clone());
    let reveal = path.clone();
    menu = menu.separator()
        .item(PopupMenuItem::new(t("Show in Explorer", "在资源管理器中显示")).on_click(move |_, _, cx| cx.reveal_path(&reveal)))
        .item(PopupMenuItem::new(t("Rename File…", "重命名文件…")).disabled(busy).on_click(move |_, window, cx| {
            let st = a.clone(); let extension = extension.clone();
            prompt_text(t("Rename File (extension is preserved)", "重命名文件（保留扩展名）"), name.clone(), window, cx, move |name, _, cx| {
                let name = if extension.is_empty() { name } else { format!("{name}.{extension}") };
                st.update(cx, |s, cx| s.run_library_action(LibraryAction::Rename { image: id, name }, cx));
            });
        }))
        .item(PopupMenuItem::new(format!("{} ({})", t("Move to Folder…", "移动到文件夹…"), ids.len())).disabled(busy).on_click(move |_, _, cx| {
            let st = b.clone(); let ids = move_ids.clone();
            choose_folder(cx, move |destination, cx| st.update(cx, |s, cx| s.run_library_action(LibraryAction::Move { images: ids, destination }, cx)));
        }))
        .item(PopupMenuItem::new(if trash { t("Restore from Trash", "从废纸篓恢复") } else { t("Move to Trash", "移到废纸篓") }).disabled(busy).on_click(move |_, _, cx| {
            c.update(cx, |s, cx| { s.selection = trash_ids.iter().copied().collect(); s.toggle_trash_selection(cx); });
        }))
        .item(PopupMenuItem::new(t("Remove from Catalog…", "从目录中移除…")).disabled(busy).on_click(move |_, window, cx| {
            let st = d.clone(); let ids = forget_ids.clone();
            confirm(t("Remove from Catalog", "从目录中移除"), format!("{} {}", ids.len(), t("variants and their adjustments will be removed from this library. Original files remain on disk.", "个变体及其编辑记录将从当前图库移除，原始文件保留在磁盘上。")), window, cx, move |cx| st.update(cx, |s, cx| s.run_library_action(LibraryAction::Forget { images: ids.clone() }, cx)));
        }));
    menu = menu.item(PopupMenuItem::new(t("Locate Original…", "定位原始文件…")).disabled(busy || !state.read(cx).library_offline.contains(&id)).on_click(move |_, _, cx| locate_original(&e, id, cx)));
    if ids.len() > 1 {
        let st = state.clone();
        let rename_ids = ids.clone();
        menu = menu.item(PopupMenuItem::new(t("Batch Rename…", "批量重命名…")).disabled(busy).on_click(move |_, window, cx| {
            let st = st.clone(); let ids = rename_ids.clone();
            prompt_text(t("Naming: [Image Name] [Counter] [Date]", "命名模板：[Image Name] [Counter] [Date]"), "[Image Name]_[Counter]".into(), window, cx, move |template, _, cx| {
                let mut seen = std::collections::HashSet::new();
                let paths = ids.iter().filter_map(|id| st.read(cx).info(*id).map(|i| (*id, i.path.clone()))).filter(|(_, p)| seen.insert(p.clone())).collect::<Vec<_>>();
                let date = super::import::today();
                let names = paths.iter().enumerate().map(|(n, (id, p))| {
                    let name = super::import::apply_naming(&template, p, n + 1, &date);
                    let name = match p.extension() { Some(ext) => format!("{name}.{}", ext.to_string_lossy()), None => name };
                    (*id, name)
                }).collect();
                st.update(cx, |s, cx| s.run_library_action(LibraryAction::RenameMany { names }, cx));
            });
        }));
    }
    if let Some(destination) = state.read(cx).engine.library_root().zip(state.read(cx).library_document.as_ref()).and_then(|(root, doc)| doc.shortcuts.get("Selects").map(|p| LibraryDocument::resolve(&root, p))) {
        let st = state.clone();
        menu = menu.item(PopupMenuItem::new(t("Move to Selects", "移到精选文件夹")).disabled(busy).on_click(move |_, _, cx| st.update(cx, |s, cx| s.run_library_action(LibraryAction::Move { images: ids.clone(), destination: destination.clone() }, cx))));
    }
    menu
}

pub fn empty_trash(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let st = state.clone();
    let count = st.read(cx).trash.len();
    confirm(t("Empty Trash", "清空废纸篓"), format!("{count} {}", t("variants and adjustments will be removed from the catalog. Original files remain on disk.", "个变体及编辑记录将从目录中移除，原始文件保留在磁盘上。")), window, cx, move |cx| st.update(cx, |s, cx| s.empty_trash(cx)));
}

pub fn shortcuts(state: &Entity<AppState>, cx: &mut App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let root = s.engine.library_root();
    let doc = s.library_document.clone();
    let mut rows = v_flex().w_full();
    if let (Some(root), Some(doc)) = (root, doc) {
        for role in ["Capture", "Selects", "Output", "Trash"] {
            if let Some(path) = doc.shortcuts.get(role) {
                let path = LibraryDocument::resolve(&root, path);
                let st = state.clone();
                let label = match role { "Capture" => t("Capture", "拍摄"), "Selects" => t("Selects", "精选"), "Output" => t("Output", "输出"), _ => t("Trash Folder", "废纸篓文件夹") };
                rows = rows.child(Button::new(format!("role-{role}")).ghost().label(label).on_click(move |_, _, cx| st.update(cx, |s, cx| s.browse_library_folder(path.clone(), cx))));
            }
        }
        for (i, path) in doc.favorites.iter().enumerate() {
            let path = LibraryDocument::resolve(&root, path);
            let label = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy().into_owned();
            let st = state.clone();
            rows = rows.child(Button::new(format!("favorite-{i}")).ghost().label(format!("★ {label}")).on_click(move |_, _, cx| st.update(cx, |s, cx| s.browse_library_folder(path.clone(), cx))));
        }
    }
    rows
}
