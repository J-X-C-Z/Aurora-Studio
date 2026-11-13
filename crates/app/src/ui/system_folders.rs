//! System folders are read one level at a time, off the UI thread.

use std::collections::BTreeMap;
use std::fs::{self, Metadata};
use std::io;
use std::path::{Path, PathBuf};

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _, button::{Button, ButtonVariants as _},
    h_flex, menu::{ContextMenuExt as _, PopupMenuItem}, v_flex,
};
use gpui_kit::{
    AnyElement, App, AppContext as _, Context, Entity, InteractiveElement, IntoElement, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, div,
    prelude::FluentBuilder as _, px,
};
use lumen_engine::LibraryAction;

use crate::state::AppState;
use crate::theme::pal;
use crate::ui::collections::DraggedImages;

#[derive(Clone)]
struct Root {
    path: PathBuf,
    label: String,
}

#[derive(Clone, Default)]
struct Branch {
    open: bool,
    loading: bool,
    children: Option<Vec<PathBuf>>,
    error: Option<String>,
}

#[derive(Default)]
struct FolderBrowser {
    roots: Vec<Root>,
    roots_loading: bool,
    roots_error: Option<String>,
    branches: BTreeMap<PathBuf, Branch>,
    selection_generation: u64,
}

fn normal_directory(metadata: &Metadata) -> bool {
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return false;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;
        if metadata.file_attributes() & 0x400 != 0 { return false; } // FILE_ATTRIBUTE_REPARSE_POINT
    }
    true
}

fn check_directory(path: &Path) -> io::Result<()> {
    // Cached descendants must not traverse an ancestor replaced by a junction.
    for ancestor in path.ancestors().filter(|p| !p.as_os_str().is_empty()) {
        if ancestor.file_name().is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case(".lumen"))
            || !normal_directory(&fs::symlink_metadata(ancestor)?)
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "不浏览内部目录、符号链接或重解析点"));
        }
    }
    Ok(())
}

struct Listing {
    children: Vec<PathBuf>,
    error: Option<String>,
}

fn read_children(path: &Path) -> io::Result<Listing> {
    check_directory(path)?;
    let mut listing = Listing { children: Vec::new(), error: None };
    for entry in fs::read_dir(path)? {
        let child = entry.and_then(|entry| {
            if entry.file_name().to_string_lossy().eq_ignore_ascii_case(".lumen") { return Ok(None); }
            let kind = entry.file_type()?;
            if !kind.is_dir() || kind.is_symlink() { return Ok(None); }
            let path = entry.path();
            Ok(normal_directory(&fs::symlink_metadata(&path)?).then_some(path))
        });
        match child {
            Ok(Some(path)) => listing.children.push(path),
            Ok(None) => {},
            Err(err) => { listing.error.get_or_insert_with(|| err.to_string()); },
        }
    }
    listing.children.sort_by_cached_key(|p| p.file_name().unwrap_or_default().to_string_lossy().to_lowercase());
    Ok(listing)
}

fn system_roots() -> io::Result<Vec<Root>> {
    let mut roots = Vec::new();
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        let path = PathBuf::from(home);
        if path.is_absolute() { roots.push(Root { path, label: "用户主目录".into() }); }
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" { fn GetLogicalDrives() -> u32; }
        // The bitmask lists assigned letters without opening local or network volumes.
        let mask = unsafe { GetLogicalDrives() };
        if mask == 0 { return Err(io::Error::last_os_error()); }
        for index in 0..26 {
            if mask & (1 << index) != 0 {
                let letter = char::from(b'A' + index);
                roots.push(Root { path: PathBuf::from(format!("{letter}:\\")), label: format!("磁盘 {letter}:") });
            }
        }
    }
    #[cfg(not(windows))]
    roots.push(Root { path: PathBuf::from("/"), label: "文件系统".into() });
    Ok(roots)
}

impl FolderBrowser {
    fn load_roots(&mut self, cx: &mut Context<Self>) {
        if self.roots_loading { return; }
        self.roots_loading = true;
        self.roots_error = None;
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async { system_roots() }).await;
            this.update(cx, |s, cx| {
                s.roots_loading = false;
                match result {
                    Ok(roots) => s.roots = roots,
                    Err(err) => s.roots_error = Some(err.to_string()),
                }
                cx.notify();
            }).ok();
        }).detach();
        cx.notify();
    }

    fn load_children(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let branch = self.branches.entry(path.clone()).or_default();
        branch.open = true;
        if branch.loading { return; }
        branch.loading = true;
        branch.error = None;
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn({ let path = path.clone(); async move { read_children(&path) } }).await;
            this.update(cx, |s, cx| {
                let branch = s.branches.entry(path).or_default();
                branch.loading = false;
                match result {
                    Ok(listing) => { branch.children = Some(listing.children); branch.error = listing.error; },
                    Err(err) => { branch.children = None; branch.error = Some(err.to_string()); },
                }
                cx.notify();
            }).ok();
        }).detach();
        cx.notify();
    }

    fn toggle(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let branch = self.branches.entry(path.clone()).or_default();
        branch.open = !branch.open;
        if branch.open && branch.children.is_none() && branch.error.is_none() {
            self.load_children(path, cx);
        }
        cx.notify();
    }

    fn browse(&mut self, state: Entity<AppState>, path: PathBuf, cx: &mut Context<Self>) {
        self.selection_generation += 1;
        let generation = self.selection_generation;
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn({ let path = path.clone(); async move { check_directory(&path) } }).await;
            this.update(cx, |s, cx| {
                if s.selection_generation != generation { return; }
                match result {
                    Ok(()) => state.update(cx, |s, cx| s.browse_library_folder(path, cx)),
                    Err(err) => {
                        let branch = s.branches.entry(path).or_default();
                        branch.open = true;
                        branch.error = Some(err.to_string());
                    },
                }
                cx.notify();
            }).ok();
        }).detach();
    }
}

/// The caller supplies the panel; this element owns only its lazy tree state.
pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let tree = window.use_keyed_state("system-folder-browser", cx, |_, cx| {
        let mut tree = FolderBrowser::default();
        tree.load_roots(cx);
        tree
    });
    let roots = tree.read(cx).roots.clone();
    let loading = tree.read(cx).roots_loading;
    let error = tree.read(cx).roots_error.clone();
    let mut content = v_flex().w_full();
    if loading { content = content.child(div().px_3().py_1().text_xs().text_color(pal(cx).text_muted).child("正在读取磁盘…")); }
    if let Some(error) = error {
        let retry = tree.clone();
        content = content.child(v_flex().px_3().gap_1().child(div().text_xs().text_color(pal(cx).danger).child(format!("无法读取磁盘：{error}")))
            .child(Button::new("system-folders-retry-roots").xsmall().ghost().label("重试").on_click(move |_, _, cx| retry.update(cx, |s, cx| s.load_roots(cx)))));
    }
    for (index, root) in roots.iter().enumerate() {
        content = content.child(folder_row(state, &tree, &root.path, &root.label, index, 0, cx));
    }
    content.into_any_element()
}

fn folder_row(state: &Entity<AppState>, tree: &Entity<FolderBrowser>, path: &Path, label: &str, root: usize, depth: usize, cx: &mut App) -> AnyElement {
    let p = pal(cx).clone();
    let branch = tree.read(cx).branches.get(path).cloned().unwrap_or_default();
    let active = state.read(cx).node(state.read(cx).active_collection).and_then(|n| n.path.as_deref()) == Some(path);
    let key = format!("system-folder-{root}-{}", path.display());
    let (expand, select, retry, refresh) = (tree.clone(), tree.clone(), tree.clone(), tree.clone());
    let (expand_path, select_path, menu_path, drop_path, retry_path, refresh_path) = (path.to_path_buf(), path.to_path_buf(), path.to_path_buf(), path.to_path_buf(), path.to_path_buf(), path.to_path_buf());
    let (select_state, menu_state, drop_state) = (state.clone(), state.clone(), state.clone());
    let row = h_flex()
        .id(SharedString::from(key.clone()))
        .h(px(26.0)).mx_2().pl(px(8.0 + depth as f32 * 14.0)).pr_2().gap_1p5().rounded(px(5.0))
        .cursor_pointer().when(active, |d| d.bg(p.accent_soft)).hover(|d| d.bg(p.hover))
        .on_click(move |_, _, cx| select.update(cx, |s, cx| s.browse(select_state.clone(), select_path.clone(), cx)))
        .drag_over::<DraggedImages>(|style, _, _, cx| style.bg(pal(cx).accent.opacity(0.35)))
        .on_drop(move |drag: &DraggedImages, _, cx| drop_state.update(cx, |s, cx| s.run_library_action(LibraryAction::Move { images: drag.0.clone(), destination: drop_path.clone() }, cx)))
        .context_menu(move |menu, _, cx| {
            let tree = refresh.clone();
            let path = refresh_path.clone();
            crate::ui::files::folder_menu(menu, &menu_state, menu_path.clone(), cx).separator()
                .item(PopupMenuItem::new("刷新文件夹").on_click(move |_, _, cx| tree.update(cx, |s, cx| s.load_children(path.clone(), cx))))
        })
        .child(div().id(SharedString::from(format!("{key}-expand"))).size(px(14.0)).flex().items_center().justify_center()
            .child(Icon::new(if branch.open { IconName::ChevronDown } else { IconName::ChevronRight }).xsmall().text_color(p.text_subtle))
            .on_click(move |_, _, cx| { cx.stop_propagation(); expand.update(cx, |s, cx| s.toggle(expand_path.clone(), cx)); }))
        .child(Icon::new(IconName::Folder).xsmall().text_color(p.text_muted))
        .child(div().flex_1().text_xs().text_color(p.text).overflow_hidden().whitespace_nowrap().text_ellipsis().child(label.to_string()));
    let mut content = v_flex().w_full().child(row);
    if branch.open {
        if branch.loading {
            content = content.child(div().pl(px(36.0 + depth as f32 * 14.0)).text_xs().text_color(p.text_muted).child("正在读取…"));
        }
        if let Some(error) = &branch.error {
            content = content.child(v_flex().pl(px(36.0 + depth as f32 * 14.0)).pr_2().gap_1()
                .child(div().text_xs().text_color(p.danger).child(format!("无法读取文件夹：{error}")))
                .child(Button::new(SharedString::from(format!("{key}-retry"))).xsmall().ghost().label("重试").on_click(move |_, _, cx| {
                    retry.update(cx, |s, cx| s.load_children(retry_path.clone(), cx));
                })));
        }
        if let Some(children) = branch.children {
            if children.is_empty() && !branch.loading && branch.error.is_none() {
                content = content.child(div().pl(px(36.0 + depth as f32 * 14.0)).text_xs().text_color(p.text_subtle).child("无子文件夹"));
            }
            for child in children {
                let label = child.file_name().unwrap_or_default().to_string_lossy().into_owned();
                content = content.child(folder_row(state, tree, &child, &label, root, depth + 1, cx));
            }
        }
    }
    content.into_any_element()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDir(PathBuf);
    impl TestDir {
        fn new() -> Self {
            let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
            let path = fs::canonicalize(std::env::temp_dir()).unwrap().join(format!("aurora-system-folders-{}-{unique}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); }
    }

    #[test]
    fn lists_only_immediate_normal_folders_in_name_order() {
        let dir = TestDir::new();
        for name in ["zebra/nested", "Alpha", ".lumen/cache", ".LUMEN"] { fs::create_dir_all(dir.0.join(name)).unwrap(); }
        fs::write(dir.0.join("photo.nef"), b"synthetic").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&dir.0, dir.0.join("loop")).unwrap();
        let listing = read_children(&dir.0).unwrap();
        assert_eq!(listing.children, vec![dir.0.join("Alpha"), dir.0.join("zebra")]);
        assert!(listing.error.is_none());
        assert!(read_children(&dir.0.join(".lumen")).is_err());
    }

    #[test]
    fn missing_or_non_directory_paths_return_errors() {
        let dir = TestDir::new();
        fs::write(dir.0.join("file"), b"synthetic").unwrap();
        assert!(read_children(&dir.0.join("file")).is_err());
        assert_eq!(read_children(&dir.0.join("missing")).err().unwrap().kind(), io::ErrorKind::NotFound);
        assert!(read_children(&dir.0).unwrap().children.is_empty());
    }
}
