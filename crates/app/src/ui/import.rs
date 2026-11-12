//! The Import window: pick a source, choose files, and decide where they go, how
//! they are named and backed up, and which metadata and style they receive.
//! File copying, backup and renaming happen here; the engine receives the final paths.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    checkbox::Checkbox,
    h_flex,
    scroll::ScrollableElement as _,
    v_flex,
};
use gpui_kit::{
    App, AppContext as _, Context, Entity, InteractiveElement, IntoElement, ObjectFit, ParentElement, PathPromptOptions,
    Render, RenderImage, SharedString, StatefulInteractiveElement, Styled, StyledImage as _, Window, div, img,
    prelude::FluentBuilder as _, px,
};

use crate::i18n::t;
use crate::state::{AppState, ImportAfter};
use crate::theme::pal;
use crate::widgets::{Segmented, caption, row, select, switch_row, text_field};

const EXTENSIONS: [&str; 16] =
    ["jpg", "jpeg", "png", "tif", "tiff", "webp", "dng", "iiq", "nef", "cr2", "cr3", "arw", "raf", "orf", "rw2", "heic"];

struct SourceFile {
    path: PathBuf,
    size: u64,
    checked: bool,
    thumb: Option<Arc<RenderImage>>,
}

pub struct ImportDialog {
    state: Entity<AppState>,
    source: Option<PathBuf>,
    include_sub: bool,
    files: Vec<SourceFile>,
    scanning: bool,
    copy: bool,
    dest: Option<PathBuf>,
    by_date: bool,
    backup: bool,
    backup_dir: Option<PathBuf>,
    rename: bool,
    naming: String,
    copyright: String,
    creator: String,
    keywords: String,
    style: Option<usize>,
    album: Option<u64>,
    working: bool,
}

fn is_image(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

fn scan(dir: &Path, recursive: bool, out: &mut Vec<(PathBuf, u64)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        if path.is_dir() {
            if recursive && !e.file_name().to_string_lossy().starts_with('.') {
                scan(&path, true, out);
            }
        } else if is_image(&path) {
            out.push((path, e.metadata().map(|m| m.len()).unwrap_or(0)));
        }
    }
}

fn thumb(path: &Path) -> Option<Arc<RenderImage>> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    if !matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "webp" | "tif" | "tiff") {
        return None;
    }
    let img = image::open(path).ok()?.thumbnail(200, 200).to_rgba8();
    let (w, h) = img.dimensions();
    let mut data = img.into_raw();
    for px in data.chunks_exact_mut(4) {
        px.swap(0, 2);
    }
    let buf = image::RgbaImage::from_raw(w, h, data)?;
    Some(Arc::new(RenderImage::new(vec![image::Frame::new(buf)])))
}

/// Applies a naming format to one file.
pub fn apply_naming(format: &str, original: &Path, counter: usize, date: &str) -> String {
    let stem = original.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    format.replace("[Image Name]", &stem).replace("[Counter]", &format!("{counter:04}")).replace("[Date]", date)
}

pub fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    // Civil date from days since the epoch (Howard Hinnant's algorithm).
    let z = secs / 86_400 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

fn pick_dir(cx: &mut Context<ImportDialog>, set: impl Fn(&mut ImportDialog, PathBuf, &mut Context<ImportDialog>) + 'static) {
    let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: None });
    cx.spawn(async move |this, cx| {
        if let Ok(Ok(Some(paths))) = rx.await
            && let Some(dir) = paths.into_iter().next()
        {
            this.update(cx, |d, cx| {
                set(d, dir, cx);
                cx.notify();
            })
            .ok();
        }
    })
    .detach();
}

impl ImportDialog {
    fn new(state: Entity<AppState>) -> Self {
        Self {
            state,
            source: None,
            include_sub: true,
            files: Vec::new(),
            scanning: false,
            copy: false,
            dest: None,
            by_date: true,
            backup: false,
            backup_dir: None,
            rename: false,
            naming: "[Image Name]".into(),
            copyright: String::new(),
            creator: String::new(),
            keywords: String::new(),
            style: None,
            album: None,
            working: false,
        }
    }

    fn set_source(&mut self, dir: PathBuf, cx: &mut Context<Self>) {
        self.source = Some(dir.clone());
        self.files.clear();
        self.scanning = true;
        let recursive = self.include_sub;
        cx.spawn(async move |this, cx| {
            let found = cx
                .background_spawn(async move {
                    let mut out = Vec::new();
                    scan(&dir, recursive, &mut out);
                    out
                })
                .await;
            let paths: Vec<PathBuf> = found.iter().take(120).map(|f| f.0.clone()).collect();
            this.update(cx, |d, cx| {
                d.scanning = false;
                d.files = found.into_iter().map(|(path, size)| SourceFile { path, size, checked: true, thumb: None }).collect();
                cx.notify();
            })
            .ok();
            // Thumbnails arrive one by one so the list is usable at once.
            for path in paths {
                let p2 = path.clone();
                let t = cx.background_spawn(async move { thumb(&p2) }).await;
                let alive = this
                    .update(cx, |d, cx| {
                        if let Some(f) = d.files.iter_mut().find(|f| f.path == path) {
                            f.thumb = t;
                        }
                        cx.notify();
                    })
                    .is_ok();
                if !alive {
                    break;
                }
            }
        })
        .detach();
    }

    fn run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let chosen: Vec<PathBuf> = self.files.iter().filter(|f| f.checked).map(|f| f.path.clone()).collect();
        if chosen.is_empty() {
            return;
        }
        let date = today();
        let dest = if self.copy {
            self.dest.clone().map(|d| if self.by_date { d.join(&date) } else { d })
        } else {
            None
        };
        let backup = self.backup.then(|| self.backup_dir.clone()).flatten();
        let rename = (self.copy && self.rename).then(|| self.naming.clone());
        let state = self.state.clone();
        let after = ImportAfter {
            copyright: self.copyright.clone(),
            creator: self.creator.clone(),
            keywords: self.keywords.split([',', '，', ';']).map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect(),
            style: self.style.and_then(|i| state.read(cx).styles.get(i).cloned()),
            album: self.album,
        };
        self.working = true;
        cx.notify();
        let handle = window.window_handle();
        cx.spawn(async move |this, cx| {
            let copied = cx
                .background_spawn(async move {
                    let mut out = Vec::new();
                    let mut errors = Vec::new();
                    for (i, src) in chosen.iter().enumerate() {
                        let ext = src.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
                        let name = match &rename {
                            Some(fmt) => format!("{}.{ext}", apply_naming(fmt, src, i + 1, &date)),
                            None => src.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                        };
                        if let Some(b) = &backup {
                            let target = b.join(src.file_name().unwrap_or_default());
                            if let Err(e) = std::fs::create_dir_all(b).and_then(|_| std::fs::copy(src, &target)) {
                                errors.push(format!("{}: {e}", src.display()));
                            }
                        }
                        match &dest {
                            Some(d) => {
                                let target = d.join(&name);
                                match std::fs::create_dir_all(d).and_then(|_| std::fs::copy(src, &target)) {
                                    Ok(_) => out.push(target),
                                    Err(e) => errors.push(format!("{}: {e}", src.display())),
                                }
                            }
                            None => out.push(src.clone()),
                        }
                    }
                    (out, errors)
                })
                .await;
            let (paths, errors) = copied;
            cx.update(|cx| {
                state.update(cx, |s, cx| {
                    if !errors.is_empty() {
                        s.notify(format!("{} {}: {}", errors.len(), t("files failed to copy", "个文件复制失败"), errors[0]), cx);
                    }
                    s.import_with(paths, after, cx)
                })
            });
            this.update(cx, |d, cx| {
                d.working = false;
                cx.notify();
            })
            .ok();
            handle.update(cx, |_, window, cx| window.close_dialog(cx)).ok();
        })
        .detach();
    }
}

impl Render for ImportDialog {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let this = cx.entity();
        let checked = self.files.iter().filter(|f| f.checked).count();
        let bytes: u64 = self.files.iter().filter(|f| f.checked).map(|f| f.size).sum();
        let sample = self.files.first().map(|f| {
            let ext = f.path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
            format!("{}.{ext}", apply_naming(&self.naming, &f.path, 1, &today()))
        });
        let styles: Vec<(Option<usize>, SharedString)> = std::iter::once((None, t("None", "无").into()))
            .chain(self.state.read(cx).styles.iter().enumerate().map(|(i, s)| (Some(i), format!("{} · {}", s.group, s.name).into())))
            .collect();
        let albums: Vec<(Option<u64>, SharedString)> = std::iter::once((None, t("None", "无").into()))
            .chain(
                self.state
                    .read(cx)
                    .nodes
                    .iter()
                    .filter(|n| n.kind == lumen_engine::CollectionKind::Album)
                    .map(|n| (Some(n.id), n.name.clone().into())),
            )
            .collect();

        let upd = |f: fn(&mut ImportDialog, bool)| {
            let this = this.clone();
            move |v: bool, _: &mut Window, cx: &mut App| {
                this.update(cx, |d, cx| {
                    f(d, v);
                    cx.notify();
                })
            }
        };
        let text = |key: &'static str, initial: String, placeholder: &'static str, f: fn(&mut ImportDialog, String), window: &mut Window, cx: &mut App| {
            let this = this.clone();
            text_field(key, initial, placeholder, move |v, _, cx| this.update(cx, |d, _| f(d, v)), window, cx)
        };
        let path_label = |p: &Option<PathBuf>, none: &'static str| p.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| none.to_string());

        let source_bar = h_flex()
            .gap_2()
            .child(Icon::new(IconName::FolderOpen).small().text_color(p.text_muted))
            .child(div().flex_1().text_sm().text_color(if self.source.is_some() { p.text } else { p.text_subtle }).overflow_hidden().whitespace_nowrap().text_ellipsis().child(path_label(&self.source, t("Choose a folder or card to import from", "选择要导入的文件夹或存储卡"))))
            .child(Button::new("imp-src").small().outline().label(t("Choose…", "选择…")).on_click(cx.listener(|_, _, _, cx| pick_dir(cx, |d, dir, cx| d.set_source(dir, cx)))));

        let grid = div()
            .id("imp-files")
            .flex_1()
            .min_h_0()
            .rounded(px(6.0))
            .bg(p.well)
            .p_2()
            .overflow_y_scrollbar()
            .child(match (self.scanning, self.files.is_empty()) {
                (true, _) => div().p_6().text_sm().text_color(p.text_muted).child(t("Scanning…", "正在扫描…")).into_any_element(),
                (false, true) => v_flex()
                    .p_6()
                    .items_center()
                    .gap_2()
                    .text_color(p.text_subtle)
                    .child(Icon::new(IconName::ImageDown).size(px(36.0)))
                    .child(div().text_sm().child(t("No images to show", "暂无可显示的图像")))
                    .into_any_element(),
                (false, false) => h_flex()
                    .flex_wrap()
                    .gap_2()
                    .children(self.files.iter().enumerate().map(|(i, f)| {
                        let this = this.clone();
                        let name = f.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                        let ext = f.path.extension().map(|e| e.to_string_lossy().to_uppercase()).unwrap_or_default();
                        v_flex()
                            .id(SharedString::from(format!("imp-file-{i}")))
                            .w(px(112.0))
                            .p_1()
                            .gap_1()
                            .rounded(px(5.0))
                            .cursor_pointer()
                            .when(f.checked, |d| d.bg(p.accent_soft))
                            .hover(|d| d.bg(p.hover))
                            .on_click(move |_, _, cx| {
                                this.update(cx, |d, cx| {
                                    if let Some(f) = d.files.get_mut(i) {
                                        f.checked = !f.checked;
                                    }
                                    cx.notify();
                                })
                            })
                            .child(
                                div()
                                    .relative()
                                    .h(px(78.0))
                                    .w_full()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .rounded(px(3.0))
                                    .bg(p.panel)
                                    .child(match f.thumb.clone() {
                                        Some(tn) => img(tn).max_w_full().max_h_full().object_fit(ObjectFit::Contain).into_any_element(),
                                        None => div().text_xs().text_color(p.text_subtle).child(ext).into_any_element(),
                                    })
                                    .child(
                                        div()
                                            .absolute()
                                            .top(px(3.0))
                                            .left(px(3.0))
                                            .size(px(14.0))
                                            .rounded(px(3.0))
                                            .border_1()
                                            .border_color(p.text_muted)
                                            .when(f.checked, |d| d.bg(p.accent).border_color(p.accent).child(Icon::new(IconName::Check).size(px(11.0)).text_color(p.on_accent))),
                                    ),
                            )
                            .child(div().text_xs().text_color(p.text_muted).overflow_hidden().whitespace_nowrap().text_ellipsis().child(name))
                    }))
                    .into_any_element(),
            });

        let this_all = this.clone();
        let this_none = this.clone();
        let left = v_flex()
            .flex_1()
            .min_w_0()
            .gap_2()
            .child(source_bar)
            .child(
                h_flex()
                    .gap_3()
                    .child(Checkbox::new("imp-sub").xsmall().checked(self.include_sub).label(t("Include subfolders", "包含子文件夹")).on_change({
                        let this = this.clone();
                        move |v: &bool, _, cx| {
                            let v = *v;
                            this.update(cx, |d, cx| {
                                d.include_sub = v;
                                if let Some(src) = d.source.clone() {
                                    d.set_source(src, cx);
                                }
                                cx.notify();
                            })
                        }
                    }))
                    .child(div().flex_1())
                    .child(div().text_xs().text_color(p.text_muted).child(format!("{checked}/{} · {:.1} MB", self.files.len(), bytes as f64 / 1e6)))
                    .child(Button::new("imp-all").xsmall().ghost().label(t("All", "全选")).on_click(move |_, _, cx| {
                        this_all.update(cx, |d, cx| {
                            d.files.iter_mut().for_each(|f| f.checked = true);
                            cx.notify();
                        })
                    }))
                    .child(Button::new("imp-none").xsmall().ghost().label(t("None", "全不选")).on_click(move |_, _, cx| {
                        this_none.update(cx, |d, cx| {
                            d.files.iter_mut().for_each(|f| f.checked = false);
                            cx.notify();
                        })
                    })),
            )
            .child(grid);

        let right = v_flex()
            .id("imp-settings")
            .w(px(320.0))
            .flex_shrink_0()
            .overflow_y_scrollbar()
            .child(caption(t("Import To", "导入到"), cx))
            .child(
                div().px_3().child(
                    Segmented::new("imp-mode", if self.copy { 1 } else { 0 }).text(t("Add in place", "就地添加")).text(t("Copy to folder", "复制到文件夹")).on_select({
                        let this = this.clone();
                        move |i, _, cx| {
                            this.update(cx, |d, cx| {
                                d.copy = i == 1;
                                cx.notify();
                            })
                        }
                    }),
                ),
            )
            .when(self.copy, |d| {
                d.child(
                    h_flex()
                        .px_3()
                        .pt_1p5()
                        .gap_2()
                        .child(div().flex_1().text_xs().text_color(p.text_muted).overflow_hidden().whitespace_nowrap().text_ellipsis().child(path_label(&self.dest, t("Choose a destination", "选择目标文件夹"))))
                        .child(Button::new("imp-dest").xsmall().outline().label(t("Choose…", "选择…")).on_click(cx.listener(|_, _, _, cx| pick_dir(cx, |d, dir, _| d.dest = Some(dir))))),
                )
                .child(switch_row("imp-bydate", t("Subfolder by date", "按日期建子文件夹"), self.by_date, upd(|d, v| d.by_date = v), cx))
                .child(switch_row("imp-rename", t("Rename files", "重命名文件"), self.rename, upd(|d, v| d.rename = v), cx))
                .when(self.rename, |d| {
                    d.child(row(t("Format", "格式"), text("imp-naming", self.naming.clone(), "[Image Name]", |d, v| d.naming = v, window, cx), cx))
                        .when_some(sample.clone(), |d, s| d.child(row(t("Sample", "示例"), div().text_xs().text_color(p.accent).child(s), cx)))
                })
            })
            .child(caption(t("Backup", "备份"), cx))
            .child(switch_row("imp-backup", t("Back up originals", "备份原始文件"), self.backup, upd(|d, v| d.backup = v), cx))
            .when(self.backup, |d| {
                d.child(
                    h_flex()
                        .px_3()
                        .gap_2()
                        .child(div().flex_1().text_xs().text_color(p.text_muted).overflow_hidden().whitespace_nowrap().text_ellipsis().child(path_label(&self.backup_dir, t("Choose a backup folder", "选择备份文件夹"))))
                        .child(Button::new("imp-bdir").xsmall().outline().label(t("Choose…", "选择…")).on_click(cx.listener(|_, _, _, cx| pick_dir(cx, |d, dir, _| d.backup_dir = Some(dir))))),
                )
            })
            .child(caption(t("Metadata", "元数据"), cx))
            .child(row(t("Copyright", "版权"), text("imp-copyright", self.copyright.clone(), "© 2026", |d, v| d.copyright = v, window, cx), cx))
            .child(row(t("Creator", "创作者"), text("imp-creator", self.creator.clone(), "", |d, v| d.creator = v, window, cx), cx))
            .child(row(t("Keywords", "关键词"), text("imp-keywords", self.keywords.clone(), "wedding, portrait", |d, v| d.keywords = v, window, cx), cx))
            .child(caption(t("Adjustments", "调整"), cx))
            .child(row(t("Style", "风格"), select("imp-style", styles, self.style, {
                let this = this.clone();
                move |v, _, cx| {
                    this.update(cx, |d, cx| {
                        d.style = v;
                        cx.notify();
                    })
                }
            }), cx))
            .child(caption(t("Collection", "集合"), cx))
            .child(row(t("Add to album", "加入相册"), select("imp-album", albums, self.album, {
                let this = this.clone();
                move |v, _, cx| {
                    this.update(cx, |d, cx| {
                        d.album = v;
                        cx.notify();
                    })
                }
            }), cx));

        let can_import = checked > 0 && !self.working && (!self.copy || self.dest.is_some()) && (!self.backup || self.backup_dir.is_some());
        v_flex()
            .h(px(560.0))
            .gap_3()
            .child(h_flex().items_start().flex_1().min_h_0().gap_3().child(left.h_full()).child(right.h_full()))
            .child(
                h_flex()
                    .justify_end()
                    .gap_2()
                    .child(Button::new("imp-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                    .child(
                        Button::new("imp-run")
                            .primary()
                            .loading(self.working)
                            .disabled(!can_import)
                            .icon(Icon::new(IconName::ImageDown))
                            .label(format!("{} {checked} {}", t("Import", "导入"), t("images", "张图像")))
                            .on_click(cx.listener(|d, _, window, cx| d.run(window, cx))),
                    ),
            )
    }
}

/// The folder the window starts in: `LUMEN_IMPORT_SOURCE`, else the user's Pictures folder.
fn default_source() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("LUMEN_IMPORT_SOURCE").map(PathBuf::from).filter(|p| p.is_dir()) {
        return Some(dir);
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from)?;
    ["Pictures", "图片"].iter().map(|n| home.join(n)).find(|p| p.is_dir())
}

pub fn open(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let view = cx.new(|cx| {
        let mut d = ImportDialog::new(state.clone());
        if let Some(dir) = default_source() {
            d.set_source(dir, cx);
        }
        d
    });
    window.open_dialog(cx, move |dialog, _, _| dialog.title(t("Import Images", "导入图像")).w(px(1040.0)).child(view.clone()));
}
