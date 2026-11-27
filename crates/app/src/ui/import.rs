//! The Import workspace. It replaces the inspector and the viewer instead of
//! opening a dialog: the sidebar holds the import settings as collapsible
//! sections (source, destination, naming, backup, metadata, adjustments,
//! collection, filters) and the centre shows the source folder as a grid to
//! choose from. File copying, backup and renaming happen here; the engine
//! receives the final paths.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    checkbox::Checkbox,
    h_flex,
    input::Input,
    scroll::ScrollableElement as _,
    tooltip::Tooltip,
    v_flex, v_virtual_list,
};
use gpui_kit::{
    AnyElement, App, AppContext as _, ClickEvent, Context, Entity, InteractiveElement, IntoElement, ObjectFit,
    ParentElement, PathPromptOptions, Render, RenderImage, SharedString, StatefulInteractiveElement, Styled,
    StyledImage as _, Window, div, img, prelude::FluentBuilder as _, px, radians, size,
};

use crate::i18n::t;
use crate::state::{AppState, ImportAfter};
use crate::theme::{Palette, metrics, pal};
use crate::widgets::{IconBtn, Segmented, row, select, switch_row, text_field};
use crate::widgets::text::field_state;

/// Thumbnail cell widths the zoom buttons step through.
const ZOOM_STEPS: [f32; 6] = [96.0, 128.0, 160.0, 208.0, 256.0, 320.0];
/// Long side of decoded thumbnails, enough for the largest cell.
const THUMB_PX: u32 = 360;
/// Thumbnails decoded at once.
const THUMB_WORKERS: usize = 4;
const NAMING_TOKENS: [&str; 3] = ["[Image Name]", "[Counter]", "[Date]"];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    Raw,
    Jpeg,
    Tiff,
    Other,
}

impl Kind {
    const ALL: [Kind; 4] = [Kind::Raw, Kind::Jpeg, Kind::Tiff, Kind::Other];

    fn of(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
            Some("jpg" | "jpeg") => Kind::Jpeg,
            Some("tif" | "tiff") => Kind::Tiff,
            Some("png" | "webp" | "heic" | "heif") => Kind::Other,
            _ => Kind::Raw,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Kind::Raw => "RAW",
            Kind::Jpeg => "JPEG",
            Kind::Tiff => "TIFF",
            Kind::Other => t("Other", "其他"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sort {
    Name,
    Date,
    Size,
}

enum Thumb {
    Pending,
    Ready(Arc<RenderImage>),
    /// No preview could be decoded; the cell shows the file type.
    None,
}

struct SourceFile {
    path: PathBuf,
    name: String,
    size: u64,
    modified: Option<SystemTime>,
    kind: Kind,
    checked: bool,
    /// The catalog already has a file with this name and size.
    duplicate: bool,
    thumb: Thumb,
}

pub struct ImportView {
    state: Entity<AppState>,
    source: Option<PathBuf>,
    include_sub: bool,
    exclude_dupes: bool,
    files: Vec<SourceFile>,
    scanning: bool,
    /// Bumped by every scan so late thumbnails of an older scan are dropped.
    scan_gen: u64,
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
    collapsed: HashSet<&'static str>,
    search: String,
    hidden_kinds: HashSet<Kind>,
    sort: Sort,
    descending: bool,
    zoom: usize,
    /// Last clicked file, for Shift-click ranges.
    anchor: Option<usize>,
    /// Files copied so far while importing, out of the total.
    progress: Option<(usize, usize)>,
}

fn scan(dir: &Path, recursive: bool, out: &mut Vec<(PathBuf, u64, Option<SystemTime>)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        if path.is_dir() {
            if recursive && !e.file_name().to_string_lossy().starts_with('.') {
                scan(&path, true, out);
            }
        } else if aurora_engine::supports_photo_path(&path) {
            let meta = e.metadata().ok();
            out.push((path, meta.as_ref().map(|m| m.len()).unwrap_or(0), meta.and_then(|m| m.modified().ok())));
        }
    }
}

/// Decodes a JPEG/PNG/TIFF/WebP upright, following its EXIF orientation.
fn decode_upright(path: &Path) -> Option<image::DynamicImage> {
    use image::ImageDecoder as _;
    let mut decoder = image::ImageReader::open(path).ok()?.with_guessed_format().ok()?.into_decoder().ok()?;
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder).ok()?;
    img.apply_orientation(orientation);
    Some(img)
}

fn thumb(path: &Path, kind: Kind) -> Option<Arc<RenderImage>> {
    let rgba = match kind {
        // A RAW shows its embedded JPEG; nothing is demosaiced here.
        Kind::Raw => {
            let preview = aurora_engine::quick_embedded_preview(path, 640).ok()?;
            image::DynamicImage::ImageRgba8(preview).thumbnail(THUMB_PX, THUMB_PX).to_rgba8()
        }
        _ => decode_upright(path)?.thumbnail(THUMB_PX, THUMB_PX).to_rgba8(),
    };
    let (w, h) = rgba.dimensions();
    let mut data = rgba.into_raw();
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

/// Copies `src` to a new file at `target`. A file already there is never
/// replaced, and a copy that fails part way removes the partial file it made.
fn copy_new(src: &Path, target: &Path) -> std::io::Result<()> {
    let mut from = std::fs::File::open(src)?;
    // `create_new` fails if `target` exists, with no gap between check and create.
    let mut to = std::fs::OpenOptions::new().write(true).create_new(true).open(target)?;
    let copied = std::io::copy(&mut from, &mut to).and_then(|_| to.sync_all());
    match copied {
        Ok(()) => {
            // Keep the original's modified time where the file system allows it;
            // the copy is complete either way.
            if let Ok(modified) = from.metadata().and_then(|m| m.modified()) {
                let _ = to.set_modified(modified);
            }
            Ok(())
        }
        Err(e) => {
            drop(to);
            let _ = std::fs::remove_file(target);
            Err(e)
        }
    }
}

fn copy_error(src: &Path, target: &Path, e: std::io::Error) -> String {
    if e.kind() == std::io::ErrorKind::AlreadyExists {
        format!("{}: {}", target.display(), t("already exists", "已存在"))
    } else {
        format!("{}: {e}", src.display())
    }
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

fn megabytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 { format!("{:.2} GB", bytes as f64 / 1e9) } else { format!("{:.1} MB", bytes as f64 / 1e6) }
}

fn pick_dir(cx: &mut Context<ImportView>, set: impl Fn(&mut ImportView, PathBuf, &mut Context<ImportView>) + 'static) {
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

/// The folder import starts in: `AURORA_IMPORT_SOURCE`, else the user's Pictures folder.
fn default_source() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("AURORA_IMPORT_SOURCE").map(PathBuf::from).filter(|p| p.is_dir()) {
        return Some(dir);
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from)?;
    ["Pictures", "图片"].iter().map(|n| home.join(n)).find(|p| p.is_dir())
}

impl ImportView {
    pub fn new(state: Entity<AppState>, cx: &mut Context<Self>) -> Self {
        let mut view = Self {
            state,
            source: None,
            include_sub: true,
            exclude_dupes: true,
            files: Vec::new(),
            scanning: false,
            scan_gen: 0,
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
            // Less used sections start folded, as in Capture One.
            collapsed: ["backup", "metadata", "adjust", "collection"].into_iter().collect(),
            search: String::new(),
            hidden_kinds: HashSet::new(),
            sort: Sort::Name,
            descending: false,
            zoom: 2,
            anchor: None,
            progress: None,
        };
        if let Some(dir) = default_source() {
            view.set_source(dir, cx);
        }
        view
    }

    /// Rescans the source, e.g. when the workspace is opened again after a card was inserted.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        if let Some(src) = self.source.clone()
            && self.progress.is_none()
        {
            self.set_source(src, cx);
        }
    }

    fn close(&mut self, cx: &mut Context<Self>) {
        self.state.update(cx, |s, cx| {
            s.importing = false;
            cx.notify();
        });
    }

    fn mark_duplicates(&mut self, cx: &App) {
        let known: HashSet<(String, u64)> =
            self.state.read(cx).library.images.iter().map(|i| (i.file_name.to_lowercase(), i.file_size)).collect();
        for f in &mut self.files {
            f.duplicate = known.contains(&(f.name.to_lowercase(), f.size));
        }
    }

    fn set_source(&mut self, dir: PathBuf, cx: &mut Context<Self>) {
        self.source = Some(dir.clone());
        self.files.clear();
        self.anchor = None;
        self.scanning = true;
        self.scan_gen += 1;
        let generation = self.scan_gen;
        let recursive = self.include_sub;
        cx.spawn(async move |this, cx| {
            let found = cx
                .background_spawn(async move {
                    let mut out = Vec::new();
                    scan(&dir, recursive, &mut out);
                    out
                })
                .await;
            let queue = this
                .update(cx, |d, cx| {
                    if d.scan_gen != generation {
                        return None;
                    }
                    d.scanning = false;
                    d.files = found
                        .into_iter()
                        .map(|(path, size, modified)| SourceFile {
                            name: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                            kind: Kind::of(&path),
                            path,
                            size,
                            modified,
                            checked: true,
                            duplicate: false,
                            thumb: Thumb::Pending,
                        })
                        .collect();
                    d.mark_duplicates(cx);
                    cx.notify();
                    // Thumbnails load in the order the grid shows them.
                    Some(d.visible().into_iter().map(|i| (d.files[i].path.clone(), d.files[i].kind)).collect::<VecDeque<_>>())
                })
                .ok()
                .flatten();
            let Some(queue) = queue else { return };
            let queue = Arc::new(Mutex::new(queue));
            for _ in 0..THUMB_WORKERS {
                let queue = queue.clone();
                cx.spawn({
                    let this = this.clone();
                    async move |cx| loop {
                        let Some((path, kind)) = queue.lock().ok().and_then(|mut q| q.pop_front()) else { break };
                        let p2 = path.clone();
                        let decoded = cx.background_spawn(async move { thumb(&p2, kind) }).await;
                        let alive = this
                            .update(cx, |d, cx| {
                                if d.scan_gen != generation {
                                    return false;
                                }
                                if let Some(f) = d.files.iter_mut().find(|f| f.path == path) {
                                    f.thumb = decoded.map_or(Thumb::None, Thumb::Ready);
                                }
                                cx.notify();
                                true
                            })
                            .unwrap_or(false);
                        if !alive {
                            break;
                        }
                    }
                })
                .detach();
            }
        })
        .detach();
    }

    /// Indices of the files the grid shows, filtered and sorted.
    fn visible(&self) -> Vec<usize> {
        let needle = self.search.trim().to_lowercase();
        let mut out: Vec<usize> = (0..self.files.len())
            .filter(|&i| {
                let f = &self.files[i];
                !(self.exclude_dupes && f.duplicate)
                    && !self.hidden_kinds.contains(&f.kind)
                    && (needle.is_empty() || f.name.to_lowercase().contains(&needle))
            })
            .collect();
        let files = &self.files;
        match self.sort {
            Sort::Name => out.sort_by(|&a, &b| files[a].name.to_lowercase().cmp(&files[b].name.to_lowercase())),
            Sort::Date => out.sort_by_key(|&i| files[i].modified),
            Sort::Size => out.sort_by_key(|&i| files[i].size),
        }
        if self.descending {
            out.reverse();
        }
        out
    }

    fn filtering(&self) -> bool {
        !self.search.trim().is_empty() || !self.hidden_kinds.is_empty() || (self.exclude_dupes && self.files.iter().any(|f| f.duplicate))
    }

    fn click_file(&mut self, i: usize, e: &ClickEvent) {
        let shift = e.modifiers().shift;
        match self.anchor.filter(|_| shift) {
            Some(anchor) => {
                // Shift-click sets the whole range to the anchor's state.
                let order = self.visible();
                let (Some(a), Some(b)) = (order.iter().position(|&x| x == anchor), order.iter().position(|&x| x == i)) else { return };
                let on = self.files[anchor].checked;
                for &x in &order[a.min(b)..=a.max(b)] {
                    self.files[x].checked = on;
                }
            }
            None => {
                self.files[i].checked = !self.files[i].checked;
                self.anchor = Some(i);
            }
        }
    }

    fn set_all(&mut self, on: bool) {
        for i in self.visible() {
            self.files[i].checked = on;
        }
    }

    fn run(&mut self, cx: &mut Context<Self>) {
        let chosen: Vec<PathBuf> = self.visible().into_iter().filter(|&i| self.files[i].checked).map(|i| self.files[i].path.clone()).collect();
        if chosen.is_empty() || self.progress.is_some() {
            return;
        }
        let date = today();
        let dest = if self.copy { self.dest.clone().map(|d| if self.by_date { d.join(&date) } else { d }) } else { None };
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
        let total = chosen.len();
        self.progress = Some((0, total));
        cx.notify();
        cx.spawn(async move |this, cx| {
            let mut paths = Vec::new();
            let mut errors = Vec::new();
            for (i, src) in chosen.into_iter().enumerate() {
                let ext = src.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
                let name = match &rename {
                    Some(fmt) => format!("{}.{ext}", apply_naming(fmt, &src, i + 1, &date)),
                    None => src.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                };
                let (backup, dest) = (backup.clone(), dest.clone());
                let (done, mut errs) = cx
                    .background_spawn(async move {
                        let mut errs = Vec::new();
                        if let Some(b) = &backup {
                            let target = b.join(src.file_name().unwrap_or_default());
                            if let Err(e) = std::fs::create_dir_all(b).and_then(|_| copy_new(&src, &target)) {
                                errs.push(copy_error(&src, &target, e));
                            }
                        }
                        let done = match &dest {
                            Some(d) => {
                                let target = d.join(&name);
                                match std::fs::create_dir_all(d).and_then(|_| copy_new(&src, &target)) {
                                    Ok(()) => Some(target),
                                    Err(e) => {
                                        errs.push(copy_error(&src, &target, e));
                                        None
                                    }
                                }
                            }
                            None => Some(src),
                        };
                        (done, errs)
                    })
                    .await;
                paths.extend(done);
                errors.append(&mut errs);
                if this
                    .update(cx, |d, cx| {
                        d.progress = Some((i + 1, total));
                        cx.notify();
                    })
                    .is_err()
                {
                    return;
                }
            }
            cx.update(|cx| {
                state.update(cx, |s, cx| {
                    if !errors.is_empty() {
                        s.notify(format!("{} {}: {}", errors.len(), t("files could not be copied", "个文件未能复制"), errors[0]), cx);
                    }
                    s.welcome = false;
                    s.importing = false;
                    s.import_with(paths, after, cx)
                })
            });
            this.update(cx, |d, cx| {
                d.progress = None;
                d.mark_duplicates(cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// A Capture One style tool section: a header that folds the body.
    fn section(&self, id: &'static str, title: &'static str, summary: Option<String>, body: Vec<AnyElement>, p: &Palette, cx: &mut Context<Self>) -> AnyElement {
        let open = !self.collapsed.contains(id);
        let header = h_flex()
            .id(SharedString::from(format!("imp-sec-{id}")))
            .h(px(metrics::panel_header()))
            .px_2()
            .gap_1p5()
            .bg(p.raised)
            .border_t_1()
            .border_color(p.edge)
            .cursor_pointer()
            .hover(|s| s.bg(p.hover))
            .on_click(cx.listener(move |d, _, _, cx| {
                if !d.collapsed.remove(id) {
                    d.collapsed.insert(id);
                }
                cx.notify();
            }))
            .child(Icon::new(IconName::ChevronRight).xsmall().text_color(p.text_subtle).when(open, |i| i.rotate(radians(std::f32::consts::FRAC_PI_2))))
            .child(div().flex_shrink_0().text_xs().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(p.text).child(title))
            // Folded sections still say what they will do.
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_ellipsis()
                    .text_right()
                    .text_xs()
                    .text_color(p.text_subtle)
                    .when(!open, |d| d.children(summary)),
            );
        v_flex()
            .w_full()
            .border_b_1()
            .border_color(p.divider)
            .child(header)
            .when(open, |d| d.child(v_flex().w_full().py_1p5().gap_0p5().bg(p.panel).children(body)))
            .into_any_element()
    }

    fn path_row(&self, id: &'static str, path: &Option<PathBuf>, none: &'static str, p: &Palette, cx: &mut Context<Self>, set: fn(&mut ImportView, PathBuf, &mut Context<ImportView>)) -> AnyElement {
        let label = path.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| none.to_string());
        let tip: SharedString = label.clone().into();
        h_flex()
            .px_3()
            .py_0p5()
            .gap_2()
            .child(
                h_flex()
                    .id(SharedString::from(format!("{id}-path")))
                    .flex_1()
                    .min_w_0()
                    .h(px(24.0))
                    .px_2()
                    .gap_1p5()
                    .rounded(px(4.0))
                    .bg(p.well)
                    .tooltip(move |window, cx| Tooltip::new(tip.clone()).build(window, cx))
                    .child(Icon::new(IconName::Folder).xsmall().text_color(p.text_muted))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .text_ellipsis()
                            .text_xs()
                            .text_color(if path.is_some() { p.text } else { p.text_subtle })
                            .child(label),
                    ),
            )
            .child(Button::new(id).xsmall().outline().label(t("Choose…", "选择…")).on_click(cx.listener(move |_, _, _, cx| pick_dir(cx, set))))
            .into_any_element()
    }

    fn note(text: impl Into<SharedString>, p: &Palette) -> AnyElement {
        div().px_3().py_0p5().text_xs().text_color(p.text_subtle).child(text.into()).into_any_element()
    }

    fn sidebar(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let this = cx.entity();
        let upd = |f: fn(&mut ImportView, bool)| {
            let this = this.clone();
            move |v: bool, _: &mut Window, cx: &mut App| {
                this.update(cx, |d, cx| {
                    f(d, v);
                    cx.notify();
                })
            }
        };
        let text = |key: &'static str, initial: String, placeholder: &'static str, f: fn(&mut ImportView, String), window: &mut Window, cx: &mut App| {
            let this = this.clone();
            text_field(key, initial, placeholder, move |v, _, cx| this.update(cx, |d, _| f(d, v)), window, cx)
        };

        // Import from.
        let source_name = self.source.as_ref().and_then(|s| s.file_name()).map(|n| n.to_string_lossy().into_owned());
        let from = vec![
            self.path_row("imp-src", &self.source, t("Choose a folder or card", "选择文件夹或存储卡"), &p, cx, |d, dir, cx| d.set_source(dir, cx)),
            h_flex()
                .px_3()
                .pt_1()
                .gap_3()
                .child(Checkbox::new("imp-sub").xsmall().checked(self.include_sub).label(t("Include subfolders", "包含子文件夹")).on_change({
                    let this = this.clone();
                    move |v: &bool, _, cx| {
                        let v = *v;
                        this.update(cx, |d, cx| {
                            d.include_sub = v;
                            d.refresh(cx);
                            cx.notify();
                        })
                    }
                }))
                .into_any_element(),
            h_flex()
                .px_3()
                .py_0p5()
                .gap_3()
                .child(Checkbox::new("imp-dupes").xsmall().checked(self.exclude_dupes).label(t("Exclude duplicates", "排除重复项")).on_change({
                    let this = this.clone();
                    move |v: &bool, _, cx| {
                        let v = *v;
                        this.update(cx, |d, cx| {
                            d.exclude_dupes = v;
                            cx.notify();
                        })
                    }
                }))
                .into_any_element(),
            Self::note(t("Duplicates have the same name and size as an image already in the catalog.", "重复项指与目录中已有图像同名且大小相同的文件。"), &p),
        ];

        // Import to.
        let mut to = vec![div()
            .px_3()
            .child(Segmented::new("imp-mode", if self.copy { 1 } else { 0 }).text(t("Add in place", "就地添加")).text(t("Copy to folder", "复制到文件夹")).on_select({
                let this = this.clone();
                move |i, _, cx| {
                    this.update(cx, |d, cx| {
                        d.copy = i == 1;
                        cx.notify();
                    })
                }
            }))
            .into_any_element()];
        if self.copy {
            to.push(self.path_row("imp-dest", &self.dest, t("Choose a destination", "选择目标文件夹"), &p, cx, |d, dir, _| d.dest = Some(dir)));
            to.push(switch_row("imp-bydate", t("Subfolder by date", "按日期建子文件夹"), self.by_date, upd(|d, v| d.by_date = v), cx));
            if let Some(dest) = &self.dest {
                let full = if self.by_date { dest.join(today()) } else { dest.clone() };
                to.push(Self::note(format!("{} {}", t("Files go to", "文件将复制到"), full.display()), &p));
            }
        } else {
            to.push(Self::note(t("Your files stay where they are.", "您的文件将保留在原处。"), &p));
        }
        let to_summary = if self.copy {
            self.dest.as_ref().and_then(|d| d.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| t("Copy", "复制").into())
        } else {
            t("In place", "就地").into()
        };

        // Naming.
        let sample = self.files.first().map(|f| {
            let ext = f.path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
            format!("{}.{ext}", apply_naming(&self.naming, &f.path, 1, &today()))
        });
        let mut naming = vec![];
        if self.copy {
            naming.push(switch_row("imp-rename", t("Rename files", "重命名文件"), self.rename, upd(|d, v| d.rename = v), cx));
            if self.rename {
                let input = field_state("imp-naming", self.naming.clone(), "[Image Name]", {
                    let this = this.clone();
                    move |v, _, cx| {
                        this.update(cx, |d, cx| {
                            d.naming = v;
                            cx.notify();
                        })
                    }
                }, window, cx);
                naming.push(row(t("Format", "格式"), Input::new(&input).xsmall(), cx));
                naming.push(
                    h_flex()
                        .px_3()
                        .gap_1()
                        .flex_wrap()
                        .children(NAMING_TOKENS.iter().map(|token| {
                            let (this, input) = (this.clone(), input.clone());
                            Button::new(SharedString::from(format!("imp-token-{token}"))).xsmall().ghost().label(*token).on_click(move |_, window, cx| {
                                let next = format!("{}{token}", this.read(cx).naming);
                                input.update(cx, |i, cx| i.set_value(next.clone(), window, cx));
                                this.update(cx, |d, cx| {
                                    d.naming = next;
                                    cx.notify();
                                });
                            })
                        }))
                        .into_any_element(),
                );
                if let Some(s) = sample.clone() {
                    naming.push(row(t("Sample", "示例"), div().text_xs().text_color(p.accent).overflow_hidden().whitespace_nowrap().text_ellipsis().child(s), cx));
                }
            }
        } else {
            naming.push(Self::note(t("Renaming applies when copying to a folder.", "复制到文件夹时才能重命名。"), &p));
        }
        let naming_summary = (self.copy && self.rename).then(|| self.naming.clone());

        // Backup.
        let mut backup = vec![switch_row("imp-backup", t("Back up originals", "备份原始文件"), self.backup, upd(|d, v| d.backup = v), cx)];
        if self.backup {
            backup.push(self.path_row("imp-bdir", &self.backup_dir, t("Choose a backup folder", "选择备份文件夹"), &p, cx, |d, dir, _| d.backup_dir = Some(dir)));
        }
        let backup_summary = self.backup.then(|| t("On", "开").to_string());

        // Metadata.
        let metadata = vec![
            row(t("Copyright", "版权"), text("imp-copyright", self.copyright.clone(), "© 2026", |d, v| d.copyright = v, window, cx), cx),
            row(t("Creator", "创作者"), text("imp-creator", self.creator.clone(), "", |d, v| d.creator = v, window, cx), cx),
            row(t("Keywords", "关键词"), text("imp-keywords", self.keywords.clone(), "wedding, portrait", |d, v| d.keywords = v, window, cx), cx),
        ];
        let meta_summary = [&self.copyright, &self.creator, &self.keywords].iter().any(|s| !s.is_empty()).then(|| t("Set", "已设置").to_string());

        // Adjustments and collection.
        let styles: Vec<(Option<usize>, SharedString)> = std::iter::once((None, t("None", "无").into()))
            .chain(self.state.read(cx).styles.iter().enumerate().map(|(i, s)| (Some(i), format!("{} · {}", s.group, s.name).into())))
            .collect();
        let style_summary = self.style.and_then(|i| styles.iter().find(|(k, _)| *k == Some(i))).map(|(_, l)| l.to_string());
        let adjust = vec![row(t("Style", "风格"), select("imp-style", styles, self.style, {
            let this = this.clone();
            move |v, _, cx| {
                this.update(cx, |d, cx| {
                    d.style = v;
                    cx.notify();
                })
            }
        }), cx)];
        let albums: Vec<(Option<u64>, SharedString)> = std::iter::once((None, t("None", "无").into()))
            .chain(self.state.read(cx).nodes.iter().filter(|n| n.kind == aurora_engine::CollectionKind::Album).map(|n| (Some(n.id), n.name.clone().into())))
            .collect();
        let album_summary = self.album.and_then(|a| albums.iter().find(|(k, _)| *k == Some(a))).map(|(_, l)| l.to_string());
        let collection = vec![row(t("Add to album", "加入相册"), select("imp-album", albums, self.album, {
            let this = this.clone();
            move |v, _, cx| {
                this.update(cx, |d, cx| {
                    d.album = v;
                    cx.notify();
                })
            }
        }), cx)];

        // Filters.
        let search = text_field("imp-search", self.search.clone(), t("Search file names", "搜索文件名"), {
            let this = this.clone();
            move |v, _, cx| {
                this.update(cx, |d, cx| {
                    d.search = v;
                    cx.notify();
                })
            }
        }, window, cx);
        let pool: Vec<&SourceFile> = self.files.iter().filter(|f| !(self.exclude_dupes && f.duplicate)).collect();
        let mut filters = vec![h_flex().px_3().py_0p5().gap_1p5().child(Icon::new(IconName::Search).xsmall().text_color(p.text_subtle)).child(div().flex_1().child(search)).into_any_element()];
        filters.push(Self::note(t("File type", "文件类型"), &p));
        for kind in Kind::ALL {
            let n = pool.iter().filter(|f| f.kind == kind).count();
            let shown = !self.hidden_kinds.contains(&kind);
            filters.push(
                h_flex()
                    .px_3()
                    .h(px(metrics::row() + 2.0))
                    .gap_2()
                    .child(div().flex_1().child(Checkbox::new(SharedString::from(format!("imp-kind-{kind:?}"))).xsmall().checked(shown).disabled(n == 0).label(kind.label()).on_change({
                        let this = this.clone();
                        move |v: &bool, _, cx| {
                            let v = *v;
                            this.update(cx, |d, cx| {
                                if v {
                                    d.hidden_kinds.remove(&kind);
                                } else {
                                    d.hidden_kinds.insert(kind);
                                }
                                cx.notify();
                            })
                        }
                    })))
                    .child(div().min_w(px(28.0)).px_1p5().rounded(px(8.0)).bg(p.well).text_center().text_xs().text_color(if n > 0 { p.text_muted } else { p.text_disabled }).child(n.to_string()))
                    .into_any_element(),
            );
        }
        let filter_summary = self.filtering().then(|| t("Active", "已启用").to_string());

        v_flex()
            .id("imp-sidebar")
            .size_full()
            .bg(p.panel)
            .overflow_y_scrollbar()
            .child(self.section("from", t("Import From", "导入自"), source_name, from, &p, cx))
            .child(self.section("to", t("Import To", "导入至"), Some(to_summary), to, &p, cx))
            .child(self.section("naming", t("Naming", "命名"), naming_summary, naming, &p, cx))
            .child(self.section("backup", t("Backup", "备份"), backup_summary, backup, &p, cx))
            .child(self.section("metadata", t("Metadata", "元数据"), meta_summary, metadata, &p, cx))
            .child(self.section("adjust", t("Adjustments", "调整"), style_summary, adjust, &p, cx))
            .child(self.section("collection", t("Collection", "集合"), album_summary, collection, &p, cx))
            .child(self.section("filter", t("Filters", "过滤器"), filter_summary, filters, &p, cx))
            .child(div().h(px(24.0)))
            .into_any_element()
    }

    fn cell(&self, i: usize, side: f32, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let f = &self.files[i];
        let thumb_h = side * 0.72;
        v_flex()
            .id(SharedString::from(format!("imp-file-{i}")))
            .w(px(side))
            .flex_shrink_0()
            .p_1()
            .gap_1()
            .rounded(px(5.0))
            .cursor_pointer()
            .border_1()
            .border_color(if f.checked { p.accent.opacity(0.55) } else { gpui_kit::transparent_black() })
            .when(f.checked, |d| d.bg(p.accent_soft))
            .hover(|d| d.bg(p.hover))
            .on_click(cx.listener(move |d, e: &ClickEvent, _, cx| {
                d.click_file(i, e);
                cx.notify();
            }))
            .child(
                div()
                    .relative()
                    .h(px(thumb_h))
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded(px(3.0))
                    .overflow_hidden()
                    .bg(p.chrome)
                    .child(match &f.thumb {
                        Thumb::Ready(tn) => img(tn.clone()).size_full().object_fit(ObjectFit::Contain).when(!f.checked, |i| i.opacity(0.55)).into_any_element(),
                        Thumb::Pending => Icon::new(IconName::LoaderCircle).small().text_color(p.text_disabled).into_any_element(),
                        Thumb::None => v_flex()
                            .items_center()
                            .gap_1()
                            .text_color(p.text_subtle)
                            .child(Icon::new(IconName::Image).small())
                            .child(div().text_xs().child(f.path.extension().map(|e| e.to_string_lossy().to_uppercase()).unwrap_or_default()))
                            .into_any_element(),
                    })
                    .child(
                        div()
                            .absolute()
                            .top(px(4.0))
                            .left(px(4.0))
                            .size(px(16.0))
                            .rounded(px(3.0))
                            .border_1()
                            .border_color(p.text_muted)
                            .bg(p.chrome.opacity(0.6))
                            .flex()
                            .items_center()
                            .justify_center()
                            .when(f.checked, |d| d.bg(p.accent).border_color(p.accent).child(Icon::new(IconName::Check).size(px(12.0)).text_color(p.on_accent))),
                    )
                    .when(f.kind == Kind::Raw, |d| {
                        d.child(div().absolute().bottom(px(4.0)).right(px(4.0)).px_1().rounded(px(3.0)).bg(p.chrome.opacity(0.75)).text_size(px(9.0)).text_color(p.text_muted).child("RAW"))
                    })
                    .when(f.duplicate, |d| {
                        d.child(div().absolute().top(px(4.0)).right(px(4.0)).px_1().rounded(px(3.0)).bg(p.warning.opacity(0.85)).text_size(px(9.0)).text_color(p.chrome).child(t("In catalog", "已在目录")))
                    }),
            )
            .child(div().w_full().text_center().text_xs().text_color(if f.checked { p.text } else { p.text_muted }).overflow_hidden().whitespace_nowrap().text_ellipsis().child(f.name.clone()))
            .into_any_element()
    }

    fn browser(&mut self, grid_w: f32, cx: &mut Context<Self>) -> AnyElement {
        let p = pal(cx).clone();
        let this = cx.entity();
        let visible = self.visible();
        let checked = visible.iter().filter(|&&i| self.files[i].checked).count();
        let bytes: u64 = visible.iter().filter(|&&i| self.files[i].checked).map(|&i| self.files[i].size).sum();
        let count_label = if self.filtering() {
            format!("{} {}", visible.len(), t("images (filtered)", "个图像（已筛选）"))
        } else {
            format!("{} {}", visible.len(), t("images", "个图像"))
        };

        let sorts: Vec<(Sort, SharedString)> = vec![(Sort::Name, t("Name", "名称").into()), (Sort::Date, t("Date", "日期").into()), (Sort::Size, t("Size", "大小").into())];
        let zoom = self.zoom;
        let top = h_flex()
            .h(px(36.0))
            .flex_shrink_0()
            .px_3()
            .gap_2()
            .bg(p.chrome)
            .border_b_1()
            .border_color(p.divider)
            .child(Icon::new(IconName::LayoutGrid).small().text_color(p.accent))
            .child(div().text_sm().text_color(p.text).child(t("Import", "导入")))
            .child(div().w(px(1.0)).h(px(16.0)).bg(p.divider))
            .child(div().w(px(96.0)).child(select("imp-sort", sorts, self.sort, {
                let this = this.clone();
                move |v, _, cx| {
                    this.update(cx, |d, cx| {
                        d.sort = v;
                        cx.notify();
                    })
                }
            })))
            .child(
                IconBtn::new("imp-order", if self.descending { IconName::ArrowDownWideNarrow } else { IconName::ArrowUpNarrowWide }, t("Reverse order", "反向排序")).on_click({
                    let this = this.clone();
                    move |_, cx| {
                        this.update(cx, |d, cx| {
                            d.descending = !d.descending;
                            cx.notify();
                        })
                    }
                }),
            )
            .child(div().flex_1().text_center().text_sm().text_color(p.text_muted).child(count_label))
            .child(Button::new("imp-all").xsmall().ghost().label(t("Select All", "全选")).on_click(cx.listener(|d, _, _, cx| {
                d.set_all(true);
                cx.notify();
            })))
            .child(Button::new("imp-none").xsmall().ghost().label(t("Select None", "全不选")).on_click(cx.listener(|d, _, _, cx| {
                d.set_all(false);
                cx.notify();
            })))
            .child(div().w(px(1.0)).h(px(16.0)).bg(p.divider))
            .child(IconBtn::new("imp-zoom-out", IconName::ZoomOut, t("Smaller thumbnails", "缩小缩略图")).disabled(zoom == 0).on_click({
                let this = this.clone();
                move |_, cx| {
                    this.update(cx, |d, cx| {
                        d.zoom = d.zoom.saturating_sub(1);
                        cx.notify();
                    })
                }
            }))
            .child(
                h_flex().w(px(64.0)).h(px(4.0)).rounded(px(2.0)).bg(p.well).child(
                    div().h_full().rounded(px(2.0)).bg(p.text_muted).w(px(64.0 * (zoom + 1) as f32 / ZOOM_STEPS.len() as f32)),
                ),
            )
            .child(IconBtn::new("imp-zoom-in", IconName::ZoomIn, t("Larger thumbnails", "放大缩略图")).disabled(zoom + 1 >= ZOOM_STEPS.len()).on_click({
                let this = this.clone();
                move |_, cx| {
                    this.update(cx, |d, cx| {
                        d.zoom = (d.zoom + 1).min(ZOOM_STEPS.len() - 1);
                        cx.notify();
                    })
                }
            }));

        let gap = 8.0;
        let inner = (grid_w - gap * 2.0).max(ZOOM_STEPS[0]);
        let want = ZOOM_STEPS[self.zoom];
        let per_row = ((inner + gap) / (want + gap)).floor().max(1.0) as usize;
        let side = ((inner - gap * (per_row as f32 - 1.0)) / per_row as f32).min(want * 1.25);
        let row_h = side * 0.72 + 8.0 + 22.0 + gap;
        let rows = visible.len().div_ceil(per_row);

        let body = if self.scanning {
            v_flex().size_full().items_center().justify_center().gap_2().text_color(p.text_muted).child(Icon::new(IconName::LoaderCircle).size(px(28.0))).child(div().text_sm().child(t("Scanning…", "正在扫描…"))).into_any_element()
        } else if self.source.is_none() {
            v_flex()
                .size_full()
                .items_center()
                .justify_center()
                .gap_3()
                .text_color(p.text_subtle)
                .child(Icon::new(IconName::ImageDown).size(px(44.0)))
                .child(div().text_sm().child(t("Choose a folder or memory card to import from", "选择要导入的文件夹或存储卡")))
                .child(Button::new("imp-src-empty").primary().icon(Icon::new(IconName::FolderOpen)).label(t("Choose Source…", "选择来源…")).on_click(cx.listener(|_, _, _, cx| pick_dir(cx, |d, dir, cx| d.set_source(dir, cx)))))
                .into_any_element()
        } else if visible.is_empty() {
            let why = if self.files.is_empty() {
                t("No supported images in this folder", "此文件夹中没有支持的图像")
            } else {
                t("Every image is filtered out or already in the catalog", "所有图像都被过滤或已在目录中")
            };
            v_flex().size_full().items_center().justify_center().gap_2().text_color(p.text_subtle).child(Icon::new(IconName::Images).size(px(40.0))).child(div().text_sm().child(why)).into_any_element()
        } else {
            let order = Rc::new(visible.clone());
            let sizes = Rc::new(vec![size(px(grid_w), px(row_h)); rows]);
            v_virtual_list(this.clone(), "imp-grid", sizes, move |view, range, _, cx| {
                range
                    .map(|r| {
                        let start = r * per_row;
                        let end = (start + per_row).min(order.len());
                        h_flex().px(px(gap)).gap(px(gap)).h(px(row_h)).items_start().children((start..end).map(|k| view.cell(order[k], side, cx)))
                    })
                    .collect()
            })
            .pt(px(gap))
            .size_full()
            .into_any_element()
        };

        // The action bar says exactly what will happen before it happens.
        let dest_ok = !self.copy || self.dest.is_some();
        let backup_ok = !self.backup || self.backup_dir.is_some();
        let working = self.progress.is_some();
        let can_import = checked > 0 && !working && dest_ok && backup_ok;
        let target = if self.copy {
            match &self.dest {
                Some(d) => format!("{} {}", t("copy to", "复制到"), if self.by_date { d.join(today()) } else { d.clone() }.display()),
                None => t("choose a destination first", "请先选择目标文件夹").to_string(),
            }
        } else {
            t("add in place", "就地添加").to_string()
        };
        let status = match self.progress {
            Some((done, total)) => format!("{} {done}/{total}", t("Importing", "正在导入")),
            None if !backup_ok => t("Choose a backup folder or turn backup off", "请选择备份文件夹或关闭备份").to_string(),
            None => format!("{checked} {} · {} · {target}", t("selected", "张已选"), megabytes(bytes)),
        };
        let bar = h_flex()
            .h(px(44.0))
            .flex_shrink_0()
            .px_3()
            .gap_2()
            .bg(p.chrome)
            .border_t_1()
            .border_color(p.divider)
            .child(div().flex_1().min_w_0().overflow_hidden().whitespace_nowrap().text_ellipsis().text_xs().text_color(if dest_ok && backup_ok { p.text_muted } else { p.warning }).child(status))
            .when_some(self.progress, |d, (done, total)| {
                d.child(h_flex().w(px(120.0)).h(px(4.0)).rounded(px(2.0)).bg(p.well).child(div().h_full().rounded(px(2.0)).bg(p.accent).w(px(120.0 * done as f32 / total.max(1) as f32))))
            })
            .child(Button::new("imp-cancel").outline().label(t("Cancel", "取消")).disabled(working).on_click(cx.listener(|d, _, _, cx| d.close(cx))))
            .child(
                Button::new("imp-run")
                    .primary()
                    .loading(working)
                    .disabled(!can_import)
                    .icon(Icon::new(IconName::ImageDown))
                    .label(format!("{} {checked} {}", t("Import", "导入"), t("images", "张图像")))
                    .on_click(cx.listener(|d, _, _, cx| d.run(cx))),
            );

        v_flex().size_full().min_w_0().bg(p.well).child(top).child(div().flex_1().min_h_0().child(body)).child(bar).into_any_element()
    }
}

impl Render for ImportView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let view = self.state.read(cx).prefs.view.clone();
        let side_w = view.inspector_width.max(300.0);
        let grid_w = (f32::from(window.viewport_size().width) - side_w - 1.0).max(200.0);
        let sidebar = div().w(px(side_w)).h_full().flex_shrink_0().child(self.sidebar(window, cx));
        let rule = div().w(px(1.0)).h_full().flex_shrink_0().bg(p.divider);
        let browser = div().flex_1().h_full().min_w_0().child(self.browser(grid_w, cx));
        h_flex().id("import-workspace").size_full().map(|d| {
            if view.inspector_right { d.child(browser).child(rule).child(sidebar) } else { d.child(sidebar).child(rule).child(browser) }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naming_replaces_every_token() {
        let name = apply_naming("[Date]_[Image Name]_[Counter]", Path::new("/a/DSC0001.NEF"), 7, "2026-11-27");
        assert_eq!(name, "2026-11-27_DSC0001_0007");
    }

    #[test]
    fn kinds_follow_the_extension() {
        assert_eq!(Kind::of(Path::new("a.JPG")), Kind::Jpeg);
        assert_eq!(Kind::of(Path::new("a.tif")), Kind::Tiff);
        assert_eq!(Kind::of(Path::new("a.heic")), Kind::Other);
        assert_eq!(Kind::of(Path::new("a.NEF")), Kind::Raw);
    }

    #[test]
    fn copies_never_replace_a_file_already_there() {
        // Synthetic bytes in a fresh folder under the system temp directory.
        let dir = std::env::temp_dir().join(format!("aurora-import-copy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (src, target) = (dir.join("src.bin"), dir.join("copy.bin"));
        std::fs::write(&src, b"new bytes").unwrap();
        copy_new(&src, &target).unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new bytes");
        std::fs::write(&target, b"kept").unwrap();
        let e = copy_new(&src, &target).unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&target).unwrap(), b"kept", "the file already there is untouched");
        assert!(copy_new(&dir.join("missing.bin"), &dir.join("other.bin")).is_err());
        assert!(!dir.join("other.bin").exists(), "a failed copy leaves no file behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn today_is_an_iso_date() {
        let d = today();
        assert_eq!(d.len(), 10);
        assert_eq!(&d[4..5], "-");
        assert_eq!(&d[7..8], "-");
    }
}
