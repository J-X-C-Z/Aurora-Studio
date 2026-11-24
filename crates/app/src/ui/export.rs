//! The Export window: export variants through a recipe, or copy originals with an
//! optional adjustments sidecar.

use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{
    App, AppContext as _, Context, Entity, IntoElement, ParentElement, PathPromptOptions, Render, SharedString, Styled,
    Window, div, prelude::FluentBuilder as _, px,
};
use aurora_engine::ExportFormat;

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::pal;
use crate::widgets::{ParamSlider, Segmented, caption, row, select, switch_row, text_field};

pub struct ExportDialog {
    state: Entity<AppState>,
    originals: bool,
    recipe: usize,
    format: Option<ExportFormat>,
    quality: u8,
    dest: Option<PathBuf>,
    naming: String,
    by_date: bool,
    sidecar: bool,
    working: bool,
}

impl ExportDialog {
    fn pick_dest(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: None });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(dir) = paths.into_iter().next()
            {
                this.update(cx, |d, cx| {
                    d.dest = Some(dir);
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }

    fn run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let state = self.state.clone();
        if !self.originals {
            let Some(mut recipe) = state.read(cx).recipes.get(self.recipe).cloned() else { return };
            if let Some(f) = self.format {
                recipe.format = f;
            }
            recipe.quality = self.quality;
            recipe.destination = self.dest.clone().or(recipe.destination);
            recipe.file_name_format = self.naming.clone();
            state.update(cx, |s, cx| s.process(vec![recipe], cx));
            window.close_dialog(cx);
            return;
        }
        let Some(dest) = self.dest.clone() else { return };
        let dest = if self.by_date { dest.join(crate::ui::import::today()) } else { dest };
        let sidecar = self.sidecar;
        let items: Vec<OriginalItem> = {
            let s = state.read(cx);
            s.selection
                .iter()
                .filter_map(|id| {
                    s.info(*id).map(|i| OriginalItem {
                        source: i.path.clone(),
                        name: i.file_name.clone(),
                        sidecar: sidecar.then(|| {
                            serde_json::to_string_pretty(
                                &s.edits.get(id).map(|e| e.adj.clone()).unwrap_or_default(),
                            )
                            .unwrap_or_default()
                        }),
                    })
                })
                .collect()
        };
        self.working = true;
        cx.notify();
        let handle = window.window_handle();
        cx.spawn(async move |this, cx| {
            let (done, errors) = cx
                .background_spawn(async move { export_originals(&dest, &items) })
                .await;
            cx.update(|cx| {
                state.update(cx, |s, cx| {
                    let mut msg = format!("{done} {}", t("originals exported", "个原始文件已导出"));
                    if let Some(first) = errors.first() {
                        msg = format!("{msg} · {} {} ({first})", errors.len(), t("failed", "个失败"));
                    }
                    s.notify(msg, cx)
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

/// One selected photograph for "Export Originals".
#[derive(Clone, Debug)]
struct OriginalItem {
    source: PathBuf,
    /// The source's file name only; never a path, so error text stays free of
    /// the folder a photograph lives in.
    name: String,
    /// Pretty-printed adjustments, written beside the copy when asked for.
    sidecar: Option<String>,
}

/// Prefix of a same-directory temporary file used while copying an original.
const COPY_PREFIX: &str = ".aurora-export-copy-";

/// Copies every item into `dest` and then publishes it.
///
/// Returns how many items were fully exported (image plus sidecar) and one
/// message per failed item, prefixed with the file name only.
fn export_originals(dest: &Path, items: &[OriginalItem]) -> (usize, Vec<String>) {
    if let Err(error) = std::fs::create_dir_all(dest) {
        return (0, vec![error.to_string()]);
    }
    sweep_stale_before(dest, SystemTime::now() - Duration::from_secs(3600));
    let mut done = 0;
    let mut errors = Vec::new();
    for item in items {
        match export_original(dest, item) {
            Ok(()) => done += 1,
            Err(error) => errors.push(format!("{}: {error}", item.name)),
        }
    }
    (done, errors)
}

/// Copy one original and its sidecar, or do nothing and keep the destination
/// as it was. The image is published first, then the sidecar; if the sidecar
/// cannot be written the already-published copy is removed again so an export
/// never looks complete while its edits are missing.
fn export_original(dest: &Path, item: &OriginalItem) -> io::Result<()> {
    let published = publish_copy(&item.source, dest, &item.name)?;
    if let Some(json) = &item.sidecar {
        let sidecar = sidecar_path(&published);
        if let Err(error) = write_new_atomic(&sidecar, json.as_bytes()) {
            remove_published(&published);
            return Err(error);
        }
    }
    Ok(())
}

/// Copies `source` into a temporary file in `dest`, flushes it, and links it
/// under the first free name. Existing files are never replaced; a second
/// `shot.jpg` becomes `shot-1.jpg`, exactly like the recipe exporter.
fn publish_copy(source: &Path, dest: &Path, name: &str) -> io::Result<PathBuf> {
    let temp = staging_path(dest);
    let result = (|| {
        // `fs::copy` also carries the file attributes; the sync below makes the
        // bytes durable before the name appears in the destination.
        std::fs::copy(source, &temp)?;
        flush_copy(&temp)?;
        publish_new(&temp, dest, name)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

fn publish_new(temp: &Path, dir: &Path, name: &str) -> io::Result<PathBuf> {
    for n in 0..10_000u32 {
        let candidate = dir.join(numbered_name(name, n));
        if candidate.exists() {
            continue;
        }
        match std::fs::hard_link(temp, &candidate) {
            Ok(()) => {
                let _ = std::fs::remove_file(temp);
                return Ok(candidate);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(_) => {
                // A filesystem without hard links still gets an atomic rename;
                // the free-name check above is repeated right before it.
                if candidate.exists() {
                    continue;
                }
                match std::fs::rename(temp, &candidate) {
                    Ok(()) => return Ok(candidate),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => return Err(error),
                }
            }
        }
    }
    Err(io::Error::new(io::ErrorKind::AlreadyExists, "no free output name"))
}

/// Writes a new file through a same-directory temporary file and a hard link,
/// so a reader never sees a half-written sidecar and an existing file is never
/// replaced.
fn write_new_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let temp = staging_path(dir);
    let result = (|| {
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&temp)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        match std::fs::hard_link(&temp, path) {
            Ok(()) => {
                let _ = std::fs::remove_file(&temp);
                Ok(())
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Err(error),
            Err(_) => {
                if path.exists() {
                    return Err(io::Error::from(io::ErrorKind::AlreadyExists));
                }
                std::fs::rename(&temp, path)
            }
        }
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

/// `sync_all` on a copied original, including when the source was read-only.
///
/// `FlushFileBuffers` needs a writable handle on Windows, so a copied
/// read-only attribute is lifted for the flush and restored before publish.
fn flush_copy(path: &Path) -> io::Result<()> {
    #[cfg(windows)]
    {
        let mut permissions = std::fs::metadata(path)?.permissions();
        let readonly = permissions.readonly();
        if readonly {
            permissions.set_readonly(false);
            std::fs::set_permissions(path, permissions)?;
        }
        let result = (|| {
            let file = std::fs::OpenOptions::new().write(true).open(path)?;
            file.sync_all()
        })();
        if readonly {
            if let Ok(metadata) = std::fs::metadata(path) {
                let mut permissions = metadata.permissions();
                permissions.set_readonly(true);
                let _ = std::fs::set_permissions(path, permissions);
            }
        }
        result
    }
    #[cfg(not(windows))]
    {
        // POSIX fsync accepts a read-only descriptor, so the mode stays intact.
        std::fs::File::open(path)?.sync_all()
    }
}

/// Removes a copy this function just published, lifting a copied read-only
/// attribute first (Windows refuses to delete such a file).
fn remove_published(path: &Path) {
    #[cfg(windows)]
    if let Ok(metadata) = std::fs::metadata(path) {
        let mut permissions = metadata.permissions();
        if permissions.readonly() {
            permissions.set_readonly(false);
            let _ = std::fs::set_permissions(path, permissions);
        }
    }
    let _ = std::fs::remove_file(path);
}

fn sidecar_path(image: &Path) -> PathBuf {
    let name = image
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".into());
    image.with_file_name(format!("{name}.portrait.json"))
}

fn numbered_name(name: &str, n: u32) -> String {
    if n == 0 {
        return name.to_owned();
    }
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && !ext.is_empty() => {
            format!("{stem}-{n}.{ext}")
        }
        _ => format!("{name}-{n}"),
    }
}

fn staging_path(dir: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    dir.join(format!(
        "{COPY_PREFIX}{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Removes copy temporaries a crashed export left in `dest`.
fn sweep_stale_before(dir: &Path, cutoff: SystemTime) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with(COPY_PREFIX) {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .is_ok_and(|modified| modified < cutoff);
        if stale && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

impl Render for ExportDialog {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let this = cx.entity();
        let s = self.state.read(cx);
        let n = s.selection.len();
        let recipes: Vec<(usize, SharedString)> = s.recipes.iter().enumerate().map(|(i, r)| (i, r.name.clone().into())).collect();
        let recipe = s.recipes.get(self.recipe).cloned();
        let format = self.format.or(recipe.as_ref().map(|r| r.format)).unwrap_or_default();
        let dest_label = self
            .dest
            .as_ref()
            .map(|d| d.display().to_string())
            .or_else(|| recipe.as_ref().and_then(|r| r.destination.as_ref()).map(|d| d.display().to_string()))
            .unwrap_or_else(|| t("Output folder next to originals", "原图旁的 Output 文件夹").to_string());
        let can_run = n > 0 && !self.working && (!self.originals || self.dest.is_some());

        let body = if self.originals {
            v_flex()
                .child(caption(t("Originals", "原始文件"), cx))
                .child(div().px_3().text_xs().text_color(p.text_muted).child(t(
                    "Copies the original files unchanged. Adjustments can travel along as a sidecar.",
                    "原样复制原始文件，可附带一个记录调整的旁车文件。",
                )))
                .child(switch_row("exp-bydate", t("Subfolder by date", "按日期建子文件夹"), self.by_date, {
                    let this = this.clone();
                    move |v, _, cx| {
                        this.update(cx, |d, cx| {
                            d.by_date = v;
                            cx.notify();
                        })
                    }
                }, cx))
                .child(switch_row("exp-sidecar", t("Include adjustments (.portrait.json)", "附带调整（.portrait.json）"), self.sidecar, {
                    let this = this.clone();
                    move |v, _, cx| {
                        this.update(cx, |d, cx| {
                            d.sidecar = v;
                            cx.notify();
                        })
                    }
                }, cx))
        } else {
            let this2 = this.clone();
            let this3 = this.clone();
            let this4 = this.clone();
            let naming = self.naming.clone();
            v_flex()
                .child(caption(t("Recipe", "配方"), cx))
                .child(row(t("Based on", "基于"), select("exp-recipe", recipes, self.recipe, move |i, _, cx| {
                    this2.update(cx, |d, cx| {
                        d.recipe = i;
                        d.format = None;
                        if let Some(r) = d.state.read(cx).recipes.get(i) {
                            d.quality = r.quality;
                            d.naming = r.file_name_format.clone();
                        }
                        cx.notify();
                    })
                }), cx))
                .child(row(t("Format", "格式"), select("exp-format", ExportFormat::ALL.iter().map(|f| (*f, crate::contract::format_label(&s.capabilities, *f).into())).collect(), format, move |f, _, cx| {
                    this3.update(cx, |d, cx| {
                        d.format = Some(f);
                        cx.notify();
                    })
                }), cx))
                .when(format.has_quality(), |d| {
                    d.child(ParamSlider::new("exp-quality", t("Quality", "质量"), self.quality as f32, 1.0, 100.0).default_value(90.0).step(1.0).on_change(move |v, _, cx| {
                        this4.update(cx, |d, cx| {
                            d.quality = v.round() as u8;
                            cx.notify();
                        })
                    }))
                })
                .child(row(t("File name", "文件名"), text_field("exp-naming", naming, "[Image Name]", {
                    let this = this.clone();
                    move |v, _, cx| this.update(cx, |d, _| d.naming = v)
                }, window, cx), cx))
        };

        v_flex()
            .gap_3()
            .child(
                div().w(px(320.0)).child(Segmented::new("exp-mode", if self.originals { 1 } else { 0 }).text(t("Export Variants", "导出变体")).text(t("Export Originals", "导出原始文件")).on_select({
                    let this = this.clone();
                    move |i, _, cx| {
                        this.update(cx, |d, cx| {
                            d.originals = i == 1;
                            cx.notify();
                        })
                    }
                })),
            )
            .child(body)
            .child(caption(t("Location", "位置"), cx))
            .child(
                h_flex()
                    .px_3()
                    .gap_2()
                    .child(Icon::new(IconName::Folder).small().text_color(p.text_muted))
                    .child(div().flex_1().text_xs().text_color(p.text_muted).overflow_hidden().whitespace_nowrap().text_ellipsis().child(dest_label))
                    .child(Button::new("exp-dest").xsmall().outline().label(t("Choose…", "选择…")).on_click(cx.listener(|d, _, _, cx| d.pick_dest(cx)))),
            )
            .child(
                h_flex()
                    .pt_2()
                    .justify_end()
                    .gap_2()
                    .child(Button::new("exp-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                    .child(
                        Button::new("exp-run")
                            .primary()
                            .loading(self.working)
                            .disabled(!can_run)
                            .icon(Icon::new(IconName::Upload))
                            .label(format!("{} {n} {}", t("Export", "导出"), if self.originals { t("originals", "个原始文件") } else { t("variants", "个变体") }))
                            .on_click(cx.listener(|d, _, window, cx| d.run(window, cx))),
                    ),
            )
    }
}

pub fn open(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let (quality, naming) = state.read(cx).recipes.first().map(|r| (r.quality, r.file_name_format.clone())).unwrap_or((90, "[Image Name]".into()));
    let view = cx.new(|_| ExportDialog {
        state: state.clone(),
        originals: false,
        recipe: 0,
        format: None,
        quality,
        dest: None,
        naming,
        by_date: false,
        sidecar: true,
        working: false,
    });
    window.open_dialog(cx, move |dialog, _, _| dialog.title(t("Export", "导出")).w(px(520.0)).child(view.clone()));
}

#[cfg(test)]
mod tests {
    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct Temp(PathBuf);

    impl Temp {
        fn new(tag: &str) -> Self {
            let id = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir()
                .join(format!("aurora-export-originals-{tag}-{}-{id}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn source(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn item(source: PathBuf, name: &str, sidecar: Option<&str>) -> OriginalItem {
        OriginalItem { source, name: name.into(), sidecar: sidecar.map(str::to_owned) }
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    fn temporaries(dir: &Path) -> Vec<String> {
        names(dir).into_iter().filter(|name| name.starts_with(COPY_PREFIX)).collect()
    }

    #[test]
    fn originals_are_copied_without_overwriting_and_duplicates_get_suffixes() {
        let temp = Temp::new("overwrite");
        let first = source(temp.path(), "first.jpg", b"first bytes");
        let second = source(temp.path(), "second.jpg", b"second bytes");
        let dest = temp.path().join("out");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("shot.jpg"), b"keep me").unwrap();

        let items = [
            item(first, "shot.jpg", None),
            item(second, "shot.jpg", None),
        ];
        let (done, errors) = export_originals(&dest, &items);

        assert_eq!((done, errors.len()), (2, 0), "{errors:?}");
        assert_eq!(std::fs::read(dest.join("shot.jpg")).unwrap(), b"keep me");
        assert_eq!(std::fs::read(dest.join("shot-1.jpg")).unwrap(), b"first bytes");
        assert_eq!(std::fs::read(dest.join("shot-2.jpg")).unwrap(), b"second bytes");
        assert!(temporaries(&dest).is_empty());
    }

    #[test]
    fn a_sidecar_follows_the_published_image_name() {
        let temp = Temp::new("sidecar");
        let first = source(temp.path(), "a.jpg", b"a");
        let second = source(temp.path(), "b.jpg", b"b");
        let dest = temp.path().join("out");

        let items = [
            item(first, "shot.jpg", Some("{\"one\":1}")),
            item(second, "shot.jpg", Some("{\"two\":2}")),
        ];
        let (done, errors) = export_originals(&dest, &items);

        assert_eq!((done, errors.len()), (2, 0), "{errors:?}");
        assert_eq!(std::fs::read_to_string(dest.join("shot.jpg.portrait.json")).unwrap(), "{\"one\":1}");
        assert_eq!(std::fs::read_to_string(dest.join("shot-1.jpg.portrait.json")).unwrap(), "{\"two\":2}");
    }

    #[test]
    fn a_missing_source_leaves_no_partial_file() {
        let temp = Temp::new("missing");
        let dest = temp.path().join("out");
        let items = [item(temp.path().join("gone.jpg"), "gone.jpg", Some("{}"))];

        let (done, errors) = export_originals(&dest, &items);

        assert_eq!(done, 0);
        assert_eq!(errors.len(), 1);
        assert!(names(&dest).is_empty(), "nothing may be published: {:?}", names(&dest));
    }

    #[test]
    fn a_failed_sidecar_rolls_the_copy_back() {
        let temp = Temp::new("sidecar-fail");
        let src = source(temp.path(), "src.jpg", b"image bytes");
        let dest = temp.path().join("out");
        std::fs::create_dir_all(&dest).unwrap();
        // A directory where the sidecar must go makes the sidecar write fail.
        std::fs::create_dir_all(dest.join("shot.jpg.portrait.json")).unwrap();

        let items = [item(src, "shot.jpg", Some("{}"))];
        let (done, errors) = export_originals(&dest, &items);

        assert_eq!(done, 0);
        assert_eq!(errors.len(), 1);
        assert!(!dest.join("shot.jpg").exists(), "the image is rolled back when its edits cannot be written");
        assert!(temporaries(&dest).is_empty());
    }

    #[test]
    fn an_existing_sidecar_is_kept_and_the_new_copy_is_rolled_back() {
        let temp = Temp::new("sidecar-exists");
        let src = source(temp.path(), "src.jpg", b"image bytes");
        let dest = temp.path().join("out");
        std::fs::create_dir_all(&dest).unwrap();
        // A sidecar left by an earlier export: it must survive, never be
        // replaced by this run's JSON.
        std::fs::write(dest.join("shot.jpg.portrait.json"), b"{\"old\":true}").unwrap();

        let items = [item(src, "shot.jpg", Some("{\"new\":true}"))];
        let (done, errors) = export_originals(&dest, &items);

        assert_eq!(done, 0);
        assert_eq!(errors.len(), 1);
        assert!(!dest.join("shot.jpg").exists(), "the image is rolled back");
        assert_eq!(
            std::fs::read_to_string(dest.join("shot.jpg.portrait.json")).unwrap(),
            "{\"old\":true}",
            "an existing sidecar is never overwritten"
        );
        assert!(temporaries(&dest).is_empty());
    }

    #[test]
    fn stale_copy_temporaries_are_swept_and_user_files_kept() {
        let temp = Temp::new("sweep");
        let dest = temp.path().join("out");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join(format!("{COPY_PREFIX}1-1.tmp")), b"partial").unwrap();
        std::fs::write(dest.join("keep.jpg"), b"user file").unwrap();

        let cutoff = SystemTime::now() + Duration::from_secs(1);
        assert_eq!(sweep_stale_before(&dest, cutoff), 1);
        assert!(temporaries(&dest).is_empty());
        assert!(dest.join("keep.jpg").exists());
    }

    #[test]
    fn failures_name_the_file_not_its_folder() {
        let temp = Temp::new("privacy");
        let nested = temp.path().join("Pictures").join("Shoot");
        std::fs::create_dir_all(&nested).unwrap();
        let dest = temp.path().join("out");

        let items = [item(nested.join("DSC_0001.NEF"), "DSC_0001.NEF", None)];
        let (_, errors) = export_originals(&dest, &items);

        assert_eq!(errors.len(), 1);
        assert!(errors[0].starts_with("DSC_0001.NEF: "), "{}", errors[0]);
        assert!(!errors[0].contains("Pictures"), "{}", errors[0]);
        assert!(!errors[0].contains(&temp.path().to_string_lossy().to_string()), "{}", errors[0]);
    }

    #[test]
    fn read_only_originals_are_copied_and_keep_the_attribute() {
        let temp = Temp::new("readonly");
        let src = source(temp.path(), "src.jpg", b"camera card");
        let mut permissions = std::fs::metadata(&src).unwrap().permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&src, permissions).unwrap();
        let dest = temp.path().join("out");

        let (done, errors) = export_originals(&dest, &[item(src, "shot.jpg", None)]);

        assert_eq!((done, errors.len()), (1, 0), "{errors:?}");
        let published = dest.join("shot.jpg");
        assert_eq!(std::fs::read(&published).unwrap(), b"camera card");
        #[cfg(windows)]
        assert!(
            std::fs::metadata(&published).unwrap().permissions().readonly(),
            "the copy keeps the read-only attribute of the original"
        );

        // Let the temporary folder clean itself up on Windows.
        let mut permissions = std::fs::metadata(&published).unwrap().permissions();
        permissions.set_readonly(false);
        let _ = std::fs::set_permissions(&published, permissions);
    }
}
