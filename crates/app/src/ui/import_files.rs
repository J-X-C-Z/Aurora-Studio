//! Import file operations, independent of the UI and tested only with synthetic files.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub fn apply_naming(format: &str, original: &Path, counter: usize, date: &str) -> String {
    let stem = original.file_stem().unwrap_or_default().to_string_lossy();
    safe_component(
        &format
            .replace("[Image Name]", &stem)
            .replace("[Counter]", &format!("{counter:04}"))
            .replace("[Date]", date),
    )
}

fn safe_component(value: &str) -> String {
    let mut name = String::new();
    let mut units = 0;
    for c in value.chars() {
        let c = if c.is_control() || "<>:\"/\\|?*".contains(c) {
            '_'
        } else {
            c
        };
        // Leave room for the extension, sidecar and collision suffix on Windows.
        units += c.len_utf16();
        if units > 180 {
            break;
        }
        name.push(c);
    }
    let mut name = name.trim_matches([' ', '.']).to_owned();
    if name.is_empty() {
        name = "image".into();
    }
    let device = name
        .split('.')
        .next()
        .unwrap_or_default()
        .trim_end()
        .to_ascii_uppercase();
    if matches!(
        device.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$" | "CONIN$" | "CONOUT$"
    ) || ["COM", "LPT"].iter().any(|p| {
        device.strip_prefix(p).is_some_and(|n| {
            matches!(
                n,
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    }) {
        name.insert(0, '_');
    }
    name
}

pub fn is_reparse(meta: &fs::Metadata) -> bool {
    if meta.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if meta.file_attributes() & 0x400 != 0 {
            return true;
        }
    }
    false
}

pub fn scan(
    dir: &Path,
    recursive: bool,
    generation: &AtomicU64,
    expected: u64,
    is_image: fn(&Path) -> bool,
) -> Vec<(PathBuf, u64)> {
    let mut out = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        if generation.load(Ordering::Relaxed) != expected {
            break;
        }
        let Ok(meta) = fs::symlink_metadata(&dir) else {
            continue;
        };
        if is_reparse(&meta) {
            continue;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            if generation.load(Ordering::Relaxed) != expected {
                return Vec::new();
            }
            let path = entry.path();
            let Ok(meta) = fs::symlink_metadata(&path) else {
                continue;
            };
            if is_reparse(&meta) {
                continue;
            }
            if meta.is_dir() {
                if recursive && !entry.file_name().to_string_lossy().starts_with('.') {
                    pending.push(path);
                }
            } else if meta.is_file() && is_image(&path) {
                out.push((path, meta.len()));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn path_key(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut normalized = PathBuf::new();
    for part in absolute.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(part.as_os_str()),
        }
    }
    let canonical = fs::canonicalize(&normalized).unwrap_or_else(|_| {
        normalized
            .parent()
            .and_then(|p| fs::canonicalize(p).ok())
            .map(|p| p.join(normalized.file_name().unwrap_or_default()))
            .unwrap_or(normalized)
    });
    #[cfg(windows)]
    {
        PathBuf::from(canonical.to_string_lossy().to_lowercase())
    }
    #[cfg(not(windows))]
    {
        canonical
    }
}

pub fn same_directory(left: &Path, right: &Path) -> bool {
    path_key(left) == path_key(right)
}

fn regular_file(path: &Path) -> io::Result<()> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || is_reparse(&meta) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "源文件不是普通文件或是链接",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum SidecarKind {
    Stem,
    Appended,
}

fn appended_xmp(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".xmp");
    PathBuf::from(name)
}

fn sidecars(src: &Path) -> io::Result<Vec<(PathBuf, SidecarKind)>> {
    let stem = src
        .with_extension("xmp")
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let appended = appended_xmp(src)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let mut found = Vec::new();
    for entry in fs::read_dir(src.parent().unwrap_or_else(|| Path::new(".")))? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let kind = if name.eq_ignore_ascii_case(&stem) {
            SidecarKind::Stem
        } else if name.eq_ignore_ascii_case(&appended) {
            SidecarKind::Appended
        } else {
            continue;
        };
        regular_file(&entry.path())?;
        if found.iter().any(|(_, k)| {
            matches!(
                (k, kind),
                (SidecarKind::Stem, SidecarKind::Stem)
                    | (SidecarKind::Appended, SidecarKind::Appended)
            )
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "存在多个同名 XMP，无法安全复制",
            ));
        }
        found.push((entry.path(), kind));
    }
    Ok(found)
}

struct CopyPlan {
    image: PathBuf,
    files: Vec<(PathBuf, PathBuf)>,
}

fn collision(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

fn plan_copy(
    src: &Path,
    target: &Path,
    xmp: &[(PathBuf, SidecarKind)],
    skip: bool,
    reserved: &mut HashSet<PathBuf>,
) -> io::Result<Option<CopyPlan>> {
    if path_key(src) == path_key(target) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "不能复制到源文件自身",
        ));
    }
    let stem = target.file_stem().unwrap_or_default().to_string_lossy();
    let ext = target.extension().unwrap_or_default().to_string_lossy();
    for counter in 0..10_000 {
        let image = if counter == 0 {
            target.to_owned()
        } else {
            target.with_file_name(format!("{stem}_{counter:04}.{ext}"))
        };
        // Even without a source sidecar, never attach an unrelated destination XMP.
        let claims = [
            image.clone(),
            image.with_extension("xmp"),
            appended_xmp(&image),
        ];
        let mut occupied = false;
        for path in &claims {
            occupied |= reserved.contains(&path_key(path)) || collision(path)?;
        }
        if occupied {
            if skip {
                return Ok(None);
            }
            continue;
        }
        reserved.extend(claims.iter().map(|p| path_key(p)));
        let mut files = vec![(src.to_owned(), image.clone())];
        files.extend(xmp.iter().map(|(p, kind)| {
            (
                p.clone(),
                match kind {
                    SidecarKind::Stem => image.with_extension("xmp"),
                    SidecarKind::Appended => appended_xmp(&image),
                },
            )
        }));
        return Ok(Some(CopyPlan { image, files }));
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "无法分配不重名的文件名",
    ))
}

fn execute(plans: &[CopyPlan]) -> io::Result<()> {
    // Reserve all outputs before copying; create_new closes the overwrite race.
    // Never unlink on failure: a user may have replaced a path in the meantime.
    let mut opened = Vec::new();
    for plan in plans {
        for (src, target) in &plan.files {
            let input = File::open(src)?;
            let output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(target)
                .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", target.display())))?;
            opened.push((input, output, target));
        }
    }
    for (mut input, mut output, target) in opened {
        io::copy(&mut input, &mut output)
            .and_then(|_| output.sync_all())
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", target.display())))?;
    }
    Ok(())
}

pub struct ImportOptions {
    pub dest: Option<PathBuf>,
    pub backup: Option<PathBuf>,
    pub naming: Option<String>,
    pub date: String,
    pub skip_duplicates: bool,
    pub known_sources: Vec<PathBuf>,
}

#[derive(Default)]
pub struct ImportResult {
    pub paths: Vec<PathBuf>,
    pub skipped: usize,
    pub errors: Vec<String>,
    pub cancelled: usize,
}

pub fn import_files(
    chosen: &[PathBuf],
    options: &ImportOptions,
    cancel: &AtomicBool,
) -> ImportResult {
    let mut result = ImportResult::default();
    let known: HashSet<_> = options.known_sources.iter().map(|p| path_key(p)).collect();
    let mut seen = HashSet::new();
    for (i, src) in chosen.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            result.cancelled = chosen.len() - i;
            break;
        }
        let key = path_key(src);
        if !seen.insert(key.clone()) || (options.skip_duplicates && known.contains(&key)) {
            result.skipped += 1;
            continue;
        }
        let operation = || -> io::Result<Option<PathBuf>> {
            regular_file(src)?;
            if let Some(parent) = src.parent() {
                for dir in options.dest.iter().chain(options.backup.iter()) {
                    if same_directory(parent, dir) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "目标或备份文件夹与源文件夹相同，请使用就地添加或选择其他文件夹",
                        ));
                    }
                }
            }
            if options.dest.is_none() && options.backup.is_none() {
                return Ok(Some(src.clone()));
            }
            let xmp = sidecars(src)?;
            let ext = safe_component(&src.extension().unwrap_or_default().to_string_lossy());
            let original_name = format!(
                "{}.{ext}",
                safe_component(&src.file_stem().unwrap_or_default().to_string_lossy())
            );
            let name = match &options.naming {
                Some(fmt) => format!("{}.{ext}", apply_naming(fmt, src, i + 1, &options.date)),
                None => original_name.clone(),
            };
            let mut plans = Vec::new();
            let mut reserved = HashSet::new();
            // Check every collision before creating directories or writing any output.
            let mut final_path = src.clone();
            if let Some(dir) = &options.dest {
                let Some(plan) = plan_copy(
                    src,
                    &dir.join(name),
                    &xmp,
                    options.skip_duplicates,
                    &mut reserved,
                )?
                else {
                    return Ok(None);
                };
                final_path = plan.image.clone();
                plans.push(plan);
            }
            if let Some(dir) = &options.backup {
                let Some(plan) = plan_copy(
                    src,
                    &dir.join(original_name),
                    &xmp,
                    options.skip_duplicates,
                    &mut reserved,
                )?
                else {
                    return Ok(None);
                };
                plans.push(plan);
            }
            for plan in &plans {
                if let Some(dir) = plan.image.parent() {
                    fs::create_dir_all(dir)?;
                }
            }
            execute(&plans)?;
            Ok(Some(final_path))
        };
        match operation() {
            Ok(Some(path)) => result.paths.push(path),
            Ok(None) => result.skipped += 1,
            Err(e) => result.errors.push(format!(
                "{}: {e}（若已创建部分副本，予以保留；不会删除或覆盖文件）",
                src.display()
            )),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let base = std::env::temp_dir();
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = base.join(format!(
                "lumen-import-{}-{nonce}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn file(&self, name: &str, contents: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, contents).unwrap();
            path
        }
        fn options(&self) -> ImportOptions {
            ImportOptions {
                dest: Some(self.0.join("out")),
                backup: None,
                naming: None,
                date: "2026-11-13".into(),
                skip_duplicates: true,
                known_sources: Vec::new(),
            }
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn names_are_one_safe_windows_component() {
        for name in [
            "../..\\evil:C?*<>|\"\0",
            ".",
            "..",
            "  ",
            "CON",
            "nul.txt",
            "LPT1",
            "com¹",
            "CONIN$",
            "name. ",
        ] {
            let value = apply_naming(name, Path::new("photo.nef"), 1, "2026-11-13");
            assert!(!value.is_empty());
            assert!(
                !value
                    .chars()
                    .any(|c| c.is_control() || "<>:\"/\\|?*".contains(c))
            );
            assert!(!value.ends_with([' ', '.']));
            assert_eq!(Path::new(&value).components().count(), 1);
        }
        assert_eq!(
            apply_naming("NUL.txt", Path::new("x.nef"), 1, "date"),
            "_NUL.txt"
        );
        assert_eq!(
            apply_naming(
                "[Date]_[Counter]_[Image Name]",
                Path::new("猫.nef"),
                12,
                "2026-11-13"
            ),
            "2026-11-13_0012_猫"
        );
        assert!(safe_component(&"猫".repeat(300)).encode_utf16().count() <= 180);
    }

    #[test]
    fn collision_skips_without_touching_image_or_xmp_or_backup() {
        let t = Temp::new();
        let src = t.file("src/a.nef", b"new raw");
        t.file("src/a.xmp", b"new metadata");
        let existing = t.file("out/a.nef", b"user raw");
        let sidecar = t.file("out/a.xmp", b"user metadata");
        let mut options = t.options();
        options.backup = Some(t.0.join("backup"));
        let r = import_files(&[src], &options, &AtomicBool::new(false));
        assert_eq!((r.paths.len(), r.skipped, r.errors.len()), (0, 1, 0));
        assert_eq!(fs::read(existing).unwrap(), b"user raw");
        assert_eq!(fs::read(sidecar).unwrap(), b"user metadata");
        assert!(!t.0.join("backup").exists());
    }

    #[test]
    fn sidecar_only_and_backup_collisions_also_skip() {
        let t = Temp::new();
        let src = t.file("src/a.nef", b"raw");
        t.file("out/a.xmp", b"unrelated");
        let r = import_files(&[src.clone()], &t.options(), &AtomicBool::new(false));
        assert_eq!(r.skipped, 1);
        assert!(!t.0.join("out/a.nef").exists());
        let mut options = t.options();
        options.dest = Some(t.0.join("fresh"));
        options.backup = Some(t.0.join("backup"));
        t.file("backup/a.nef", b"old backup");
        let r = import_files(&[src], &options, &AtomicBool::new(false));
        assert_eq!(r.skipped, 1);
        assert!(!t.0.join("fresh").exists());
        assert_eq!(fs::read(t.0.join("backup/a.nef")).unwrap(), b"old backup");
    }

    #[test]
    fn renamed_xmps_and_backup_are_byte_exact() {
        let t = Temp::new();
        let src = t.file("src/a.NEF", b"raw bytes");
        t.file("src/a.XMP", b"<xmp>vendor data</xmp>");
        t.file("src/a.NEF.XMP", b"appended metadata");
        let mut options = t.options();
        options.naming = Some("[Date]_[Counter]".into());
        options.backup = Some(t.0.join("backup"));
        let r = import_files(&[src.clone()], &options, &AtomicBool::new(false));
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert_eq!(r.paths, [t.0.join("out/2026-11-13_0001.NEF")]);
        for (name, contents) in [
            ("2026-11-13_0001.NEF", &b"raw bytes"[..]),
            ("2026-11-13_0001.xmp", &b"<xmp>vendor data</xmp>"[..]),
            ("2026-11-13_0001.NEF.xmp", &b"appended metadata"[..]),
        ] {
            assert_eq!(fs::read(t.0.join("out").join(name)).unwrap(), contents);
        }
        assert_eq!(
            fs::read(t.0.join("backup/a.xmp")).unwrap(),
            b"<xmp>vendor data</xmp>"
        );
        assert_eq!(
            fs::read(t.0.join("backup/a.NEF.xmp")).unwrap(),
            b"appended metadata"
        );
        assert_eq!(fs::read(src).unwrap(), b"raw bytes");
    }

    #[test]
    fn unchecked_skip_numbers_collisions_without_overwriting() {
        let t = Temp::new();
        let a = t.file("src/a.nef", b"one");
        let b = t.file("other/a.nef", b"two");
        t.file("out/a.nef", b"existing");
        t.file("src/a.xmp", b"one xmp");
        t.file("other/a.xmp", b"two xmp");
        let mut options = t.options();
        options.skip_duplicates = false;
        let r = import_files(&[a, b], &options, &AtomicBool::new(false));
        assert_eq!((r.paths.len(), r.skipped, r.errors.len()), (2, 0, 0));
        assert_eq!(fs::read(t.0.join("out/a.nef")).unwrap(), b"existing");
        assert_eq!(fs::read(t.0.join("out/a_0001.xmp")).unwrap(), b"one xmp");
        assert_eq!(fs::read(t.0.join("out/a_0002.xmp")).unwrap(), b"two xmp");
    }

    #[test]
    fn self_copy_is_an_error_even_when_skip_is_off() {
        let t = Temp::new();
        let src = t.file("src/a.nef", b"original");
        for skip in [true, false] {
            let mut options = t.options();
            options.dest = Some(t.0.join("src"));
            options.skip_duplicates = skip;
            let r = import_files(&[src.clone()], &options, &AtomicBool::new(false));
            assert_eq!((r.paths.len(), r.skipped, r.errors.len()), (0, 0, 1));
        }
        assert_eq!(fs::read(src).unwrap(), b"original");
    }

    #[test]
    fn known_sources_and_batch_aliases_are_deduplicated() {
        let t = Temp::new();
        let a = t.file("src/a.nef", b"a");
        let b = t.file("src/b.nef", b"b");
        let alias = t.0.join("src/./b.nef");
        let mut options = t.options();
        options.dest = None;
        options.known_sources.push(a.clone());
        let r = import_files(
            &[a.clone(), b.clone(), alias],
            &options,
            &AtomicBool::new(false),
        );
        assert_eq!((r.paths, r.skipped, r.errors.len()), (vec![b], 2, 0));
        options.skip_duplicates = false;
        let r = import_files(&[a.clone(), a], &options, &AtomicBool::new(false));
        assert_eq!((r.paths.len(), r.skipped), (1, 1));
    }

    #[test]
    fn same_source_folder_is_rejected_even_with_renaming_or_backup() {
        let t = Temp::new();
        let src = t.file("src/a.nef", b"original");
        assert!(same_directory(&t.0.join("src"), &t.0.join("src/./")));
        let mut options = t.options();
        options.dest = Some(t.0.join("src"));
        options.naming = Some("renamed_[Counter]".into());
        options.skip_duplicates = false;
        let r = import_files(&[src.clone()], &options, &AtomicBool::new(false));
        assert_eq!((r.paths.len(), r.errors.len()), (0, 1));
        assert!(!t.0.join("src/renamed_0001.nef").exists());
        options.dest = None;
        options.backup = Some(t.0.join("src/./"));
        let r = import_files(&[src.clone()], &options, &AtomicBool::new(false));
        assert_eq!((r.paths.len(), r.errors.len()), (0, 1));
        assert_eq!(fs::read(src).unwrap(), b"original");
    }

    #[test]
    fn cancellation_leaves_sources_and_existing_outputs_untouched() {
        let t = Temp::new();
        let src = t.file("src/a.nef", b"original");
        let r = import_files(&[src.clone()], &t.options(), &AtomicBool::new(true));
        assert_eq!(
            (r.paths.len(), r.skipped, r.errors.len(), r.cancelled),
            (0, 0, 0, 1)
        );
        assert!(!t.0.join("out").exists());
        assert_eq!(fs::read(src).unwrap(), b"original");
    }

    #[test]
    fn atomic_reservation_refuses_a_late_collision() {
        let t = Temp::new();
        let src = t.file("src/a.nef", b"raw");
        let options = t.options();
        fs::create_dir_all(options.dest.as_ref().unwrap()).unwrap();
        let target = t.0.join("out/a.nef");
        let plan = plan_copy(&src, &target, &[], true, &mut HashSet::new())
            .unwrap()
            .unwrap();
        fs::write(&target, b"arrived after planning").unwrap();
        assert!(execute(&[plan]).is_err());
        assert_eq!(fs::read(target).unwrap(), b"arrived after planning");
    }

    #[test]
    fn late_sidecar_collision_does_not_delete_or_overwrite_user_files() {
        let t = Temp::new();
        let src = t.file("src/a.nef", b"original raw");
        let xmp = t.file("src/a.xmp", b"original xmp");
        let target = t.0.join("out/a.nef");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        let plan = plan_copy(
            &src,
            &target,
            &sidecars(&src).unwrap(),
            true,
            &mut HashSet::new(),
        )
        .unwrap()
        .unwrap();
        let user_xmp = t.file("out/a.xmp", b"new user metadata");
        assert!(execute(&[plan]).is_err());
        assert_eq!(fs::read(user_xmp).unwrap(), b"new user metadata");
        assert_eq!(fs::read(src).unwrap(), b"original raw");
        assert_eq!(fs::read(xmp).unwrap(), b"original xmp");
        // Our reserved output is retained, never confused with a user replacement.
        assert!(target.exists());
    }

    #[test]
    fn one_failure_does_not_hide_later_success_or_skip_counts() {
        let t = Temp::new();
        let missing = t.0.join("src/missing.nef");
        let skip = t.file("src/skip.nef", b"skip");
        t.file("out/skip.nef", b"existing");
        let success = t.file("src/ok.nef", b"ok");
        let r = import_files(
            &[missing, skip, success],
            &t.options(),
            &AtomicBool::new(false),
        );
        assert_eq!(
            (r.paths.len(), r.skipped, r.errors.len(), r.cancelled),
            (1, 1, 1, 0)
        );
        assert_eq!(fs::read(&r.paths[0]).unwrap(), b"ok");
    }

    #[test]
    fn stale_scan_returns_no_files() {
        let t = Temp::new();
        t.file("src/a.nef", b"a");
        assert!(scan(&t.0, true, &AtomicU64::new(2), 1, |_| true).is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn scan_skips_windows_junction_loops() {
        use std::os::windows::process::CommandExt;
        let t = Temp::new();
        let src = t.file("src/a.nef", b"a");
        let junction = t.0.join("src").join("loop");
        let output = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(&junction)
            .arg(t.0.join("src"))
            .creation_flags(0x08000000)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(is_reparse(&fs::symlink_metadata(&junction).unwrap()));
        let files = scan(&t.0, true, &AtomicU64::new(1), 1, |_| true);
        assert_eq!(files, [(src, 1)]);
        assert!(scan(&junction, true, &AtomicU64::new(1), 1, |_| true).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn scan_skips_symlink_loops_and_linked_files() {
        use std::os::unix::fs::symlink;
        let t = Temp::new();
        t.file("src/a.nef", b"a");
        symlink(t.0.join("src"), t.0.join("src/loop")).unwrap();
        symlink(t.0.join("src/a.nef"), t.0.join("src/link.nef")).unwrap();
        let files = scan(&t.0, true, &AtomicU64::new(1), 1, |_| true);
        assert_eq!(files.len(), 1);
    }
}
