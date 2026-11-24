//! BO-4: destructive fault matrix for export, Catalog and Session writes.
//!
//! Each test names the row of review §8 it covers. Rows that need a fault
//! injected below the public API (full disk, crash leftovers, EXIF step) are
//! unit-tested in `src/output.rs`, `src/batch.rs` and `src/file_engine.rs`;
//! this file exercises the same guarantees through the public engine API.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use aurora_engine::{
    AdjustmentSet, ExportJob, FileEngine, ImageEngine, JobStatus, MetadataPolicy, ProcessRecipe,
    Store,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct Temp(PathBuf);

impl Temp {
    fn new(tag: &str) -> Self {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir()
            .join(format!("aurora-bo4-{tag}-{}-{id}", std::process::id()));
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

fn write_jpeg(path: &Path) {
    write_jpeg_sized(path, 8, 6);
}

fn write_jpeg_sized(path: &Path, width: u32, height: u32) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    image::RgbImage::from_pixel(width, height, image::Rgb([40, 80, 120]))
        .save(path)
        .unwrap();
}

fn recipe(destination: PathBuf) -> ProcessRecipe {
    ProcessRecipe {
        destination: Some(destination),
        include_metadata: MetadataPolicy::StripAll,
        ..ProcessRecipe::default()
    }
}

fn wait(engine: &FileEngine, job_id: u64) -> JobStatus {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match engine.job_status(job_id) {
            JobStatus::Queued | JobStatus::Running { .. } => {
                assert!(Instant::now() < deadline, "export job {job_id} timed out");
                std::thread::sleep(Duration::from_millis(5));
            }
            status => return status,
        }
    }
}

/// §8 row 1: cancelling mid-export never truncates the target.
#[test]
fn row01_cancel_keeps_the_existing_output_bytes() {
    let temp = Temp::new("cancel");
    // Large enough that the render cannot finish before the cancel lands.
    write_jpeg_sized(&temp.path().join("shot.jpg"), 2000, 1500);
    // Open before the marker file exists: a `.jpg` that is not an image would
    // otherwise be picked up by the scan.
    let engine = FileEngine::open(temp.path()).unwrap();
    let image = engine.images()[0].id;
    let out = temp.path().join("Output");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("shot.jpg"), b"previous bytes").unwrap();
    engine
        .start_export(ExportJob { job_id: 1, image, recipe: recipe(out.clone()) })
        .unwrap();
    let cancelled = engine.cancel_job(1);

    // Give the worker time to reach (and skip) its publish point.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let staging: Vec<_> = std::fs::read_dir(&out)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(".aurora-export-"))
            })
            .collect();
        if staging.is_empty() {
            break;
        }
        assert!(Instant::now() < deadline, "a cancelled export left its staging folder behind");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(std::fs::read(out.join("shot.jpg")).unwrap(), b"previous bytes");
    if cancelled {
        let entries: Vec<_> = std::fs::read_dir(&out).unwrap().flatten().collect();
        assert_eq!(entries.len(), 1, "a cancelled export publishes no second file");
    }
}

/// §8 rows 3, 4 and 8: an existing target is never overwritten, a name clash
/// between two photographs resolves to a predictable suffix, and the original
/// file keeps its bytes.
#[test]
fn row03_row04_row08_existing_targets_and_duplicate_names_are_predictable() {
    let temp = Temp::new("collision");
    write_jpeg(&temp.path().join("one").join("shot.jpg"));
    write_jpeg(&temp.path().join("two").join("shot.jpg"));
    let engine = FileEngine::open(temp.path()).unwrap();
    let images = engine.images();
    assert_eq!(images.len(), 2, "two files share one stem");

    let out = temp.path().join("Output");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("shot.jpg"), b"keep me").unwrap();

    let mut written = Vec::new();
    for (index, info) in images.iter().enumerate() {
        let job_id = 10 + index as u64;
        engine
            .start_export(ExportJob { job_id, image: info.id, recipe: recipe(out.clone()) })
            .unwrap();
        let JobStatus::Done { output } = wait(&engine, job_id) else {
            panic!("export {job_id} failed: {:?}", engine.job_status(job_id));
        };
        written.push(output);
    }

    assert_eq!(std::fs::read(out.join("shot.jpg")).unwrap(), b"keep me");
    let names: Vec<String> = written
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(names.contains(&"shot-1.jpg".to_string()), "{names:?}");
    assert!(names.contains(&"shot-2.jpg".to_string()), "{names:?}");
    for path in &written {
        assert!(image::image_dimensions(path).is_ok(), "{} is a complete image", path.display());
    }
}

/// §8 row 5: opening a Capture One catalog read-only and exporting it as a
/// session leaves the source database byte-identical.
#[test]
fn row05_read_only_catalog_open_and_session_export_keep_the_database() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-sample.cocatalogdb");
    if !fixture.is_file() {
        eprintln!("skipped: testdata/c1-sample.cocatalogdb is not present");
        return;
    }
    let temp = Temp::new("catalog-readonly");
    let bundle = temp.path().join("One.cocatalogdb");
    std::fs::create_dir_all(&bundle).unwrap();
    let database = bundle.join("One.cocatalogdb");
    std::fs::copy(&fixture, &database).unwrap();
    let before = std::fs::read(&database).unwrap();

    {
        let engine = FileEngine::open_c1_catalog(&database).unwrap();
        assert!(!engine.images().is_empty());
        let _ = engine.load_adjustments(engine.images()[0].id);
    }
    assert_eq!(std::fs::read(&database).unwrap(), before, "read-only open wrote to the catalog");

    let import = aurora_engine::C1Import::open(&database).unwrap();
    let target = temp.path().join("Bundle.cosessiondb");
    aurora_engine::export_session(&import, &target).unwrap();
    assert!(target.is_dir());
    assert_eq!(
        std::fs::read(&database).unwrap(),
        before,
        "exporting a session wrote to the source catalog"
    );
}

/// §8 row 5 for our own `.portrait` catalog: opening and reading it writes
/// nothing, and a failed edit write keeps the previous edit readable.
#[test]
fn row05_row06_store_reads_and_failed_writes_keep_the_previous_state() {
    let temp = Temp::new("store");
    std::fs::create_dir_all(temp.path().join("shoot")).unwrap();
    std::fs::write(temp.path().join("shoot/img_001.cr3"), b"not a real raw").unwrap();
    let store = Store::open(temp.path()).unwrap();

    let mut first = AdjustmentSet::default();
    first.exposure.exposure = 1.25;
    store.save_adjustments("shoot/img_001.cr3", &first).unwrap();
    let catalog_before = snapshot(&store.root().join("adjust"));

    // Reopen and read: nothing may be rewritten.
    let again = Store::open(temp.path()).unwrap();
    let loaded = again.load_adjustments("shoot/img_001.cr3").unwrap();
    let _ = again.load_index().unwrap();
    assert_eq!(loaded.exposure.exposure, 1.25);
    assert_eq!(snapshot(&again.root().join("adjust")), catalog_before, "reads rewrite files");

    // Block the next write, then prove the earlier edit still loads.
    let blocked = block_writes(&again.root().join("adjust"));
    let mut second = AdjustmentSet::default();
    second.exposure.exposure = 2.0;
    let failed = again.save_adjustments("shoot/img_001.cr3", &second);
    assert!(failed.is_err(), "the blocked write must fail");
    blocked.unblock();
    let kept = again.load_adjustments("shoot/img_001.cr3").unwrap();
    assert_eq!(kept.exposure.exposure, 1.25, "the previous edit survives a failed write");
    assert_eq!(snapshot(&again.root().join("adjust")), catalog_before);
}

fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(snapshot(&path));
            } else {
                out.push((path.clone(), std::fs::read(&path).unwrap()));
            }
        }
    }
    out.sort();
    out
}

/// Makes the next write into `dir` fail, platform by platform.
struct Blocked {
    path: PathBuf,
    #[cfg(unix)]
    mode: u32,
}

impl Blocked {
    fn unblock(self) {
        #[cfg(windows)]
        {
            // The one file in `adjust` was made read-only; renaming over it fails.
            if let Ok(entries) = std::fs::read_dir(&self.path) {
                for entry in entries.flatten() {
                    if let Ok(mut permissions) = entry.metadata().map(|meta| meta.permissions()) {
                        permissions.set_readonly(false);
                        let _ = std::fs::set_permissions(entry.path(), permissions);
                    }
                }
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&self.path, std::fs::Permissions::from_mode(self.mode));
        }
    }
}

fn block_writes(dir: &Path) -> Blocked {
    #[cfg(windows)]
    {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let mut permissions = entry.metadata().unwrap().permissions();
            permissions.set_readonly(true);
            std::fs::set_permissions(entry.path(), permissions).unwrap();
        }
        Blocked { path: dir.to_path_buf() }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(dir).unwrap().permissions().mode();
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        Blocked { path: dir.to_path_buf(), mode }
    }
}
