//! Sequential batch export on top of [`ImageEngine`].
//!
//! One worker thread runs one image at a time and reports through an atomic
//! progress snapshot. The engine keeps its existing per-image contract: it
//! writes `<output>.tmp` and renames it, so a failed image leaves no partial
//! file behind except that temp, which this module deletes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::{ExportJob, ImageEngine, ImageId, ImageInfo, JobStatus, ProcessRecipe};

const POLL: Duration = Duration::from_millis(5);

static NEXT_BATCH: AtomicU64 = AtomicU64::new(1);
static NEXT_JOB: AtomicU64 = AtomicU64::new(1);
static REGISTRY: OnceLock<Mutex<HashMap<BatchId, Arc<BatchState>>>> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BatchId(pub u64);

#[derive(Clone, Debug)]
pub struct BatchJob {
    pub images: Vec<ImageId>,
    pub recipe: ProcessRecipe,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchProgress {
    /// Images exported successfully.
    pub done: usize,
    /// Images that failed; `failures` holds one reason per image.
    pub failed: usize,
    pub total: usize,
    /// The image being exported right now.
    pub current: Option<ImageId>,
    pub cancelled: bool,
    pub finished: bool,
    pub failures: Vec<(ImageId, String)>,
}

struct BatchState {
    progress: Mutex<BatchProgress>,
    cancel: AtomicBool,
}

impl BatchState {
    fn update(&self, f: impl FnOnce(&mut BatchProgress)) {
        f(&mut self.progress.lock().unwrap());
    }

    fn snapshot(&self) -> BatchProgress {
        self.progress.lock().unwrap().clone()
    }
}

/// Starts a batch and returns immediately; work happens on one background
/// thread, so images are processed in order and never in parallel.
pub struct Batch;

impl Batch {
    pub fn start(engine: Arc<dyn ImageEngine>, job: BatchJob) -> BatchId {
        let id = BatchId(NEXT_BATCH.fetch_add(1, Ordering::Relaxed));
        let state = Arc::new(BatchState {
            progress: Mutex::new(BatchProgress { total: job.images.len(), ..Default::default() }),
            cancel: AtomicBool::new(false),
        });
        registry().lock().unwrap().insert(id, state.clone());
        std::thread::spawn(move || run(engine, job, &state));
        id
    }

    pub fn progress(id: BatchId) -> Option<BatchProgress> {
        registry().lock().unwrap().get(&id).map(|state| state.snapshot())
    }

    /// Idempotent: cancelling twice, or after the batch finished, is safe.
    /// Returns whether the batch exists.
    pub fn cancel(id: BatchId) -> bool {
        match registry().lock().unwrap().get(&id) {
            Some(state) => {
                state.cancel.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }
}

fn registry() -> &'static Mutex<HashMap<BatchId, Arc<BatchState>>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

enum Outcome {
    Done,
    Failed(String),
    Cancelled,
}

fn run(engine: Arc<dyn ImageEngine>, job: BatchJob, state: &BatchState) {
    let infos = engine.images();
    for image in job.images.iter().copied() {
        if state.cancel.load(Ordering::SeqCst) {
            break;
        }
        state.update(|p| p.current = Some(image));
        let job_id = NEXT_JOB.fetch_add(1, Ordering::Relaxed);
        let started = engine.start_export(ExportJob {
            job_id,
            image,
            recipe: job.recipe.clone(),
        });
        let outcome = match started {
            Ok(()) => wait(engine.as_ref(), job_id, state),
            Err(err) => Outcome::Failed(err.to_string()),
        };
        match outcome {
            Outcome::Done => state.update(|p| p.done += 1),
            Outcome::Failed(message) => {
                remove_partial(&job.recipe, infos.iter().find(|info| info.id == image));
                state.update(|p| {
                    p.failed += 1;
                    p.failures.push((image, message));
                });
            }
            Outcome::Cancelled => break,
        }
    }
    let cancelled = state.cancel.load(Ordering::SeqCst);
    state.update(|p| {
        p.current = None;
        p.cancelled = cancelled;
        p.finished = true;
    });
}

fn wait(engine: &dyn ImageEngine, job_id: u64, state: &BatchState) -> Outcome {
    loop {
        match engine.job_status(job_id) {
            JobStatus::Done { .. } => return Outcome::Done,
            JobStatus::Failed { message } => return Outcome::Failed(message),
            JobStatus::Queued | JobStatus::Running { .. } => {
                if state.cancel.load(Ordering::SeqCst) {
                    return Outcome::Cancelled;
                }
                std::thread::sleep(POLL);
            }
        }
    }
}

/// The engine writes through `<output>.tmp`; a failed image must not leave it.
fn remove_partial(recipe: &ProcessRecipe, info: Option<&ImageInfo>) {
    let Some(info) = info else { return };
    let dir = recipe
        .destination
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("lumen-output"));
    let name = crate::output_name(recipe, info, info.variant as usize);
    let tmp: PathBuf = dir.join(format!("{name}.{}.tmp", recipe.format.extension()));
    let _ = std::fs::remove_file(tmp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use crate::{AdjustmentSet, AutoResult, AutoTarget, Collection, PreviewFrame, PreviewRequest, RgbaImage};

    struct MockEngine {
        infos: Vec<ImageInfo>,
        fail: Vec<u64>,
        delay: Duration,
        started: Mutex<Vec<u64>>,
        jobs: Mutex<HashMap<u64, (u64, Instant)>>,
    }

    impl MockEngine {
        fn new(count: u64, fail: &[u64], delay_ms: u64) -> Arc<Self> {
            let infos = (1..=count)
                .map(|id| ImageInfo {
                    id: ImageId(id),
                    path: PathBuf::from(format!("C:/mock/{id}.jpg")),
                    file_name: format!("{id}.jpg"),
                    kind: crate::FileKind::Jpeg,
                    width: 100,
                    height: 100,
                    file_size: 1,
                    exif: Default::default(),
                    iptc: Default::default(),
                    keywords: Vec::new(),
                    rating: 0,
                    color_tag: Default::default(),
                    variant: 1,
                    has_adjustments: false,
                    trashed: false,
                    heif: None,
                })
                .collect();
            Arc::new(Self {
                infos,
                fail: fail.to_vec(),
                delay: Duration::from_millis(delay_ms),
                started: Mutex::new(Vec::new()),
                jobs: Mutex::new(HashMap::new()),
            })
        }

        fn started(&self) -> Vec<u64> {
            self.started.lock().unwrap().clone()
        }
    }

    impl ImageEngine for MockEngine {
        fn name(&self) -> String {
            "mock".into()
        }

        fn images(&self) -> Vec<ImageInfo> {
            self.infos.clone()
        }

        fn import(&self, _paths: &[std::path::PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
            Ok(Vec::new())
        }

        fn collections(&self) -> Vec<Collection> {
            Vec::new()
        }

        fn update_info(&self, _info: &ImageInfo) -> anyhow::Result<()> {
            Ok(())
        }

        fn load_adjustments(&self, _image: ImageId) -> AdjustmentSet {
            AdjustmentSet::default()
        }

        fn save_adjustments(&self, _image: ImageId, _adjustments: &AdjustmentSet) -> anyhow::Result<()> {
            Ok(())
        }

        fn new_variant(&self, _image: ImageId, _adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo> {
            anyhow::bail!("not supported")
        }

        fn thumbnail(&self, _image: ImageId, _adjustments: &AdjustmentSet, _max_size: u32) -> anyhow::Result<RgbaImage> {
            Ok(RgbaImage::new(1, 1, vec![0; 4]))
        }

        fn render_preview(&self, _request: &PreviewRequest) -> anyhow::Result<PreviewFrame> {
            anyhow::bail!("not supported")
        }

        fn sample_color(&self, _image: ImageId, _adjustments: &AdjustmentSet, _point: [f32; 2]) -> anyhow::Result<[f32; 3]> {
            anyhow::bail!("not supported")
        }

        fn auto_adjust(&self, _image: ImageId, _current: &AdjustmentSet, _target: AutoTarget) -> anyhow::Result<AutoResult> {
            anyhow::bail!("not supported")
        }

        fn pick_white_balance(&self, _image: ImageId, _point: [f32; 2]) -> anyhow::Result<(f32, f32)> {
            anyhow::bail!("not supported")
        }

        fn compute_ai_mask(&self, _image: ImageId, _kind: crate::AiMaskKind) -> anyhow::Result<String> {
            anyhow::bail!("not supported")
        }

        fn icc_profiles(&self) -> Vec<String> {
            Vec::new()
        }

        fn lens_profiles(&self) -> Vec<String> {
            Vec::new()
        }

        fn styles(&self) -> Vec<crate::StyleInfo> {
            Vec::new()
        }

        fn start_export(&self, job: ExportJob) -> anyhow::Result<()> {
            self.started.lock().unwrap().push(job.image.0);
            self.jobs.lock().unwrap().insert(job.job_id, (job.image.0, Instant::now()));
            Ok(())
        }

        fn job_status(&self, job_id: u64) -> JobStatus {
            let jobs = self.jobs.lock().unwrap();
            let Some((image, at)) = jobs.get(&job_id) else {
                return JobStatus::Failed { message: "unknown job".into() };
            };
            if at.elapsed() < self.delay {
                return JobStatus::Running { progress: 0.5 };
            }
            if self.fail.contains(image) {
                JobStatus::Failed { message: format!("image {image} failed") }
            } else {
                JobStatus::Done { output: PathBuf::from(format!("C:/mock/out/{image}.jpg")) }
            }
        }
    }

    fn job(images: &[u64]) -> BatchJob {
        BatchJob {
            images: images.iter().copied().map(ImageId).collect(),
            recipe: ProcessRecipe::default(),
        }
    }

    fn wait_done(id: BatchId) -> BatchProgress {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let progress = Batch::progress(id).expect("batch is registered");
            if progress.finished {
                return progress;
            }
            assert!(Instant::now() < deadline, "batch did not finish");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn three_images_all_succeed_in_order() {
        let engine = MockEngine::new(3, &[], 1);
        let id = Batch::start(engine.clone(), job(&[1, 2, 3]));
        let progress = wait_done(id);

        assert_eq!((progress.done, progress.failed, progress.total), (3, 0, 3));
        assert!(progress.failures.is_empty());
        assert!(progress.finished && !progress.cancelled);
        assert_eq!(engine.started(), vec![1, 2, 3]);
    }

    #[test]
    fn middle_failure_continues_and_keeps_the_reason() {
        let engine = MockEngine::new(3, &[2], 1);
        let id = Batch::start(engine.clone(), job(&[1, 2, 3]));
        let progress = wait_done(id);

        assert_eq!((progress.done, progress.failed, progress.total), (2, 1, 3));
        assert_eq!(progress.failures.len(), 1);
        assert_eq!(progress.failures[0].0, ImageId(2));
        assert!(progress.failures[0].1.contains("failed"));
        assert_eq!(engine.started(), vec![1, 2, 3], "the image after the failure still ran");
    }

    #[test]
    fn cancel_stops_the_queue() {
        let engine = MockEngine::new(5, &[], 40);
        let id = Batch::start(engine.clone(), job(&[1, 2, 3, 4, 5]));
        assert!(Batch::cancel(id));
        let progress = wait_done(id);

        assert!(progress.cancelled);
        assert!(progress.done < 5, "cancel must stop before the whole queue runs");
        assert!(engine.started().len() < 5);
    }

    #[test]
    fn progress_counts_only_move_forward() {
        let engine = MockEngine::new(4, &[], 15);
        let id = Batch::start(engine, job(&[1, 2, 3, 4]));
        let (mut last_done, mut last_failed, mut last_total) = (0, 0, 0);
        loop {
            let progress = Batch::progress(id).unwrap();
            assert!(progress.done >= last_done);
            assert!(progress.failed >= last_failed);
            assert!(progress.total >= last_total);
            assert!(progress.done + progress.failed <= progress.total);
            last_done = progress.done;
            last_failed = progress.failed;
            last_total = progress.total;
            if progress.finished {
                assert_eq!(last_total, 4);
                break;
            }
            std::thread::sleep(Duration::from_millis(3));
        }
        assert_eq!(last_done, 4);
    }

    #[test]
    fn cancel_twice_is_safe() {
        let engine = MockEngine::new(2, &[], 40);
        let id = Batch::start(engine, job(&[1, 2]));
        assert!(Batch::cancel(id));
        assert!(Batch::cancel(id));
        let progress = wait_done(id);
        assert!(progress.cancelled);
        assert!(Batch::cancel(id), "cancelling a finished batch stays harmless");
        assert!(!Batch::cancel(BatchId(99_999)), "unknown ids are reported, not panicked on");
    }

    #[test]
    fn failure_deletes_the_partial_temp_file() {
        let engine = MockEngine::new(2, &[1], 1);
        let dir = std::env::temp_dir().join(format!("lumen-batch-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut recipe = ProcessRecipe::default();
        recipe.destination = Some(dir.clone());
        let tmp = dir.join("1.jpg.tmp");
        std::fs::write(&tmp, b"partial").unwrap();

        let id = Batch::start(engine, BatchJob { images: vec![ImageId(1)], recipe });
        let progress = wait_done(id);
        assert_eq!(progress.failed, 1);
        assert!(!tmp.exists(), "the failed image's .tmp file must be deleted");
        let _ = std::fs::remove_dir_all(dir);
    }
}
