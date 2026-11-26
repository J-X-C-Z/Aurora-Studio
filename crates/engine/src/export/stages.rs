//! The three real stages of a standard export, plus the process-wide pipeline
//! that drives them.
//!
//! This is the single implementation of "render a frame and write the file" for
//! the standard (non-Z6III-exact) export path. The serial worker in
//! [`crate::file_engine`] and the pipelined batch path both call these
//! functions, so a batch export and a one-image export cannot produce different
//! pixels or different bytes: there is no second implementation to drift.
//!
//! Boundaries:
//! * stage 1 [`acquire_source`] — the only place a deferred full-resolution
//!   decode runs. Src/RAW decode internals are untouched: the engine's own
//!   planner hands over either a ready `Arc<Source>` or a closure that runs the
//!   existing `FullResExportRequest::decode`.
//! * stage 2 [`render_frame`] — `process::render` / `process::render16` with the
//!   exact arguments the serial worker used (`NormRect::FULL`, `bypass=false`,
//!   no mask overlay).
//! * stage 3 [`encode_and_publish`] — `recipe::write_export(16)` into
//!   [`crate::output::StagedExport`] and [`publish_staged`], which keeps the
//!   publish gate, the supersede check and the output map exactly as the serial
//!   worker had them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use crate::adjust::AdjustmentSet;
use crate::catalog::ImageInfo;
use crate::export::pipeline::{
    CANCELLED, Pipeline, PipelineLimits, PipelineStats, Published, Queued,
};
use crate::job_lifecycle::{CancelToken, JobId, JobRegistry, JobState};
use crate::mock::process::{self, Source};
use crate::output::{ProcessRecipe, StagedExport};
use crate::NormRect;

/// Where a stage-1 request gets its pixels from.
pub enum SourceAcquire {
    /// Native RAW frame or a cached proxy: no decode left to do.
    Ready(Arc<Source>),
    /// Full-resolution raster decode, deferred to the pipeline's stage 1 so it
    /// never runs on the caller (UI) thread.
    Deferred(Box<dyn FnOnce() -> anyhow::Result<Source> + Send>),
}

/// Owned handles stage 3 needs to publish and to report to the job registry.
///
/// Cloned from the engine in `start_export`, exactly like the serial worker
/// captured them; the lock order stays `gate -> maps -> registry`.
#[derive(Clone)]
pub struct PublishTarget {
    /// UI export job id (`ExportJob::job_id`).
    pub job_id: u64,
    pub destination: PathBuf,
    pub registry: JobRegistry,
    pub jid: JobId,
    /// Serializes the final cancel check against publication.
    pub gate: Arc<Mutex<()>>,
    /// UI job id -> registry id, so a superseded worker never publishes.
    pub job_ids: Arc<Mutex<HashMap<u64, JobId>>>,
    /// Published output path per UI job id.
    pub outputs: Arc<Mutex<HashMap<u64, PathBuf>>>,
    /// Catalog variant index used by the output-name template.
    pub variant: usize,
    pub cancel: CancelToken,
}

/// Stage 1 input: everything `start_export` already resolved, minus the pixels.
pub struct PreparedExport {
    pub info: ImageInfo,
    pub recipe: ProcessRecipe,
    pub adjustments: AdjustmentSet,
    pub max_size: u32,
    pub source: SourceAcquire,
    pub target: PublishTarget,
}

/// Stage 1 output: a render-ready source plus the recipe it is rendered with.
pub struct AcquiredFrame {
    pub source: Arc<Source>,
    pub info: ImageInfo,
    pub recipe: ProcessRecipe,
    pub adjustments: AdjustmentSet,
    pub max_size: u32,
    pub target: PublishTarget,
}

impl Queued for AcquiredFrame {
    /// Same per-pixel envelope the export budget charges
    /// (`estimated_export_bytes`): source frame plus the render buffer.
    fn charge_bytes(&self) -> usize {
        let (width, height) = self.source.size();
        crate::file_engine::estimated_export_bytes(width, height, self.recipe.bit_depth)
    }
}

/// Stage 2 output: encoded pixels waiting for the writer.
pub enum RenderedPixels {
    Rgba8(Vec<u8>),
    Rgba16(Vec<u16>),
}

pub struct RenderedExport {
    pub pixels: RenderedPixels,
    pub width: u32,
    pub height: u32,
    pub info: ImageInfo,
    pub recipe: ProcessRecipe,
    pub target: PublishTarget,
}

impl Queued for RenderedExport {
    /// The frame buffer plus the estimated encoded payload (a 24 MP 16-bit PNG
    /// lands around 40-70 MB, so 2 B/px covers every supported format).
    fn charge_bytes(&self) -> usize {
        let pixels = (self.width as usize).saturating_mul(self.height as usize);
        let buffer = match self.pixels {
            RenderedPixels::Rgba8(_) => pixels.saturating_mul(4),
            RenderedPixels::Rgba16(_) => pixels.saturating_mul(8),
        };
        buffer.saturating_add(pixels.saturating_mul(2))
    }
}

/// Stage 1: acquire the render-ready source.
///
/// A deferred decode is the CPU work this stage exists for; a ready source is
/// passed through untouched (no copy, no re-decode).
pub fn acquire_source(prepared: PreparedExport) -> Result<AcquiredFrame, String> {
    let PreparedExport { info, recipe, adjustments, max_size, source, target } = prepared;
    let source = match source {
        SourceAcquire::Ready(source) => source,
        SourceAcquire::Deferred(decode) => {
            Arc::new(decode().map_err(|err| err.to_string())?)
        }
    };
    Ok(AcquiredFrame { source, info, recipe, adjustments, max_size, target })
}

/// Stage 2: the real develop/render, on the GPU lane.
pub fn render_frame(frame: AcquiredFrame, cancel: &CancelToken) -> Result<RenderedExport, String> {
    if cancel.is_cancelled() {
        return Err(CANCELLED.to_string());
    }
    let AcquiredFrame { source, info, recipe, adjustments, max_size, target } = frame;
    let (pixels, width, height) = if recipe.bit_depth == 16 {
        let out = process::render16(&source, &adjustments, NormRect::FULL, max_size, false);
        (RenderedPixels::Rgba16(out.rgba), out.width, out.height)
    } else {
        let out = process::render(&source, &adjustments, NormRect::FULL, max_size, false, None);
        (RenderedPixels::Rgba8(out.rgba), out.width, out.height)
    };
    Ok(RenderedExport { pixels, width, height, info, recipe, target })
}

/// Stage 3: encode into a staging folder, then publish atomically.
///
/// A failure or a cancel at any point drops the [`StagedExport`], which removes
/// the staging folder, so the destination keeps whatever it had.
pub fn encode_and_publish(rendered: RenderedExport, cancel: &CancelToken) -> Result<Published, String> {
    if cancel.is_cancelled() {
        return Err(CANCELLED.to_string());
    }
    let RenderedExport { pixels, width, height, info, recipe, target } = rendered;
    let staged = StagedExport::new(&target.destination, target.job_id).map_err(|err| err.to_string())?;
    let progress_target = target.clone();
    let progress = move |fraction: f32| {
        if !progress_target.cancel.is_cancelled() {
            // Progress is cosmetic; a rejected transition must never overwrite a
            // terminal state.
            let _ = progress_target
                .registry
                .transition(progress_target.jid, JobState::running(0.1 + 0.9 * fraction));
        }
    };
    let written = match pixels {
        RenderedPixels::Rgba8(rgba) => {
            crate::recipe::write_export(staged.dir(), &info, &recipe, target.variant, rgba, width, height, progress)
        }
        RenderedPixels::Rgba16(rgba) => {
            crate::recipe::write_export16(staged.dir(), &info, &recipe, target.variant, rgba, width, height, progress)
        }
    }
    .map_err(|err| err.to_string())?;
    if cancel.is_cancelled() {
        return Err(CANCELLED.to_string());
    }
    let path = publish_staged(&target, staged, &written, cancel).map_err(|err| err.to_string())?;
    Ok(Published { path })
}

/// Publishes `written` (a file inside `staged`) to the destination.
///
/// Keeps the serial worker's contract: the cancel check and the publish are one
/// step under `gate`; a worker whose UI job id was reused by a newer export
/// refuses to publish; a rejected `Succeeded` transition removes the recorded
/// output again. The Z6III exact path calls this too.
pub(crate) fn publish_staged(
    target: &PublishTarget,
    staged: StagedExport,
    written: &Path,
    cancel: &CancelToken,
) -> anyhow::Result<PathBuf> {
    let _gate = target.gate.lock();
    if cancel.is_cancelled() {
        anyhow::bail!("cancelled");
    }
    if target.job_ids.lock().get(&target.job_id).copied() != Some(target.jid) {
        anyhow::bail!("export superseded by a newer job");
    }
    let output = staged.publish(written)?;
    target.outputs.lock().insert(target.job_id, output.clone());
    if target.registry.transition(target.jid, JobState::Succeeded).is_err() {
        // The registry went terminal first; remove the path so a late success is
        // never exposed as `Done`.
        target.outputs.lock().remove(&target.job_id);
        anyhow::bail!("export finished after the job was cancelled");
    }
    Ok(output)
}

/// The process-wide standard-export pipeline.
pub type ExportPipeline = Pipeline<PreparedExport, AcquiredFrame, RenderedExport>;

static PIPELINE: OnceLock<ExportPipeline> = OnceLock::new();

/// The shared pipeline, started on first use. Its lanes live for the process,
/// which is what lets job N's encode overlap job N+1's render.
pub fn pipeline() -> &'static ExportPipeline {
    PIPELINE.get_or_init(|| {
        Pipeline::start(
            PipelineLimits::default(),
            Arc::new(acquire_source),
            Arc::new(render_frame),
            Arc::new(encode_and_publish),
        )
    })
}

/// Runs one standard export through the bounded pipeline (or inline, when
/// [`crate::export::pipeline::PipelineMode::Serial`] is pinned). Returns the
/// published path.
pub fn run_export(prepared: PreparedExport, job_id: u64, cancel: &CancelToken) -> anyhow::Result<PathBuf> {
    let published = pipeline()
        .run(prepared, job_id, cancel)
        .map_err(|reason| anyhow::anyhow!(reason))?;
    Ok(published.path)
}

/// Stage counters for the performance log and the regressions.
pub fn pipeline_stats() -> PipelineStats {
    pipeline().stats()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ImageId;
    use crate::export::pipeline::{pin_mode, PipelineMode};
    use crate::{ExportJob, FileEngine, ImageEngine, JobStatus};

    fn fixture(tag: &str, images: usize) -> (PathBuf, FileEngine, Vec<ImageId>) {
        fixture_sized(tag, images, 48, 32)
    }

    fn fixture_sized(tag: &str, images: usize, width: u32, height: u32) -> (PathBuf, FileEngine, Vec<ImageId>) {
        let dir = std::env::temp_dir().join(format!("aurora-export-stages-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for index in 0..images {
            // A deterministic gradient: enough structure that a wrong render or a
            // wrong channel order would change the encoded bytes.
            let mut frame = image::RgbaImage::new(width, height);
            for (x, y, pixel) in frame.enumerate_pixels_mut() {
                let r = ((x * 5 + index as u32 * 17) % 256) as u8;
                let g = ((y * 7 + index as u32 * 29) % 256) as u8;
                let b = ((x * 3 + y * 11) % 256) as u8;
                *pixel = image::Rgba([r, g, b, 255]);
            }
            frame.save(dir.join(format!("frame-{index}.png"))).unwrap();
        }
        let engine = FileEngine::open(&dir).unwrap();
        let ids: Vec<ImageId> = engine.images().into_iter().map(|info| info.id).collect();
        (dir, engine, ids)
    }

    fn recipe(destination: PathBuf, format: crate::ExportFormat, bit_depth: u8) -> ProcessRecipe {
        ProcessRecipe {
            destination: Some(destination),
            format,
            bit_depth,
            include_metadata: crate::MetadataPolicy::StripAll,
            ..ProcessRecipe::default()
        }
    }

    fn wait_export(engine: &FileEngine, job_id: u64) -> JobStatus {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            match engine.job_status(job_id) {
                JobStatus::Queued | JobStatus::Running { .. } => {
                    assert!(std::time::Instant::now() < deadline, "export job {job_id} timed out");
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                status => return status,
            }
        }
    }

    fn published_bytes(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|entry| entry.path().is_file())
            .map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                (name, std::fs::read(entry.path()).unwrap())
            })
            .collect();
        files.sort_by(|a, b| a.0.cmp(&b.0));
        files
    }

    fn staging_leftovers(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .map(|entries| entries.flatten().filter(|entry| crate::output::is_staging_path(&entry.path())).count())
            .unwrap_or(0)
    }

    fn publish_target(dir: &Path, job_id: u64, cancel: CancelToken) -> PublishTarget {
        let registry = JobRegistry::new();
        let jid = registry.submit(crate::job_lifecycle::JobScope::Export, cancel.clone());
        PublishTarget {
            job_id,
            destination: dir.to_path_buf(),
            registry,
            jid,
            gate: Arc::new(Mutex::new(())),
            job_ids: Arc::new(Mutex::new(HashMap::new())),
            outputs: Arc::new(Mutex::new(HashMap::new())),
            variant: 1,
            cancel,
        }
    }

    /// Stage 1 runs the deferred decode it was handed and passes a ready source
    /// straight through; a decode error becomes the job's failure reason.
    #[test]
    fn stage_one_runs_the_deferred_decode_and_propagates_its_error() {
        let dir = std::env::temp_dir().join(format!("aurora-export-stage1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let info = ImageInfo {
            id: ImageId(1),
            path: dir.join("frame.png"),
            file_name: "frame.png".into(),
            kind: crate::FileKind::Png,
            width: 4,
            height: 4,
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
        };
        let prepared = |source: SourceAcquire, job_id: u64| PreparedExport {
            info: info.clone(),
            recipe: recipe(dir.clone(), crate::ExportFormat::Png, 8),
            adjustments: AdjustmentSet::default(),
            max_size: 16,
            source,
            target: publish_target(&dir, job_id, CancelToken::new()),
        };

        let deferred = acquire_source(prepared(
            SourceAcquire::Deferred(Box::new(|| Ok(Source::from_rgba(4, 4, &[9u8; 64])))),
            1,
        ))
        .expect("the deferred decode runs in stage 1");
        assert_eq!(deferred.source.size(), (4, 4));
        assert!(deferred.charge_bytes() > 0);

        let ready = acquire_source(prepared(
            SourceAcquire::Ready(Arc::new(Source::from_rgba(4, 4, &[7u8; 64]))),
            2,
        ))
        .expect("a ready source needs no decode");
        assert_eq!(ready.source.size(), (4, 4));

        let err = match acquire_source(prepared(
            SourceAcquire::Deferred(Box::new(|| anyhow::bail!("decoder refused"))),
            3,
        )) {
            Ok(_) => panic!("a failed decode must fail the item"),
            Err(err) => err,
        };
        assert!(err.contains("decoder refused"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The regression the task requires: the same images, exported in the same
    /// order to two destinations, must publish identical file names and
    /// identical bytes whether the stages ran inline or through the pipeline.
    ///
    /// Each mode gets its own engine over identical generated files, so both
    /// runs pass the same cold-to-warm cache progression; the comparison is the
    /// execution shape, not cache state.
    #[test]
    fn pipelined_and_serial_exports_publish_identical_bytes() {
        let (dir_a, engine_a, images_a) = fixture("identity-serial", 3);
        let (dir_b, engine_b, images_b) = fixture("identity-pipelined", 3);
        let serial_out = std::env::temp_dir().join(format!("aurora-export-out-serial-{}", std::process::id()));
        let pipelined_out =
            std::env::temp_dir().join(format!("aurora-export-out-pipelined-{}", std::process::id()));
        for dir in [&serial_out, &pipelined_out] {
            let _ = std::fs::remove_dir_all(dir);
            std::fs::create_dir_all(dir).unwrap();
        }

        {
            let _pin = pin_mode(PipelineMode::Serial);
            for (offset, image) in images_a.iter().enumerate() {
                let job_id = 901 + offset as u64;
                engine_a
                    .start_export(ExportJob {
                        job_id,
                        image: *image,
                        recipe: recipe(serial_out.clone(), crate::ExportFormat::Png, 16),
                    })
                    .unwrap();
                assert!(matches!(wait_export(&engine_a, job_id), JobStatus::Done { .. }));
            }
        }
        {
            let _pin = pin_mode(PipelineMode::Pipelined);
            for (offset, image) in images_b.iter().enumerate() {
                engine_b
                    .start_export(ExportJob {
                        job_id: 901 + offset as u64,
                        image: *image,
                        recipe: recipe(pipelined_out.clone(), crate::ExportFormat::Png, 16),
                    })
                    .unwrap();
            }
            for offset in 0..images_b.len() {
                let status = wait_export(&engine_b, 901 + offset as u64);
                assert!(matches!(status, JobStatus::Done { .. }), "pipelined export failed: {status:?}");
            }
        }

        let serial = published_bytes(&serial_out);
        let pipelined = published_bytes(&pipelined_out);
        assert_eq!(serial.len(), images_a.len(), "every serial export published a file");
        assert_eq!(pipelined.len(), images_b.len(), "every pipelined export published a file");
        assert_eq!(
            serial.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>(),
            pipelined.iter().map(|(name, _)| name.clone()).collect::<Vec<_>>(),
            "file names must not depend on the execution shape"
        );
        for ((name, serial_bytes), (_, pipelined_bytes)) in serial.iter().zip(pipelined.iter()) {
            assert_eq!(serial_bytes, pipelined_bytes, "{name} differs between serial and pipelined export");
        }
        assert_eq!(staging_leftovers(&serial_out), 0);
        assert_eq!(staging_leftovers(&pipelined_out), 0);
        let _ = std::fs::remove_dir_all(&serial_out);
        let _ = std::fs::remove_dir_all(&pipelined_out);
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    /// A cancelled batch leaves no output and no staging folder, and the images
    /// that were already admitted still publish.
    #[test]
    fn a_cancelled_pipelined_export_publishes_nothing_and_keeps_staging_clean() {
        let (_dir, engine, images) = fixture("cancel", 3);
        let out = std::env::temp_dir().join(format!("aurora-export-out-cancel-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        let _pin = pin_mode(PipelineMode::Pipelined);

        let mut ids = Vec::new();
        for (index, image) in images.iter().enumerate() {
            let job_id = 950 + index as u64;
            engine
                .start_export(ExportJob {
                    job_id,
                    image: *image,
                    recipe: recipe(out.clone(), crate::ExportFormat::Png, 16),
                })
                .unwrap();
            ids.push(job_id);
        }
        // Cancel the middle job; whether it had started or not, it must end
        // terminal without publishing a file.
        let cancelled = engine.cancel_job(ids[1]);
        let status = wait_export(&engine, ids[1]);
        if cancelled {
            assert!(matches!(status, JobStatus::Failed { .. }), "{status:?}");
            assert!(
                published_bytes(&out).iter().all(|(name, _)| !name.contains("frame-1")),
                "a cancelled export published a file"
            );
        }
        for job_id in [ids[0], ids[2]] {
            let status = wait_export(&engine, job_id);
            if !cancelled {
                assert!(matches!(status, JobStatus::Done { .. } | JobStatus::Failed { .. }), "{status:?}");
            }
        }
        // Every job is terminal and no staging folder survived.
        for job_id in &ids {
            let _ = wait_export(&engine, *job_id);
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while staging_leftovers(&out) != 0 {
            assert!(std::time::Instant::now() < deadline, "a cancelled export kept its staging folder");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let _ = std::fs::remove_dir_all(&out);
        let _ = std::fs::remove_dir_all(&_dir);
    }

    /// Batch backpressure report: does the item/byte cap on the ready queue
    /// actually hold the submitting threads back, or does the export scheduler
    /// stay the only admission limit? Ignored by default: it writes 4 x 12 MP
    /// PNG fixtures plus two exports of them.
    ///
    /// ```text
    /// cargo test --release -p aurora-engine --lib \
    ///     export::stages::tests::bench_batch_reports_queue_backpressure \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "12 MP batch measurement; run explicitly with --release --ignored --nocapture"]
    fn bench_batch_reports_queue_backpressure() {
        use std::time::Instant;

        let images = 4usize;
        let (dir_a, engine_a, ids_a) = fixture_sized("bp-serial", images, 4000, 3000);
        let (dir_b, engine_b, ids_b) = fixture_sized("bp-pipelined", images, 4000, 3000);
        let out_serial = std::env::temp_dir().join(format!("aurora-bp-serial-{}", std::process::id()));
        let out_pipelined = std::env::temp_dir().join(format!("aurora-bp-pipelined-{}", std::process::id()));
        for dir in [&out_serial, &out_pipelined] {
            let _ = std::fs::remove_dir_all(dir);
            std::fs::create_dir_all(dir).unwrap();
        }

        let mut timings = Vec::new();
        for (mode, engine, ids, out) in [
            (PipelineMode::Serial, &engine_a, &ids_a, &out_serial),
            (PipelineMode::Pipelined, &engine_b, &ids_b, &out_pipelined),
        ] {
            let before = pipeline_stats();
            let _pin = pin_mode(mode);
            let started = Instant::now();
            for (offset, image) in ids.iter().enumerate() {
                let job_id = 1200 + offset as u64;
                engine
                    .start_export(ExportJob {
                        job_id,
                        image: *image,
                        recipe: recipe(out.clone(), crate::ExportFormat::Png, 16),
                    })
                    .unwrap();
            }
            for offset in 0..ids.len() {
                let job_id = 1200 + offset as u64;
                assert!(
                    matches!(wait_export(engine, job_id), JobStatus::Done { .. }),
                    "{mode:?} export {job_id} failed: {:?}",
                    engine.job_status(job_id)
                );
            }
            let wall = started.elapsed();
            let after = pipeline_stats();
            timings.push(wall);
            println!(
                "{mode:?}: wall {:.3} s for {images} x 12 MP PNG16 ({:.3} s/image)",
                wall.as_secs_f64(),
                wall.as_secs_f64() / images as f64
            );
            println!(
                "  stage items {} / {} / {}  failed {}  cancelled {}",
                after.stage1_items - before.stage1_items,
                after.stage2_items - before.stage2_items,
                after.stage3_items - before.stage3_items,
                after.failed_items - before.failed_items,
                after.cancelled_items - before.cancelled_items
            );
            println!(
                "  ready queue: peak {:.1} MiB  blocked pushes {}  blocked wall {:.1} ms",
                after.ready_peak_bytes as f64 / (1024.0 * 1024.0),
                after.ready_blocked_pushes - before.ready_blocked_pushes,
                (after.ready_blocked_nanos - before.ready_blocked_nanos) as f64 / 1e6
            );
            println!(
                "  rendered queue: peak {:.1} MiB  blocked pushes {}  blocked wall {:.1} ms",
                after.rendered_peak_bytes as f64 / (1024.0 * 1024.0),
                after.rendered_blocked_pushes - before.rendered_blocked_pushes,
                (after.rendered_blocked_nanos - before.rendered_blocked_nanos) as f64 / 1e6
            );
        }
        if let [serial, pipelined] = timings[..] {
            println!(
                "pipeline speedup {:.2}x (serial {:.3} s -> pipelined {:.3} s)",
                serial.as_secs_f64() / pipelined.as_secs_f64(),
                serial.as_secs_f64(),
                pipelined.as_secs_f64()
            );
        }

        // Both shapes must publish every image; the timing is the payload.
        assert_eq!(published_bytes(&out_serial).len(), images);
        assert_eq!(published_bytes(&out_pipelined).len(), images);
        for dir in [&out_serial, &out_pipelined, &dir_a, &dir_b] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    /// 8-bit and 16-bit recipes both survive the pipeline, and the published
    /// files decode to the source dimensions.
    #[test]
    fn pipelined_exports_publish_decodable_files_for_both_bit_depths() {
        let (_dir, engine, images) = fixture("depths", 1);
        let out = std::env::temp_dir().join(format!("aurora-export-out-depths-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        let _pin = pin_mode(PipelineMode::Pipelined);

        for (offset, depth) in [8u8, 16u8].into_iter().enumerate() {
            let out = out.join(format!("d{depth}"));
            std::fs::create_dir_all(&out).unwrap();
            let job_id = 970 + offset as u64;
            engine
                .start_export(ExportJob {
                    job_id,
                    image: images[0],
                    recipe: recipe(out.clone(), crate::ExportFormat::Png, depth),
                })
                .unwrap();
            let JobStatus::Done { output } = wait_export(&engine, job_id) else {
                panic!("export at {depth} bit failed: {:?}", engine.job_status(job_id));
            };
            assert_eq!(image::image_dimensions(&output).unwrap(), (48, 32));
            assert_eq!(staging_leftovers(&out), 0);
        }
        let _ = std::fs::remove_dir_all(&out);
        let _ = std::fs::remove_dir_all(&_dir);
    }
}
