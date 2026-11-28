//! A5 Architecture Regression Integration Test Suite.
//!
//! Exercises the public `aurora_engine::*` contracts end-to-end:
//! 1. Legacy save-file & unknown extension round-trip (`AdjustmentSet`).
//! 2. Persistent document vs. per-job `RuntimeInputs` isolation.
//! 3. Concurrent `FileEngine` preview/reads with bounded `save_adjustments_queued`
//!    + `flush_saves`, plus bounded flush timeout / close reporting on `SaveQueue`.
//! 4. Cancellation-safe atomic export publishing (`AtomicPublisher`).
//! 5. Explicit `ExportPlan` pre-flight rejection of invalid bit depth, zero
//!    dimensions, invalid destination directories, and unsupported formats.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use aurora_engine::document::{
    document_fingerprint, DocumentRevision, DocumentSnapshot, RenderGeneration, RenderIntent,
    RenderRequest, RuntimeInputs, SourceId,
};
use aurora_engine::export::{
    run_staged_export, AtomicPublisher, ExportPlan, ExportPlanError, ExportStage, PublishError,
    StageStatus,
};
use aurora_engine::job_lifecycle::CancelToken;
use aurora_engine::{
    write_export, AdjustmentSet, ColorTag, Exif, ExportFormat, FileEngine, FileKind, Geometry,
    ImageEngine, ImageId, ImageInfo, Iptc, NormRect, PathState, PreviewRequest, ProcessRecipe,
    ProducerInputs, QueuedSaveOutcome, RenderQuality, SaveQueue, SaveWriter, ScaleMode,
    SmileProducerInputs, SubmitOutcome, WhiteBalanceMode,
};

static SUITE_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempWorkspace(PathBuf);

impl TempWorkspace {
    fn new(name: &str) -> Self {
        let id = SUITE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir()
            .join(format!("aurora-arch-reg-{name}-{}-{id}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp workspace");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Writes a small valid PNG into `dir` using the public exporter so no extra
/// test dependency is needed.
fn write_test_png(dir: &Path, stem: &str, width: u32, height: u32) -> PathBuf {
    let mut recipe = ProcessRecipe::default();
    recipe.format = ExportFormat::Png;
    recipe.bit_depth = 8;
    recipe.file_name_format = stem.to_string();

    let info = ImageInfo {
        id: ImageId(1),
        path: dir.join(format!("{stem}.png")),
        file_name: format!("{stem}.png"),
        kind: FileKind::Png,
        width,
        height,
        file_size: 0,
        exif: Exif::default(),
        iptc: Iptc::default(),
        keywords: Vec::new(),
        rating: 0,
        color_tag: ColorTag::None,
        variant: 1,
        has_adjustments: false,
        trashed: false,
        heif: None,
    };

    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            rgba[idx] = ((x * 7) % 256) as u8;
            rgba[idx + 1] = ((y * 11) % 256) as u8;
            rgba[idx + 2] = (((x + y) * 5) % 256) as u8;
            rgba[idx + 3] = 255;
        }
    }

    write_export(dir, &info, &recipe, 1, rgba, width, height, |_| {})
        .expect("write synthetic test PNG")
}

// ---------------------------------------------------------------------------
// 1. Legacy archive & unknown extension field round-trip
// ---------------------------------------------------------------------------

#[test]
fn a5_legacy_document_and_unknown_extensions_round_trip() {
    // Simulate a sparse legacy `.portrait` document that omits most sections,
    // carries `extra` vendor/portrait keys, and includes unknown top-level
    // extension fields written by a newer or third-party tool.
    let legacy_json = r#"{
        "exposure": { "exposure": 0.75, "contrast": 12.0 },
        "extra": {
            "pc:21093": 35.5,
            "BlemishRemovalAmount": 42.0
        },
        "vendor_future_flag": true,
        "vendor_nested": { "version": 3, "tags": ["portrait", "studio"] }
    }"#;

    let loaded: AdjustmentSet =
        serde_json::from_str(legacy_json).expect("legacy JSON must deserialize cleanly");

    // Explicit fields keep their values.
    assert_eq!(loaded.exposure.exposure, 0.75);
    assert_eq!(loaded.exposure.contrast, 12.0);
    // Omitted fields keep default semantics.
    assert_eq!(loaded.exposure.brightness, 0.0);
    assert_eq!(loaded.white_balance.mode, WhiteBalanceMode::Shot);
    assert_eq!(loaded.white_balance.kelvin, 5600.0);
    // `extra` keys survive.
    assert_eq!(loaded.extra.get("pc:21093"), Some(&35.5));
    assert_eq!(loaded.extra.get("BlemishRemovalAmount"), Some(&42.0));
    // Unknown top-level extension keys survive verbatim.
    assert_eq!(loaded.extensions.get("vendor_future_flag"), Some(&serde_json::json!(true)));
    assert_eq!(
        loaded.extensions.get("vendor_nested"),
        Some(&serde_json::json!({ "version": 3, "tags": ["portrait", "studio"] }))
    );

    // Round-trip serialization preserves both typed fields and unknown extensions.
    let reserialized = serde_json::to_string(&loaded).expect("reserialize document");
    let reloaded: AdjustmentSet =
        serde_json::from_str(&reserialized).expect("deserialize round-tripped JSON");

    assert_eq!(loaded, reloaded);
    assert_eq!(document_fingerprint(&loaded), document_fingerprint(&reloaded));
    assert_eq!(reloaded.extra.get("pc:21093"), Some(&35.5));
    assert_eq!(reloaded.extensions.get("vendor_future_flag"), Some(&serde_json::json!(true)));
}

// ---------------------------------------------------------------------------
// 2. Persistent document vs. RuntimeInputs isolation
// ---------------------------------------------------------------------------

#[test]
fn a5_persistent_document_is_isolated_from_runtime_inputs() {
    const SENTINEL_A: &str = "RUNTIME_PRODUCER_SENTINEL_ALPHA";
    const SENTINEL_B: &str = "RUNTIME_PRODUCER_SENTINEL_BETA";

    let mut adj = AdjustmentSet::default();
    adj.exposure.exposure = 1.25;
    adj.extra.insert("pc:30025".to_string(), 50.0);

    let producer_a = ProducerInputs {
        smile: Some(SmileProducerInputs {
            ticket: 101,
            material: Some(SENTINEL_A.to_string()),
            ..SmileProducerInputs::default()
        }),
        ..ProducerInputs::default()
    };
    let producer_b = ProducerInputs {
        smile: Some(SmileProducerInputs {
            ticket: 202,
            material: Some(SENTINEL_B.to_string()),
            ..SmileProducerInputs::default()
        }),
        ..ProducerInputs::default()
    };

    let runtime_a = RuntimeInputs::new(Some(Arc::new(producer_a)));
    let runtime_b = RuntimeInputs::new(Some(Arc::new(producer_b)));

    let snap = DocumentSnapshot::from_adjustments(adj.clone(), DocumentRevision::new(1));
    let req_a = RenderRequest {
        source: SourceId::new("catalog:img-1"),
        document: snap.clone(),
        region: NormRect::FULL,
        generation: RenderGeneration::new(1),
        intent: RenderIntent::Final,
        runtime: runtime_a.clone(),
    };
    let req_b = RenderRequest {
        source: SourceId::new("catalog:img-1"),
        document: snap,
        region: NormRect::FULL,
        generation: RenderGeneration::new(2),
        intent: RenderIntent::Final,
        runtime: runtime_b.clone(),
    };

    // Identical persistent document => identical document equality & fingerprint.
    assert_eq!(req_a.document.adjustments(), req_b.document.adjustments());
    assert_eq!(req_a.document.document_fingerprint(), req_b.document.document_fingerprint());

    // Different runtime inputs => distinct runtime fingerprints.
    assert!(req_a.runtime.fingerprint().is_some());
    assert!(req_b.runtime.fingerprint().is_some());
    assert_ne!(req_a.runtime.fingerprint(), req_b.runtime.fingerprint());

    // Persisted JSON is never polluted by RuntimeInputs.
    let json = serde_json::to_string(req_a.document.adjustments()).expect("serialize document");
    assert!(!json.contains("portrait_producer_inputs"), "{json}");
    assert!(!json.contains(SENTINEL_A), "{json}");
    assert!(!json.contains(SENTINEL_B), "{json}");
}

// ---------------------------------------------------------------------------
// 3. Concurrent FileEngine preview + save_adjustments_queued + flush_saves
//    and bounded close/timeout reporting
// ---------------------------------------------------------------------------

#[test]
fn a5_concurrent_file_engine_preview_and_queued_save_flush() {
    let ws = TempWorkspace::new("file-engine-conc");
    let _img1 = write_test_png(ws.path(), "shot_a", 64, 48);
    let _img2 = write_test_png(ws.path(), "shot_b", 64, 48);

    let engine = Arc::new(FileEngine::open(ws.path()).expect("open FileEngine"));
    let images = engine.images();
    assert_eq!(images.len(), 2, "expected 2 images in workspace");
    let id_a = images[0].id;
    let id_b = images[1].id;

    // Reader thread: repeatedly renders previews while saves are queued.
    let reader_engine = Arc::clone(&engine);
    let reader = thread::spawn(move || {
        for generation_id in 1..=6u64 {
            let target = if generation_id % 2 == 0 { id_a } else { id_b };
            let req = PreviewRequest {
                image: target,
                adjustments: reader_engine.load_adjustments(target),
                region: NormRect::FULL,
                max_size: 64,
                quality: RenderQuality::Draft,
                bypass: false,
                mask_overlay: None,
                generation: generation_id,
                display_profile: None,
                proof_profile: None,
                proof_paper_white: 1.0,
                proof_black_point: 0.0,
                rendering_intent: None,
                black_point_compensation: false,
            };
            let frame = reader_engine.render_preview(&req).expect("render preview");
            assert!(frame.image.width > 0 && frame.image.height > 0);
            assert_eq!(frame.generation, generation_id);
        }
    });

    // Writer thread: queues multiple revisions for both images.
    let writer_engine = Arc::clone(&engine);
    let writer = thread::spawn(move || {
        let mut last_rev_a = None;
        for step in 1..=4u32 {
            let mut adj_a = AdjustmentSet::default();
            adj_a.exposure.exposure = step as f32 * 0.25;
            let mut adj_b = AdjustmentSet::default();
            adj_b.exposure.contrast = step as f32 * 5.0;

            let summary = writer_engine.save_adjustments_queued(&[(id_a, adj_a), (id_b, adj_b)]);
            assert_eq!(summary.documents.len(), 2);
            for doc in &summary.documents {
                match &doc.outcome {
                    QueuedSaveOutcome::Accepted { revision } => {
                        if doc.image == id_a {
                            if let Some(prev) = last_rev_a {
                                assert!(*revision > prev, "revisions must strictly increase");
                            }
                            last_rev_a = Some(*revision);
                        }
                    }
                    other => panic!("unexpected queued save outcome: {other:?}"),
                }
            }
        }
    });

    reader.join().expect("reader thread panicked");
    writer.join().expect("writer thread panicked");

    // Flush and verify complete persistence report.
    let report = engine
        .flush_saves(Duration::from_secs(5))
        .expect("FileEngine must have an active save queue");
    assert!(report.is_complete(), "flush report must complete cleanly: {report:?}");
    assert_eq!(report.failed(), 0);
    assert_eq!(report.not_written(), 0);
    assert!(report.index_error.is_none());

    // Reopen from disk to confirm the newest edits actually landed.
    drop(engine);
    let reopened = FileEngine::open(ws.path()).expect("reopen FileEngine");
    let reloaded_images = reopened.images();
    assert_eq!(reloaded_images.len(), 2);

    let mut exposures = BTreeMap::new();
    for info in &reloaded_images {
        let adj = reopened.load_adjustments(info.id);
        exposures.insert(info.file_name.clone(), (adj.exposure.exposure, adj.exposure.contrast));
    }
    assert_eq!(exposures.get("shot_a.png"), Some(&(1.0, 0.0)));
    assert_eq!(exposures.get("shot_b.png"), Some(&(0.0, 20.0)));
}

struct BlockingSaveWriter {
    entered: (Mutex<bool>, Condvar),
    gate: (Mutex<bool>, Condvar),
    fail_index: AtomicBool,
}

impl BlockingSaveWriter {
    fn new() -> Self {
        Self {
            entered: (Mutex::new(false), Condvar::new()),
            gate: (Mutex::new(false), Condvar::new()),
            fail_index: AtomicBool::new(false),
        }
    }

    fn wait_until_entered(&self) {
        let (lock, cv) = &self.entered;
        let mut entered = lock.lock().unwrap();
        while !*entered {
            entered = cv.wait(entered).unwrap();
        }
    }

    fn release(&self) {
        let (lock, cv) = &self.gate;
        let mut open = lock.lock().unwrap();
        *open = true;
        cv.notify_all();
    }
}

impl SaveWriter for BlockingSaveWriter {
    fn write(&self, _rel_path: &str, _adjustments: &AdjustmentSet) -> anyhow::Result<()> {
        {
            let (lock, cv) = &self.entered;
            let mut entered = lock.lock().unwrap();
            *entered = true;
            cv.notify_all();
        }
        let (lock, cv) = &self.gate;
        let mut open = lock.lock().unwrap();
        while !*open {
            open = cv.wait(open).unwrap();
        }
        Ok(())
    }

    fn commit_index(&self) -> anyhow::Result<()> {
        if self.fail_index.load(Ordering::SeqCst) {
            anyhow::bail!("simulated index commit failure")
        }
        Ok(())
    }
}

#[test]
fn a5_save_queue_flush_timeout_and_close_report_unwritten_paths() {
    let writer = Arc::new(BlockingSaveWriter::new());
    let queue = SaveQueue::new(Arc::clone(&writer) as Arc<dyn SaveWriter>, 8);

    let outcome = queue.submit(
        "folder/a.nef",
        AdjustmentSet::default(),
        DocumentRevision::new(1),
    );
    assert_eq!(outcome, SubmitOutcome::Accepted { revision: DocumentRevision::new(1) });

    // Wait until the single writer thread is blocked inside `write`.
    writer.wait_until_entered();

    // Queue a second document while the first is in-flight.
    let outcome_b = queue.submit(
        "folder/b.nef",
        AdjustmentSet::default(),
        DocumentRevision::new(1),
    );
    assert_eq!(outcome_b, SubmitOutcome::Accepted { revision: DocumentRevision::new(1) });

    // Bounded flush while blocked must time out and explicitly mark paths as NotWritten.
    let timed_out_report = queue.flush(Duration::from_millis(40));
    assert!(timed_out_report.timed_out, "flush must report timeout when writer is blocked");
    assert!(!timed_out_report.is_complete());
    assert_eq!(timed_out_report.state_of("folder/a.nef"), Some(&PathState::NotWritten));
    assert_eq!(timed_out_report.state_of("folder/b.nef"), Some(&PathState::NotWritten));

    // Unblock the writer and close cleanly; everything drains and completes.
    writer.release();
    let final_report = queue.close(Duration::from_secs(2));
    assert!(final_report.is_complete(), "close report should succeed after unblock: {final_report:?}");
    assert_eq!(final_report.saved(), 2);
    assert_eq!(final_report.not_written(), 0);
}

// ---------------------------------------------------------------------------
// 4. Export cancellation safety (`AtomicPublisher`)
// ---------------------------------------------------------------------------

#[test]
fn a5_export_cancellation_never_publishes_partial_or_corrupts_existing_file() {
    let ws = TempWorkspace::new("export-cancel");
    let existing = ws.path().join("portrait.jpg");
    let original_bytes = b"ORIGINAL_USER_EXPORT_DO_NOT_TOUCH";
    std::fs::write(&existing, original_bytes).expect("seed existing export");

    // Case A: Pre-cancelled token rejected before staging or publishing.
    let pre_cancelled = CancelToken::new();
    pre_cancelled.cancel();
    let pre_err = AtomicPublisher::new(ws.path(), "pre-cancel", pre_cancelled).unwrap_err();
    assert_eq!(pre_err, PublishError::Cancelled);
    assert_eq!(std::fs::read(&existing).unwrap(), original_bytes);

    // Case B: Cancelled mid-write after partial bytes are written to staging.
    let mid_cancel = CancelToken::new();
    let publisher = AtomicPublisher::new(ws.path(), "mid-cancel", mid_cancel.clone())
        .expect("create publisher");
    let mid_err = publisher
        .publish_with("portrait.jpg", Some(64), |file, cancel| {
            file.write_all(b"PARTIAL_GARBAGE_BYTES")
                .map_err(|e| PublishError::Io(e.to_string()))?;
            cancel.cancel();
            Ok(())
        })
        .unwrap_err();
    assert_eq!(mid_err, PublishError::Cancelled);

    // Original file is byte-for-byte intact and no partial/staging leftovers exist.
    assert_eq!(std::fs::read(&existing).unwrap(), original_bytes);
    let mut names: Vec<String> = std::fs::read_dir(ws.path())
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["portrait.jpg".to_string()]);

    // Case C: Full staged export helper records cancellation stage cleanly.
    let staged_cancel = CancelToken::new();
    let recipe = ProcessRecipe::default();
    let report = run_staged_export(
        None,
        &recipe,
        (1000, 800),
        &Geometry::default(),
        ws.path(),
        "portrait.jpg",
        staged_cancel.clone(),
        |_plan, cancel| {
            cancel.cancel();
            Ok(vec![1, 2, 3, 4])
        },
    );
    assert!(!report.is_succeeded());
    assert!(report.cancelled);
    assert_eq!(report.failure_stage, Some(ExportStage::Publish));
    assert_eq!(report.stage_status(ExportStage::Publish), Some(&StageStatus::Cancelled));
    assert_eq!(std::fs::read(&existing).unwrap(), original_bytes);
}

// ---------------------------------------------------------------------------
// 5. ExportPlan explicit rejection of invalid bit depth / zero size / bad dir
// ---------------------------------------------------------------------------

#[test]
fn a5_export_plan_explicitly_rejects_invalid_bit_depth_zero_size_and_bad_directory() {
    let ws = TempWorkspace::new("export-plan-errors");
    let geom = Geometry::default();

    // 1) Invalid bit depth: JPEG 16-bit must error, never silently downgrade to 8-bit.
    let mut bad_depth = ProcessRecipe::default();
    bad_depth.format = ExportFormat::Jpeg;
    bad_depth.bit_depth = 16;
    assert!(matches!(
        ExportPlan::from_recipe(&bad_depth, (2000, 1500), &geom),
        Err(ExportPlanError::UnsupportedBitDepth {
            format: ExportFormat::Jpeg,
            bit_depth: 16,
            ..
        })
    ));

    // 2) Zero source dimensions.
    let ok_recipe = ProcessRecipe::default();
    assert_eq!(
        ExportPlan::from_recipe(&ok_recipe, (0, 1000), &geom),
        Err(ExportPlanError::ZeroSourceDimensions { width: 0, height: 1000 })
    );
    assert_eq!(
        ExportPlan::from_recipe(&ok_recipe, (1000, 0), &geom),
        Err(ExportPlanError::ZeroSourceDimensions { width: 1000, height: 0 })
    );

    // 3) Invalid destination directory: empty path and existing regular file.
    let mut empty_dir = ProcessRecipe::default();
    empty_dir.destination = Some(PathBuf::from(""));
    assert!(matches!(
        ExportPlan::from_recipe(&empty_dir, (1000, 800), &geom),
        Err(ExportPlanError::InvalidDestination { .. })
    ));

    let regular_file = ws.path().join("occupied.txt");
    std::fs::write(&regular_file, b"not-a-directory").unwrap();
    let mut file_as_dir = ProcessRecipe::default();
    file_as_dir.destination = Some(regular_file.clone());
    assert!(matches!(
        ExportPlan::from_recipe(&file_as_dir, (1000, 800), &geom),
        Err(ExportPlanError::InvalidDestination { .. })
    ));
    assert_eq!(std::fs::read(&regular_file).unwrap(), b"not-a-directory");

    // 4) Unsupported format & invalid scale parameter.
    let mut unsupported = ProcessRecipe::default();
    unsupported.format = ExportFormat::JpegXl;
    assert!(matches!(
        ExportPlan::from_recipe(&unsupported, (1000, 800), &geom),
        Err(ExportPlanError::UnsupportedFormat { format: ExportFormat::JpegXl, .. })
    ));

    let mut bad_scale = ProcessRecipe::default();
    bad_scale.scale = ScaleMode::Dimensions;
    bad_scale.scale_value = (1200, 0);
    assert!(matches!(
        ExportPlan::from_recipe(&bad_scale, (1000, 800), &geom),
        Err(ExportPlanError::InvalidScaleValue { mode: ScaleMode::Dimensions, value: (1200, 0) })
    ));

    // 5) Valid 16-bit PNG plan succeeds and reports explicit memory budget.
    let mut valid_png16 = ProcessRecipe::default();
    valid_png16.format = ExportFormat::Png;
    valid_png16.bit_depth = 16;
    valid_png16.destination = Some(ws.path().to_path_buf());
    let plan = ExportPlan::from_recipe(&valid_png16, (1600, 1200), &geom)
        .expect("16-bit PNG with valid directory should plan cleanly");
    assert_eq!(plan.output_size, (1600, 1200));
    assert!(plan.memory_budget.peak_bytes > plan.memory_budget.source_bytes);
}
