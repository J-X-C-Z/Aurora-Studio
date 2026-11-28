//! Layered export pipeline facade: [`ExportPlan`], [`AtomicPublisher`] and
//! [`ExportExecutionReport`].
//!
//! This module provides explicit stage boundaries (`Plan`, `SourceAcquire`,
//! `Render`, `Encode`, `Metadata`, `Publish`), pre-flight validation with
//! memory-budget accounting, and cancellation-safe atomic publishing while
//! reusing the proven primitives in [`crate::recipe`] and [`crate::output`].
//!
//! [`pipeline`] is the bounded three-stage executor the standard export path
//! runs on (source acquire -> render -> encode/publish), and [`stages`] is the
//! single implementation of those three stage bodies.

pub mod pipeline;
pub mod plan;
pub mod publisher;
pub mod stages;

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::adjust::Geometry;
use crate::job_lifecycle::{CancelToken, JobId};
use crate::output::ProcessRecipe;

pub use plan::{ExportMemoryBudget, ExportPlan, ExportPlanError, ExportStage};
pub use pipeline::{
    mode as export_pipeline_mode, pin_mode as pin_export_pipeline_mode, PipelineLimits as ExportPipelineLimits,
    PipelineMode as ExportPipelineMode, PipelineStats as ExportPipelineStats,
};
pub use publisher::{AtomicPublisher, PublishError, PublishOutcome};
pub use stages::{
    pipeline_stats as export_pipeline_stats, run_export as run_standard_export, PublishTarget,
};

/// Outcome of a single stage in an [`ExportExecutionReport`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    Pending,
    Running,
    Succeeded,
    Cancelled,
    Failed { reason: String },
}

/// Recorded execution status for one [`ExportStage`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRecord {
    pub stage: ExportStage,
    pub status: StageStatus,
    pub duration_ms: Option<u64>,
    pub detail: Option<String>,
}

/// Structured execution report across the layered export stages.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExportExecutionReport {
    pub job_id: Option<JobId>,
    pub plan: Option<ExportPlan>,
    pub stages: Vec<StageRecord>,
    pub output_path: Option<PathBuf>,
    pub bytes_written: u64,
    pub cancelled: bool,
    pub failure_stage: Option<ExportStage>,
    pub failure_reason: Option<String>,
}

impl ExportExecutionReport {
    /// Creates a new report with all six stages initialized to [`StageStatus::Pending`].
    pub fn new(job_id: Option<JobId>) -> Self {
        let stages = ExportStage::ALL
            .into_iter()
            .map(|stage| StageRecord {
                stage,
                status: StageStatus::Pending,
                duration_ms: None,
                detail: None,
            })
            .collect();
        Self {
            job_id,
            plan: None,
            stages,
            output_path: None,
            bytes_written: 0,
            cancelled: false,
            failure_stage: None,
            failure_reason: None,
        }
    }

    pub fn set_plan(&mut self, plan: ExportPlan) {
        self.plan = Some(plan);
    }

    pub fn record_stage(
        &mut self,
        stage: ExportStage,
        status: StageStatus,
        duration_ms: Option<u64>,
        detail: Option<String>,
    ) {
        if let Some(record) = self.stages.iter_mut().find(|r| r.stage == stage) {
            record.status = status;
            record.duration_ms = duration_ms;
            record.detail = detail;
        }
    }

    pub fn mark_succeeded(&mut self, outcome: &PublishOutcome) {
        self.output_path = Some(outcome.final_path.clone());
        self.bytes_written = outcome.bytes_written;
        self.cancelled = false;
        self.failure_stage = None;
        self.failure_reason = None;
    }

    pub fn mark_cancelled(&mut self, stage: ExportStage) {
        self.cancelled = true;
        self.failure_stage = Some(stage);
        self.failure_reason = Some(format!("cancelled at stage {stage}"));
        self.record_stage(stage, StageStatus::Cancelled, None, None);
    }

    pub fn mark_failed(&mut self, stage: ExportStage, reason: impl Into<String>) {
        let reason = reason.into();
        self.cancelled = false;
        self.failure_stage = Some(stage);
        self.failure_reason = Some(reason.clone());
        self.record_stage(stage, StageStatus::Failed { reason }, None, None);
    }

    pub fn is_succeeded(&self) -> bool {
        self.output_path.is_some() && !self.cancelled && self.failure_stage.is_none()
    }

    pub fn stage_status(&self, stage: ExportStage) -> Option<&StageStatus> {
        self.stages.iter().find(|r| r.stage == stage).map(|r| &r.status)
    }
}

/// Runs the planning + cancellation-checked atomic publish lifecycle and
/// produces an [`ExportExecutionReport`].
///
/// The caller supplies `produce_bytes` which runs after `Plan` succeeds and
/// produces the encoded + metadata-stamped bytes to publish.
pub fn run_staged_export<F>(
    job_id: Option<JobId>,
    recipe: &ProcessRecipe,
    source_size: (u32, u32),
    geometry: &Geometry,
    fallback_dir: &Path,
    file_name: &str,
    cancel: CancelToken,
    produce_bytes: F,
) -> ExportExecutionReport
where
    F: FnOnce(&ExportPlan, &CancelToken) -> Result<Vec<u8>, String>,
{
    let mut report = ExportExecutionReport::new(job_id);

    if cancel.is_cancelled() {
        report.mark_cancelled(ExportStage::Plan);
        return report;
    }

    let plan_start = Instant::now();
    let plan = match ExportPlan::from_recipe(recipe, source_size, geometry) {
        Ok(plan) => plan,
        Err(err) => {
            report.mark_failed(ExportStage::Plan, err.to_string());
            return report;
        }
    };
    let dest = match plan.resolve_destination(fallback_dir) {
        Ok(dir) => dir,
        Err(err) => {
            report.mark_failed(ExportStage::Plan, err.to_string());
            return report;
        }
    };
    report.set_plan(plan.clone());
    report.record_stage(
        ExportStage::Plan,
        StageStatus::Succeeded,
        Some(plan_start.elapsed().as_millis() as u64),
        None,
    );

    if cancel.is_cancelled() {
        report.mark_cancelled(ExportStage::SourceAcquire);
        return report;
    }

    let encode_start = Instant::now();
    let bytes = match produce_bytes(&plan, &cancel) {
        Ok(bytes) => bytes,
        Err(reason) => {
            if cancel.is_cancelled() {
                report.mark_cancelled(ExportStage::Encode);
            } else {
                report.mark_failed(ExportStage::Encode, reason);
            }
            return report;
        }
    };
    let encode_ms = encode_start.elapsed().as_millis() as u64;
    report.record_stage(ExportStage::SourceAcquire, StageStatus::Succeeded, Some(0), None);
    report.record_stage(ExportStage::Render, StageStatus::Succeeded, Some(0), None);
    report.record_stage(ExportStage::Encode, StageStatus::Succeeded, Some(encode_ms), None);
    report.record_stage(ExportStage::Metadata, StageStatus::Succeeded, Some(0), None);

    if cancel.is_cancelled() {
        report.mark_cancelled(ExportStage::Publish);
        return report;
    }

    let publish_start = Instant::now();
    let tag = job_id.map_or_else(|| "staged".to_string(), |id| format!("job-{}", id.serial));
    let publisher = match AtomicPublisher::new(&dest, tag, cancel) {
        Ok(publisher) => publisher,
        Err(PublishError::Cancelled) => {
            report.mark_cancelled(ExportStage::Publish);
            return report;
        }
        Err(err) => {
            report.mark_failed(ExportStage::Publish, err.to_string());
            return report;
        }
    };

    match publisher.publish_bytes(file_name, &bytes) {
        Ok(outcome) => {
            report.record_stage(
                ExportStage::Publish,
                StageStatus::Succeeded,
                Some(publish_start.elapsed().as_millis() as u64),
                None,
            );
            report.mark_succeeded(&outcome);
        }
        Err(PublishError::Cancelled) => {
            report.mark_cancelled(ExportStage::Publish);
        }
        Err(err) => {
            report.mark_failed(ExportStage::Publish, err.to_string());
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjust::NormRect;
    use crate::output::{ExportFormat, ScaleMode};
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir()
                .join(format!("aurora-export-mod-{label}-{}-{id}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn plan_rejects_zero_source_dimensions() {
        let recipe = ProcessRecipe::default();
        let geom = Geometry::default();
        assert_eq!(
            ExportPlan::from_recipe(&recipe, (0, 1080), &geom),
            Err(ExportPlanError::ZeroSourceDimensions { width: 0, height: 1080 })
        );
        assert_eq!(
            ExportPlan::from_recipe(&recipe, (1920, 0), &geom),
            Err(ExportPlanError::ZeroSourceDimensions { width: 1920, height: 0 })
        );
    }

    #[test]
    fn plan_rejects_invalid_bit_depth_without_silent_downgrade() {
        let mut recipe = ProcessRecipe::default();
        recipe.format = ExportFormat::Jpeg;
        recipe.bit_depth = 16;
        let err = ExportPlan::from_recipe(&recipe, (1200, 800), &Geometry::default()).unwrap_err();
        assert!(matches!(
            err,
            ExportPlanError::UnsupportedBitDepth {
                format: ExportFormat::Jpeg,
                bit_depth: 16,
                ..
            }
        ));
    }

    #[test]
    fn plan_rejects_unsupported_format_and_bad_scale() {
        let mut psd = ProcessRecipe::default();
        psd.format = ExportFormat::Psd;
        assert!(matches!(
            ExportPlan::from_recipe(&psd, (800, 600), &Geometry::default()),
            Err(ExportPlanError::UnsupportedFormat { format: ExportFormat::Psd, .. })
        ));

        let mut bad_scale = ProcessRecipe::default();
        bad_scale.scale = ScaleMode::Width;
        bad_scale.scale_value = (0, 0);
        assert!(matches!(
            ExportPlan::from_recipe(&bad_scale, (800, 600), &Geometry::default()),
            Err(ExportPlanError::InvalidScaleValue { mode: ScaleMode::Width, value: (0, 0) })
        ));
    }

    #[test]
    fn plan_rejects_empty_and_file_destination() {
        let temp = TempDir::new("dest");
        let file = temp.path().join("not-a-dir.txt");
        std::fs::write(&file, b"keep").unwrap();

        let mut empty_dest = ProcessRecipe::default();
        empty_dest.destination = Some(PathBuf::from(""));
        assert!(matches!(
            ExportPlan::from_recipe(&empty_dest, (800, 600), &Geometry::default()),
            Err(ExportPlanError::InvalidDestination { .. })
        ));

        let mut file_dest = ProcessRecipe::default();
        file_dest.destination = Some(file.clone());
        assert!(matches!(
            ExportPlan::from_recipe(&file_dest, (800, 600), &Geometry::default()),
            Err(ExportPlanError::InvalidDestination { .. })
        ));
        assert_eq!(std::fs::read(&file).unwrap(), b"keep");
    }

    #[test]
    fn plan_computes_oriented_cropped_size_and_memory_budget() {
        let mut recipe = ProcessRecipe::default();
        recipe.format = ExportFormat::Png;
        recipe.bit_depth = 16;
        recipe.scale = ScaleMode::LongEdge;
        recipe.scale_value = (1000, 0);

        let mut geom = Geometry::default();
        geom.orientation = 1; // swaps 4000x2000 -> 2000x4000
        geom.crop = NormRect { x: 0.0, y: 0.0, width: 0.5, height: 0.5 }; // -> 1000x2000

        let plan = ExportPlan::from_recipe(&recipe, (4000, 2000), &geom).expect("valid plan");
        assert_eq!(plan.developed_size, (1000, 2000));
        assert_eq!(plan.output_size, (500, 1000));
        assert!(plan.memory_budget.source_bytes > 0);
        assert!(plan.memory_budget.render_buffer_bytes > 0);
        assert!(plan.memory_budget.encode_buffer_bytes > 0);
        assert_eq!(
            plan.memory_budget.peak_bytes,
            plan.memory_budget.source_bytes
                + plan.memory_budget.render_buffer_bytes
                + plan.memory_budget.encode_buffer_bytes
        );
    }

    #[test]
    fn atomic_publisher_preserves_existing_file_and_cleans_staging_on_cancel() {
        let temp = TempDir::new("cancel");
        let existing = temp.path().join("photo.jpg");
        std::fs::write(&existing, b"original-photo").unwrap();

        let token = CancelToken::new();
        let publisher = AtomicPublisher::new(temp.path(), "cancel-mid", token.clone()).unwrap();

        let err = publisher
            .publish_with("photo.jpg", Some(16), |file, cancel| {
                file.write_all(b"half-way").map_err(|e| PublishError::Io(e.to_string()))?;
                cancel.cancel();
                Ok(())
            })
            .unwrap_err();
        assert_eq!(err, PublishError::Cancelled);
        assert_eq!(std::fs::read(&existing).unwrap(), b"original-photo");

        let entries: Vec<_> = std::fs::read_dir(temp.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["photo.jpg".to_string()]);
    }

    #[test]
    fn run_staged_export_records_success_and_bytes() {
        let temp = TempDir::new("ok");
        let recipe = ProcessRecipe::default();
        let report = run_staged_export(
            None,
            &recipe,
            (640, 480),
            &Geometry::default(),
            temp.path(),
            "frame.jpg",
            CancelToken::new(),
            |_plan, _cancel| Ok(vec![0xFF, 0xD8, 0xFF, 0xD9]),
        );

        assert!(report.is_succeeded(), "{report:?}");
        assert_eq!(report.bytes_written, 4);
        assert_eq!(report.stage_status(ExportStage::Plan), Some(&StageStatus::Succeeded));
        assert_eq!(report.stage_status(ExportStage::Publish), Some(&StageStatus::Succeeded));
        let path = report.output_path.as_ref().unwrap();
        assert_eq!(std::fs::read(path).unwrap(), vec![0xFF, 0xD8, 0xFF, 0xD9]);
    }
}
