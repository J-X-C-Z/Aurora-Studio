//! Asynchronous HDR/panorama jobs and persisted soft-proof settings.
//!
//! The worker owns its own thread; the UI never needs a GPUI executor. Progress
//! and results are shared through `Arc<Mutex<..>>`, cancellation through an
//! `AtomicBool`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::mock::process::Source;
use crate::{
    DeghostMode, HdrOptions, ImageId, PanoramaOptions, Projection,
};

pub type MergeJobId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeKind {
    Hdr,
    Panorama,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MergeRequest {
    pub images: Vec<ImageId>,
    pub output_path: Option<PathBuf>,
    pub alignment: bool,
    pub deghost: DeghostMode,
    pub projection: Projection,
}

impl Default for MergeRequest {
    fn default() -> Self {
        Self {
            images: Vec::new(),
            output_path: None,
            alignment: true,
            deghost: DeghostMode::Auto,
            projection: Projection::Perspective,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStage {
    Load,
    Align,
    Blend,
    Encode,
    Done,
    Cancelled,
    Failed,
}

impl MergeStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Align => "align",
            Self::Blend => "blend",
            Self::Encode => "encode",
            Self::Done => "done",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeProgress {
    pub job_id: MergeJobId,
    pub stage: MergeStage,
    pub done: u32,
    pub total: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MergeOutcome {
    pub job_id: MergeJobId,
    pub output: PathBuf,
    pub elapsed_ms: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeErrorCode {
    InvalidInput,
    UnknownImage,
    Io,
    Blend,
    Encode,
    Cancelled,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeError {
    pub code: MergeErrorCode,
    pub message: String,
}

impl MergeError {
    pub fn new(code: MergeErrorCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }

    pub fn cancelled() -> Self {
        Self::new(MergeErrorCode::Cancelled, "merge cancelled")
    }
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for MergeError {}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftProofSettings {
    pub profile: Option<String>,
    pub intent: u8,
}

#[derive(Clone)]
pub(crate) struct MergeJob {
    id: MergeJobId,
    progress: Arc<Mutex<MergeProgress>>,
    result: Arc<Mutex<Option<std::result::Result<MergeOutcome, MergeError>>>>,
    cancel: Arc<AtomicBool>,
}

impl MergeJob {
    fn new(id: u64, total: usize) -> Self {
        Self {
            id,
            progress: Arc::new(Mutex::new(MergeProgress {
                job_id: id,
                stage: MergeStage::Load,
                done: 0,
                total: total as u32,
            })),
            result: Arc::new(Mutex::new(None)),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    fn progress(&self, stage: MergeStage, done: usize, total: usize) {
        *self.progress.lock() = MergeProgress { job_id: self.id, stage, done: done as u32, total: total as u32 };
    }

    fn finish(&self, result: std::result::Result<MergeOutcome, MergeError>) {
        let stage = match &result {
            Ok(_) => MergeStage::Done,
            Err(error) if error.code == MergeErrorCode::Cancelled => MergeStage::Cancelled,
            Err(_) => MergeStage::Failed,
        };
        let mut progress = self.progress.lock();
        progress.stage = stage;
        progress.done = progress.total;
        drop(progress);
        *self.result.lock() = Some(result);
    }
}

#[derive(Default)]
pub(crate) struct MergeRegistry {
    next: AtomicU64,
    jobs: Mutex<HashMap<MergeJobId, MergeJob>>,
}

impl MergeRegistry {
    pub(crate) fn start(
        &self,
        kind: MergeKind,
        sources: Vec<Arc<Source>>,
        stops: Vec<f32>,
        first_path: PathBuf,
        request: MergeRequest,
    ) -> MergeJobId {
        let id = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        let job = MergeJob::new(id, sources.len());
        self.jobs.lock().insert(id, job.clone());
        std::thread::spawn(move || run(kind, job, sources, stops, first_path, request));
        id
    }

    pub(crate) fn progress(&self, id: MergeJobId) -> Option<MergeProgress> {
        self.jobs.lock().get(&id).map(|job| job.progress.lock().clone())
    }

    pub(crate) fn cancel(&self, id: MergeJobId) -> bool {
        let Some(job) = self.jobs.lock().get(&id).cloned() else { return false };
        if job.result.lock().is_some() {
            return false;
        }
        job.cancel.store(true, Ordering::SeqCst);
        true
    }

    pub(crate) fn result(&self, id: MergeJobId) -> Option<std::result::Result<MergeOutcome, MergeError>> {
        self.jobs.lock().get(&id).and_then(|job| job.result.lock().clone())
    }
}

fn run(
    kind: MergeKind,
    job: MergeJob,
    sources: Vec<Arc<Source>>,
    stops: Vec<f32>,
    first_path: PathBuf,
    request: MergeRequest,
) {
    let started = Instant::now();
    let total = sources.len();
    if sources.is_empty() {
        job.finish(Err(MergeError::new(MergeErrorCode::InvalidInput, "merge needs at least one image")));
        return;
    }
    job.progress(MergeStage::Load, 0, total);
    for (index, _source) in sources.iter().enumerate() {
        if job.cancel.load(Ordering::SeqCst) {
            job.finish(Err(MergeError::cancelled()));
            return;
        }
        job.progress(MergeStage::Load, index + 1, total);
    }
    if job.cancel.load(Ordering::SeqCst) {
        job.finish(Err(MergeError::cancelled()));
        return;
    }
    job.progress(MergeStage::Align, total, total);
    if job.cancel.load(Ordering::SeqCst) {
        job.finish(Err(MergeError::cancelled()));
        return;
    }
    let frames: Vec<&Source> = sources.iter().map(|source| source.as_ref()).collect();
    let fused = match kind {
        MergeKind::Hdr => {
            let options = HdrOptions {
                align: request.alignment,
                deghost: request.deghost,
                deghost_reference: None,
                auto_tone: false,
            };
            crate::hdr::fuse_with_options(&frames, &stops, &options)
                .map_err(|error| MergeError::new(MergeErrorCode::Blend, error.to_string()))
        }
        MergeKind::Panorama => {
            let options = PanoramaOptions {
                projection: request.projection,
                boundary_warp: 0.0,
                auto_crop: true,
            };
            crate::pano::stitch_with_options(&frames, &options)
                .map_err(|error| MergeError::new(MergeErrorCode::Blend, error.to_string()))
        }
    };
    let fused = match fused {
        Ok(fused) => fused,
        Err(error) => {
            job.finish(Err(error));
            return;
        }
    };
    job.progress(MergeStage::Blend, total, total);
    if job.cancel.load(Ordering::SeqCst) {
        job.finish(Err(MergeError::cancelled()));
        return;
    }
    let output = match output_path(&request, &first_path, kind) {
        Ok(output) => output,
        Err(error) => {
            job.finish(Err(error));
            return;
        }
    };
    if let Some(parent) = output.parent() {
        if !parent.is_dir() {
            job.finish(Err(MergeError::new(MergeErrorCode::Io, format!("output folder unavailable: {}", parent.display()))));
            return;
        }
    }
    if output.exists() {
        job.finish(Err(MergeError::new(MergeErrorCode::Io, format!("output already exists: {}", output.display()))));
        return;
    }
    let temp = output.with_file_name(format!(".{}.tmp", output.file_name().and_then(|name| name.to_str()).unwrap_or("merge.tif")));
    job.progress(MergeStage::Encode, total, total);
    let encoded = match kind {
        MergeKind::Hdr => write_tiff16(&fused, &temp),
        MergeKind::Panorama => write_tiff8(&fused, &temp),
    };
    if let Err(error) = encoded {
        let _ = std::fs::remove_file(&temp);
        job.finish(Err(MergeError::new(MergeErrorCode::Encode, error.to_string())));
        return;
    }
    if job.cancel.load(Ordering::SeqCst) {
        let _ = std::fs::remove_file(&temp);
        job.finish(Err(MergeError::cancelled()));
        return;
    }
    if let Err(error) = std::fs::rename(&temp, &output) {
        let _ = std::fs::remove_file(&temp);
        job.finish(Err(MergeError::new(MergeErrorCode::Io, error.to_string())));
        return;
    }
    job.finish(Ok(MergeOutcome { job_id: job.id, output, elapsed_ms: started.elapsed().as_secs_f32() * 1000.0 }));
}

fn output_path(request: &MergeRequest, first_path: &Path, kind: MergeKind) -> std::result::Result<PathBuf, MergeError> {
    if let Some(output) = &request.output_path {
        return Ok(output.clone());
    }
    let dir = first_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = first_path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("merge");
    let tag = match kind {
        MergeKind::Hdr => "HDR",
        MergeKind::Panorama => "Pano",
    };
    let mut path = dir.join(format!("{stem}_{tag}.tif"));
    let mut counter = 2;
    while path.exists() {
        if counter > 999 {
            return Err(MergeError::new(MergeErrorCode::Io, format!("no free output name beside {}", first_path.display())));
        }
        path = dir.join(format!("{stem}_{tag}-{counter}.tif"));
        counter += 1;
    }
    Ok(path)
}

fn write_tiff16(fused: &crate::hdr::FusedFrame, path: &Path) -> Result<()> {
    use image::ImageEncoder as _;
    let bytes: Vec<u8> = crate::hdr::to_srgb16(fused).iter().flat_map(|value| value.to_ne_bytes()).collect();
    let mut output = std::io::Cursor::new(Vec::new());
    image::codecs::tiff::TiffEncoder::new(&mut output)
        .write_image(&bytes, fused.width, fused.height, image::ExtendedColorType::Rgb16)?;
    std::fs::write(path, output.into_inner())?;
    Ok(())
}

fn write_tiff8(fused: &crate::hdr::FusedFrame, path: &Path) -> Result<()> {
    let bytes = crate::pano::to_srgb8(fused);
    let image = image::RgbImage::from_raw(fused.width, fused.height, bytes)
        .context("panorama output has the wrong dimensions")?;
    image.save_with_format(path, image::ImageFormat::Tiff)?;
    Ok(())
}

pub(crate) fn settings_with_profile(profile: Option<String>, intent: u8) -> Result<SoftProofSettings> {
    ensure!(intent <= 3, "soft-proof intent must be 0..=3");
    Ok(SoftProofSettings { profile, intent })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AdjustmentSet;
    use crate::ImageEngine;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "aurora-merge-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn image(&self, name: &str) {
            image::RgbaImage::from_pixel(32, 24, image::Rgba([20, 40, 60, 255]))
                .save(self.0.join(name))
                .unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn wait(engine: &dyn ImageEngine, id: MergeJobId) -> std::result::Result<MergeOutcome, MergeError> {
        for _ in 0..400 {
            if let Some(result) = engine.merge_result(id) {
                return result;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("merge job {id} did not finish");
    }

    #[test]
    fn async_types_keep_the_stage_names_and_error_codes_stable() {
        assert_eq!(MergeStage::Load.as_str(), "load");
        assert_eq!(MergeStage::Align.as_str(), "align");
        assert_eq!(MergeStage::Blend.as_str(), "blend");
        assert_eq!(MergeStage::Encode.as_str(), "encode");
        assert!(MergeError::cancelled().code == MergeErrorCode::Cancelled);
    }

    #[test]
    fn soft_proof_settings_reject_an_out_of_range_intent() {
        assert!(settings_with_profile(None, 4).is_err());
        assert!(settings_with_profile(Some("x".into()), 3).is_ok());
    }

    #[test]
    fn hdr_job_completes_and_writes_the_requested_output() {
        let fixture = Fixture::new("hdr");
        fixture.image("a.png");
        fixture.image("b.png");
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        let ids: Vec<_> = engine.images().iter().map(|image| image.id).collect();
        let output = fixture.0.join("merged.tif");
        let request = MergeRequest { output_path: Some(output.clone()), ..Default::default() };
        let job = engine.merge_hdr(&ids, &request).unwrap();
        let result = wait(&engine, job).unwrap();
        assert_eq!(result.job_id, job);
        assert_eq!(result.output, output);
        assert!(output.is_file());
        assert_eq!(engine.merge_progress(job).unwrap().stage, MergeStage::Done);
    }

    #[test]
    fn panorama_job_completes_and_writes_the_requested_output() {
        let fixture = Fixture::new("pano");
        fixture.image("a.png");
        fixture.image("b.png");
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        let ids: Vec<_> = engine.images().iter().map(|image| image.id).collect();
        let output = fixture.0.join("panorama.tif");
        let request = MergeRequest { output_path: Some(output.clone()), projection: Projection::Cylindrical, ..Default::default() };
        let job = engine.merge_panorama(&ids, &request).unwrap();
        let result = wait(&engine, job).unwrap();
        assert_eq!(result.output, output);
        assert!(output.is_file());
    }

    #[test]
    fn panorama_missing_output_folder_is_a_structured_io_error() {
        let fixture = Fixture::new("pano-io");
        fixture.image("a.png");
        fixture.image("b.png");
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        let ids: Vec<_> = engine.images().iter().map(|image| image.id).collect();
        let request = MergeRequest { output_path: Some(fixture.0.join("missing/pano.tif")), ..Default::default() };
        let job = engine.merge_panorama(&ids, &request).unwrap();
        assert_eq!(wait(&engine, job).unwrap_err().code, MergeErrorCode::Io);
    }

    #[test]
    fn missing_output_folder_is_a_structured_io_error() {
        let fixture = Fixture::new("io");
        fixture.image("a.png");
        fixture.image("b.png");
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        let ids: Vec<_> = engine.images().iter().map(|image| image.id).collect();
        let request = MergeRequest { output_path: Some(fixture.0.join("missing/out.tif")), ..Default::default() };
        let job = engine.merge_hdr(&ids, &request).unwrap();
        let error = wait(&engine, job).unwrap_err();
        assert_eq!(error.code, MergeErrorCode::Io);
        assert!(!fixture.0.join("missing/out.tif").exists());
    }

    #[test]
    fn unknown_job_cancel_and_result_are_safe() {
        let fixture = Fixture::new("unknown");
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        assert!(!engine.cancel_merge(999));
        assert!(engine.merge_result(999).is_none());
        assert!(engine.merge_progress(999).is_none());
    }

    #[test]
    fn cancel_merge_marks_an_active_job_as_cancelled_or_it_already_finished() {
        let fixture = Fixture::new("cancel");
        for index in 0..32 {
            fixture.image(&format!("{index:02}.png"));
        }
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        let ids: Vec<_> = engine.images().iter().map(|image| image.id).collect();
        let request = MergeRequest { output_path: Some(fixture.0.join("cancelled.tif")), ..Default::default() };
        let job = engine.merge_hdr(&ids, &request).unwrap();
        if engine.cancel_merge(job) {
            let result = wait(&engine, job);
            assert_eq!(result.unwrap_err().code, MergeErrorCode::Cancelled);
            assert!(!fixture.0.join("cancelled.tif").exists());
        } else {
            let result = wait(&engine, job);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn soft_proof_profile_survives_reopen_and_missing_profile_degrades() {
        let fixture = Fixture::new("soft-proof");
        fixture.image("a.png");
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        let id = engine.images()[0].id;
        engine.set_soft_proof_profile(Some("No Such Profile".into()), 1).unwrap();
        drop(engine);
        let reopened = crate::FileEngine::open(&fixture.0).unwrap();
        assert_eq!(reopened.soft_proof_profile().profile.as_deref(), Some("No Such Profile"));
        let frame = reopened.soft_proof_preview(id, &AdjustmentSet::default(), 64).unwrap();
        assert!(frame.degraded.is_some());
        assert!(reopened.soft_proof_gamut_warning(id, &AdjustmentSet::default(), 64).unwrap().is_none());
        reopened.set_soft_proof_profile(None, 1).unwrap();
        let frame = reopened.soft_proof_preview(id, &AdjustmentSet::default(), 64).unwrap();
        assert!(frame.degraded.is_none());
    }

    #[test]
    fn soft_proof_profile_rejects_an_invalid_intent() {
        let fixture = Fixture::new("proof-intent");
        fixture.image("a.png");
        let engine = crate::FileEngine::open(&fixture.0).unwrap();
        assert!(engine.set_soft_proof_profile(None, 4).is_err());
    }
}
