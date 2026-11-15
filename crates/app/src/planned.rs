//! Engine calls the interface is already built for, ahead of the engine.
//!
//! Each function is the frontend's half of a method `ImageEngine` does not have
//! yet; its doc comment gives the signature the engine is asked to add
//! (`docs/frontend-contract/engine-requests.md`, "前端先行"). Until then every
//! call answers "not connected" and the matching `contract` status marks the
//! control. When the engine adds a method, its body here becomes a plain call
//! and its `READY` flag turns true; nothing else in the interface changes.

use std::path::Path;

use aurora_engine::{EngineCapabilities, ImageEngine, ImageId, ImageInfo, StyleInfo};

use crate::i18n::t;

fn not_connected(what: &str) -> anyhow::Error {
    anyhow::anyhow!("{what}: {}", t("the engine does not provide this yet", "引擎尚未提供此功能"))
}

/// `fn capabilities(&self) -> EngineCapabilities`: what this engine can really
/// do on this machine. The engine checks model files but does not run inference.
pub fn capabilities(engine: &dyn ImageEngine) -> EngineCapabilities {
    engine.capabilities()
}

/// `fn film_curves(&self) -> Vec<String>`: the `.fcrv` names
/// `BaseCharacteristics::film_curve` accepts. Empty until then.
pub fn film_curves(engine: &dyn ImageEngine) -> Vec<String> {
    engine.film_curves()
}

pub const STYLE_IMPORT_READY: bool = true;

/// `fn import_style(&self, path: &Path) -> anyhow::Result<StyleInfo>`: reads a
/// Capture One `.costyle` / `.copreset` and adds it to `styles()`.
pub fn import_style(engine: &dyn ImageEngine, path: &Path) -> anyhow::Result<StyleInfo> {
    engine.import_style(path)
}

pub const CANCEL_READY: bool = true;

/// `fn cancel_job(&self, job_id: u64) -> bool`: stops a queued or running
/// export; the job then reports `JobStatus::Failed { message: "cancelled" }`.
pub fn cancel_job(engine: &dyn ImageEngine, job: u64) -> anyhow::Result<()> {
    if engine.cancel_job(job) { Ok(()) } else { Err(anyhow::anyhow!(t("That export is no longer active", "该导出已不在进行中"))) }
}

/// `fn source_status(&self, image: ImageId) -> Option<String>`: why the engine
/// shows something other than the full sensor data (an embedded preview after
/// a failed RAW decode, an unsupported compression). `None` when all is well.
pub fn source_status(engine: &dyn ImageEngine, image: ImageId) -> Option<String> {
    engine.source_status(image)
}

pub use aurora_engine::{HdrOptions, PanoramaOptions, Projection};

pub fn projection_label(projection: Projection) -> &'static str {
    match projection {
        Projection::Spherical => t("Spherical", "球面"),
        Projection::Cylindrical => t("Cylindrical", "圆柱"),
        Projection::Perspective => t("Perspective", "透视"),
    }
}

/// Soft proofing asks for `PreviewRequest::proof_profile: Option<String>`: the
/// preview is rendered through that output profile and back to the display,
/// so out-of-gamut colors look as they will print.
pub const PROOF_READY: bool = true;

pub const MERGE_READY: bool = true;

/// `fn merge_hdr_with_options(&self, images: &[ImageId], options: &HdrOptions)`:
/// writes a 16-bit TIFF beside the first frame and catalogs it. `align: false`
/// keeps the engine's existing no-alignment path; `deghost` strength and
/// `auto_tone` are still not consumed by the engine.
pub fn merge_hdr(engine: &dyn ImageEngine, images: &[ImageId], options: &HdrOptions) -> anyhow::Result<ImageInfo> {
    engine.merge_hdr_with_options(images, options)
}

/// `fn merge_panorama_with_options(&self, images: &[ImageId], options: &PanoramaOptions)`:
/// writes an 8-bit TIFF beside the first frame and catalogs it. `auto_crop:
/// false` keeps the stitched canvas edges; projection and boundary warp are
/// still not consumed by the engine.
pub fn merge_panorama(engine: &dyn ImageEngine, images: &[ImageId], options: &PanoramaOptions) -> anyhow::Result<ImageInfo> {
    engine.merge_panorama_with_options(images, options)
}
