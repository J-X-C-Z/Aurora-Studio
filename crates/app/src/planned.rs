//! Engine calls the interface is already built for, ahead of the engine.
//!
//! Connected resources delegate to `ImageEngine`. The remaining calls retain
//! explicit readiness flags until their backend implementation is available.

use std::path::Path;

use lumen_engine::{ImageEngine, ImageId, ImageInfo, StyleInfo};

use crate::i18n::t;

fn not_connected(what: &str) -> anyhow::Error {
    anyhow::anyhow!("{what}: {}", t("the engine does not provide this yet", "引擎尚未提供此功能"))
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

pub const CANCEL_READY: bool = false;

/// `fn cancel_job(&self, job_id: u64) -> bool`: stops a queued or running
/// export; the job then reports `JobStatus::Failed { message: "cancelled" }`.
pub fn cancel_job(_engine: &dyn ImageEngine, _job: u64) -> anyhow::Result<()> {
    Err(not_connected(t("Cancel export", "取消导出")))
}

/// `fn source_status(&self, image: ImageId) -> Option<String>`: why the engine
/// shows something other than the full sensor data (an embedded preview after
/// a failed RAW decode, an unsupported compression). `None` when all is well.
pub fn source_status(_engine: &dyn ImageEngine, _image: ImageId) -> Option<String> {
    None
}

/// How the frames of an HDR merge are combined. Mirrors the engine type the
/// request asks for (`HdrOptions`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrOptions {
    /// Align the frames before merging (hand-held brackets).
    pub align: bool,
    /// 0 off, 1 low, 2 medium, 3 high: how hard moving things are suppressed.
    pub deghost: u8,
    /// Set a starting tone on the merged result.
    pub auto_tone: bool,
}

impl Default for HdrOptions {
    fn default() -> Self {
        Self { align: true, deghost: 0, auto_tone: true }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Projection {
    #[default]
    Spherical,
    Cylindrical,
    Perspective,
}

impl Projection {
    pub const ALL: [Self; 3] = [Self::Spherical, Self::Cylindrical, Self::Perspective];

    pub fn label(self) -> &'static str {
        match self {
            Self::Spherical => t("Spherical", "球面"),
            Self::Cylindrical => t("Cylindrical", "圆柱"),
            Self::Perspective => t("Perspective", "透视"),
        }
    }
}

/// How the frames of a panorama are stitched (`PanoramaOptions` in the request).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanoramaOptions {
    pub projection: Projection,
    /// 0 ..= 100: warp the edges to fill the rectangle.
    pub boundary_warp: f32,
    /// Crop to the largest rectangle with no empty corners.
    pub auto_crop: bool,
}

impl Default for PanoramaOptions {
    fn default() -> Self {
        Self { projection: Projection::Spherical, boundary_warp: 0.0, auto_crop: true }
    }
}

/// Soft proofing asks for `PreviewRequest::proof_profile: Option<String>`: the
/// preview is rendered through that output profile and back to the display,
/// so out-of-gamut colors look as they will print. Until the field exists the
/// viewer only marks that proofing is on.
pub const PROOF_READY: bool = false;

pub const MERGE_READY: bool = false;

/// `fn merge_hdr(&self, images: &[ImageId], options: &HdrOptions) -> anyhow::Result<ImageInfo>`:
/// writes a linear DNG/TIFF beside the first frame and catalogs it.
pub fn merge_hdr(_engine: &dyn ImageEngine, _images: &[ImageId], _options: &HdrOptions) -> anyhow::Result<ImageInfo> {
    Err(not_connected(t("HDR merge", "HDR 合并")))
}

/// `fn merge_panorama(&self, images: &[ImageId], options: &PanoramaOptions) -> anyhow::Result<ImageInfo>`.
pub fn merge_panorama(_engine: &dyn ImageEngine, _images: &[ImageId], _options: &PanoramaOptions) -> anyhow::Result<ImageInfo> {
    Err(not_connected(t("Panorama merge", "全景合并")))
}
