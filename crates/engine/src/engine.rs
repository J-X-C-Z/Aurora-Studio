//! The trait a processing engine implements for the interface.
//!
//! The interface calls these methods from background threads and never holds a
//! lock across calls, so an implementation must be `Send + Sync` and may run
//! requests concurrently. Rendering calls are latency-sensitive: the interface
//! issues a `Draft` preview on every slider movement and a `Final` one once the
//! pointer is released, and drops results superseded by a newer `generation`.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    AdjustmentSet, AiMaskKind, Collection, ExportJob, ImageId, ImageInfo, JobStatus, MatchLook,
    NormRect, TetherStatus, UserCollection,
};

/// 8-bit straight-alpha RGBA pixels, row-major, no padding.
#[derive(Clone, Debug)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<Vec<u8>>,
}

impl RgbaImage {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        debug_assert_eq!(pixels.len(), (width * height * 4) as usize);
        Self { width, height, pixels: Arc::new(pixels) }
    }

    /// Wraps pixels already shared with another frame, such as a preview cache hit.
    pub fn from_shared(width: u32, height: u32, pixels: Arc<Vec<u8>>) -> Self {
        debug_assert_eq!(pixels.len(), (width * height * 4) as usize);
        Self { width, height, pixels }
    }
}

/// 256 bins per channel, counted on the processed preview.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    pub red: Vec<u32>,
    pub green: Vec<u32>,
    pub blue: Vec<u32>,
    pub luma: Vec<u32>,
    /// Share of pixels clipped to black / white, 0 ..= 1.
    pub clipped_shadows: f32,
    pub clipped_highlights: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RenderQuality {
    /// Fast path while the user drags a control.
    Draft,
    #[default]
    Final,
}

/// What the viewer needs drawn.
#[derive(Clone, Debug)]
pub struct PreviewRequest {
    pub image: ImageId,
    pub adjustments: AdjustmentSet,
    /// Region of the uncropped image to render, normalized. `NormRect::FULL` for the whole frame.
    pub region: NormRect,
    /// Longest side of the output in device pixels.
    pub max_size: u32,
    pub quality: RenderQuality,
    /// Render without any adjustment (the "before" image).
    pub bypass: bool,
    /// Draw the selected layer's mask as a red overlay.
    pub mask_overlay: Option<u64>,
    /// Monotonic counter; results with a lower generation than the latest request are discarded.
    pub generation: u64,
}

#[derive(Clone, Debug)]
pub struct PreviewFrame {
    pub image: RgbaImage,
    pub histogram: Histogram,
    pub generation: u64,
    /// Engine time spent, for the status bar.
    pub elapsed_ms: f32,
}

/// Camera tethering status; engines without tethering report `Disconnected`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum CameraStatus {
    #[default]
    Disconnected,
    Connected {
        model: String,
        battery: f32,
        shots_remaining: u32,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CameraSettings {
    pub iso: String,
    pub shutter: String,
    pub aperture: String,
    pub exposure_comp: f32,
    pub white_balance: String,
    pub drive_mode: String,
}

/// Results the engine computes on request for "Auto" buttons.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AutoResult {
    pub adjustments: AdjustmentSet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AutoTarget {
    All,
    WhiteBalance,
    Exposure,
    HighDynamicRange,
    Levels,
    Rotation,
    Keystone,
}

pub trait ImageEngine: Send + Sync + 'static {
    /// Human-readable engine name and version, shown in About.
    fn name(&self) -> String;

    // ── Catalog ──

    /// Images already in the session when the interface opens.
    fn images(&self) -> Vec<ImageInfo>;
    /// Adds files (or every image in a folder) and returns what was added.
    fn import(&self, paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>>;
    fn collections(&self) -> Vec<Collection>;
    fn library_root(&self) -> Option<PathBuf> {
        None
    }
    fn collection_path(&self, _id: u64) -> Option<PathBuf> {
        None
    }
    fn user_collections(&self) -> Vec<UserCollection> {
        Vec::new()
    }
    fn manage_files(&self, _action: crate::LibraryAction) -> anyhow::Result<Vec<ImageInfo>> {
        anyhow::bail!("library filesystem operations are unsupported by this engine")
    }
    /// Images listed by one engine collection; `None` when the interface should
    /// derive membership itself (all images, recent imports, folders).
    fn collection_images(&self, _collection: u64) -> Option<Vec<ImageId>> {
        None
    }
    /// Persists the albums, projects and groups the user made. Their rules are
    /// opaque here: the interface evaluates them, the engine only stores them.
    fn save_user_collections(&self, _collections: &[UserCollection]) -> anyhow::Result<()> {
        Ok(())
    }
    /// Soft delete: nothing is removed from disk.
    fn set_trashed(&self, _ids: &[ImageId], _trashed: bool) -> anyhow::Result<()> {
        Ok(())
    }
    /// Persists rating, tag, keywords and IPTC changes made in the interface.
    fn update_info(&self, info: &ImageInfo) -> anyhow::Result<()>;
    /// Adjustments stored for an image, so reopening a session restores them.
    fn load_adjustments(&self, image: ImageId) -> AdjustmentSet;
    fn save_adjustments(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<()>;
    /// Creates a new variant of `image` with the given adjustments; returns its info.
    fn new_variant(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo>;

    // ── Rendering ──

    /// A thumbnail whose longest side is at most `max_size`, with adjustments applied.
    fn thumbnail(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<RgbaImage>;
    fn render_preview(&self, request: &PreviewRequest) -> anyhow::Result<PreviewFrame>;
    /// Compatibility entry point; interactive previews preserve full pixel quality.
    /// Implementations may reuse cached work, but must not reduce resolution.
    fn render_preview_fast(&self, request: &PreviewRequest, _fast: bool) -> anyhow::Result<PreviewFrame> {
        self.render_preview(request)
    }
    /// Processed sRGB color under a normalized image point, 0 ..= 1 per channel.
    fn sample_color(&self, image: ImageId, adjustments: &AdjustmentSet, point: [f32; 2]) -> anyhow::Result<[f32; 3]>;

    // ── Analysis ──

    fn auto_adjust(&self, image: ImageId, current: &AdjustmentSet, target: AutoTarget) -> anyhow::Result<AutoResult>;
    /// White balance that neutralizes the color under `point`: (kelvin, tint).
    fn pick_white_balance(&self, image: ImageId, point: [f32; 2]) -> anyhow::Result<(f32, f32)>;
    /// Adjustments that carry `reference`'s look onto `target`, plus what moved.
    fn match_look(&self, _reference: ImageId, _target: ImageId) -> anyhow::Result<MatchLook> {
        anyhow::bail!("match look is not supported by this engine")
    }
    /// Computes an AI mask and returns its cache key for [`crate::AiMask::cache_key`].
    fn compute_ai_mask(&self, image: ImageId, kind: AiMaskKind) -> anyhow::Result<String>;
    /// Capture One's closed-eye check for a tethering review. Reports only; it
    /// never blocks an import.
    fn eye_report(&self, _image: ImageId) -> anyhow::Result<crate::EyeReport> {
        anyhow::bail!("this engine has no closed-eye detection")
    }
    /// Suggested clockwise quarter turn (0, 90, 180 or 270) from Capture One's
    /// `autoRotation.onnx`. Only a suggestion: the interface decides whether to
    /// apply it and save the adjustments.
    fn auto_rotate(&self, _image: ImageId) -> anyhow::Result<u16> {
        anyhow::bail!("auto rotation is unsupported by this engine")
    }

    // ── Resources ──

    fn icc_profiles(&self) -> Vec<String>;
    fn lens_profiles(&self) -> Vec<String>;
    fn styles(&self) -> Vec<StyleInfo>;
    /// Available film-curve file stems; empty without local calibration assets.
    fn film_curves(&self) -> Vec<String> { Vec::new() }
    /// Reads a style into this engine's session. The source file stays unchanged.
    fn import_style(&self, _path: &std::path::Path) -> anyhow::Result<StyleInfo> {
        anyhow::bail!("style import is unsupported by this engine")
    }
    /// Slider ranges and defaults, for drawing the inspector without guessing.
    fn param_metadata(&self) -> &'static [crate::ParamMeta] {
        crate::param_meta::catalog()
    }
    /// This image's own ranges and defaults from its RAW info block; empty
    /// when the engine cannot read them.
    fn param_overrides(&self, _image: ImageId) -> crate::ParamOverrides {
        crate::ParamOverrides::default()
    }

    // ── Output ──

    fn start_export(&self, job: ExportJob) -> anyhow::Result<()>;
    fn job_status(&self, job_id: u64) -> JobStatus;

    // ── Capture ──

    /// Whether libgphoto2 can really open a camera; the interface shows the green
    /// dot from this, not from [`Self::camera_status`].
    fn tether_status(&self) -> TetherStatus {
        TetherStatus::NoLibrary
    }
    fn camera_status(&self) -> CameraStatus {
        CameraStatus::Disconnected
    }
    fn camera_settings(&self) -> Option<CameraSettings> {
        None
    }
    fn capture(&self) -> anyhow::Result<ImageInfo> {
        anyhow::bail!("no camera connected")
    }
    /// One live-view frame at most `width` wide; `None` without tethering.
    fn live_view_frame(&self, _width: u32) -> Option<RgbaImage> {
        None
    }
    /// Moves focus by `steps`. TODO: no autofocus until a camera backend exists.
    fn live_view_focus(&self, _steps: i32) {}
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StyleInfo {
    pub name: String,
    pub group: String,
    pub adjustments: AdjustmentSet,
}
