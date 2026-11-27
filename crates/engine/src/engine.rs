//! The trait a processing engine implements for the interface.
//!
//! The interface calls these methods from background threads and never holds a
//! lock across calls, so an implementation must be `Send + Sync` and may run
//! requests concurrently. Rendering calls are latency-sensitive: the interface
//! issues a `Draft` preview on every slider movement and a `Final` one once the
//! pointer is released, and drops results superseded by a newer `generation`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    AdjustmentSet, AiMaskKind, Collection, ExportFormat, ExportJob, ImageId, ImageInfo, JobStatus,
    CaptureDevice, HdrOptions, MatchLook, MergeError, MergeJobId, MergeOutcome, MergeProgress,
    MergeRequest, NormRect, PanoramaOptions, PortraitCapability, PrintLayout, PrintPreview,
    PrintRequest, SoftProofSettings, TetherStatus, UserCollection,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// ICC display profile to convert the sRGB preview into. `None` keeps sRGB.
    #[serde(default)]
    pub display_profile: Option<PathBuf>,
    /// Soft-proof profile name (file stem in the profile folder). `None` skips proofing.
    #[serde(default)]
    pub proof_profile: Option<String>,
    /// Soft-proof white level; 1.0 is neutral. A bounded linear approximation,
    /// not Capture One's paper/ink simulation.
    #[serde(default = "unit")]
    pub proof_paper_white: f32,
    /// Soft-proof black level; 0.0 is neutral.
    #[serde(default)]
    pub proof_black_point: f32,
    /// Soft-proof rendering intent: 0 perceptual, 1 relative (default),
    /// 2 saturation, 3 absolute. Only relative is implemented; every other
    /// value falls back to it so a request never shifts pixels by accident.
    #[serde(default)]
    pub rendering_intent: Option<u8>,
    /// Apply black point compensation while proofing. Ignored without a proof
    /// profile or without a `bkpt` tag; the frame then reports `degraded`.
    #[serde(default)]
    pub black_point_compensation: bool,
}

fn unit() -> f32 {
    1.0
}

/// Map a request intent onto the ICC request. C1's enum order is perceptual,
/// relative, saturation, absolute; unknown values and intents that need
/// `A2B1`/`B2A1` tables stay on relative and report why.
fn proof_rendering(intent: Option<u8>, black_point_compensation: bool) -> (crate::icc::Rendering, Option<String>) {
    use crate::icc::{Intent, Rendering};
    let note = match intent {
        None | Some(1) => None,
        Some(0) => Some("rendering intent perceptual: unsupported, using relative".to_string()),
        Some(2) => Some("rendering intent saturation: unsupported, using relative".to_string()),
        Some(3) => Some("rendering intent absolute: unsupported, using relative".to_string()),
        Some(other) => Some(format!("rendering intent {other}: unknown, using relative")),
    };
    (Rendering { intent: Intent::Relative, black_point_compensation }, note)
}

/// Rendering to use for a request; intent and BPC only apply while proofing.
pub(crate) fn request_rendering(request: &PreviewRequest) -> (crate::icc::Rendering, Option<String>) {
    if request.proof_profile.is_none() {
        return (crate::icc::Rendering::default(), None);
    }
    proof_rendering(request.rendering_intent, request.black_point_compensation)
}

#[derive(Clone, Debug)]
pub struct PreviewFrame {
    pub image: RgbaImage,
    pub histogram: Histogram,
    pub generation: u64,
    /// Engine time spent, for the status bar.
    pub elapsed_ms: f32,
    /// Why the requested display profile was not applied; the frame is sRGB.
    pub degraded: Option<String>,
    /// Pixels that change materially through an sRGB -> proof -> sRGB round trip.
    /// `None` when proofing is off or the proof profile is sRGB.
    pub gamut_warning: Option<GamutWarning>,
    /// True when `image.pixels` is already in BGRA byte order for GPUI display.
    pub bgra: bool,
    /// Viewer scope counts taken from exactly these pixels on the device, when
    /// the request asked for them and the render could promise that the pixels
    /// are the displayed ones. `None` otherwise — CPU-only path, ICC/soft-proof
    /// conversion, a cached frame, a post-8-bit stage or an unsupported kind —
    /// and the caller then counts the scopes itself, as before.
    pub scopes: Option<crate::scopes::ScopeMaps>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GamutWarning {
    pub out_of_gamut: u64,
    pub total: u64,
}

/// Workspace sRGB -> proof profile -> display profile, 8-bit. The cached frame
/// stays sRGB; only requested conversions copy the pixels. Proof failure keeps
/// the original frame and reports why.
pub(crate) fn with_display_profile(
    image: RgbaImage,
    proof: Option<&str>,
    path: Option<&Path>,
    levels: (f32, f32),
    rendering: crate::icc::Rendering,
) -> (RgbaImage, Option<String>) {
    let mut image = image;
    let mut degraded = None;
    if let Some(name) = proof {
        let Some(profile) = crate::icc::profile(name) else {
            return (image, Some(format!("proof profile {name}: not found")));
        };
        if !crate::icc::is_srgb(&profile) {
            let pixels = Arc::make_mut(&mut image.pixels);
            if crate::icc::convert_srgb_to_with(pixels, &profile, rendering).is_none() {
                crate::icc::convert_srgb_to(pixels, &profile);
                degraded = Some("proof rendering not supported by profile: using relative".to_string());
            }
        }
        apply_paper_levels(&mut image, levels.0, levels.1);
    }
    let Some(path) = path else { return (image, degraded) };
    match crate::icc::load_profile(path) {
        Ok(profile) => {
            if !crate::icc::is_srgb(&profile) {
                let pixels = Arc::make_mut(&mut image.pixels);
                crate::icc::convert_srgb_to(pixels, &profile);
            }
            (image, degraded)
        }
        Err(err) => {
            let reason = format!("display profile {}: {err}", path.display());
            (image, Some(degraded.map_or(reason.clone(), |d| format!("{d}; {reason}"))))
        }
    }
}

/// Bounded linear paper white / black point on proofed pixels. Neutral is an
/// exact no-op; this is a stand-in for, not a copy of, C1's paper/ink model.
fn apply_paper_levels(image: &mut RgbaImage, paper_white: f32, black_point: f32) {
    use rayon::prelude::*;

    let white = paper_white.clamp(0.0, 1.0);
    let black = black_point.clamp(0.0, 1.0).min(white);
    if white >= 1.0 && black <= 0.0 {
        return;
    }
    Arc::make_mut(&mut image.pixels).par_chunks_exact_mut(4).for_each(|px| {
        for channel in &mut px[..3] {
            *channel = (black * 255.0 + *channel as f32 * (white - black)).round().clamp(0.0, 255.0) as u8;
        }
    });
}

/// Key for one converted frame: the sRGB frame's content key (image, params,
/// region, size) plus the display/proof settings. `generation` is a request
/// stamp and is deliberately absent; a content change alters the frame key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProfileKey {
    frame: crate::preview_cache::PreviewKey,
    display: Option<PathBuf>,
    proof: Option<String>,
    levels: (u32, u32),
    rendering: crate::icc::Rendering,
}

impl ProfileKey {
    pub(crate) fn new(
        frame: crate::preview_cache::PreviewKey,
        proof: Option<&str>,
        display: Option<&Path>,
        levels: (f32, f32),
        rendering: crate::icc::Rendering,
    ) -> Self {
        let levels = if proof.is_some() { (levels.0.to_bits(), levels.1.to_bits()) } else { (1.0f32.to_bits(), 0.0f32.to_bits()) };
        let rendering = if proof.is_some() { rendering } else { crate::icc::Rendering::default() };
        Self { frame, display: display.map(Path::to_path_buf), proof: proof.map(str::to_string), levels, rendering }
    }
}

/// Small LRU of converted frames, so a repaint or a drag back to a value
/// already shown does not repeat the 8-bit ICC conversion.
pub(crate) struct ProfileCache {
    entries: Vec<(ProfileKey, RgbaImage, Option<String>)>,
}

impl Default for ProfileCache {
    fn default() -> Self {
        Self { entries: Vec::new() }
    }
}

impl ProfileCache {
    /// Returns the converted frame, converting only on a miss. A request with
    /// no proof or display profile keeps the original path and skips the cache.
    pub(crate) fn converted(
        &mut self,
        key: ProfileKey,
        image: RgbaImage,
        proof: Option<&str>,
        display: Option<&Path>,
        levels: (f32, f32),
        rendering: crate::icc::Rendering,
    ) -> (RgbaImage, Option<String>) {
        if proof.is_none() && display.is_none() {
            return (image, None);
        }
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            let entry = self.entries.remove(pos);
            let out = (entry.1.clone(), entry.2.clone());
            self.entries.push(entry);
            return out;
        }
        let (out, degraded) = with_display_profile(image, proof, display, levels, rendering);
        self.entries.push((key, out.clone(), degraded.clone()));
        const CAPACITY: usize = 4;
        const MAX_BYTES: usize = 256 << 20;
        while self.entries.len() > CAPACITY {
            self.entries.remove(0);
        }
        while self.entries.len() > 1
            && self.entries.iter().map(|e| e.1.pixels.len()).sum::<usize>() > MAX_BYTES
        {
            self.entries.remove(0);
        }
        (out, degraded)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Counts pixels whose sRGB -> proof -> sRGB round trip shifts any channel by
/// more than the threshold. A minimal stand-in for C1's gamut warning, not its
/// paper/ink-aware algorithm.
pub fn gamut_warning(image: &RgbaImage, proof: Option<&str>) -> Option<GamutWarning> {
    let profile = crate::icc::profile(proof?)?;
    if crate::icc::is_srgb(&profile) {
        return None;
    }
    let mut round = image.pixels.as_ref().clone();
    crate::icc::convert_srgb_to(&mut round, &profile);
    crate::icc::convert_to_srgb(&mut round, &profile);
    const THRESHOLD: i32 = 12;
    use rayon::prelude::*;
    let out_of_gamut = image
        .pixels
        .par_chunks_exact(4)
        .zip(round.par_chunks_exact(4))
        .filter(|(src, back)| {
            (0..3).any(|i| (src[i] as i32 - back[i] as i32).abs() > THRESHOLD)
        })
        .count() as u64;
    Some(GamutWarning { out_of_gamut, total: (image.pixels.len() / 4) as u64 })
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
    /// Read-only focus mode; empty when the driver does not report it.
    pub focus_mode: String,
    /// Read-only focus distance; empty when unknown.
    pub focus_distance: String,
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

/// A layer imported from a Capture One catalog, shown read-only in the Layers
/// panel. The mask pixels are not decoded yet, so nothing renders from it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSummary {
    pub name: String,
    pub enabled: bool,
    /// Capture One's `ZMASKUUID`; the `.comask` payload is not resolved yet.
    pub mask_uuid: Option<String>,
}

/// What an engine can actually do on this machine, so the interface can grey
/// out menus instead of guessing. Detection must stay cheap: check that model
/// files and platform libraries exist, never run inference.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EngineCapabilities {
    /// Kinds this engine can actually produce a mask for.
    pub ai_mask_kinds: Vec<AiMaskKind>,
    pub eye_report: bool,
    pub auto_rotate: bool,
    pub match_look: bool,
    /// Whether this platform has a tether backend, regardless of a connected camera.
    pub tether: bool,
    /// The 16-bit gain-map blemish path is compiled and can be enabled.
    pub blemish_gainmap: bool,
    /// The shipped `blemishRemoval.onnx` is present; false means the engine
    /// must refuse/degrade the gain-map path instead of silently skipping it.
    pub blemish_gainmap_model: bool,
    pub export_formats: Vec<ExportFormat>,
}

/// Two actual executions of a generated three-operator graph, not an installed-model score.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceProbe {
    pub requested: String,
    pub executed: String,
    pub adapter: Option<String>,
    pub first_us: u64,
    pub warm_us: u64,
    pub earlier_failures: Vec<String>,
}

impl InferenceProbe {
    pub fn run(backend: &str) -> Result<Self, String> {
        let req: aurora_portrait::native_infer::backend::BackendRequest = backend.parse()?;
        let r = aurora_portrait::native_infer::selfcheck::run(req)?;
        Ok(Self { requested: req.label().into(), executed: r.actual.label().into(), adapter:r.adapter, first_us: r.first_us,
            warm_us: r.warm_us, earlier_failures: r.prior.into_iter().map(|(k,e)| format!("{}: {e}", k.label())).collect() })
    }
}

pub trait ImageEngine: Send + Sync + 'static {
    /// Opt-in diagnostic. Call off the UI thread; no user photos, private assets, or ORT are required.
    fn check_inference_device(&self, backend: &str) -> Result<InferenceProbe, String> { InferenceProbe::run(backend) }

    /// A folder-backed host can supply its sorted/filtered next RAW paths. Demo engines have nothing to prefetch.
    fn prefetch_visible_neighbors(&self, _targets:Vec<PathBuf>) {}

    /// Human-readable engine name and version, shown in About.
    fn name(&self) -> String;

    /// See [`EngineCapabilities`]; engines override with what they can really do.
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities::default()
    }

    /// Local layers that came from a Capture One catalog. Read-only: the mask
    /// pixels are not decoded, so these are listed but never rendered.
    fn layer_summaries(&self, _image: ImageId) -> Vec<LayerSummary> {
        Vec::new()
    }

    /// Read-only `ZRETOUCHINGLAYER` mask summaries: skin evening and blemish
    /// removal UUIDs, plus whether a `.comask` record has a Map grid. Nothing
    /// here reaches the renderer.
    fn retouch_masks(&self, _image: ImageId) -> Vec<crate::c1_retouch::RetouchMaskSummary> {
        Vec::new()
    }

    /// Merges bracketed exposures into one 16-bit TIFF beside the first frame
    /// and catalogs it. Alignment, deghosting and auto tone are not applied yet.
    fn merge_hdr_frames(&self, _ids: &[ImageId]) -> anyhow::Result<ImageInfo> {
        anyhow::bail!("HDR merge is not supported by this engine")
    }

    /// Like [`Self::merge_hdr_frames`], with the interface's merge options.
    /// Engines that have not consumed the options keep the old behaviour.
    fn merge_hdr_with_options(
        &self,
        ids: &[ImageId],
        _options: &HdrOptions,
    ) -> anyhow::Result<ImageInfo> {
        self.merge_hdr_frames(ids)
    }

    /// Stitches frames left to right into one 8-bit TIFF beside the first frame
    /// and catalogs it. Horizontal alignment and overlap feathering only:
    /// distortion, perspective and exposure matching are not applied yet.
    fn merge_panorama_frames(&self, _ids: &[ImageId]) -> anyhow::Result<ImageInfo> {
        anyhow::bail!("Panorama merge is not supported by this engine")
    }

    /// Like [`Self::merge_panorama_frames`], with the interface's merge options.
    /// Engines that have not consumed the options keep the old behaviour.
    fn merge_panorama_with_options(
        &self,
        ids: &[ImageId],
        _options: &PanoramaOptions,
    ) -> anyhow::Result<ImageInfo> {
        self.merge_panorama_frames(ids)
    }

    // ── Catalog ──

    /// Images already in the session when the interface opens.
    fn images(&self) -> Vec<ImageInfo>;
    /// Adds files (or every image in a folder) and returns what was added.
    fn import(&self, paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>>;
    fn collections(&self) -> Vec<Collection>;
    /// The library root this engine reads. `None` for engines with no on-disk
    /// library (the demo/mock engine).
    ///
    /// The root is set by [`crate::FileEngine::open`] / [`crate::FileEngine::open_c1_catalog`];
    /// changing it means reopening the library, not mutating a live engine.
    fn library_root(&self) -> Option<PathBuf> {
        None
    }
    /// Session / Catalog profile (kind, Capture/Selects/Output/Trash dirs, favorites).
    fn library_profile(&self) -> Option<crate::LibraryProfile> {
        None
    }
    /// Moves images into one of the Session's role directories (`Capture`, `Selects`, `Output`, `Trash`).
    fn move_to_session_role(
        &self,
        _images: &[ImageId],
        _role: crate::SessionFolderRole,
    ) -> anyhow::Result<Vec<ImageInfo>> {
        anyhow::bail!("session role folders are unsupported by this engine")
    }
    fn add_favorite_folder(&self, _rel_path: &str, _label: &str) -> anyhow::Result<()> {
        Ok(())
    }
    fn remove_favorite_folder(&self, _rel_path: &str) -> anyhow::Result<()> {
        Ok(())
    }
    /// Absolute folder path behind a folder collection id. `None` for a
    /// C1-only album/project, which has no filesystem path.
    fn collection_path(&self, _id: u64) -> Option<PathBuf> {
        None
    }
    /// Albums, projects and groups the user made, read back from the store.
    fn user_collections(&self) -> Vec<UserCollection> {
        Vec::new()
    }
    /// Move, copy, rename, relink, forget or create a library folder. The
    /// operation is all-or-nothing: any collision or I/O failure rolls back
    /// every file already changed in the batch.
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
    /// Persists metadata for a multi-image edit. Engines that can commit one
    /// catalog snapshot override this; simple engines share the single path.
    fn update_info_batch(&self, infos: &[ImageInfo]) -> anyhow::Result<()> {
        for info in infos {
            self.update_info(info)?;
        }
        Ok(())
    }
    /// Adjustments stored for an image, so reopening a session restores them.
    fn load_adjustments(&self, image: ImageId) -> AdjustmentSet;
    /// Adjustments for many images in one call, returned in `images` order.
    ///
    /// The startup path uses this so an engine backed by a catalog can take one
    /// snapshot of its entry list instead of locking and scanning it once per
    /// image. The default preserves the single-image behaviour and the
    /// parallel fan-out the startup path had before; engines whose load also
    /// applies import defaults can resolve the whole batch together.
    fn load_adjustments_batch(&self, images: &[ImageId]) -> Vec<(ImageId, AdjustmentSet)> {
        use rayon::prelude::*;
        images
            .par_iter()
            .map(|image| (*image, self.load_adjustments(*image)))
            .collect()
    }
    fn save_adjustments(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<()>;
    /// Persists a multi-image adjustment edit with one catalog commit when the
    /// engine supports it.
    fn save_adjustments_batch(&self, items: &[(ImageId, AdjustmentSet)]) -> anyhow::Result<()> {
        for (image, adjustments) in items {
            self.save_adjustments(*image, adjustments)?;
        }
        Ok(())
    }
    /// Persists a multi-image adjustment edit and reports every document.
    ///
    /// Unlike [`Self::save_adjustments_batch`], a partially failed batch keeps
    /// its per-document results instead of collapsing to one error: the caller
    /// can distinguish "written", "refused by the disk" and "superseded by a
    /// newer save". The default implementation is the single-document path;
    /// engines that commit a batch (a real catalog) override it.
    fn save_adjustments_batch_report(
        &self,
        items: &[(ImageId, AdjustmentSet)],
    ) -> crate::SaveReport {
        let documents = items
            .iter()
            .map(|(image, adjustments)| match self.save_adjustments(*image, adjustments) {
                Ok(()) => crate::DocumentSaveResult::saved(*image, String::new()),
                Err(error) => {
                    crate::DocumentSaveResult::failed(*image, String::new(), format!("{error:#}"))
                }
            })
            .collect();
        crate::SaveReport::new(documents, None)
    }
    /// Queues a multi-image edit for the engine's bounded single-writer save
    /// queue, returning per-image submit results.
    ///
    /// The in-memory edit is applied before the queue is consulted, so the
    /// user's latest content is visible even when the queue is full or closed.
    /// Engines without a queue return `None`; the caller then uses
    /// [`Self::save_adjustments_batch_report`].
    fn save_adjustments_queued(
        &self,
        _items: &[(ImageId, AdjustmentSet)],
    ) -> Option<crate::QueuedSaveSummary> {
        None
    }
    /// Waits up to `timeout` for queued saves and reports what reached disk.
    ///
    /// `None` means this engine has no queue (the synchronous path owns saving).
    fn flush_pending_saves(
        &self,
        _timeout: std::time::Duration,
    ) -> Option<crate::FlushReport> {
        None
    }
    /// Cheap snapshot of the save queue, or `None` without a queue.
    fn pending_save_status(&self) -> Option<crate::QueueStatus> {
        None
    }
    /// Creates a new variant of `image` with the given adjustments; returns its info.
    fn new_variant(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo>;

    /// Write the image's adjustments into its Capture One `.cos` sidecar.
    fn save_c1_settings(
        &self,
        _image: ImageId,
        _adjustments: &AdjustmentSet,
    ) -> anyhow::Result<crate::c1_cos::CosSaveOutcome> {
        anyhow::bail!("this engine cannot write Capture One `.cos` sidecars")
    }

    /// Write the image's adjustments into the open Capture One catalog.
    fn save_c1_catalog(
        &self,
        _image: ImageId,
        _adjustments: &AdjustmentSet,
    ) -> anyhow::Result<crate::c1_writeback::CatalogSaveOutcome> {
        Ok(crate::c1_writeback::CatalogSaveOutcome::NotACatalog)
    }

    // ── Lens cast calibration ──

    /// Reads a `.lcc` and returns its Uniform Light grid. This is intentionally
    /// stateless: `set_lcc_calibration` is the only command that changes the
    /// image's persisted calibration.
    fn load_lcc_calibration(
        &self,
        path: &Path,
    ) -> Result<crate::dust_lcc::LightFalloff, crate::dust_lcc::LccLoadError> {
        crate::dust_lcc::load_light_falloff(path)
    }

    /// Stores the grid as the image's `.lcc` sidecar and points the C1 `.cos`
    /// global layer at it through `Lccid`. `None` clears that pointer.
    fn set_lcc_calibration(
        &self,
        _image: ImageId,
        _calibration: Option<crate::dust_lcc::LightFalloff>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("lens cast calibration persistence is unsupported by this engine")
    }

    /// Returns the calibration referenced by the image's `.cos` `Lccid`.
    fn lcc_calibration(&self, _image: ImageId) -> Option<crate::dust_lcc::LightFalloff> {
        None
    }

    // ── Async merge / soft proof ──

    /// Starts a background HDR merge and returns its job id. The worker thread
    /// checks cancellation at stage boundaries and after each source load.
    fn merge_hdr(&self, images: &[ImageId], options: &MergeRequest) -> anyhow::Result<MergeJobId> {
        let _ = (images, options);
        anyhow::bail!("async merge is unsupported by this engine")
    }

    /// Starts a background panorama merge and returns its job id.
    fn merge_panorama(&self, images: &[ImageId], options: &MergeRequest) -> anyhow::Result<MergeJobId> {
        let _ = (images, options);
        anyhow::bail!("async panorama merge is unsupported by this engine")
    }

    fn merge_progress(&self, _job_id: MergeJobId) -> Option<MergeProgress> {
        None
    }

    fn cancel_merge(&self, _job_id: MergeJobId) -> bool {
        false
    }

    fn merge_result(&self, _job_id: MergeJobId) -> Option<std::result::Result<MergeOutcome, MergeError>> {
        None
    }

    /// Persist the soft-proof profile stem and rendering intent. The engine
    /// stores them beside the library; the app settings still own the display
    /// profile separately.
    fn set_soft_proof_profile(&self, _profile: Option<String>, _intent: u8) -> anyhow::Result<()> {
        anyhow::bail!("soft-proof persistence is unsupported by this engine")
    }

    fn soft_proof_profile(&self) -> SoftProofSettings {
        SoftProofSettings::default()
    }

    /// Render with the persisted proof profile. Missing profile files do not
    /// fail the frame: the engine returns sRGB pixels plus `degraded`.
    fn soft_proof_preview(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<PreviewFrame> {
        let request = PreviewRequest {
            image,
            adjustments: adjustments.clone(),
            region: NormRect::FULL,
            max_size: max_size.max(1),
            quality: RenderQuality::Final,
            bypass: false,
            mask_overlay: None,
            generation: 1,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
            rendering_intent: Some(1),
            black_point_compensation: false,
        };
        self.render_preview(&request)
    }

    fn soft_proof_gamut_warning(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<Option<GamutWarning>> {
        Ok(self.soft_proof_preview(image, adjustments, max_size)?.gamut_warning)
    }

    /// Pure paper/grid calculation for the print dialog.
    fn print_layout(&self, request: &PrintRequest) -> anyhow::Result<PrintLayout> {
        Ok(crate::panel_commands::print_template(request)?.1)
    }

    /// Render one print page to RGBA for the dialog's page preview.
    fn print_preview(&self, request: &PrintRequest, page: usize) -> anyhow::Result<PrintPreview> {
        crate::panel_commands::print_preview(self, request, page)
    }

    /// Write one print page as a PDF. Only the first page is written; the
    /// current `write_image_pdf` path is single-page.
    fn print_to_pdf(&self, request: &PrintRequest, output: &Path) -> anyhow::Result<crate::print::RenderedPage> {
        crate::panel_commands::print_to_pdf(self, request, output)
    }

    /// Print one rendered page to a system printer. Multi-page requests return
    /// `Unsupported` until the print module grows a page queue.
    fn print_to_printer(
        &self,
        request: &PrintRequest,
        printer: Option<&str>,
    ) -> Result<(), crate::print::PrintError> {
        crate::panel_commands::print_to_printer(self, request, printer)
    }

    /// Devices WIA/libgphoto2 can enumerate, independent of a connected session.
    fn capture_devices(&self) -> Vec<CaptureDevice> {
        Vec::new()
    }

    /// Start a capture session. The Windows backend has one-shot capture but no
    /// persistent WIA session, so engines that cannot do this must say so.
    fn capture_start(&self) -> anyhow::Result<()> {
        anyhow::bail!("capture session persistence is not implemented; use capture() for a one-shot capture")
    }

    /// Stop a capture session. One-shot engines make this a no-op.
    fn capture_stop(&self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Poll the next captured frame. One-shot engines return `None`.
    fn capture_arrived(&self) -> Option<ImageInfo> {
        None
    }

    /// Read-only portrait stage registry: what can run with explicit inputs.
    fn portrait_capabilities(&self) -> Vec<PortraitCapability> {
        Vec::new()
    }

    /// Per-photo portrait capability snapshot: what this engine can really run
    /// for `image` right now.
    ///
    /// The interface asks for this off the UI thread and greys out exactly the
    /// rows it reports as unavailable, so an engine must answer from evidence
    /// about *this* photo, not from the static catalogue bridge in
    /// [`crate::panel_commands::portrait_capabilities`].
    ///
    /// The default is conservative on every axis at once: no backend, no
    /// models and no same-frame inputs are claimed, which is deliberately not
    /// [`crate::portrait_registry::RuntimeProbeContext::fully_ready`]. Engines
    /// that can observe their own host and frame override this; an engine that
    /// cannot is expected to report the gap instead of optimism.
    fn portrait_capability_snapshot(
        &self,
        _image: ImageId,
    ) -> crate::portrait_registry::PortraitCapabilitySnapshot {
        crate::portrait_registry::capability_snapshot(
            &crate::portrait_registry::RuntimeProbeContext::default(),
        )
    }

    /// Per-photo capability for one document state.
    ///
    /// `detail` asks the engine to spend a real production pass on the rows it
    /// reports; `false` must stay cheap and is what the automatic path uses. An
    /// engine with per-row evidence overrides this; the default delegates to
    /// [`Self::portrait_capability_snapshot`], so an engine without one keeps its
    /// conservative answer and never invents a positive.
    fn portrait_capability_for(
        &self,
        image: ImageId,
        _adjustments: &AdjustmentSet,
        _detail: bool,
    ) -> crate::portrait_registry::PortraitCapabilitySnapshot {
        self.portrait_capability_snapshot(image)
    }

    // ── Rendering ──

    /// A thumbnail whose longest side is at most `max_size`, with adjustments applied.
    fn thumbnail(&self, image: ImageId, adjustments: &AdjustmentSet, max_size: u32) -> anyhow::Result<RgbaImage>;
    /// [`Self::thumbnail`] converted for the display and, while proofing, the
    /// proof profile. Engines that only cache unprofiled pixels keep the default.
    fn thumbnail_profiled(
        &self,
        image: ImageId,
        adjustments: &AdjustmentSet,
        max_size: u32,
        display_profile: Option<&Path>,
        proof_profile: Option<&str>,
        levels: (f32, f32),
    ) -> anyhow::Result<RgbaImage> {
        let img = self.thumbnail(image, adjustments, max_size)?;
        Ok(with_display_profile(img, proof_profile, display_profile, levels, crate::icc::Rendering::default()).0)
    }
    fn render_preview(&self, request: &PreviewRequest) -> anyhow::Result<PreviewFrame>;
    /// [`Self::render_preview`] that also asks for viewer scope counts of the
    /// returned pixels.
    ///
    /// The counts are only filled when the engine can prove they were taken
    /// from the very pixels in `frame.image`; otherwise `frame.scopes` stays
    /// `None` and the caller counts on the CPU. This default keeps every engine
    /// without a device path on that fallback.
    fn render_preview_scoped(
        &self,
        request: &PreviewRequest,
        _scopes: crate::scopes::ScopeWants,
    ) -> anyhow::Result<PreviewFrame> {
        let mut frame = self.render_preview(request)?;
        frame.scopes = None;
        Ok(frame)
    }
    /// Legacy interactive hint. Both settings preserve the exact sampling grid,
    /// processing stages and visible pixels of `render_preview`.
    fn render_preview_fast(&self, request: &PreviewRequest, _fast: bool) -> anyhow::Result<PreviewFrame> {
        self.render_preview(request)
    }
    /// Fast embedded-JPEG preview for a cold RAW image whose 16-bit source is
    /// not yet in RAM or the 16-bit disk cache. Returns `None` when the image
    /// is already warm, is not a RAW, or has no embedded JPEG preview.
    fn quick_cold_raw_preview(&self, _request: &PreviewRequest) -> Option<PreviewFrame> {
        None
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
    /// Every `.fcrv` stem the base-characteristics film curve accepts.
    fn film_curves(&self) -> Vec<String> {
        Vec::new()
    }
    /// Reads a Capture One `.costyle` / `.copreset` into [`Self::styles`].
    fn import_style(&self, _path: &Path) -> anyhow::Result<StyleInfo> {
        anyhow::bail!("style import is not supported by this engine")
    }
    /// The style's adjustments for `image`. An AICGR style is re-loaded against
    /// the image's current adjustments and white balance; every other style
    /// returns its already loaded adjustments.
    fn style_for_image(
        &self,
        _image: ImageId,
        style: &StyleInfo,
        _target: &AdjustmentSet,
    ) -> anyhow::Result<AdjustmentSet> {
        Ok(style.adjustments.clone())
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
    /// Runtime portrait capability registry table (TSV) for frontend capability discovery.
    fn portrait_capability_registry_tsv(&self) -> &'static str {
        aurora_portrait::capability_registry::REGISTRY_TSV
    }

    // ── Output ──

    fn start_export(&self, job: ExportJob) -> anyhow::Result<()>;
    fn job_status(&self, job_id: u64) -> JobStatus;
    /// Stops a queued or running export. It will report
    /// `JobStatus::Failed { message: "cancelled" }`.
    fn cancel_job(&self, _job_id: u64) -> bool {
        false
    }
    /// Why an image is shown from other than full sensor data, such as an
    /// embedded preview after a failed RAW decode. `None` when all is well.
    fn source_status(&self, _image: ImageId) -> Option<String> {
        None
    }

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
    /// Moves focus by `steps`. WIA only reports focus mode/distance read-only and
    /// has no autofocus command, so this stays a no-op.
    fn live_view_focus(&self, _steps: i32) {}
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StyleInfo {
    pub name: String,
    pub group: String,
    pub adjustments: AdjustmentSet,
    /// The `.costyle` / `.copreset` this style came from, when it has one.
    #[serde(default)]
    pub source: Option<PathBuf>,
    /// `<AICGR>` was present, so applying this style can use the target image.
    #[serde(default)]
    pub has_look: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::icc::Rendering;

    fn request() -> PreviewRequest {
        PreviewRequest {
            image: ImageId(1),
            adjustments: AdjustmentSet::default(),
            region: NormRect::FULL,
            max_size: 512,
            quality: RenderQuality::Draft,
            bypass: false,
            mask_overlay: None,
            generation: 1,
            display_profile: None,
            proof_profile: None,
            proof_paper_white: 1.0,
            proof_black_point: 0.0,
            rendering_intent: None,
            black_point_compensation: false,
        }
    }

    #[test]
    fn old_json_without_display_profile_is_srgb() {
        let mut value = serde_json::to_value(request()).unwrap();
        value.as_object_mut().unwrap().remove("display_profile");
        value.as_object_mut().unwrap().remove("proof_profile");
        value.as_object_mut().unwrap().remove("proof_paper_white");
        value.as_object_mut().unwrap().remove("proof_black_point");
        value.as_object_mut().unwrap().remove("rendering_intent");
        value.as_object_mut().unwrap().remove("black_point_compensation");
        let request: PreviewRequest = serde_json::from_value(value).unwrap();
        assert_eq!(request.display_profile, None);
        assert_eq!(request.proof_profile, None);
        assert_eq!(request.proof_paper_white, 1.0);
        assert_eq!(request.proof_black_point, 0.0);
        assert_eq!(request.rendering_intent, None);
        assert!(!request.black_point_compensation);
    }

    #[test]
    fn relative_without_bpc_matches_the_legacy_conversion() {
        let Some(profile) = crate::icc::profile("NikonZ6III-Generic") else { return };
        let image = RgbaImage::new(2, 1, vec![128, 64, 32, 255, 0, 255, 128, 255]);
        let (out, degraded) = with_display_profile(image.clone(), Some("NikonZ6III-Generic"), None, (1.0, 0.0), Rendering::default());
        let mut legacy = image.pixels.as_ref().clone();
        crate::icc::convert_srgb_to(&mut legacy, &profile);
        assert_eq!(&*out.pixels, &legacy, "relative + no BPC must equal the old path");
        assert!(degraded.is_none());
    }

    #[test]
    fn unknown_intent_falls_back_to_relative_without_panic() {
        let (fallback, note) = proof_rendering(Some(250), false);
        assert_eq!(fallback, Rendering::default());
        assert!(note.as_deref().is_some_and(|n| n.contains("unknown")));
        let (perceptual, note) = proof_rendering(Some(0), false);
        assert_eq!(perceptual, Rendering::default());
        assert!(note.as_deref().is_some_and(|n| n.contains("unsupported")));
        let (bpc_only, note) = proof_rendering(Some(250), true);
        assert!(bpc_only.black_point_compensation, "fallback keeps BPC");
        assert!(note.is_some());
    }

    #[test]
    fn bpc_without_bkpt_falls_back_to_relative_and_reports() {
        let Some(profile) = crate::icc::profile("NikonZ6III-Generic") else { return };
        assert!(profile.bkpt.is_none(), "LUT camera profile has no bkpt tag");
        let image = RgbaImage::new(1, 1, vec![128, 64, 32, 255]);
        let (plain, _) = with_display_profile(image.clone(), Some("NikonZ6III-Generic"), None, (1.0, 0.0), Rendering::default());
        let bpc = Rendering { black_point_compensation: true, ..Rendering::default() };
        let (out, degraded) = with_display_profile(image, Some("NikonZ6III-Generic"), None, (1.0, 0.0), bpc);
        assert_eq!(out.pixels, plain.pixels, "unsupported BPC must not change pixels");
        assert!(degraded.as_deref().is_some_and(|d| d.contains("not supported")));
    }

    #[test]
    fn display_profile_round_trips() {
        let mut request = request();
        request.display_profile = Some(PathBuf::from("C:/icc/display.icc"));
        let text = serde_json::to_string(&request).unwrap();
        let back: PreviewRequest = serde_json::from_str(&text).unwrap();
        assert_eq!(back.display_profile, request.display_profile);
    }

    #[test]
    fn missing_profile_keeps_srgb_and_reports_reason() {
        let image = RgbaImage::new(1, 1, vec![10, 20, 30, 255]);
        let (out, degraded) = with_display_profile(image, None, Some(Path::new("C:/no/such/profile.icc")), (1.0, 0.0), Rendering::default());
        assert_eq!(&*out.pixels, &[10, 20, 30, 255]);
        assert!(degraded.is_some(), "missing profile must report a reason");
    }

    #[test]
    fn system_srgb_profile_is_a_no_op() {
        let path = Path::new(r"C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm");
        if !path.exists() {
            return;
        }
        let image = RgbaImage::new(1, 1, vec![12, 200, 99, 255]);
        let (out, degraded) = with_display_profile(image, None, Some(path), (1.0, 0.0), Rendering::default());
        assert_eq!(&*out.pixels, &[12, 200, 99, 255]);
        assert!(degraded.is_none());
    }

    #[test]
    fn no_proof_profile_keeps_pixels() {
        let image = RgbaImage::new(1, 1, vec![7, 8, 9, 255]);
        let (out, degraded) = with_display_profile(image, None, None, (1.0, 0.0), Rendering::default());
        assert_eq!(&*out.pixels, &[7, 8, 9, 255]);
        assert!(degraded.is_none());
    }

    #[test]
    fn unknown_proof_profile_keeps_pixels_and_reports_reason() {
        let image = RgbaImage::new(1, 1, vec![7, 8, 9, 255]);
        let (out, degraded) = with_display_profile(image, Some("No Such Profile"), None, (1.0, 0.0), Rendering::default());
        assert_eq!(&*out.pixels, &[7, 8, 9, 255]);
        assert!(degraded.as_deref().is_some_and(|d| d.contains("No Such Profile")));
    }

    #[test]
    fn known_proof_profile_changes_pixels() {
        if crate::icc::profile("NikonZ6III-Generic").is_none() {
            return;
        }
        let image = RgbaImage::new(1, 1, vec![128, 64, 32, 255]);
        let (out, degraded) = with_display_profile(image, Some("NikonZ6III-Generic"), None, (1.0, 0.0), Rendering::default());
        assert_ne!(&*out.pixels, &[128, 64, 32, 255]);
        assert!(degraded.is_none());
    }

    #[test]
    fn neutral_paper_levels_are_byte_identical() {
        let mut image = RgbaImage::new(1, 1, vec![5, 128, 250, 255]);
        apply_paper_levels(&mut image, 1.0, 0.0);
        assert_eq!(&*image.pixels, &[5, 128, 250, 255]);
    }

    #[test]
    fn non_neutral_paper_levels_change_and_stay_in_range() {
        let mut image = RgbaImage::new(2, 1, vec![0, 64, 255, 255, 255, 0, 128, 255]);
        apply_paper_levels(&mut image, 0.8, 0.1);
        assert_eq!(&*image.pixels, &[26, 70, 204, 255, 204, 26, 115, 255]);
        let mut clamped = RgbaImage::new(1, 1, vec![128, 64, 32, 255]);
        apply_paper_levels(&mut clamped, 2.0, 2.0);
        assert_eq!(&*clamped.pixels, &[255, 255, 255, 255]);
    }

    fn apply_paper_levels_serial(image: &mut RgbaImage, paper_white: f32, black_point: f32) {
        let white = paper_white.clamp(0.0, 1.0);
        let black = black_point.clamp(0.0, 1.0).min(white);
        for px in Arc::make_mut(&mut image.pixels).chunks_exact_mut(4) {
            for channel in &mut px[..3] {
                *channel = (black * 255.0 + *channel as f32 * (white - black))
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }

    #[test]
    fn parallel_paper_levels_match_the_serial_original_exactly() {
        let (w, h) = (73u32, 41u32);
        let pixels: Vec<u8> = (0..w * h * 4)
            .map(|i| i.wrapping_mul(73).wrapping_add(i / 11) as u8)
            .collect();
        for levels in [(0.8, 0.1), (0.53, 0.07), (0.0, 0.0), (1.0, 0.0)] {
            let mut expected =
                RgbaImage::new(w, h, pixels.clone());
            let mut actual =
                RgbaImage::new(w, h, pixels.clone());
            apply_paper_levels_serial(&mut expected, levels.0, levels.1);
            apply_paper_levels(&mut actual, levels.0, levels.1);
            assert_eq!(actual.pixels, expected.pixels, "{levels:?}");
        }
    }

    #[test]
    #[ignore]
    fn bench_parallel_paper_levels_24mp() {
        use std::time::Instant;

        let (w, h) = (6000u32, 4000u32);
        let pixels: Vec<u8> = (0..w * h * 4)
            .map(|i| i.wrapping_mul(40503).wrapping_add(i / 17) as u8)
            .collect();
        let measure = |parallel: bool| {
            let mut times = Vec::with_capacity(7);
            for _ in 0..7 {
                let mut image = RgbaImage::new(w, h, pixels.clone());
                let started = Instant::now();
                if parallel {
                    apply_paper_levels(&mut image, 0.8, 0.1);
                } else {
                    apply_paper_levels_serial(&mut image, 0.8, 0.1);
                }
                times.push(started.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(image);
            }
            times.sort_by(f64::total_cmp);
            (times[times.len() / 2], times[(times.len() * 95 / 100).max(times.len() - 1)])
        };
        let (serial50, serial95) = measure(false);
        let (parallel50, parallel95) = measure(true);
        println!(
            "6000x4000 paper levels: serial {serial50:.3}/{serial95:.3} ms, parallel \
             {parallel50:.3}/{parallel95:.3} ms, {:.2}x P50",
            serial50 / parallel50
        );
    }

    #[test]
    fn profile_cache_hits_same_key_and_misses_other_profile() {
        let frame = crate::preview_cache::key(ImageId(1), &AdjustmentSet::default(), NormRect::FULL, 64, false, None, false);
        let mut cache = ProfileCache::default();
        let image = RgbaImage::new(1, 1, vec![128, 64, 32, 255]);
        let key = ProfileKey::new(frame, Some("NikonZ6III-Generic"), None, (0.8, 0.1), Rendering::default());
        let (a, _) = cache.converted(key.clone(), image.clone(), Some("NikonZ6III-Generic"), None, (0.8, 0.1), Rendering::default());
        assert_eq!(cache.len(), 1);
        let (b, _) = cache.converted(key, image.clone(), Some("NikonZ6III-Generic"), None, (0.8, 0.1), Rendering::default());
        assert_eq!(cache.len(), 1, "same key must hit");
        assert_eq!(a.pixels, b.pixels);
        let other = ProfileKey::new(frame, Some("NikonZ6III-Generic"), None, (0.6, 0.2), Rendering::default());
        cache.converted(other, image.clone(), Some("NikonZ6III-Generic"), None, (0.6, 0.2), Rendering::default());
        assert_eq!(cache.len(), 2, "different paper levels must miss");
        let display = ProfileKey::new(frame, None, Some(Path::new("C:/icc/a.icc")), (1.0, 0.0), Rendering::default());
        cache.converted(display, image.clone(), None, Some(Path::new("C:/icc/a.icc")), (1.0, 0.0), Rendering::default());
        assert_eq!(cache.len(), 3, "different display profile must miss");
        let plain = ProfileKey::new(frame, None, None, (1.0, 0.0), Rendering::default());
        cache.converted(plain, image, None, None, (1.0, 0.0), Rendering::default());
        assert_eq!(cache.len(), 3, "no profile skips the cache");
    }

    #[test]
    fn gamut_warning_counts_out_of_gamut_pixels() {
        let gray = RgbaImage::new(1, 1, vec![128, 128, 128, 255]);
        assert!(gamut_warning(&gray, None).is_none(), "proofing off must cost nothing");
        if crate::icc::profile("NikonZ6III-Generic").is_none() {
            return;
        }
        let neutral = gamut_warning(&gray, Some("NikonZ6III-Generic")).unwrap();
        assert_eq!(neutral.total, 1);
        assert_eq!(neutral.out_of_gamut, 0, "mid grey must stay inside");
        if crate::icc::profile("Phase One Effects - Sepia").is_none() {
            return;
        }
        let saturated = RgbaImage::new(1, 1, vec![0, 255, 0, 255]);
        let warning = gamut_warning(&saturated, Some("Phase One Effects - Sepia")).unwrap();
        assert!(warning.out_of_gamut > 0, "saturated green through a sepia proof must warn");
    }
}
