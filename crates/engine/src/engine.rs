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
    MatchLook, NormRect, TetherStatus, UserCollection,
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
}

fn unit() -> f32 {
    1.0
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
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GamutWarning {
    pub out_of_gamut: u64,
    pub total: u64,
}

/// Workspace sRGB -> proof profile -> display profile, 8-bit. The cached frame
/// stays sRGB; only requested conversions copy the pixels. Proof failure keeps
/// the original frame and reports why.
pub(crate) fn with_display_profile(image: RgbaImage, proof: Option<&str>, path: Option<&Path>, levels: (f32, f32)) -> (RgbaImage, Option<String>) {
    let mut image = image;
    if let Some(name) = proof {
        let Some(profile) = crate::icc::profile(name) else {
            return (image, Some(format!("proof profile {name}: not found")));
        };
        if !crate::icc::is_srgb(&profile) {
            let pixels = Arc::make_mut(&mut image.pixels);
            crate::icc::convert_srgb_to(pixels, &profile);
        }
        apply_paper_levels(&mut image, levels.0, levels.1);
    }
    let Some(path) = path else { return (image, None) };
    match crate::icc::load_profile(path) {
        Ok(profile) => {
            if !crate::icc::is_srgb(&profile) {
                let pixels = Arc::make_mut(&mut image.pixels);
                crate::icc::convert_srgb_to(pixels, &profile);
            }
            (image, None)
        }
        Err(err) => (image, Some(format!("display profile {}: {err}", path.display()))),
    }
}

/// Bounded linear paper white / black point on proofed pixels. Neutral is an
/// exact no-op; this is a stand-in for, not a copy of, C1's paper/ink model.
fn apply_paper_levels(image: &mut RgbaImage, paper_white: f32, black_point: f32) {
    let white = paper_white.clamp(0.0, 1.0);
    let black = black_point.clamp(0.0, 1.0).min(white);
    if white >= 1.0 && black <= 0.0 {
        return;
    }
    for px in Arc::make_mut(&mut image.pixels).chunks_exact_mut(4) {
        for channel in &mut px[..3] {
            *channel = (black * 255.0 + *channel as f32 * (white - black)).round().clamp(0.0, 255.0) as u8;
        }
    }
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
}

impl ProfileKey {
    pub(crate) fn new(frame: crate::preview_cache::PreviewKey, proof: Option<&str>, display: Option<&Path>, levels: (f32, f32)) -> Self {
        let levels = if proof.is_some() { (levels.0.to_bits(), levels.1.to_bits()) } else { (1.0f32.to_bits(), 0.0f32.to_bits()) };
        Self { frame, display: display.map(Path::to_path_buf), proof: proof.map(str::to_string), levels }
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
        let (out, degraded) = with_display_profile(image, proof, display, levels);
        self.entries.push((key, out.clone(), degraded.clone()));
        const CAPACITY: usize = 4;
        const MAX_BYTES: usize = 256 << 20;
        while self.entries.len() > CAPACITY {
            self.entries.remove(0);
        }
        while self.entries.len() > 1 && self.entries.iter().map(|e| e.1.pixels.len()).sum::<usize>() > MAX_BYTES {
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
    let mut out_of_gamut = 0u64;
    for (src, back) in image.pixels.chunks_exact(4).zip(round.chunks_exact(4)) {
        if (0..3).any(|i| (src[i] as i32 - back[i] as i32).abs() > THRESHOLD) {
            out_of_gamut += 1;
        }
    }
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
    pub export_formats: Vec<ExportFormat>,
}

pub trait ImageEngine: Send + Sync + 'static {
    /// Explicit-input portrait stages; availability is separate from automatic UI processing.
    fn portrait_capabilities(&self) -> Vec<crate::portrait::Capability> {
        crate::portrait::capabilities()
    }
    fn render_portrait_stage(&self, request: &crate::portrait::Request<'_>) -> anyhow::Result<crate::portrait::Frame> {
        crate::portrait::render(request)
    }
    fn export_portrait_stage(&self, request: &crate::portrait::Request<'_>, info: &ImageInfo,
        recipe: &crate::ProcessRecipe, directory: &Path) -> anyhow::Result<PathBuf> {
        crate::portrait::export(request, info, recipe, directory)
    }
    /// Human-readable engine name and version, shown in About.
    fn name(&self) -> String;

    /// See [`EngineCapabilities`]; engines override with what they can really do.
    fn capabilities(&self) -> EngineCapabilities {
        EngineCapabilities::default()
    }

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
        Ok(with_display_profile(img, proof_profile, display_profile, levels).0)
    }
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
    /// Every `.fcrv` stem the base-characteristics film curve accepts.
    fn film_curves(&self) -> Vec<String> {
        Vec::new()
    }
    /// Reads a Capture One `.costyle` / `.copreset` into [`Self::styles`].
    fn import_style(&self, _path: &Path) -> anyhow::Result<StyleInfo> {
        anyhow::bail!("style import is not supported by this engine")
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    #[test]
    fn old_json_without_display_profile_is_srgb() {
        let mut value = serde_json::to_value(request()).unwrap();
        value.as_object_mut().unwrap().remove("display_profile");
        value.as_object_mut().unwrap().remove("proof_profile");
        value.as_object_mut().unwrap().remove("proof_paper_white");
        value.as_object_mut().unwrap().remove("proof_black_point");
        let request: PreviewRequest = serde_json::from_value(value).unwrap();
        assert_eq!(request.display_profile, None);
        assert_eq!(request.proof_profile, None);
        assert_eq!(request.proof_paper_white, 1.0);
        assert_eq!(request.proof_black_point, 0.0);
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
        let (out, degraded) = with_display_profile(image, None, Some(Path::new("C:/no/such/profile.icc")), (1.0, 0.0));
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
        let (out, degraded) = with_display_profile(image, None, Some(path), (1.0, 0.0));
        assert_eq!(&*out.pixels, &[12, 200, 99, 255]);
        assert!(degraded.is_none());
    }

    #[test]
    fn no_proof_profile_keeps_pixels() {
        let image = RgbaImage::new(1, 1, vec![7, 8, 9, 255]);
        let (out, degraded) = with_display_profile(image, None, None, (1.0, 0.0));
        assert_eq!(&*out.pixels, &[7, 8, 9, 255]);
        assert!(degraded.is_none());
    }

    #[test]
    fn unknown_proof_profile_keeps_pixels_and_reports_reason() {
        let image = RgbaImage::new(1, 1, vec![7, 8, 9, 255]);
        let (out, degraded) = with_display_profile(image, Some("No Such Profile"), None, (1.0, 0.0));
        assert_eq!(&*out.pixels, &[7, 8, 9, 255]);
        assert!(degraded.as_deref().is_some_and(|d| d.contains("No Such Profile")));
    }

    #[test]
    fn known_proof_profile_changes_pixels() {
        if crate::icc::profile("NikonZ6III-Generic").is_none() {
            return;
        }
        let image = RgbaImage::new(1, 1, vec![128, 64, 32, 255]);
        let (out, degraded) = with_display_profile(image, Some("NikonZ6III-Generic"), None, (1.0, 0.0));
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

    #[test]
    fn profile_cache_hits_same_key_and_misses_other_profile() {
        let frame = crate::preview_cache::key(ImageId(1), &AdjustmentSet::default(), NormRect::FULL, 64, false, None, false);
        let mut cache = ProfileCache::default();
        let image = RgbaImage::new(1, 1, vec![128, 64, 32, 255]);
        let key = ProfileKey::new(frame, Some("NikonZ6III-Generic"), None, (0.8, 0.1));
        let (a, _) = cache.converted(key.clone(), image.clone(), Some("NikonZ6III-Generic"), None, (0.8, 0.1));
        assert_eq!(cache.len(), 1);
        let (b, _) = cache.converted(key, image.clone(), Some("NikonZ6III-Generic"), None, (0.8, 0.1));
        assert_eq!(cache.len(), 1, "same key must hit");
        assert_eq!(a.pixels, b.pixels);
        let other = ProfileKey::new(frame, Some("NikonZ6III-Generic"), None, (0.6, 0.2));
        cache.converted(other, image.clone(), Some("NikonZ6III-Generic"), None, (0.6, 0.2));
        assert_eq!(cache.len(), 2, "different paper levels must miss");
        let display = ProfileKey::new(frame, None, Some(Path::new("C:/icc/a.icc")), (1.0, 0.0));
        cache.converted(display, image.clone(), None, Some(Path::new("C:/icc/a.icc")), (1.0, 0.0));
        assert_eq!(cache.len(), 3, "different display profile must miss");
        let plain = ProfileKey::new(frame, None, None, (1.0, 0.0));
        cache.converted(plain, image, None, None, (1.0, 0.0));
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
