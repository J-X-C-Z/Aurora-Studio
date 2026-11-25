//! The application model: catalog, selection, edits with undo, and the preview pipeline.
//!
//! Views hold an `Entity<AppState>` and observe it. All engine calls run on the
//! background executor; results come back through `cx.spawn` and replace state
//! only if they are still the newest request.

mod binding;
mod clipboard;
mod collections;
mod editor;
pub(crate) mod render;
mod settings;
mod settings_state;
mod source;
mod arrange;
mod panel_order;
mod speed;
mod show;
mod nodes;
mod analysis;
mod cull;
mod ahead;
mod tasks;
mod device;
mod preview;
mod library;
pub(crate) mod scopes;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use gpui_kit::{App, AppContext, Context, Entity, EventEmitter, Pixels, RenderImage, SharedString};
use aurora_engine::*;
use rayon::prelude::*;

pub use binding::*;
pub use clipboard::ClipGroup;
pub use collections::CollNode;
pub use editor::{Edit, EditorState};
pub use settings::*;
pub use settings_state::SettingsState;
pub use source::Source;
pub use arrange::{Arrangement, Stack};
pub use speed::{SpeedEdit, SpeedParam};
pub use cull::Cull;
pub use ahead::MergeKind;
pub use tasks::TaskCenter;
pub use device::DeviceState;
pub use preview::PreviewState;
pub use library::LibraryState;
pub use scopes::ScopeKind;
pub use nodes::{GraphExt, NewNode, add_shape, flattened, flattened_checked, is_full};
pub mod open {
    pub use super::source::{demo, display_name, is_catalog, open};
}

use crate::i18n::t;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolTab {
    Library,
    Capture,
    Lens,
    Grading,
    Details,
    Portrait,
    Adjustments,
    Metadata,
    Output,
    Batch,
    /// The user's own selection of tool panels (Capture One's Quick tab).
    Quick,
}

impl ToolTab {
    /// Last of all is Quick, so the Ctrl+1…9 shortcuts keep their tabs.
    pub const ALL: [ToolTab; 11] = [
        ToolTab::Library,
        ToolTab::Capture,
        ToolTab::Lens,
        ToolTab::Grading,
        ToolTab::Details,
        ToolTab::Portrait,
        ToolTab::Adjustments,
        ToolTab::Metadata,
        ToolTab::Output,
        ToolTab::Batch,
        ToolTab::Quick,
    ];

    pub fn label(self) -> &'static str {
        match self {
            ToolTab::Library => t("Library", "图库"),
            ToolTab::Capture => t("Capture", "拍摄"),
            ToolTab::Lens => t("Lens", "镜头"),
            ToolTab::Grading => t("Grading", "调色"),
            ToolTab::Details => t("Details", "细节"),
            ToolTab::Portrait => t("Portrait", "人像"),
            ToolTab::Adjustments => t("Adjustments", "调整"),
            ToolTab::Metadata => t("Metadata", "元数据"),
            ToolTab::Output => t("Output", "输出"),
            ToolTab::Batch => t("Batch", "批处理"),
            ToolTab::Quick => t("Quick", "快速"),
        }
    }

    pub fn icon(self) -> gpui_kit::assets::IconName {
        use gpui_kit::assets::IconName as I;
        match self {
            ToolTab::Library => I::LibraryBig,
            ToolTab::Capture => I::Camera,
            ToolTab::Lens => I::Aperture,
            ToolTab::Grading => I::Palette,
            ToolTab::Details => I::ScanSearch,
            ToolTab::Portrait => I::ScanFace,
            ToolTab::Adjustments => I::SlidersHorizontal,
            ToolTab::Metadata => I::Info,
            ToolTab::Output => I::Upload,
            ToolTab::Batch => I::ListChecks,
            ToolTab::Quick => I::Bookmark,
        }
    }

    pub fn shortcut(self) -> Option<&'static str> {
        let i = ToolTab::ALL.iter().position(|t| *t == self)?;
        ["ctrl-1", "ctrl-2", "ctrl-3", "ctrl-4", "ctrl-5", "ctrl-6", "ctrl-7", "ctrl-8", "ctrl-9"].get(i).copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CursorTool {
    Select,
    Pan,
    Zoom,
    Crop,
    Straighten,
    Keystone,
    Spot,
    Brush,
    Eraser,
    LinearGradient,
    RadialGradient,
    ColorPicker,
    WhiteBalance,
    Annotate,
    Loupe,
}

impl CursorTool {
    pub fn label(self) -> &'static str {
        match self {
            CursorTool::Select => t("Select", "选择"),
            CursorTool::Pan => t("Pan", "平移"),
            CursorTool::Zoom => t("Zoom", "缩放"),
            CursorTool::Crop => t("Crop", "裁剪"),
            CursorTool::Straighten => t("Straighten", "拉直"),
            CursorTool::Keystone => t("Keystone", "梯形校正"),
            CursorTool::Spot => t("Heal & Clone", "修复与仿制"),
            CursorTool::Brush => t("Draw Mask", "绘制蒙版"),
            CursorTool::Eraser => t("Erase Mask", "擦除蒙版"),
            CursorTool::LinearGradient => t("Linear Gradient", "线性渐变"),
            CursorTool::RadialGradient => t("Radial Gradient", "径向渐变"),
            CursorTool::ColorPicker => t("Color Picker", "颜色取样"),
            CursorTool::WhiteBalance => t("Pick White Balance", "白平衡取样"),
            CursorTool::Annotate => t("Annotate", "标注"),
            CursorTool::Loupe => t("Loupe", "放大镜"),
        }
    }

    pub fn icon(self) -> gpui_kit::assets::IconName {
        use gpui_kit::assets::IconName as I;
        match self {
            CursorTool::Select => I::MousePointer2,
            CursorTool::Pan => I::Hand,
            CursorTool::Zoom => I::ZoomIn,
            CursorTool::Crop => I::Crop,
            CursorTool::Straighten => I::Ruler,
            CursorTool::Keystone => I::Grid2x2Check,
            CursorTool::Spot => I::Bandage,
            CursorTool::Brush => I::Brush,
            CursorTool::Eraser => I::Eraser,
            CursorTool::LinearGradient => I::Blend,
            CursorTool::RadialGradient => I::CircleDot,
            CursorTool::ColorPicker => I::Pipette,
            CursorTool::WhiteBalance => I::Pipette,
            CursorTool::Annotate => I::PenLine,
            CursorTool::Loupe => I::ScanEye,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            CursorTool::Select => "V",
            CursorTool::Pan => "H",
            CursorTool::Zoom => "Z",
            CursorTool::Crop => "C",
            CursorTool::Straighten => "R",
            CursorTool::Keystone => "K",
            CursorTool::Spot => "Q",
            CursorTool::Brush => "B",
            CursorTool::Eraser => "E",
            CursorTool::LinearGradient => "L",
            CursorTool::RadialGradient => "T",
            CursorTool::ColorPicker => "I",
            CursorTool::WhiteBalance => "W",
            CursorTool::Annotate => "A",
            CursorTool::Loupe => "O",
        }
    }

    /// Tools that paint into the selected layer's mask.
    pub fn is_mask_tool(self) -> bool {
        matches!(self, CursorTool::Brush | CursorTool::Eraser | CursorTool::LinearGradient | CursorTool::RadialGradient)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrowserMode {
    /// Filmstrip below the viewer.
    Filmstrip,
    /// Grid fills the work area; the viewer hides.
    Grid,
    /// Detailed list of files.
    List,
    Hidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareMode {
    Off,
    /// A draggable before/after divider over one image.
    Split,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    Name,
    CaptureDate,
    Rating,
    ColorTag,
    FileSize,
    /// The order the user dragged thumbnails into.
    Custom,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Filter {
    pub min_rating: i8,
    pub color_tag: Option<ColorTag>,
    pub search: String,
    pub edited_only: bool,
}

impl Default for Filter {
    fn default() -> Self {
        Self { min_rating: -1, color_tag: None, search: String::new(), edited_only: false }
    }
}

/// Zoom and pan of the viewer, in developed-frame space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewZoom {
    /// `None` fits the image; otherwise device pixels per frame pixel (1.0 = 100%).
    pub scale: Option<f32>,
    /// Normalized frame point at the viewport center.
    pub center: (f32, f32),
}

impl Default for ViewZoom {
    fn default() -> Self {
        Self { scale: None, center: (0.5, 0.5) }
    }
}

pub struct Frame {
    pub image: Arc<RenderImage>,
    /// Identity of the engine RGBA buffer used to build `image`.
    pub(crate) source_pixels: usize,
    /// The developed-frame region this frame shows.
    pub region: NormRect,
    pub(crate) geometry: Geometry,
    pub(crate) portrait_applied: bool,
    pub image_id: ImageId,
    /// Clipped highlights (red) and shadows (blue), when the exposure warning is on.
    pub warning: Option<Arc<RenderImage>>,
    /// Sharp edges in green, when the focus mask is on.
    pub focus: Option<Arc<RenderImage>>,
    /// Why the display profile was not applied; the frame is sRGB.
    pub degraded: Option<SharedString>,
    pub(crate) stamp: crate::perf::Stamp,
}

impl Frame {
    pub(crate) fn release(self, cx: &mut gpui_kit::App) {
        let Frame { image, warning, focus, .. } = self;
        cx.drop_image(image, None);
        if let Some(warning) = warning {
            cx.drop_image(warning, None);
        }
        if let Some(focus) = focus {
            cx.drop_image(focus, None);
        }
    }

    pub(crate) fn release_reusing_image(self, cx: &mut gpui_kit::App) {
        self.release_overlays(cx);
    }

    fn release_overlays(self, cx: &mut gpui_kit::App) {
        if let Some(w) = self.warning {
            cx.drop_image(w, None);
        }
        if let Some(f) = self.focus {
            cx.drop_image(f, None);
        }
    }
}

#[derive(Default)]
pub struct Preview {
    pub frame: Option<Frame>,
    pub before: Option<Frame>,
    pub(crate) full_after: Option<Frame>,
    pub(crate) full_before: Option<Frame>,
    pub histogram: Histogram,
    pub elapsed_ms: f32,
    pub(crate) generation: u64,
    pub(crate) in_flight: bool,
    pub(crate) pending: Option<(RenderQuality, std::time::Instant)>,
    pub(crate) before_dirty: bool,
    pub rendering: bool,
    /// The last render requested was a draft; the next gesture end refines it.
    pub(crate) draft: bool,
    /// The scope for the latest frame, when the scopes panel shows one.
    pub scope: Option<std::sync::Arc<RenderImage>>,
    /// Why the latest render of an image failed; cleared by its next frame.
    pub error: Option<(ImageId, SharedString)>,
    /// Inputs of the request that produced the shown after frame. Toggling
    /// warning/focus reuses that frame instead of rendering it again.
    pub(crate) last_request: Option<PreviewRequest>,
    /// Overlay settings used for the shown frame's warning and focus layers.
    pub(crate) last_overlay: (Option<(u8, u8, bool)>, Option<f32>),
    /// Scope kind drawn for the shown frame.
    pub(crate) last_scope: ScopeKind,
}

#[derive(Clone, Debug)]
pub struct Slideshow {
    pub playing: bool,
    /// Seconds per image.
    pub interval: f32,
    /// Distinguishes this run's timer from an earlier one.
    pub(crate) run: u64,
    /// The window was full screen before the show started.
    pub(crate) was_fullscreen: bool,
}

#[derive(Clone, Debug)]
pub struct JobView {
    pub id: u64,
    pub name: String,
    pub recipe: String,
    pub status: JobStatus,
}

/// Per-panel UI choices that are not adjustments.
#[derive(Clone, Debug)]
pub struct PanelPrefs {
    pub collapsed: HashSet<&'static str>,
    pub curve_channel: ToneChannel,
    pub levels_channel: ToneChannel,
    pub color_balance_mode: ColorBalanceMode,
    pub color_editor_mode: u8,
    pub selected_sector: usize,
    pub histogram_mode: u8,
    pub brush_size: f32,
    /// The spot tool clones from a clicked source point instead of healing.
    pub spot_clone: bool,
    pub brush_hardness: f32,
    pub brush_opacity: f32,
    pub brush_flow: f32,
    pub brush_auto_mask: bool,
    pub mask_overlay: bool,
    pub selected_color_range: Option<u64>,
    pub capture_format: String,
    pub capture_counter: u32,
    pub capture_adjust: u8,
    pub capture_folder: String,
    pub stack_styles: bool,
    /// The next color pick creates a color editor range (or picks the basic sector).
    pub pick_to_range: bool,
    /// Filter typed into the Portrait tab's search field.
    pub portrait_query: String,
    /// The Portrait tab shows the whole parameter catalogue ("All"), not only
    /// the controls the renderer uses ("Live").
    pub portrait_open: bool,
    /// Face reshape moves left and right together.
    pub portrait_link_lr: bool,
    /// AI mask kinds ticked in the mask panel, waiting for "Create".
    pub ai_pick: Vec<AiMaskKind>,
    /// What the scopes panel draws; `Off` costs nothing per frame.
    pub scope_kind: ScopeKind,
}

impl Default for PanelPrefs {
    fn default() -> Self {
        Self {
            // The portrait bands hold ~250 controls; they start closed.
            collapsed: crate::contract::BANDS.iter().map(|b| b.id).collect(),
            curve_channel: ToneChannel::Rgb,
            levels_channel: ToneChannel::Rgb,
            color_balance_mode: ColorBalanceMode::ThreeWay,
            color_editor_mode: 0,
            selected_sector: 0,
            histogram_mode: 0,
            brush_size: 0.05,
            spot_clone: false,
            brush_hardness: 50.0,
            brush_opacity: 100.0,
            brush_flow: 100.0,
            brush_auto_mask: false,
            mask_overlay: false,
            selected_color_range: None,
            capture_format: "[Job]_[Counter]".into(),
            capture_counter: 1,
            capture_adjust: 0,
            capture_folder: "Capture".into(),
            portrait_query: String::new(),
            portrait_open: false,
            portrait_link_lr: true,
            ai_pick: Vec::new(),
            scope_kind: ScopeKind::Off,
            pick_to_range: false,
            stack_styles: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ViewPrefs {
    /// The composition overlay image is shown over the viewer.
    pub overlay: bool,
    /// The node editor sits wide under the viewer instead of in the Layers tab.
    pub node_dock: bool,
    pub browser: BrowserMode,
    pub show_inspector: bool,
    pub compare: CompareMode,
    pub split: f32,
    pub exposure_warning: bool,
    pub proof: bool,
    /// Soft proofing with the selected recipe's output profile.
    pub soft_proof: bool,
    pub grid: bool,
    pub focus_mask: bool,
    pub thumb_size: f32,
    pub show_info_overlay: bool,
    pub viewer_gray: f32,
    pub inspector_width: f32,
    pub filmstrip_height: f32,
    /// Show every selected image side by side (Capture One's multiple view).
    pub multi: bool,
    /// The variant pinned as the reference in multiple view.
    pub compare_variant: Option<ImageId>,
    /// Tools on the right of the viewer instead of the left.
    pub inspector_right: bool,
}

impl Default for ViewPrefs {
    fn default() -> Self {
        Self {
            overlay: false,
            node_dock: false,
            browser: BrowserMode::Filmstrip,
            show_inspector: true,
            compare: CompareMode::Off,
            split: 0.5,
            exposure_warning: false,
            proof: false,
            soft_proof: false,
            grid: false,
            focus_mask: false,
            thumb_size: 120.0,
            show_info_overlay: true,
            viewer_gray: 0.11,
            inspector_width: 320.0,
            filmstrip_height: 132.0,
            multi: true,
            compare_variant: None,
            inspector_right: false,
        }
    }
}

/// What to do with images once they are imported.
#[derive(Clone, Default)]
pub struct ImportAfter {
    pub copyright: String,
    pub creator: String,
    pub keywords: Vec<String>,
    pub style: Option<StyleInfo>,
    pub album: Option<u64>,
}

pub enum StateEvent {
    /// A short message for the status bar or a toast.
    Notice(SharedString),
    CurrentChanged,
}

pub struct AppState {
    pub engine: Arc<dyn ImageEngine>,
    /// The catalog's images, selection, current image and browser arrangement.
    pub library: LibraryState,
    /// Per-image edits with undo history, and the editor's own selection.
    pub editor: EditorState,
    pub tool_tab: ToolTab,
    pub cursor: CursorTool,
    /// Preferences, per-panel choices and the engine identity read at start.
    pub prefs: SettingsState,
    /// Camera/tether status and the window's display identity.
    pub device: DeviceState,
    /// Viewer frames, thumbnails, zoom/pan and the color-picker sample.
    pub preview_state: PreviewState,
    pub capturing: bool,
    pub styles: Vec<StyleInfo>,
    pub icc_profiles: Vec<String>,
    pub icc_group: usize,
    pub icc_query: String,
    pub lens_profiles: Vec<String>,
    pub recipes: Vec<ProcessRecipe>,
    pub selected_recipe: usize,
    /// Export jobs and the background-task busy flags.
    pub tasks: TaskCenter,
    pub notice: Option<SharedString>,
    /// Last background save error already shown, so a sticky queue error is
    /// not re-notified on every commit.
    last_save_error: Option<String>,
    /// Result of the last "save to Capture One" action.
    pub c1_save_status: Option<C1SaveStatus>,
    /// Keys or layers the last save refused to write.
    pub c1_save_warnings: Vec<SharedString>,
    /// True while a C1 save runs; a second action is ignored.
    pub c1_save_busy: bool,
    /// AI mask kinds the engine refused this session (no model, no runtime).
    pub ai_unavailable: HashSet<AiMaskKind>,
    /// Closed-eye results for this session, shown on the thumbnails.
    pub eyes: HashMap<ImageId, EyeReport>,
    /// Images whose thumbnail the engine could not make.
    pub thumb_failed: HashSet<ImageId>,
    /// What this window has open.
    pub source: Source,
    /// The welcome screen covers the demo images until one is opened or the
    /// user chooses to look at the demo.
    pub welcome: bool,
    /// A folder or catalog being opened from the welcome screen.
    pub opening: Option<std::path::PathBuf>,
    /// Why the last open failed, shown on the welcome screen.
    pub open_error: Option<String>,
    /// A Speed Edit key is held.
    pub speed: Option<SpeedEdit>,
    /// Bumped when metadata changes outside the fields, so they reload.
    pub metadata_rev: u64,
    pub slideshow: Option<Slideshow>,
    pub cull: Option<Cull>,
    /// The cursor before the last change, and when it changed (Speed Edit
    /// puts it back after a key that also selects a cursor tool).
    pub(crate) prev_cursor: CursorTool,
    pub(crate) cursor_changed: std::time::Instant,
}

impl EventEmitter<StateEvent> for AppState {}

// Transitional seam: `tools/portrait.rs` is owned by another lane and still
// reaches `AppState.panels` directly. These impls let that field read resolve
// through the extracted `SettingsState` until that file migrates to `prefs.*`;
// no other module should rely on them.
impl AppState {
    pub fn new(engine: Arc<dyn ImageEngine>, source: Source, settings: Settings, cx: &mut Context<Self>) -> Self {
        let mut this = Self::assemble(engine, source, settings);
        let all: Vec<_> = this.library.images.iter().map(|i| i.id).collect();
        this.queue_thumbs(all, cx);
        this.request_render(RenderQuality::Final, cx);
        this.probe_tether(cx);
        this
    }

    /// Builds the model without touching GPUI: no thumbnails, preview or
    /// tether probe are started. [`Self::new`] does those; tests use this to
    /// exercise pure state transitions.
    fn assemble(engine: Arc<dyn ImageEngine>, source: Source, settings: Settings) -> Self {
        let images = engine.images();
        let edits = images
            .par_iter()
            .map(|i| (i.id, Edit { adj: engine.load_adjustments(i.id), ..Default::default() }))
            .collect();
        let current = images.first().map(|i| i.id);
        let recipes = vec![
            ProcessRecipe { id: 1, name: "JPEG · 全尺寸".into(), ..Default::default() },
            ProcessRecipe {
                id: 2,
                name: "TIFF 16-bit".into(),
                format: ExportFormat::Tiff,
                bit_depth: 16,
                icc_profile: "Adobe RGB (1998)".into(),
                enabled: false,
                ..Default::default()
            },
            ProcessRecipe {
                id: 3,
                name: "Web · 2048px".into(),
                scale: ScaleMode::LongEdge,
                scale_value: (2048, 0),
                quality: 82,
                output_sharpening: OutputSharpening::Screen,
                enabled: false,
                ..Default::default()
            },
        ];
        let camera_settings = engine.camera_settings();
        let engine_name = engine.name();
        let capabilities = crate::planned::capabilities(engine.as_ref());
        let collection_nodes = collections::flatten(engine.as_ref(), &engine.collections());
        let mut this = Self {
            styles: engine.styles(),
            icc_profiles: engine.icc_profiles(),
            icc_group: settings.icc_group.min(3) as usize,
            icc_query: String::new(),
            lens_profiles: engine.lens_profiles(),
            ai_unavailable: HashSet::new(),
            eyes: HashMap::new(),
            thumb_failed: HashSet::new(),
            engine,
            library: LibraryState {
                images,
                nodes: collection_nodes,
                selection: current.into_iter().collect(),
                current,
                ..LibraryState::default()
            },
            editor: EditorState::with_edits(edits),
            tool_tab: tab_from(&settings.workspace.tab).unwrap_or(ToolTab::Grading),
            cursor: CursorTool::Select,
            prefs: SettingsState::from_parts(engine_name, capabilities, settings),
            // Trust `tether_status` first: mock/imaging-only devices may report
            // `camera_status` without a camera the tether backend can open.
            device: DeviceState {
                camera: CameraStatus::Disconnected,
                camera_settings,
                ..DeviceState::default()
            },
            preview_state: PreviewState::default(),
            capturing: false,
            recipes,
            selected_recipe: 0,
            tasks: TaskCenter::default(),
            notice: None,
            last_save_error: None,
            c1_save_status: None,
            c1_save_warnings: Vec::new(),
            c1_save_busy: false,
            welcome: source == Source::Demo,
            source,
            opening: None,
            open_error: None,
            speed: None,
            metadata_rev: 0,
            slideshow: None,
            cull: None,
            prev_cursor: CursorTool::Select,
            cursor_changed: std::time::Instant::now(),
        };
        let layout = this.prefs.settings.workspace.clone();
        this.load_layout(&layout);
        this.load_arrangement();
        this.prefs.view.viewer_gray = this.prefs.settings.viewer_gray;
        this.prefs.view.thumb_size = this.prefs.settings.thumb_size;
        this.prefs.view.show_info_overlay = this.prefs.settings.info_overlay;
        this
    }

    // ───────────── Catalog ─────────────

    pub fn info(&self, id: ImageId) -> Option<&ImageInfo> {
        self.library.info(id)
    }

    pub fn current_info(&self) -> Option<&ImageInfo> {
        self.library.current_info()
    }

    /// Images passing the filter, in sort order.
    pub fn visible_images(&self) -> Vec<&ImageInfo> {
        let f = &self.library.filter;
        let search = f.search.to_lowercase();
        let in_trash = self.node(self.library.active_collection).is_some_and(|n| n.kind == CollectionKind::Trash);
        let members = self.members(self.library.active_collection);
        let mut out: Vec<&ImageInfo> = self.library
            .images
            .iter()
            .filter(|i| members.as_ref().is_none_or(|m| m.contains(&i.id)))
            .filter(|i| in_trash || !self.library.trash.contains(&i.id))
            .filter(|i| i.rating >= f.min_rating)
            .filter(|i| f.color_tag.is_none_or(|t| i.color_tag == t))
            .filter(|i| !f.edited_only || self.editor.edits.get(&i.id).is_some_and(|e| !e.adj.is_default()))
            .filter(|i| {
                search.is_empty()
                    || i.file_name.to_lowercase().contains(&search)
                    || i.keywords.iter().any(|k| k.contains(&search))
            })
            .collect();
        match self.library.sort {
            SortKey::Name => out.sort_by(|a, b| a.file_name.cmp(&b.file_name)),
            SortKey::CaptureDate => out.sort_by(|a, b| a.exif.captured_at.cmp(&b.exif.captured_at)),
            SortKey::Rating => out.sort_by(|a, b| b.rating.cmp(&a.rating)),
            SortKey::ColorTag => out.sort_by_key(|i| i.color_tag as u8),
            SortKey::FileSize => out.sort_by_key(|i| i.file_size),
            SortKey::Custom => out.sort_by_key(|i| (self.custom_rank(i.id), i.file_name.clone())),
        }
        if self.library.sort_desc {
            out.reverse();
        }
        self.fold_stacks(out)
    }

    pub fn select(&mut self, id: ImageId, extend: bool, range: bool, cx: &mut Context<Self>) {
        self.preview_state.hover_look = None;
        if range {
            if let Some(anchor) = self.library.current {
                let visible: Vec<ImageId> = self.visible_images().iter().map(|i| i.id).collect();
                let a = visible.iter().position(|i| *i == anchor);
                let b = visible.iter().position(|i| *i == id);
                if let (Some(a), Some(b)) = (a, b) {
                    let (lo, hi) = (a.min(b), a.max(b));
                    self.library.selection = visible[lo..=hi].iter().copied().collect();
                }
            }
        } else if extend {
            if !self.library.selection.remove(&id) {
                self.library.selection.insert(id);
            }
        } else {
            self.library.selection.clear();
            self.library.selection.insert(id);
        }
        self.set_current(id, cx);
        cx.notify();
    }

    /// Makes `id` the image the tools edit, leaving the selection alone.
    pub fn set_current(&mut self, id: ImageId, cx: &mut Context<Self>) {
        if self.library.current != Some(id) {
            self.end_gesture(cx);
            self.library.current = Some(id);
            self.preview_state.thumb_queue.prioritize(id);
            self.editor.active_layer = None;
            self.editor.active_node = None;
            self.preview_state.zoom = ViewZoom::default();
            self.preview_state.preview.before_dirty = true;
            for frame in [self.preview_state.preview.full_after.take(), self.preview_state.preview.full_before.take()].into_iter().flatten() {
                frame.release(cx);
            }
            self.preview_state.sampled_color = None;
            self.preview_state.picker_point = None;
            cx.emit(StateEvent::CurrentChanged);
            self.request_render(RenderQuality::Final, cx);
        }
        cx.notify();
    }

    /// The images multiple view shows, the pinned compare variant first; empty
    /// when a single image fills the viewer.
    pub fn multi_images(&self) -> Vec<ImageId> {
        if !self.prefs.view.multi {
            return Vec::new();
        }
        let pinned = self.prefs.view.compare_variant.filter(|id| self.info(*id).is_some());
        let mut ids: Vec<ImageId> = pinned.into_iter().collect();
        ids.extend(self.library.selection.iter().copied().filter(|id| Some(*id) != pinned).take(16));
        if ids.len() < 2 { Vec::new() } else { ids }
    }

    /// Moves the current image by `delta` within the visible order.
    pub fn step(&mut self, delta: isize, cx: &mut Context<Self>) {
        let visible: Vec<ImageId> = self.visible_images().iter().map(|i| i.id).collect();
        if visible.is_empty() {
            return;
        }
        let pos = self.library.current.and_then(|c| visible.iter().position(|i| *i == c)).unwrap_or(0) as isize;
        let next = (pos + delta).clamp(0, visible.len() as isize - 1) as usize;
        self.select(visible[next], false, false, cx);
    }

    pub fn select_all(&mut self, cx: &mut Context<Self>) {
        self.library.selection = self.visible_images().iter().map(|i| i.id).collect();
        cx.notify();
    }

    pub fn set_rating(&mut self, rating: i8, cx: &mut Context<Self>) {
        let targets: Vec<ImageId> = self.library.selection.iter().copied().collect();
        let mut updated = Vec::with_capacity(targets.len());
        for id in targets {
            if let Some(info) = self.library.images.iter_mut().find(|i| i.id == id) {
                info.rating = rating;
                updated.push(info.clone());
            }
        }
        if let Err(err) = self.engine.update_info_batch(&updated) {
            self.notify(format!("{} {err}", t("Could not save rating", "无法存储评级：")), cx);
        }
        cx.notify();
    }

    pub fn set_color_tag(&mut self, tag: ColorTag, cx: &mut Context<Self>) {
        let targets: Vec<ImageId> = self.library.selection.iter().copied().collect();
        let mut updated = Vec::with_capacity(targets.len());
        for id in targets {
            if let Some(info) = self.library.images.iter_mut().find(|i| i.id == id) {
                info.color_tag = tag;
                updated.push(info.clone());
            }
        }
        if let Err(err) = self.engine.update_info_batch(&updated) {
            self.notify(format!("{} {err}", t("Could not save color tag", "无法存储色标：")), cx);
        }
        cx.notify();
    }

    /// Saves the current image's IPTC fields and keywords as a metadata preset.
    pub fn save_metadata_preset(&mut self, name: String, cx: &mut Context<Self>) {
        let Some(info) = self.current_info() else { return };
        let preset = MetadataPreset { name: name.clone(), iptc: info.iptc.clone(), keywords: info.keywords.clone() };
        match self.prefs.settings.metadata_presets.iter_mut().find(|p| p.name == name) {
            Some(existing) => *existing = preset,
            None => self.prefs.settings.metadata_presets.push(preset),
        }
        self.prefs.settings.save();
        self.notify(format!("{} {name}", t("Saved metadata preset", "已存储元数据预设")), cx);
    }

    /// Applies metadata preset `index` to every selected image.
    pub fn apply_metadata_preset(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(preset) = self.prefs.settings.metadata_presets.get(index).cloned() else { return };
        let fill = |dst: &mut String, src: &String| {
            if !src.trim().is_empty() {
                *dst = src.clone();
            }
        };
        let ids: Vec<ImageId> = if self.library.selection.is_empty() { self.library.current.into_iter().collect() } else { self.library.selection.iter().copied().collect() };
        let mut updated = Vec::with_capacity(ids.len());
        for info in self.library.images.iter_mut().filter(|i| ids.contains(&i.id)) {
            let (d, s) = (&mut info.iptc, &preset.iptc);
            fill(&mut d.title, &s.title);
            fill(&mut d.description, &s.description);
            fill(&mut d.creator, &s.creator);
            fill(&mut d.copyright, &s.copyright);
            fill(&mut d.headline, &s.headline);
            fill(&mut d.city, &s.city);
            fill(&mut d.country, &s.country);
            fill(&mut d.job_id, &s.job_id);
            for k in &preset.keywords {
                if !info.keywords.contains(k) {
                    info.keywords.push(k.clone());
                }
            }
            updated.push(info.clone());
        }
        let save_error = self.engine.update_info_batch(&updated).err();
        // Fields keep their own text; a new revision starts them from the new values.
        self.metadata_rev += 1;
        let msg = if save_error.is_none() {
            format!("{} {} · {} {}", t("Applied", "已应用"), preset.name, ids.len(), t("images", "张图像"))
        } else {
            format!("{} {} {}", t("Could not save metadata for", "无法存储元数据："), ids.len(), t("images", "张图像"))
        };
        self.notify(msg, cx);
    }

    pub fn rename_metadata_preset(&mut self, index: usize, name: String, cx: &mut Context<Self>) {
        if let Some(p) = self.prefs.settings.metadata_presets.get_mut(index) {
            p.name = name;
            self.prefs.settings.save();
            cx.notify();
        }
    }

    pub fn delete_metadata_preset(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.prefs.settings.metadata_presets.len() {
            let p = self.prefs.settings.metadata_presets.remove(index);
            self.prefs.settings.save();
            self.notify(format!("{} {}", t("Deleted preset", "已删除预设"), p.name), cx);
        }
    }

    pub fn update_current_info(&mut self, f: impl FnOnce(&mut ImageInfo), cx: &mut Context<Self>) {
        if let Some(id) = self.library.current
            && let Some(info) = self.library.images.iter_mut().find(|i| i.id == id)
        {
            f(info);
            let _ = self.engine.update_info(info);
            cx.notify();
        }
    }

    pub fn import(&mut self, paths: Vec<std::path::PathBuf>, cx: &mut Context<Self>) {
        self.import_with(paths, ImportAfter::default(), cx);
    }

    /// Imports files, then applies metadata, a style and an album to what arrived.
    pub fn import_with(&mut self, paths: Vec<std::path::PathBuf>, after: ImportAfter, cx: &mut Context<Self>) {
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move { import_with_edits(engine.as_ref(), &paths) })
                .await;
            this.update(cx, |this, cx| match result {
                Ok(added) => {
                    let added: Vec<_> = added
                        .into_iter()
                        .map(|(info, edit)| {
                            this.editor.edits.insert(info.id, edit);
                            info
                        })
                        .collect();
                    let ids: Vec<_> = added.iter().map(|i| i.id).collect();
                    let n = added.len();
                    this.library.recent_imports.extend(ids.iter().copied());
                    this.library.images.extend(added);
                    for id in &ids {
                        if let Some(info) = this.library.images.iter_mut().find(|i| i.id == *id) {
                            if !after.copyright.is_empty() {
                                info.iptc.copyright = after.copyright.clone();
                            }
                            if !after.creator.is_empty() {
                                info.iptc.creator = after.creator.clone();
                            }
                            for k in &after.keywords {
                                if !info.keywords.contains(k) {
                                    info.keywords.push(k.clone());
                                }
                            }
                            let _ = this.engine.update_info(info);
                        }
                    }
                    if let Some(album) = after.album {
                        this.add_to_album(album, &ids, cx);
                    }
                    if let Some(style) = &after.style {
                        this.library.selection = ids.iter().copied().collect();
                        this.apply_style(style, false, cx);
                    }
                    this.queue_thumbs(ids.clone(), cx);
                    if let Some(first) = ids.first() {
                        this.select(*first, false, false, cx);
                    }
                    this.notify(format!("{} {n} {}", t("Imported", "已导入"), t("images", "张图像")), cx);
                }
                Err(err) => {
                    log::warn!("import: {err:#}");
                    let reason = match ImportErrorKind::of(&err) {
                        ImportErrorKind::NoLibRaw => t("LibRaw is not installed, so RAW files cannot be decoded", "本机缺少 LibRaw，无法解码 RAW"),
                        ImportErrorKind::DecodeFailed => t("the file could not be decoded", "文件无法解码"),
                        ImportErrorKind::TooLarge => t("the file is too large for this build", "文件超出本版本可处理的规模"),
                        ImportErrorKind::Unsupported => t("the format is not supported", "不支持该格式"),
                        ImportErrorKind::Io => t("the file could not be read", "文件无法读取"),
                        ImportErrorKind::Other => t("see the log for details", "详情见日志"),
                    };
                    this.notify(format!("{}: {reason}", t("Import failed", "导入失败")), cx)
                }
            })
            .ok();
        })
        .detach();
    }

    pub fn new_variant(&mut self, clean: bool, cx: &mut Context<Self>) {
        let Some(id) = self.library.current else { return };
        let adj = if clean { AdjustmentSet::default() } else { self.adj().clone() };
        match self.engine.new_variant(id, &adj) {
            Ok(info) => {
                let new_id = info.id;
                let pos = self.library.images.iter().position(|i| i.id == id).map_or(self.library.images.len(), |p| p + 1);
                self.library.images.insert(pos, info);
                self.editor.edits.insert(new_id, Edit { adj, ..Default::default() });
                self.queue_thumbs(vec![new_id], cx);
                self.select(new_id, false, false, cx);
            }
            Err(err) => self.notify(err.to_string(), cx),
        }
    }

    pub fn notify(&mut self, message: impl Into<SharedString>, cx: &mut Context<Self>) {
        let message = message.into();
        self.notice = Some(message.clone());
        cx.emit(StateEvent::Notice(message));
        cx.notify();
    }

    // ───────────── Editing ─────────────

    pub fn adj(&self) -> &AdjustmentSet {
        static EMPTY: std::sync::OnceLock<AdjustmentSet> = std::sync::OnceLock::new();
        self.library.current
            .and_then(|id| self.editor.edit_for(id))
            .map(|e| &e.adj)
            .unwrap_or_else(|| EMPTY.get_or_init(AdjustmentSet::default))
    }

    pub fn scope(&self) -> Scope {
        match self.editor.active_layer {
            Some(id) if self.adj().layers.iter().any(|l| l.id == id) => Scope::Layer(id),
            _ => Scope::Global,
        }
    }

    pub fn icc_picker(&self, current: Option<&str>, query: &str) -> Vec<String> {
        use aurora_engine::icc::ProfileGroup;
        let group = match self.icc_group {
            1 => Some(ProfileGroup::Camera),
            2 => Some(ProfileGroup::Standard),
            3 => Some(ProfileGroup::Output),
            _ => None,
        };
        let query = query.trim();
        let mut names = aurora_engine::icc::profile_menu(group, (!query.is_empty()).then_some(query), 300);
        if let Some(current) = current {
            if !names.iter().any(|name| name == current) {
                names.insert(0, current.to_string());
            }
        }
        names
    }

    pub fn active_layer_ref(&self) -> Option<&Layer> {
        let id = self.editor.active_layer?;
        self.adj().layers.iter().find(|l| l.id == id)
    }

    pub fn can_undo(&self) -> bool {
        self.library.current.is_some_and(|id| self.editor.can_undo(id))
    }

    pub fn can_redo(&self) -> bool {
        self.library.current.is_some_and(|id| self.editor.can_redo(id))
    }

    pub fn history(&self) -> (Vec<SharedString>, Vec<SharedString>) {
        self.library.current.map(|id| self.editor.history_entries(id)).unwrap_or_default()
    }

    /// Applies a change while a control is being dragged. The first call of a gesture
    /// records the state to undo to; [`Self::end_gesture`] commits it as one step.
    pub fn edit_live(&mut self, label: impl Into<SharedString>, f: impl FnOnce(&mut AdjustmentSet), cx: &mut Context<Self>) {
        self.edit_live_inner(label, f, Some(cx));
    }

    /// The state half of [`Self::edit_live`]. `None` skips the GPUI render and
    /// notification so tests can drive the same transition without a window.
    fn edit_live_inner(
        &mut self,
        label: impl Into<SharedString>,
        f: impl FnOnce(&mut AdjustmentSet),
        cx: Option<&mut Context<Self>>,
    ) {
        self.apply_gesture_edit(label, f);
        if let Some(cx) = cx {
            self.request_render(RenderQuality::Draft, cx);
            cx.notify();
        }
    }

    fn apply_gesture_edit(&mut self, label: impl Into<SharedString>, f: impl FnOnce(&mut AdjustmentSet)) {
        let Some(id) = self.library.current else { return };
        let edit = self.editor.edits.entry(id).or_default();
        if edit.gesture.is_none() {
            edit.gesture = Some((label.into(), edit.adj.clone()));
        }
        f(&mut edit.adj);
    }

    /// Ends the current gesture, if any, as a single undo step.
    #[track_caller]
    pub fn end_gesture(&mut self, cx: &mut Context<Self>) {
        self.end_gesture_inner(Some(cx));
    }

    /// The state half of [`Self::end_gesture`]; see [`Self::edit_live_inner`].
    #[track_caller]
    fn end_gesture_inner(&mut self, mut cx: Option<&mut Context<Self>>) {
        let Some(id) = self.library.current else { return };
        let Some(edit) = self.editor.edits.get_mut(&id) else { return };
        let Some((label, before)) = edit.gesture.take() else {
            // Panning has no undo step but still leaves a draft on screen.
            if self.preview_state.preview.draft {
                if let Some(cx) = cx {
                    self.request_render(RenderQuality::Final, cx);
                }
            }
            return;
        };
        if crate::perf::enabled() {
            crate::perf::record(serde_json::json!({"event": "gesture_commit", "caller": std::panic::Location::caller().to_string()}));
        }
        if before != edit.adj {
            edit.undo.push((label, before));
            edit.redo.clear();
            if edit.undo.len() > self.prefs.settings.undo_levels.max(10) {
                edit.undo.remove(0);
            }
            self.after_commit_inner(id, cx.as_deref_mut());
        }
        if let Some(cx) = cx {
            self.request_render(RenderQuality::Final, cx);
            cx.notify();
        }
    }

    /// A discrete change: one undo step, final-quality render.
    pub fn edit(&mut self, label: impl Into<SharedString>, f: impl FnOnce(&mut AdjustmentSet), cx: &mut Context<Self>) {
        self.edit_inner(label, f, Some(cx));
    }

    /// The state half of [`Self::edit`]; see [`Self::edit_live_inner`].
    fn edit_inner(
        &mut self,
        label: impl Into<SharedString>,
        f: impl FnOnce(&mut AdjustmentSet),
        mut cx: Option<&mut Context<Self>>,
    ) {
        self.end_gesture_inner(cx.as_deref_mut());
        self.apply_gesture_edit(label, f);
        self.end_gesture_inner(cx);
    }

    /// Applies one change to every selected image (Capture One's "Edit selected variants").
    pub fn edit_selected(&mut self, label: impl Into<SharedString>, f: impl Fn(&mut AdjustmentSet), cx: &mut Context<Self>) {
        self.edit_selected_inner(label, f, Some(cx));
    }

    /// The state half of [`Self::edit_selected`]; see [`Self::edit_live_inner`].
    fn edit_selected_inner(
        &mut self,
        label: impl Into<SharedString>,
        f: impl Fn(&mut AdjustmentSet),
        mut cx: Option<&mut Context<Self>>,
    ) {
        let label = label.into();
        self.end_gesture_inner(cx.as_deref_mut());
        let ids: Vec<ImageId> = self.library.selection.iter().copied().collect();
        let mut changed = Vec::new();
        for id in ids {
            let edit = self.editor.edits.entry(id).or_default();
            let before = edit.adj.clone();
            f(&mut edit.adj);
            if before != edit.adj {
                edit.undo.push((label.clone(), before));
                edit.redo.clear();
                changed.push(id);
            }
        }
        self.after_commit_many_inner(changed, cx.as_deref_mut());
        self.preview_state.preview.before_dirty = true;
        if let Some(cx) = cx {
            self.request_render(RenderQuality::Final, cx);
            cx.notify();
        }
    }

    pub(crate) fn after_commit(&mut self, id: ImageId, cx: &mut Context<Self>) {
        self.after_commit_inner(id, Some(cx));
    }

    fn after_commit_inner(&mut self, id: ImageId, cx: Option<&mut Context<Self>>) {
        self.after_commit_many_inner(vec![id], cx);
    }

    fn after_commit_many(&mut self, ids: Vec<ImageId>, cx: &mut Context<Self>) {
        self.after_commit_many_inner(ids, Some(cx));
    }

    /// Commits `ids` to the engine. The save path is identical with or
    /// without a GPUI context; `None` only skips notices, thumbnail refreshes
    /// and the render request.
    fn after_commit_many_inner(&mut self, ids: Vec<ImageId>, mut cx: Option<&mut Context<Self>>) {
        if ids.is_empty() {
            return;
        }
        let items: Vec<(ImageId, AdjustmentSet)> = ids
            .iter()
            .filter_map(|id| self.editor.edits.get(id).map(|edit| (*id, edit.adj.clone())))
            .collect();
        // Engines with a bounded writer use it so the UI thread does not wait
        // for N files + the catalog index. The synchronous report remains the
        // fallback (MockEngine and any engine without a queue).
        if let Some(summary) = self.engine.save_adjustments_queued(&items) {
            if let Some(message) = queued_save_notice(&summary) {
                if let Some(cx) = cx.as_deref_mut() {
                    self.notify(message, cx);
                }
            }
            // A sticky background error (disk/index) is surfaced once; a later
            // successful status clears it.
            if let Some(status) = self.engine.pending_save_status() {
                let error = status.index_error.clone().or(status.last_error.clone());
                if error != self.last_save_error {
                    if let Some(error) = &error {
                        if let Some(cx) = cx.as_deref_mut() {
                            self.notify(
                                format!("{} {error}", t("Could not save adjustments", "无法存储调整：")),
                                cx,
                            );
                        }
                    }
                    self.last_save_error = error;
                }
            }
        } else {
            let report = self.engine.save_adjustments_batch_report(&items);
            if let Some(message) = save_report_notice(&report) {
                if let Some(cx) = cx.as_deref_mut() {
                    self.notify(message, cx);
                }
            }
            self.last_save_error = None;
        }
        for id in &ids {
            if let Some(edit) = self.editor.edits.get(id) {
                let changed = !edit.adj.is_default();
                if let Some(info) = self.library.images.iter_mut().find(|info| info.id == *id) {
                    info.has_adjustments = changed;
                }
            }
        }
        if let Some(cx) = cx {
            self.queue_thumbs(ids, cx);
        }
    }

    /// Flushes queued saves (bounded) and quits. The UI may block for up to the
    /// timeout at close; a failure is logged and shown so unsaved work is not
    /// silently dropped.
    pub fn flush_and_quit(&mut self, cx: &mut Context<Self>) {
        if let Some(report) =
            self.engine.flush_pending_saves(std::time::Duration::from_secs(2))
        {
            if let Some(message) = flush_report_notice(&report) {
                log::warn!("{message}");
                self.notify(message, cx);
            }
        }
        cx.quit();
    }

    pub fn undo(&mut self, cx: &mut Context<Self>) {
        self.undo_inner(Some(cx));
    }

    /// The state half of [`Self::undo`]; see [`Self::edit_live_inner`].
    fn undo_inner(&mut self, mut cx: Option<&mut Context<Self>>) {
        self.end_gesture_inner(cx.as_deref_mut());
        let Some(id) = self.library.current else { return };
        let Some(label) = self.editor.undo_step(id) else { return };
        self.after_commit_inner(id, cx.as_deref_mut());
        if let Some(cx) = cx {
            self.request_render(RenderQuality::Final, cx);
            self.notify(format!("{} {label}", t("Undo", "撤销")), cx);
        }
    }

    pub fn redo(&mut self, cx: &mut Context<Self>) {
        self.redo_inner(Some(cx));
    }

    /// The state half of [`Self::redo`]; see [`Self::edit_live_inner`].
    fn redo_inner(&mut self, mut cx: Option<&mut Context<Self>>) {
        self.end_gesture_inner(cx.as_deref_mut());
        let Some(id) = self.library.current else { return };
        let Some(label) = self.editor.redo_step(id) else { return };
        self.after_commit_inner(id, cx.as_deref_mut());
        if let Some(cx) = cx {
            self.request_render(RenderQuality::Final, cx);
            self.notify(format!("{} {label}", t("Redo", "重做")), cx);
        }
    }

    pub fn copy_adjustments(&mut self, cx: &mut Context<Self>) {
        self.editor.clipboard = Some(self.adj().clone());
        self.notify(t("Adjustments copied", "已拷贝调整"), cx);
    }

    pub fn apply_adjustments(&mut self, cx: &mut Context<Self>) {
        let Some(clip) = self.editor.clipboard.clone() else { return };
        let groups: Vec<ClipGroup> = self.editor.clip_groups.iter().copied().collect();
        if groups.is_empty() {
            self.notify(t("Choose at least one tool to apply", "请至少选择一个要应用的工具"), cx);
            return;
        }
        self.edit_selected(t("Apply adjustments", "应用调整"), move |a| {
            for g in &groups {
                g.copy(a, &clip);
            }
        }, cx);
        self.notify(t("Adjustments applied to selection", "已将调整应用到所选图像"), cx);
    }

    /// Undoes or redoes until `undo_len` steps remain on the undo stack.
    pub fn jump_history(&mut self, undo_len: usize, cx: &mut Context<Self>) {
        self.end_gesture(cx);
        let Some(id) = self.library.current else { return };
        if !self.editor.jump_history_step(id, undo_len) {
            return;
        }
        self.after_commit(id, cx);
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    /// Applies a style's look while keeping composition and local layers.
    /// Saves the current image's `groups` as a user style, replacing one of the same name.
    pub fn save_user_style(&mut self, name: String, groups: Vec<ClipGroup>, cx: &mut Context<Self>) {
        let src = self.adj().clone();
        let mut adjustments = AdjustmentSet::default();
        for g in &groups {
            g.copy(&mut adjustments, &src);
        }
        let style = UserStyle { name: name.clone(), groups: groups.iter().map(|g| g.id()).collect(), adjustments };
        match self.prefs.settings.user_styles.iter_mut().find(|s| s.name == name) {
            Some(existing) => *existing = style,
            None => self.prefs.settings.user_styles.push(style),
        }
        self.prefs.settings.save();
        self.notify(format!("{} {name}", t("Saved style", "已存储风格")), cx);
    }

    /// Applies user style `index` to the selection: only its saved tools change.
    pub fn apply_user_style(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(style) = self.prefs.settings.user_styles.get(index).cloned() else { return };
        let name = style.name.clone();
        self.preview_state.hover_look = None;
        self.edit_selected(format!("{} {name}", t("Style", "风格")), move |a| user_style_onto(a, &style), cx);
    }

    pub fn rename_user_style(&mut self, index: usize, name: String, cx: &mut Context<Self>) {
        if let Some(style) = self.prefs.settings.user_styles.get_mut(index) {
            style.name = name;
            self.prefs.settings.save();
            cx.notify();
        }
    }

    pub fn delete_user_style(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.prefs.settings.user_styles.len() {
            let style = self.prefs.settings.user_styles.remove(index);
            self.prefs.settings.save();
            self.notify(format!("{} {}", t("Deleted style", "已删除风格"), style.name), cx);
        }
    }

    pub fn apply_style(&mut self, style: &StyleInfo, stack: bool, cx: &mut Context<Self>) {
        let style = style.clone();
        self.preview_state.hover_look = None;
        let label = format!("{} {}", t("Style", "风格"), style.name);
        if !style.has_look || style.source.is_none() {
            self.edit_selected(label, move |a| style_onto(a, &style, stack), cx);
            return;
        }
        let targets: Vec<(ImageId, AdjustmentSet)> = self.library
            .selection
            .iter()
            .copied()
            .map(|id| {
                let target = self.editor.edits.get(&id).map(|edit| edit.adj.clone()).unwrap_or_default();
                (id, target)
            })
            .collect();
        if targets.is_empty() {
            return;
        }
        let engine = self.engine.clone();
        let plain = style.clone();
        cx.spawn(async move |this, cx| {
            let loaded = cx
                .background_spawn(async move {
                    targets
                        .into_iter()
                        .map(|(id, target)| {
                            let effective = style_for_target(engine.as_ref(), &style, id, &target);
                            (id, target, effective)
                        })
                        .collect::<Vec<_>>()
                })
                .await;
            this.update(cx, |this, cx| this.apply_loaded_style(label, plain, stack, loaded, cx))
                .ok();
        })
        .detach();
    }

    fn apply_loaded_style(
        &mut self,
        label: String,
        plain: StyleInfo,
        stack: bool,
        loaded: Vec<(ImageId, AdjustmentSet, StyleInfo)>,
        cx: &mut Context<Self>,
    ) {
        let mut changed = Vec::new();
        for (id, target, effective) in loaded {
            let edit = self.editor.edits.entry(id).or_default();
            let before = edit.adj.clone();
            if edit.adj == target {
                style_onto(&mut edit.adj, &effective, stack);
            } else {
                // The image changed while the look was being loaded; the old
                // target no longer describes it, so use the preloaded style.
                style_onto(&mut edit.adj, &plain, stack);
            }
            if before != edit.adj {
                edit.undo.push((SharedString::from(label.clone()), before));
                edit.redo.clear();
                changed.push(id);
            }
        }
        self.after_commit_many(changed, cx);
        self.preview_state.preview.before_dirty = true;
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    /// Previews a style on the current image while the pointer rests on it, as
    /// Lightroom does; `None` returns to the image's own adjustments. Nothing is
    /// committed or saved.
    pub fn preview_look(&mut self, look: Option<HoverLook>, cx: &mut Context<Self>) {
        let adj = look.map(|look| {
            let mut a = self.adj().clone();
            match look {
                HoverLook::Style(style, stack) => {
                    let effective = match self.library.current {
                        Some(image) => style_for_target(self.engine.as_ref(), &style, image, &a),
                        None => style,
                    };
                    style_onto(&mut a, &effective, stack);
                }
                HoverLook::User(index) => {
                    if let Some(style) = self.prefs.settings.user_styles.get(index) {
                        user_style_onto(&mut a, style);
                    }
                }
            }
            a
        });
        if self.preview_state.hover_look == adj {
            return;
        }
        self.preview_state.hover_look = adj;
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    /// Whether tethered capture can be offered: the engine's tethering backend has
    /// a camera open (`tether_status`, not `camera_status`, per the engine contract).
    pub fn camera_ready(&self) -> bool {
        self.device.camera_ready()
    }

    /// Saves the current image's adjustments into its Capture One `.cos`.
    pub fn save_to_c1_settings(&mut self, cx: &mut Context<Self>) {
        self.save_c1(false, cx);
    }

    /// Saves the current image's adjustments into the open Capture One catalog.
    pub fn save_to_c1_catalog(&mut self, cx: &mut Context<Self>) {
        self.save_c1(true, cx);
    }

    fn save_c1(&mut self, catalog: bool, cx: &mut Context<Self>) {
        let is_catalog = matches!(self.source, Source::Catalog(_));
        if !c1_save_enabled(self.c1_save_busy, self.library.current.is_some(), catalog, is_catalog) {
            let reason = if self.c1_save_busy {
                t("A Capture One save is already running", "已有 Capture One 保存在进行")
            } else if self.library.current.is_none() {
                t("Open an image first", "请先打开图像")
            } else {
                t("This window has no Capture One catalog open", "当前窗口没有打开 C1 catalog")
            };
            self.notify(reason, cx);
            return;
        }
        let Some(image) = self.library.current else { return };
        let adjustments = self.adj().clone();
        let engine = self.engine.clone();
        self.c1_save_busy = true;
        self.c1_save_warnings.clear();
        cx.spawn(async move |this, cx| {
            let (status, warnings) = cx
                .background_spawn(async move {
                    if catalog {
                        catalog_save_status(&engine.save_c1_catalog(image, &adjustments))
                    } else {
                        cos_save_status(&engine.save_c1_settings(image, &adjustments))
                    }
                })
                .await;
            this.update(cx, |this, cx| {
                this.c1_save_busy = false;
                this.c1_save_warnings = warnings.into_iter().map(SharedString::from).collect();
                let message = status.message();
                this.c1_save_status = Some(status);
                this.notify(message, cx);
            })
            .ok();
        })
        .detach();
    }

    /// Why capture is not offered right now, for disabled controls and notices.
    pub fn tether_note(&self) -> &'static str {
        match self.device.tether {
            None => t("Looking for a camera…", "正在检测相机…"),
            Some(TetherStatus::NoLibrary) => t("Tethering needs libgphoto2, which was not found (Linux only for now)", "联机拍摄需要 libgphoto2，本机未找到（目前仅支持 Linux）"),
            Some(TetherStatus::NoCamera) => t("No camera answered; connect one by USB and check again", "未检测到相机；请通过 USB 连接后重新检测"),
            Some(TetherStatus::Available { .. }) => t("Capture", "拍摄"),
        }
    }

    /// Asks the engine whether a camera can be opened. Opening one can take a few
    /// seconds, so it runs in the background.
    pub fn probe_tether(&mut self, cx: &mut Context<Self>) {
        if self.device.tether_probing {
            return;
        }
        self.device.set_tether_probing(true);
        cx.notify();
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let (tether, camera, settings) = cx
                .background_spawn(async move {
                    let tether = engine.tether_status();
                    let open = matches!(tether, TetherStatus::Available { .. });
                    let camera = if open { engine.camera_status() } else { CameraStatus::Disconnected };
                    let settings = if open { engine.camera_settings() } else { None };
                    (tether, camera, settings)
                })
                .await;
            this.update(cx, |this, cx| {
                this.device.set_tether_probing(false);
                this.device.set_tether(Some(tether));
                this.device.set_camera(camera, settings);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub fn capture(&mut self, cx: &mut Context<Self>) {
        if self.capturing {
            return;
        }
        self.capturing = true;
        cx.notify();
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async move { engine.capture() }).await;
            this.update(cx, |this, cx| {
                this.capturing = false;
                match result {
                    Ok(info) => {
                        let id = info.id;
                        this.editor.edits.insert(id, Edit::default());
                        this.library.recent_captures.insert(id);
                        this.library.images.push(info);
                        this.queue_thumbs(vec![id], cx);
                        this.select(id, false, false, cx);
                        this.notify(t("Captured", "已拍摄"), cx);
                        // Capture One reviews each capture for closed eyes; it only reports.
                        this.run_eye_check(vec![id], false, cx);
                    }
                    Err(err) => {
                        // The log keeps the driver error; the notice stays translatable.
                        log::warn!("capture: {err:#}");
                        let reason = match TetherErrorKind::of(&err) {
                            TetherErrorKind::NoLibrary => t("tethering is not available on this system", "本机没有可用的联机后端"),
                            TetherErrorKind::NoCamera => t("no camera is connected", "未连接相机"),
                            TetherErrorKind::Unsupported => t("the camera driver does not support this capture", "该相机驱动不支持此拍摄"),
                            TetherErrorKind::Timeout => t("the camera stopped responding", "相机无响应"),
                            TetherErrorKind::Other => t("see the log for details", "详情见日志"),
                        };
                        this.notify(format!("{}: {reason}", t("Capture failed", "拍摄失败")), cx)
                    }
                }
            })
            .ok();
        })
        .detach();
    }

    pub fn reset_all(&mut self, cx: &mut Context<Self>) {
        self.edit(t("Reset", "复位"), |a| *a = AdjustmentSet::default(), cx);
        self.editor.active_layer = None;
        self.editor.active_node = None;
    }

    pub fn auto_adjust(&mut self, target: AutoTarget, cx: &mut Context<Self>) {
        let Some(id) = self.library.current else { return };
        let engine = self.engine.clone();
        let current = self.adj().clone();
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async move { engine.auto_adjust(id, &current, target) }).await;
            this.update(cx, |this, cx| match result {
                Ok(r) => {
                    if this.library.current == Some(id) {
                        this.edit(t("Auto adjust", "自动调整"), |a| *a = r.adjustments, cx);
                    }
                }
                Err(err) => this.notify(err.to_string(), cx),
            })
            .ok();
        })
        .detach();
    }

    // ───────────── Layers ─────────────

    pub fn add_layer(&mut self, kind: LayerKind, name: Option<String>, mask: Option<Mask>, cx: &mut Context<Self>) -> u64 {
        let id = self.fresh_node_id();
        let count = self.adj().layers.len() + 1;
        let name = name.unwrap_or_else(|| match kind {
            LayerKind::Adjustment => format!("{} {count}", t("Layer", "图层")),
            LayerKind::Heal => format!("{} {count}", t("Heal", "修复")),
            LayerKind::Clone => format!("{} {count}", t("Clone", "仿制")),
            LayerKind::Fill => format!("{} {count}", t("Fill", "填充")),
        });
        self.edit(t("Add layer", "新建图层"), |a| {
            a.layers.push(Layer { id, name, kind, mask: mask.unwrap_or_default(), ..Default::default() });
        }, cx);
        self.editor.active_layer = Some(id);
        self.editor.active_node = None;
        cx.notify();
        id
    }

    /// Turns a sampled color into a color editor selection: the nearest basic
    /// sector, or a new advanced / skin range centered on its hue.
    pub fn range_from_sample(&mut self, rgb: [f32; 3], cx: &mut Context<Self>) {
        self.prefs.panels.pick_to_range = false;
        let [r, g, b] = rgb.map(|v| v.clamp(0.0, 1.0));
        let (max, min) = (r.max(g).max(b), r.min(g).min(b));
        let d = max - min;
        let hue = if d < 1e-5 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / d).rem_euclid(6.0))
        } else if max == g {
            60.0 * ((b - r) / d + 2.0)
        } else {
            60.0 * ((r - g) / d + 4.0)
        };
        let sat = if max > 0.0 { d / max } else { 0.0 };
        let mode = self.prefs.panels.color_editor_mode;
        if mode == 0 {
            self.prefs.panels.selected_sector = HueSector::nearest(hue) as usize;
            self.notify(format!("{} · {hue:.0}°", t("Picked the nearest color", "已选中最接近的颜色")), cx);
        } else {
            let skin = mode == 2;
            let scope = self.scope();
            let id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_micros() as u64).unwrap_or(1);
            self.edit(t("Pick color range", "取样创建色彩范围"), |a| {
                with_scoped(a, scope, |a| &mut a.color_editor, |l| &mut l.color_editor, |ce| {
                    let r = ColorRange {
                        id,
                        hue,
                        // Muted samples get a wider range so the selection is not a sliver.
                        width: if skin { 30.0 } else { (50.0 - 25.0 * sat).clamp(20.0, 50.0) },
                        uniformity: skin.then(SkinUniformity::default),
                        ..Default::default()
                    };
                    if skin { ce.skin.push(r) } else { ce.advanced.push(r) }
                })
            }, cx);
            self.prefs.panels.selected_color_range = Some(id);
            self.notify(format!("{} · {hue:.0}°", t("Color range created from the sample", "已从取样创建色彩范围")), cx);
        }
        self.set_cursor(CursorTool::Select, cx);
        cx.notify();
    }


    pub fn edit_layer(&mut self, label: impl Into<SharedString>, id: u64, live: bool, f: impl FnOnce(&mut Layer), cx: &mut Context<Self>) {
        let apply = move |a: &mut AdjustmentSet| {
            if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                f(l);
            }
        };
        if live { self.edit_live(label, apply, cx) } else { self.edit(label, apply, cx) }
    }

    /// Ticks or unticks an AI mask kind for the next "Create" in the mask panel.
    pub fn toggle_ai_pick(&mut self, kind: AiMaskKind, cx: &mut Context<Self>) {
        let pick = &mut self.prefs.panels.ai_pick;
        if let Some(i) = pick.iter().position(|k| *k == kind) {
            pick.remove(i);
        } else {
            pick.push(kind);
            pick.sort_by_key(|k| AiMaskKind::ALL.iter().position(|a| a == k));
        }
        cx.notify();
    }

    /// Computes every ticked AI mask, one after another off the UI thread, and
    /// adds them to the layer that was selected when this started as one undo
    /// step (from the base, they go on one new layer). A result for an image
    /// no longer shown, or a layer since deleted, is dropped.
    pub fn compute_ai_masks(&mut self, kinds: Vec<AiMaskKind>, cx: &mut Context<Self>) {
        let Some(image) = self.library.current else { return };
        if kinds.is_empty() || self.tasks.ai_busy {
            return;
        }
        let engine = self.engine.clone();
        let scope = self.scope();
        self.tasks.ai_busy = true;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let results = cx
                .background_spawn(async move { kinds.into_iter().map(|k| (k, engine.compute_ai_mask(image, k))).collect::<Vec<_>>() })
                .await;
            this.update(cx, |this, cx| {
                this.tasks.ai_busy = false;
                if this.library.current != Some(image) {
                    cx.notify();
                    return;
                }
                let mut shapes = Vec::new();
                let mut failed = Vec::new();
                for (kind, result) in results {
                    match result {
                        Ok(key) => {
                            this.ai_unavailable.remove(&kind);
                            shapes.push((kind, MaskShape::Ai(AiMask { op: MaskOp::Add, kind, cache_key: Some(key) })));
                        }
                        Err(err) => {
                            this.ai_unavailable.insert(kind);
                            failed.push(format!("{kind:?}: {err}"));
                        }
                    }
                }
                this.prefs.panels.ai_pick.clear();
                if !shapes.is_empty() {
                    let label = t("AI mask", "AI 蒙版");
                    match scope {
                        Scope::Layer(id) if this.adj().layers.iter().any(|l| l.id == id) => this.edit_layer(label, id, false, |l| {
                            for (_, shape) in shapes {
                                nodes::add_shape(&mut l.mask, shape);
                            }
                        }, cx),
                        Scope::Layer(_) => {}
                        Scope::Global => {
                            let name = format!("AI · {}", shapes.iter().map(|(k, _)| format!("{k:?}")).collect::<Vec<_>>().join(" + "));
                            let mut mask = Mask::default();
                            for (_, shape) in shapes {
                                nodes::add_shape(&mut mask, shape);
                            }
                            this.add_layer(LayerKind::Adjustment, Some(name), Some(mask), cx);
                        }
                    }
                }
                if !failed.is_empty() {
                    this.notify(format!("{}: {}", t("AI mask unavailable", "AI 蒙版不可用"), failed.join("; ")), cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    // ───────────── Viewer ─────────────

    pub fn set_cursor(&mut self, tool: CursorTool, cx: &mut Context<Self>) {
        if self.cursor == tool {
            return;
        }
        let was_crop = self.cursor == CursorTool::Crop;
        self.prev_cursor = self.cursor;
        self.cursor_changed = std::time::Instant::now();
        self.cursor = tool;
        if tool != CursorTool::Spot {
            self.editor.spot_anchor = None;
        }
        if tool == CursorTool::Crop || was_crop {
            // The crop tool shows the uncropped frame with an overlay.
            self.preview_state.zoom = ViewZoom::default();
            self.request_render(RenderQuality::Final, cx);
        }
        if tool.is_mask_tool() && self.scope() == Scope::Global {
            self.add_layer(LayerKind::Adjustment, None, None, cx);
        }
        cx.notify();
    }

    pub fn set_zoom(&mut self, scale: Option<f32>, center: Option<(f32, f32)>, cx: &mut Context<Self>) {
        self.preview_state.zoom.scale = scale.map(|s| s.clamp(0.05, 8.0));
        if let Some(c) = center {
            self.preview_state.zoom.center = c;
        }
        if scale.is_none() {
            self.preview_state.zoom.center = (0.5, 0.5);
        }
        self.clamp_center();
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    pub fn pan_by(&mut self, dx: f32, dy: f32, cx: &mut Context<Self>) {
        if self.preview_state.zoom.scale.is_none() {
            return;
        }
        self.preview_state.zoom.center.0 += dx;
        self.preview_state.zoom.center.1 += dy;
        self.clamp_center();
        self.request_render(RenderQuality::Draft, cx);
        cx.notify();
    }

    pub fn set_viewport(&mut self, size: (Pixels, Pixels), scale_factor: f32, cx: &mut Context<Self>) {
        let size = (f32::from(size.0), f32::from(size.1));
        if (size.0 - self.preview_state.viewport.0).abs() > 1.0 || (size.1 - self.preview_state.viewport.1).abs() > 1.0 || scale_factor != self.device.scale_factor {
            self.preview_state.viewport = size;
            self.device.set_scale_factor(scale_factor);
            self.preview_state.preview.before_dirty = true;
            crate::state::invalidate_display_profile_cache();
            self.request_render(RenderQuality::Final, cx);
        }
    }

    pub fn toggle_panel(&mut self, id: &'static str, cx: &mut Context<Self>) {
        if !self.prefs.panels.collapsed.remove(id) {
            self.prefs.panels.collapsed.insert(id);
        }
        cx.notify();
    }

    /// Opens `id` and folds every other panel; when it is already the only
    /// open one, opens them all again (Alt-click on a panel header).
    pub fn solo_panel(&mut self, id: &'static str, cx: &mut Context<Self>) {
        let others = || crate::tools::QUICK_PANELS.iter().copied().filter(|p| *p != id);
        let alone = !self.prefs.panels.collapsed.contains(id) && others().all(|p| self.prefs.panels.collapsed.contains(p));
        if alone {
            self.set_all_panels(true, cx);
        } else {
            self.prefs.panels.collapsed.extend(others());
            self.prefs.panels.collapsed.remove(id);
            cx.notify();
        }
    }

    /// Opens or folds every adjustment panel at once.
    pub fn set_all_panels(&mut self, open: bool, cx: &mut Context<Self>) {
        for id in crate::tools::QUICK_PANELS {
            if open {
                self.prefs.panels.collapsed.remove(id);
            } else {
                self.prefs.panels.collapsed.insert(id);
            }
        }
        cx.notify();
    }

    pub fn filter_active(&self) -> bool {
        self.library.filter != Filter::default()
    }

    pub fn clear_filter(&mut self, cx: &mut Context<Self>) {
        self.library.filter = Filter::default();
        cx.notify();
    }

    // ───────────── Output ─────────────

    pub fn process_selection(&mut self, cx: &mut Context<Self>) {
        let recipes: Vec<ProcessRecipe> = self.recipes.iter().filter(|r| r.enabled).cloned().collect();
        self.process(recipes, cx);
    }

    /// Queues every selected image through each of `recipes`.
    pub fn process(&mut self, recipes: Vec<ProcessRecipe>, cx: &mut Context<Self>) {
        let images: Vec<ImageId> = self.library.selection.iter().copied().collect();
        if recipes.is_empty() {
            self.notify(t("Enable at least one process recipe", "请至少启用一个处理配方"), cx);
            return;
        }
        for image in images {
            let name = self.info(image).map(|i| i.file_name.clone()).unwrap_or_default();
            for recipe in &recipes {
                let id = self.tasks.reserve_job_id();
                let job = ExportJob { job_id: id, image, recipe: recipe.clone() };
                if let Err(err) = self.engine.start_export(job) {
                    self.notify(err.to_string(), cx);
                    continue;
                }
                self.tasks.push_job(JobView { id, name: name.clone(), recipe: recipe.name.clone(), status: JobStatus::Queued });
            }
        }
        self.tool_tab = ToolTab::Batch;
        self.poll_jobs(cx);
        cx.notify();
    }

    fn poll_jobs(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(std::time::Duration::from_millis(120)).await;
                let busy = this
                    .update(cx, |this, cx| {
                        let mut busy = false;
                        for job in &mut this.tasks.jobs {
                            if matches!(job.status, JobStatus::Queued | JobStatus::Running { .. }) {
                                job.status = this.engine.job_status(job.id);
                                if let JobStatus::Failed { message } = &job.status {
                                    log::warn!("export {}: {message}", job.id);
                                }
                                busy |= matches!(job.status, JobStatus::Queued | JobStatus::Running { .. });
                            }
                        }
                        cx.notify();
                        busy
                    })
                    .unwrap_or(false);
                if !busy {
                    break;
                }
            }
        })
        .detach();
    }
}

pub fn init_state(engine: Arc<dyn ImageEngine>, source: Source, cx: &mut App) -> Entity<AppState> {
    let mut settings = Settings::load();
    if let Some(path) = source.path() {
        settings.remember(path);
        settings.save();
    }
    crate::widgets::param_c1::reset_overrides();
    crate::i18n::set_locale(if settings.chinese { crate::i18n::Locale::ZhCn } else { crate::i18n::Locale::En });
    crate::motion::set_speed(settings.motion_speed);
    if let Some(accent) = settings.accent {
        crate::theme::set_accent(accent, cx);
    }
    crate::actions::bind_keys(cx, &settings.shortcuts);
    cx.new(|cx| AppState::new(engine, source, settings, cx))
}

impl AppState {
    /// Writes the current view into the settings and saves them.
    pub fn save_settings(&mut self) {
        let layout = self.current_layout(self.prefs.settings.workspace.name.clone());
        let st = &mut self.prefs.settings;
        st.workspace = layout;
        st.viewer_gray = self.prefs.view.viewer_gray;
        st.thumb_size = self.prefs.view.thumb_size;
        st.info_overlay = self.prefs.view.show_info_overlay;
        st.chinese = crate::i18n::locale() == crate::i18n::Locale::ZhCn;
        st.motion_speed = crate::motion::speed();
        st.save();
    }

    pub fn current_layout(&self, name: String) -> WorkspaceLayout {
        WorkspaceLayout {
            name,
            inspector_right: self.prefs.view.inspector_right,
            show_inspector: self.prefs.view.show_inspector,
            browser: browser_name(self.prefs.view.browser).into(),
            hidden_tabs: self.prefs.settings.workspace.hidden_tabs.clone(),
            tab_order: self.prefs.settings.workspace.tab_order.clone(),
            quick_tools: self.prefs.settings.workspace.quick_tools.clone(),
            panel_order: self.prefs.settings.workspace.panel_order.clone(),
            pinned_panels: self.prefs.settings.workspace.pinned_panels.clone(),
            tab: tab_id(self.tool_tab),
            inspector_width: self.prefs.view.inspector_width,
            filmstrip_height: self.prefs.view.filmstrip_height,
            multi: self.prefs.view.multi,
        }
    }

    fn load_layout(&mut self, l: &WorkspaceLayout) {
        self.prefs.view.inspector_right = l.inspector_right;
        self.prefs.view.show_inspector = l.show_inspector;
        self.prefs.view.browser = browser_from(&l.browser);
        self.prefs.view.inspector_width = l.inspector_width.clamp(260.0, 640.0);
        self.prefs.view.filmstrip_height = l.filmstrip_height.clamp(96.0, 420.0);
        self.prefs.view.multi = l.multi;
        self.prefs.settings.workspace = l.clone();
        if let Some(tab) = tab_from(&l.tab) {
            self.tool_tab = tab;
        }
        if self.tab_hidden(self.tool_tab) {
            self.tool_tab = self.visible_tabs().first().copied().unwrap_or(ToolTab::Grading);
        }
    }

    /// Built-in workspaces followed by the user's saved ones.
    pub fn workspace_list(&self) -> Vec<WorkspaceLayout> {
        let mut all = builtin_workspaces();
        all.extend(self.prefs.settings.workspaces.iter().cloned());
        all
    }

    pub fn apply_workspace(&mut self, index: usize, cx: &mut Context<Self>) {
        if let Some(l) = self.workspace_list().get(index).cloned() {
            self.load_layout(&l);
            self.save_settings();
            self.notify(format!("{} · {}", t("Workspace", "工作区"), l.name), cx);
            cx.notify();
        }
    }

    pub fn save_workspace_as(&mut self, name: String, cx: &mut Context<Self>) {
        let layout = self.current_layout(name.clone());
        self.prefs.settings.workspaces.retain(|w| w.name != name);
        self.prefs.settings.workspaces.push(layout.clone());
        self.prefs.settings.workspace = layout;
        self.save_settings();
        self.notify(format!("{} · {name}", t("Workspace saved", "工作区已保存")), cx);
    }

    pub fn delete_workspace(&mut self, name: &str, cx: &mut Context<Self>) {
        self.prefs.settings.workspaces.retain(|w| w.name != name);
        self.save_settings();
        cx.notify();
    }

    pub fn tab_hidden(&self, tab: ToolTab) -> bool {
        self.prefs.settings.workspace.hidden_tabs.iter().any(|h| *h == tab_id(tab))
    }

    /// The shown tabs in the workspace's order; tabs the order does not name
    /// yet keep their default place.
    pub fn visible_tabs(&self) -> Vec<ToolTab> {
        self.tab_order().into_iter().filter(|t| !self.tab_hidden(*t)).collect()
    }

    pub fn tab_order(&self) -> Vec<ToolTab> {
        let mut order: Vec<ToolTab> = Vec::new();
        for tab in self.prefs.settings.workspace.tab_order.iter().filter_map(|id| tab_from(id)) {
            if !order.contains(&tab) {
                order.push(tab);
            }
        }
        for (i, tab) in default_tab_order().into_iter().enumerate() {
            if !order.contains(&tab) {
                order.insert(i.min(order.len()), tab);
            }
        }
        order
    }

    /// Moves `tab` in front of `before` (to the end with `None`).
    pub fn move_tab(&mut self, tab: ToolTab, before: Option<ToolTab>, cx: &mut Context<Self>) {
        if Some(tab) == before {
            return;
        }
        let mut order = self.tab_order();
        order.retain(|t| *t != tab);
        let at = before.and_then(|b| order.iter().position(|t| *t == b)).unwrap_or(order.len());
        order.insert(at, tab);
        self.prefs.settings.workspace.tab_order = order.into_iter().map(tab_id).collect();
        self.save_settings();
        cx.notify();
    }

    pub fn in_quick(&self, panel: &str) -> bool {
        self.prefs.settings.workspace.quick_tools.iter().any(|p| p == panel)
    }

    /// Adds a tool panel to the Quick tab, or takes it off.
    pub fn toggle_quick(&mut self, panel: &str, cx: &mut Context<Self>) {
        let tools = &mut self.prefs.settings.workspace.quick_tools;
        if let Some(i) = tools.iter().position(|p| p == panel) {
            tools.remove(i);
        } else {
            tools.push(panel.to_string());
            if self.tab_hidden(ToolTab::Quick) {
                self.set_tab_hidden(ToolTab::Quick, false, cx);
            }
        }
        self.save_settings();
        cx.notify();
    }

    pub fn set_tab_hidden(&mut self, tab: ToolTab, hidden: bool, cx: &mut Context<Self>) {
        let id = tab_id(tab);
        self.prefs.settings.workspace.hidden_tabs.retain(|h| *h != id);
        if hidden && self.visible_tabs().len() > 1 {
            self.prefs.settings.workspace.hidden_tabs.push(id);
            if self.tool_tab == tab {
                self.tool_tab = self.visible_tabs().first().copied().unwrap_or(ToolTab::Grading);
            }
        }
        self.save_settings();
        cx.notify();
    }
}

/// What the pointer rests on in the styles lists.
#[derive(Clone, Debug)]
pub enum HoverLook {
    /// An engine style or preset, and whether it stacks on the current look.
    Style(StyleInfo, bool),
    /// A user style, by index.
    User(usize),
}

/// Outcome of one "save to Capture One" action, for the status bar and any
/// detail strip the menus build on top of it.
#[derive(Clone, Debug, PartialEq)]
pub enum C1SaveStatus {
    Unchanged,
    Saved { cells: usize },
    NotACatalog,
    Failed { message: String },
}

impl C1SaveStatus {
    pub fn message(&self) -> String {
        match self {
            Self::Unchanged => t("Nothing changed", "没有变化").to_owned(),
            Self::Saved { cells } => format!("{} {cells}", t("Saved to Capture One", "已保存到 Capture One")),
            Self::NotACatalog => t("No Capture One catalog is open", "当前没有打开的 C1 catalog").to_owned(),
            Self::Failed { message } => format!("{}: {message}", t("Save failed", "保存失败")),
        }
    }
}

/// The action runs only with an image, a free busy flag and, for the catalog
/// route, an engine that was opened on a catalog.
pub fn c1_save_enabled(busy: bool, has_image: bool, wants_catalog: bool, is_catalog: bool) -> bool {
    !busy && has_image && (!wants_catalog || is_catalog)
}

pub fn cos_save_status(
    result: &anyhow::Result<aurora_engine::c1_cos::CosSaveOutcome>,
) -> (C1SaveStatus, Vec<String>) {
    match result {
        Ok(aurora_engine::c1_cos::CosSaveOutcome::Unchanged(report)) => {
            (C1SaveStatus::Unchanged, report.warnings.clone())
        }
        Ok(aurora_engine::c1_cos::CosSaveOutcome::Written(report)) => (
            C1SaveStatus::Saved { cells: report.edits },
            report.warnings.clone(),
        ),
        Err(error) => (C1SaveStatus::Failed { message: format!("{error:#}") }, Vec::new()),
    }
}

pub fn catalog_save_status(
    result: &anyhow::Result<aurora_engine::c1_writeback::CatalogSaveOutcome>,
) -> (C1SaveStatus, Vec<String>) {
    use aurora_engine::c1_writeback::CatalogSaveOutcome;
    match result {
        Ok(CatalogSaveOutcome::Unchanged { warnings }) => (C1SaveStatus::Unchanged, warnings.clone()),
        Ok(CatalogSaveOutcome::Saved { columns, .. }) => {
            (C1SaveStatus::Saved { cells: *columns }, Vec::new())
        }
        Ok(CatalogSaveOutcome::NotACatalog) => (C1SaveStatus::NotACatalog, Vec::new()),
        Err(error) => (C1SaveStatus::Failed { message: format!("{error:#}") }, Vec::new()),
    }
}

/// What applying `style` does to one image's adjustments.
fn style_onto(a: &mut AdjustmentSet, style: &StyleInfo, stack: bool) {
    for g in ClipGroup::ALL {
        if g.is_composition() || g == ClipGroup::Layers {
            continue;
        }
        if !stack || g.is_set(&style.adjustments) {
            g.copy(a, &style.adjustments);
        }
    }
    a.style = Some(style.name.clone());
}

/// A style's adjustments for one target image. AICGR styles are reloaded
/// against the target; any failure falls back to the preloaded adjustments.
fn style_for_target(
    engine: &dyn ImageEngine,
    style: &StyleInfo,
    image: ImageId,
    target: &AdjustmentSet,
) -> StyleInfo {
    if !style.has_look || style.source.is_none() {
        return style.clone();
    }
    match engine.style_for_image(image, style, target) {
        Ok(adjustments) => StyleInfo { adjustments, ..style.clone() },
        Err(err) => {
            log::warn!("style {} for image {}: {err:#}; using the preloaded style", style.name, image.0);
            style.clone()
        }
    }
}

fn user_style_onto(a: &mut AdjustmentSet, style: &UserStyle) {
    for g in style.groups.iter().filter_map(|g| ClipGroup::from_id(g)) {
        g.copy(a, &style.adjustments);
    }
    a.style = Some(style.name.clone());
}

/// Source-minus-target offset in oriented, uncropped source pixels — what
/// `SpotKind::Clone` stores. Both points are crop-normalised, so the crop scale
/// comes back out here and the fine rotation is undone: the engine turns the
/// offset by `rotation` again when it draws.
pub fn clone_offset(src: [f32; 2], dst: [f32; 2], g: &Geometry, source_size: (u32, u32)) -> (i32, i32) {
    let (sw, sh) = source_size;
    if sw == 0 || sh == 0 {
        return (0, 0);
    }
    let (ow, oh) = if g.orientation % 2 == 1 {
        (sh as f32, sw as f32)
    } else {
        (sw as f32, sh as f32)
    };
    let dx = (src[0] - dst[0]) * g.crop.width * ow;
    let dy = (src[1] - dst[1]) * g.crop.height * oh;
    let (sin, cos) = g.rotation.to_radians().sin_cos();
    let (x, y) = (cos * dx + sin * dy, cos * dy - sin * dx);
    if !x.is_finite() || !y.is_finite() {
        return (0, 0);
    }
    (x.round() as i32, y.round() as i32)
}

/// Localized notice for a batch save: per-document counts, the catalog-index
/// error and the first document failure. `None` when everything reached disk.
fn save_report_notice(report: &SaveReport) -> Option<String> {
    if report.is_complete() {
        return None;
    }
    let mut message = format!(
        "{} {} / {} {} / {} {}",
        t("Saved", "已保存"),
        report.saved(),
        t("failed", "失败"),
        report.failed(),
        t("skipped", "已跳过"),
        report.superseded(),
    );
    if let Some(index) = &report.index_error {
        message.push_str(&format!("; {} {index}", t("catalog index not saved", "目录索引未存储：")));
    }
    if let Some(first) = report.first_failure() {
        let error = first.outcome.error().unwrap_or_default();
        if first.path.is_empty() {
            message.push_str(&format!("; {error}"));
        } else {
            message.push_str(&format!("; {}: {error}", first.path));
        }
    }
    Some(message)
}

/// Localized notice for a queued submission: capacity/closed/failed only.
/// Accepted-only submissions are silent; the writer reports later failures
/// through `pending_save_status` or the quit flush.
fn queued_save_notice(summary: &QueuedSaveSummary) -> Option<String> {
    if summary.full() == 0 && summary.closed() == 0 && summary.failed() == 0 {
        return None;
    }
    let mut message = format!(
        "{} {} / {} {} / {} {}",
        t("Queued", "已排队"),
        summary.accepted(),
        t("queue full", "队列已满"),
        summary.full(),
        t("queue closed", "队列已关闭"),
        summary.closed(),
    );
    if let Some(first) = summary.first_failure() {
        let error = match &first.outcome {
            QueuedSaveOutcome::Failed { error } => error.clone(),
            _ => String::new(),
        };
        if first.path.is_empty() {
            message.push_str(&format!("; {error}"));
        } else {
            message.push_str(&format!("; {}: {error}", first.path));
        }
    }
    Some(message)
}

/// Localized notice for a quit/flush report. `None` when everything confirmed.
fn flush_report_notice(report: &FlushReport) -> Option<String> {
    if report.is_complete() {
        return None;
    }
    let mut message = format!(
        "{} {} / {} {} / {} {}",
        t("Saved", "已保存"),
        report.saved(),
        t("failed", "失败"),
        report.failed(),
        t("not written", "未写入"),
        report.not_written(),
    );
    if let Some(index) = &report.index_error {
        message.push_str(&format!("; {} {index}", t("catalog index not saved", "目录索引未存储：")));
    }
    if let Some(first) = report
        .documents
        .iter()
        .find(|document| matches!(document.state, PathState::Failed { .. }))
    {
        if let PathState::Failed { error } = &first.state {
            message.push_str(&format!("; {}: {error}", first.rel_path));
        }
    }
    Some(message)
}

#[cfg(test)]
mod save_notice_tests {
    use super::{flush_report_notice, queued_save_notice, save_report_notice};
    use aurora_engine::{
        DocumentSaveResult, ImageId, PathState, PathStatus, QueuedDocumentResult,
        QueuedSaveOutcome, QueuedSaveSummary, SaveReport,
    };
    use aurora_engine::document::DocumentRevision;
    use aurora_engine::FlushReport;

    #[test]
    fn complete_reports_are_silent() {
        let report = SaveReport::new(
            vec![DocumentSaveResult::saved(ImageId(1), "a.cr3")],
            None,
        );
        assert!(save_report_notice(&report).is_none());
    }

    #[test]
    fn partial_reports_name_counts_index_and_first_failure() {
        let report = SaveReport::new(
            vec![
                DocumentSaveResult::saved(ImageId(1), "a.cr3"),
                DocumentSaveResult::saved(ImageId(2), "b.cr3"),
                DocumentSaveResult::saved(ImageId(3), "c.cr3"),
                DocumentSaveResult::saved(ImageId(4), "d.cr3"),
                DocumentSaveResult::failed(ImageId(5), "e.cr3", "read-only"),
                DocumentSaveResult::failed(ImageId(6), "f.cr3", "gone"),
                DocumentSaveResult::superseded(ImageId(7), "g.cr3"),
            ],
            Some("index locked".into()),
        );
        let message = save_report_notice(&report).expect("a partial batch is reported");
        // Counts are language-independent; labels come from t().
        assert!(message.contains('4'), "{message}");
        assert!(message.contains('2'), "{message}");
        assert!(message.contains('1'), "{message}");
        assert!(message.contains("index locked"), "{message}");
        assert!(message.contains("e.cr3: read-only"), "{message}");
    }

    #[test]
    fn index_failure_is_reported_without_a_document_failure() {
        let report = SaveReport::new(
            vec![DocumentSaveResult::saved(ImageId(1), "a.cr3")],
            Some("locked".into()),
        );
        let message = save_report_notice(&report).expect("an index failure is reported");
        assert!(message.contains("locked"), "{message}");
        assert!(!message.contains("a.cr3"), "{message}");
    }

    fn queued(path: &str, outcome: QueuedSaveOutcome) -> QueuedDocumentResult {
        QueuedDocumentResult { image: ImageId(1), path: path.into(), outcome }
    }

    #[test]
    fn accepted_queued_submissions_are_silent() {
        let summary = QueuedSaveSummary {
            documents: vec![queued(
                "a.cr3",
                QueuedSaveOutcome::Accepted { revision: DocumentRevision::new(1) },
            )],
        };
        assert!(queued_save_notice(&summary).is_none());
    }

    #[test]
    fn full_and_failed_queued_submissions_name_the_reason() {
        let summary = QueuedSaveSummary {
            documents: vec![
                queued("a.cr3", QueuedSaveOutcome::Accepted { revision: DocumentRevision::new(2) }),
                queued("b.cr3", QueuedSaveOutcome::Full { capacity: 64 }),
                queued("c.cr3", QueuedSaveOutcome::Failed { error: "unknown image".into() }),
            ],
        };
        let message = queued_save_notice(&summary).expect("not silent");
        // Counts are language-independent; accepted == full == 1 here.
        assert!(message.contains('1'), "{message}");
        assert!(message.contains("c.cr3"), "{message}");
        assert!(message.contains("unknown image"), "{message}");
    }

    #[test]
    fn quit_flush_reports_failed_and_not_written() {
        let report = FlushReport {
            documents: vec![
                PathStatus {
                    rel_path: "a.cr3".into(),
                    revision: DocumentRevision::new(1),
                    state: PathState::Saved,
                },
                PathStatus {
                    rel_path: "b.cr3".into(),
                    revision: DocumentRevision::new(2),
                    state: PathState::Failed { error: "disk full".into() },
                },
                PathStatus {
                    rel_path: "c.cr3".into(),
                    revision: DocumentRevision::new(3),
                    state: PathState::NotWritten,
                },
            ],
            index_error: Some("index locked".into()),
            timed_out: true,
            pending: 1,
        };
        let message = flush_report_notice(&report).expect("incomplete flush");
        assert!(message.contains("disk full"), "{message}");
        assert!(message.contains("index locked"), "{message}");
        assert!(message.contains("b.cr3"), "{message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: (u32, u32) = (100, 60);

    /// A plain frame: the offset is the clicked delta in source pixels.
    #[test]
    fn clone_offset_is_the_clicked_delta() {
        let g = Geometry::default();
        assert_eq!(clone_offset([0.6, 0.5], [0.4, 0.5], &g, SRC), (20, 0));
        assert_eq!(clone_offset([0.5, 0.4], [0.5, 0.5], &g, SRC), (0, -6));
    }

    /// A quarter turn swaps the long side the normalised x is measured against.
    #[test]
    fn clone_offset_follows_the_orientation() {
        let g = Geometry { orientation: 1, ..Default::default() };
        assert_eq!(clone_offset([0.6, 0.5], [0.4, 0.5], &g, SRC), (12, 0));
    }

    /// Crop-normalised points scale by the crop, not the whole image.
    #[test]
    fn clone_offset_undoes_the_crop() {
        let g = Geometry { crop: NormRect { x: 0.25, y: 0.0, width: 0.5, height: 1.0 }, ..Default::default() };
        assert_eq!(clone_offset([0.6, 0.5], [0.4, 0.5], &g, SRC), (10, 0));
    }

    /// The engine rotates the offset before drawing, so the helper must undo it.
    #[test]
    fn clone_offset_undoes_the_fine_rotation() {
        let g = Geometry { rotation: 90.0, ..Default::default() };
        assert_eq!(clone_offset([0.5, 0.6], [0.5, 0.5], &g, SRC), (6, 0));
    }

    #[test]
    fn clone_offset_survives_a_missing_image() {
        assert_eq!(clone_offset([0.6, 0.5], [0.4, 0.5], &Geometry::default(), (0, 0)), (0, 0));
    }
}

/// `TaskCenter` owns the job list and the busy flags; these pin the id and
/// ordering contract the output queue and AI panels rely on.
#[cfg(test)]
mod task_center_tests {
    use super::TaskCenter;
    use aurora_engine::JobStatus;

    #[test]
    fn ids_are_monotonic_and_jobs_keep_submission_order() {
        let mut tasks = TaskCenter::default();
        assert_eq!(tasks.start_job("first.cr3".into(), "JPEG".into()), 1);
        assert_eq!(tasks.start_job("second.cr3".into(), "TIFF".into()), 2);
        assert_eq!(tasks.start_job("third.cr3".into(), "JPEG".into()), 3);
        let ids: Vec<u64> = tasks.jobs.iter().map(|job| job.id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
        let names: Vec<&str> = tasks.jobs.iter().map(|job| job.name.as_str()).collect();
        assert_eq!(names, vec!["first.cr3", "second.cr3", "third.cr3"]);
    }

    #[test]
    fn a_reserved_id_is_consumed_even_when_the_export_is_rejected() {
        let mut tasks = TaskCenter::default();
        let rejected = tasks.reserve_job_id();
        assert_eq!(rejected, 1);
        assert!(tasks.jobs.is_empty());
        assert_eq!(tasks.start_job("next.cr3".into(), "JPEG".into()), 2);
    }

    #[test]
    fn finish_job_updates_only_a_listed_job() {
        let mut tasks = TaskCenter::default();
        let id = tasks.start_job("a.cr3".into(), "JPEG".into());
        assert!(tasks.finish_job(id, JobStatus::Failed { message: "cancelled".into() }));
        assert!(matches!(tasks.jobs[0].status, JobStatus::Failed { .. }));
        assert!(!tasks.finish_job(id + 1, JobStatus::Queued));
    }

    #[test]
    fn busy_flags_round_trip() {
        let mut tasks = TaskCenter::default();
        assert!(!tasks.ai_busy && !tasks.analysing);
        tasks.ai_busy = true;
        tasks.analysing = true;
        assert!(tasks.ai_busy && tasks.analysing);
        tasks.ai_busy = false;
        assert!(!tasks.ai_busy && tasks.analysing);
    }
}

/// Gesture/command invariants. These drive the real `AppState` transition
/// methods with the GPUI side effects suppressed (`*_inner(..., None)`) and a
/// recording engine, so undo depth and save calls are asserted on the same code
/// the UI runs.
#[cfg(test)]
mod commit_invariant_tests {
    use super::*;
    use aurora_engine::document::DocumentRevision;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Calls {
        /// Single-document `save_adjustments` writes.
        singles: Vec<(ImageId, AdjustmentSet)>,
        /// Synchronous `save_adjustments_batch_report` calls.
        sync_batches: Vec<Vec<(ImageId, AdjustmentSet)>>,
        /// `save_adjustments_queued` calls that actually queued.
        queued_batches: Vec<Vec<(ImageId, AdjustmentSet)>>,
        /// Every `save_adjustments_queued` call, including engines with no queue.
        queued_offers: usize,
    }

    /// A minimal `ImageEngine` that records save calls and keeps the last
    /// written adjustments so `load_adjustments` round-trips.
    struct RecordingEngine {
        images: Vec<ImageInfo>,
        adjustments: Mutex<HashMap<ImageId, AdjustmentSet>>,
        calls: Mutex<Calls>,
        /// Scenario switch: `true` makes `save_adjustments_queued` return a
        /// summary, `false` returns `None` so the synchronous report is used.
        has_queue: bool,
    }

    impl RecordingEngine {
        fn new(has_queue: bool, images: Vec<ImageInfo>) -> Self {
            Self { images, adjustments: Mutex::new(HashMap::new()), calls: Mutex::new(Calls::default()), has_queue }
        }

        fn save_counts(&self) -> (usize, usize, usize) {
            let calls = self.calls.lock().unwrap();
            (calls.singles.len(), calls.sync_batches.len(), calls.queued_batches.len())
        }

        fn queued_offers(&self) -> usize {
            self.calls.lock().unwrap().queued_offers
        }

        fn last_sync_batch(&self) -> Vec<(ImageId, AdjustmentSet)> {
            self.calls.lock().unwrap().sync_batches.last().cloned().unwrap_or_default()
        }

        fn last_queued_batch(&self) -> Vec<(ImageId, AdjustmentSet)> {
            self.calls.lock().unwrap().queued_batches.last().cloned().unwrap_or_default()
        }
    }

    impl ImageEngine for RecordingEngine {
        fn name(&self) -> String {
            "recording".into()
        }

        fn images(&self) -> Vec<ImageInfo> {
            self.images.clone()
        }

        fn import(&self, _paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
            Ok(Vec::new())
        }

        fn collections(&self) -> Vec<Collection> {
            Vec::new()
        }

        fn update_info(&self, _info: &ImageInfo) -> anyhow::Result<()> {
            Ok(())
        }

        fn load_adjustments(&self, image: ImageId) -> AdjustmentSet {
            self.adjustments.lock().unwrap().get(&image).cloned().unwrap_or_default()
        }

        fn save_adjustments(&self, image: ImageId, adjustments: &AdjustmentSet) -> anyhow::Result<()> {
            self.calls.lock().unwrap().singles.push((image, adjustments.clone()));
            self.adjustments.lock().unwrap().insert(image, adjustments.clone());
            Ok(())
        }

        fn save_adjustments_batch_report(&self, items: &[(ImageId, AdjustmentSet)]) -> SaveReport {
            self.calls.lock().unwrap().sync_batches.push(items.to_vec());
            let documents = items
                .iter()
                .map(|(image, adjustments)| {
                    self.adjustments.lock().unwrap().insert(*image, adjustments.clone());
                    DocumentSaveResult::saved(*image, String::new())
                })
                .collect();
            SaveReport::new(documents, None)
        }

        fn save_adjustments_queued(&self, items: &[(ImageId, AdjustmentSet)]) -> Option<QueuedSaveSummary> {
            let mut calls = self.calls.lock().unwrap();
            calls.queued_offers += 1;
            if !self.has_queue {
                return None;
            }
            calls.queued_batches.push(items.to_vec());
            drop(calls);
            for (image, adjustments) in items {
                self.adjustments.lock().unwrap().insert(*image, adjustments.clone());
            }
            Some(QueuedSaveSummary {
                documents: items
                    .iter()
                    .map(|(image, _)| QueuedDocumentResult {
                        image: *image,
                        path: format!("image-{}.jpg", image.0),
                        outcome: QueuedSaveOutcome::Accepted { revision: DocumentRevision::new(1) },
                    })
                    .collect(),
            })
        }

        fn new_variant(&self, _image: ImageId, _adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo> {
            anyhow::bail!("not used by these tests")
        }

        fn thumbnail(&self, _image: ImageId, _adjustments: &AdjustmentSet, _max_size: u32) -> anyhow::Result<RgbaImage> {
            anyhow::bail!("not used by these tests")
        }

        fn render_preview(&self, _request: &PreviewRequest) -> anyhow::Result<PreviewFrame> {
            anyhow::bail!("not used by these tests")
        }

        fn sample_color(&self, _image: ImageId, _adjustments: &AdjustmentSet, _point: [f32; 2]) -> anyhow::Result<[f32; 3]> {
            anyhow::bail!("not used by these tests")
        }

        fn auto_adjust(&self, _image: ImageId, _current: &AdjustmentSet, _target: AutoTarget) -> anyhow::Result<AutoResult> {
            anyhow::bail!("not used by these tests")
        }

        fn pick_white_balance(&self, _image: ImageId, _point: [f32; 2]) -> anyhow::Result<(f32, f32)> {
            anyhow::bail!("not used by these tests")
        }

        fn compute_ai_mask(&self, _image: ImageId, _kind: AiMaskKind) -> anyhow::Result<String> {
            anyhow::bail!("not used by these tests")
        }

        fn icc_profiles(&self) -> Vec<String> {
            Vec::new()
        }

        fn lens_profiles(&self) -> Vec<String> {
            Vec::new()
        }

        fn styles(&self) -> Vec<StyleInfo> {
            Vec::new()
        }

        fn start_export(&self, _job: ExportJob) -> anyhow::Result<()> {
            Ok(())
        }

        fn job_status(&self, _job_id: u64) -> JobStatus {
            JobStatus::Queued
        }
    }

    fn image(id: u64) -> ImageInfo {
        ImageInfo {
            id: ImageId(id),
            path: PathBuf::from(format!("image-{id}.jpg")),
            file_name: format!("image-{id}.jpg"),
            kind: FileKind::Jpeg,
            width: 4,
            height: 4,
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
        }
    }

    fn assemble_state(engine: &Arc<RecordingEngine>) -> AppState {
        let engine: Arc<dyn ImageEngine> = engine.clone();
        AppState::assemble(engine, Source::Demo, Settings::default())
    }

    /// `edit_live` × N followed by one `end_gesture` is one undo step and one
    /// save, and the last live value is what gets written.
    #[test]
    fn live_gesture_commits_one_step_and_one_save() {
        let fake = Arc::new(RecordingEngine::new(false, vec![image(1)]));
        let mut state = assemble_state(&fake);
        let id = state.library.current.expect("assemble selects the first image");

        state.edit_live_inner("Exposure", |a| a.exposure.exposure = 0.25, None);
        state.edit_live_inner("Exposure", |a| a.exposure.exposure = 0.75, None);
        state.edit_live_inner("Exposure", |a| a.exposure.exposure = 1.5, None);
        // Live changes stay pending until the gesture ends.
        assert!(state.editor.edits[&id].undo.is_empty());
        assert_eq!(fake.save_counts(), (0, 0, 0));

        state.end_gesture_inner(None);

        let edit = &state.editor.edits[&id];
        assert_eq!(edit.undo.len(), 1, "one undo step per gesture");
        assert_eq!(edit.undo[0].1.exposure.exposure, 0.0, "undo restores the value before the drag");
        assert_eq!(edit.adj.exposure.exposure, 1.5, "the last live value wins");
        assert_eq!(fake.save_counts(), (0, 1, 0), "exactly one save for the gesture");
        let batch = fake.last_sync_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].0, id);
        assert_eq!(batch[0].1.exposure.exposure, 1.5);

        // A repeated end_gesture must not add another undo step or save.
        state.end_gesture_inner(None);
        assert_eq!(state.editor.edits[&id].undo.len(), 1);
        assert_eq!(fake.save_counts(), (0, 1, 0));
    }

    /// A discrete `edit` is one undo step and one save; `undo` and `redo` both
    /// restore the value and commit once each.
    #[test]
    fn edit_undo_and_redo_each_commit_once() {
        let fake = Arc::new(RecordingEngine::new(false, vec![image(7)]));
        let mut state = assemble_state(&fake);
        let id = state.library.current.expect("assemble selects the first image");

        state.edit_inner("Exposure", |a| a.exposure.exposure = 2.0, None);
        assert_eq!(state.editor.edits[&id].undo.len(), 1);
        assert_eq!(state.editor.edits[&id].adj.exposure.exposure, 2.0);
        assert_eq!(fake.save_counts(), (0, 1, 0));

        state.undo_inner(None);
        assert_eq!(state.editor.edits[&id].adj.exposure.exposure, 0.0, "undo restores the default");
        assert!(state.editor.edits[&id].undo.is_empty());
        assert_eq!(fake.save_counts(), (0, 2, 0), "undo commits once");

        state.redo_inner(None);
        assert_eq!(state.editor.edits[&id].adj.exposure.exposure, 2.0, "redo restores the edit");
        assert_eq!(state.editor.edits[&id].undo.len(), 1);
        assert_eq!(fake.save_counts(), (0, 3, 0), "redo commits once");
    }

    /// `edit_selected` gives every affected image one undo step and writes a
    /// single batch containing the whole selection.
    #[test]
    fn edit_selected_writes_one_batch_for_the_selection() {
        let fake = Arc::new(RecordingEngine::new(false, (1..=4).map(image).collect()));
        let mut state = assemble_state(&fake);
        state.library.selection = [ImageId(1), ImageId(2), ImageId(3)].into_iter().collect();

        state.edit_selected_inner("Exposure", |a| a.exposure.exposure = 1.25, None);

        for id in [ImageId(1), ImageId(2), ImageId(3)] {
            assert_eq!(state.editor.edits[&id].undo.len(), 1, "{id:?} gets one undo step");
            assert_eq!(state.editor.edits[&id].adj.exposure.exposure, 1.25, "{id:?} has the new value");
        }
        assert_eq!(state.editor.edits[&ImageId(4)].undo.len(), 0, "unselected image is untouched");

        assert_eq!(fake.save_counts(), (0, 1, 0), "one batch save for the selection");
        let batch = fake.last_sync_batch();
        let ids: Vec<ImageId> = batch.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![ImageId(1), ImageId(2), ImageId(3)]);
        assert!(batch.iter().all(|(_, adj)| adj.exposure.exposure == 1.25));
    }

    /// The same multi-image edit uses one queued summary with every item
    /// accepted when the engine exposes a queue.
    #[test]
    fn edit_selected_uses_the_queue_when_the_engine_has_one() {
        let fake = Arc::new(RecordingEngine::new(true, (1..=3).map(image).collect()));
        let mut state = assemble_state(&fake);
        state.library.selection = [ImageId(1), ImageId(2), ImageId(3)].into_iter().collect();

        state.edit_selected_inner("Exposure", |a| a.exposure.exposure = -0.5, None);

        for id in [ImageId(1), ImageId(2), ImageId(3)] {
            assert_eq!(state.editor.edits[&id].undo.len(), 1);
        }
        assert_eq!(fake.save_counts(), (0, 0, 1), "queued instead of synchronous");
        let batch = fake.last_queued_batch();
        assert_eq!(batch.len(), 3, "one item per selected image");
        let summary = QueuedSaveSummary {
            documents: batch
                .iter()
                .map(|(image, _)| QueuedDocumentResult {
                    image: *image,
                    path: String::new(),
                    outcome: QueuedSaveOutcome::Accepted { revision: DocumentRevision::new(1) },
                })
                .collect(),
        };
        assert_eq!(summary.accepted(), 3);
    }

    #[test]
    fn after_commit_many_uses_the_queued_path_when_available() {
        let fake = Arc::new(RecordingEngine::new(true, vec![image(1)]));
        let mut state = assemble_state(&fake);
        let id = state.library.current.expect("assemble selects the first image");
        state.editor.edits.get_mut(&id).unwrap().adj.exposure.exposure = 0.8;

        state.after_commit_many_inner(vec![id], None);

        assert_eq!(fake.queued_offers(), 1);
        assert_eq!(fake.save_counts(), (0, 0, 1), "queue consulted and used");
        assert_eq!(fake.last_queued_batch().len(), 1);
        assert!(fake.last_sync_batch().is_empty(), "no synchronous report");
    }

    #[test]
    fn after_commit_many_falls_back_to_the_sync_report() {
        let fake = Arc::new(RecordingEngine::new(false, vec![image(1)]));
        let mut state = assemble_state(&fake);
        let id = state.library.current.expect("assemble selects the first image");
        state.editor.edits.get_mut(&id).unwrap().adj.exposure.exposure = 0.8;

        state.after_commit_many_inner(vec![id], None);

        assert_eq!(fake.queued_offers(), 1, "the queue is offered first");
        assert_eq!(fake.save_counts(), (0, 1, 0), "no queue means the synchronous report");
        let batch = fake.last_sync_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].1.exposure.exposure, 0.8);

        // An empty list commits nothing.
        state.after_commit_many_inner(Vec::new(), None);
        assert_eq!(fake.save_counts(), (0, 1, 0));
    }
}

fn import_with_edits(
    engine: &dyn ImageEngine,
    paths: &[std::path::PathBuf],
) -> anyhow::Result<Vec<(ImageInfo, Edit)>> {
    let images = engine.import(paths)?;
    Ok(images
        .into_iter()
        .map(|info| {
            let mut edit = Edit::default();
            if info.has_adjustments {
                edit.adj = engine.load_adjustments(info.id);
            }
            (info, edit)
        })
        .collect())
}

#[cfg(test)]
mod import_edits_tests {
    use super::import_with_edits;
    use aurora_engine::FileEngine;
    use std::{fs, process::Command};

    #[test]
    fn eip_variants_fill_the_preview_edits_immediately() {
        let dir = std::env::temp_dir().join(format!("aurora-eip-import-preview-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let engine = FileEngine::open(&dir).unwrap();
        let output = Command::new("python")
            .args([
                "-B",
                "-c",
                r#"
import base64, pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1])
png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAEElEQVR4AQEFAPr/AICAgP8FhAKAFCF59QAAAABJRU5ErkJggg==")
(root / "single.png").write_bytes(png)
with zipfile.ZipFile(root / "variants.eip", "w") as package:
    package.writestr("manifest1680.xml", """<Manifest><File><RawPath>photo.png</RawPath>
        <SettingsPath>different/settings.cos</SettingsPath></File></Manifest>""")
    package.writestr("photo.png", png)
    package.writestr("different/settings.cos", """<IMG>
        <VAR><DL><E K="Exposure" V="1.25"/></DL></VAR>
        <VAR><DL><E K="Exposure" V="-2.0"/></DL></VAR></IMG>""")
"#,
            ])
            .arg(&dir)
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let added = import_with_edits(&engine, &[dir.join("variants.eip"), dir.join("single.png")]).unwrap();
        let [(first, first_edit), (second, second_edit), (single, single_edit)] = added.as_slice() else {
            panic!("expected two EIP variants and one standalone image");
        };
        assert_ne!(first.id, second.id);
        assert_eq!((first.width, first.height), (1, 1));
        assert_eq!((first.variant, second.variant), (1, 2));
        assert_eq!(first_edit.adj.exposure.exposure, 1.25);
        assert_eq!(second_edit.adj.exposure.exposure, -2.0);
        assert_eq!(single.variant, 1);
        assert!(single_edit.adj.is_default());
        drop(engine);
        fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod c1_save_tests {
    use super::*;
    use aurora_engine::c1_cos::{CosSaveOutcome, CosSaveReport};
    use aurora_engine::c1_writeback::CatalogSaveOutcome;

    #[test]
    fn catalog_outcomes_map_to_statuses() {
        let saved = Ok(CatalogSaveOutcome::Saved {
            columns: 3,
            backup: std::path::PathBuf::from("x.bak"),
        });
        assert_eq!(catalog_save_status(&saved), (C1SaveStatus::Saved { cells: 3 }, Vec::new()));
        let unchanged = Ok(CatalogSaveOutcome::Unchanged { warnings: vec!["w".to_owned()] });
        assert_eq!(
            catalog_save_status(&unchanged),
            (C1SaveStatus::Unchanged, vec!["w".to_owned()])
        );
        assert_eq!(catalog_save_status(&Ok(CatalogSaveOutcome::NotACatalog)).0, C1SaveStatus::NotACatalog);
        let error: anyhow::Result<CatalogSaveOutcome> = Err(anyhow::anyhow!("boom"));
        assert!(matches!(catalog_save_status(&error).0, C1SaveStatus::Failed { .. }));
    }

    #[test]
    fn settings_outcomes_map_to_statuses() {
        let written = Ok(CosSaveOutcome::Written(CosSaveReport {
            edits: 2,
            layers: vec!["AL".to_owned()],
            warnings: Vec::new(),
        }));
        assert_eq!(cos_save_status(&written), (C1SaveStatus::Saved { cells: 2 }, Vec::new()));
        let unchanged = Ok(CosSaveOutcome::Unchanged(CosSaveReport::default()));
        assert_eq!(cos_save_status(&unchanged).0, C1SaveStatus::Unchanged);
        let error: anyhow::Result<CosSaveOutcome> = Err(anyhow::anyhow!("boom"));
        assert!(matches!(cos_save_status(&error).0, C1SaveStatus::Failed { .. }));
    }

    #[test]
    fn the_action_needs_an_image_and_a_catalog() {
        assert!(c1_save_enabled(false, true, false, false));
        assert!(!c1_save_enabled(false, false, false, false));
        assert!(!c1_save_enabled(false, true, true, false));
        assert!(c1_save_enabled(false, true, true, true));
    }

    #[test]
    fn a_running_save_ignores_a_second_call() {
        assert!(!c1_save_enabled(true, true, true, true));
        assert!(!c1_save_enabled(true, true, false, false));
    }
}
