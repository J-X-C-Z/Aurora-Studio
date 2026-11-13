//! The application model: catalog, selection, edits with undo, and the preview pipeline.
//!
//! Views hold an `Entity<AppState>` and observe it. All engine calls run on the
//! background executor; results come back through `cx.spawn` and replace state
//! only if they are still the newest request.

mod binding;
mod clipboard;
mod collections;
mod files;
pub(crate) mod library_document;
pub(crate) mod render;
mod settings;
mod source;
mod arrange;
mod speed;
mod show;
mod nodes;
mod analysis;
mod cull;
mod ahead;
pub(crate) mod scopes;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use gpui_kit::{App, AppContext, Context, Entity, EventEmitter, Pixels, RenderImage, SharedString, Task};
use lumen_engine::*;

pub use binding::*;
pub use clipboard::ClipGroup;
pub use collections::CollNode;
pub use library_document::{DocumentKind, LibraryDocument};
pub use settings::*;
pub use source::Source;
pub use arrange::{Arrangement, Stack};
pub use speed::{SpeedEdit, SpeedParam};
pub use cull::Cull;
pub use ahead::MergeKind;
pub use scopes::ScopeKind;
pub use nodes::{GraphExt, NewNode, add_shape, flattened, is_full};
pub mod open {
    pub use super::source::{demo, display_name, is_catalog, open};
}

use crate::i18n::t;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToolTab {
    Library,
    Capture,
    Lens,
    Color,
    Exposure,
    Details,
    Layers,
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
    pub const ALL: [ToolTab; 13] = [
        ToolTab::Library,
        ToolTab::Capture,
        ToolTab::Lens,
        ToolTab::Color,
        ToolTab::Exposure,
        ToolTab::Details,
        ToolTab::Layers,
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
            ToolTab::Color => t("Color", "色彩"),
            ToolTab::Exposure => t("Exposure", "曝光"),
            ToolTab::Details => t("Details", "细节"),
            ToolTab::Layers => t("Nodes", "节点"),
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
            ToolTab::Color => I::Palette,
            ToolTab::Exposure => I::SunMedium,
            ToolTab::Details => I::ScanSearch,
            ToolTab::Layers => I::Workflow,
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
    pub rejected_only: bool,
    pub color_tag: Option<ColorTag>,
    pub search: String,
    pub edited_only: bool,
}

impl Default for Filter {
    fn default() -> Self {
        Self { min_rating: -1, rejected_only: false, color_tag: None, search: String::new(), edited_only: false }
    }
}

impl Filter {
    fn matches_rating(&self, rating: i8) -> bool { rating >= self.min_rating && (!self.rejected_only || rating < 0) }
}

#[test]
fn rejected_filter_and_rating_threshold_are_distinct() {
    let all = Filter::default();
    assert!((-1..=5).all(|rating| all.matches_rating(rating)));
    let rejected = Filter { rejected_only: true, ..Default::default() };
    assert!(rejected.matches_rating(-1));
    assert!((0..=5).all(|rating| !rejected.matches_rating(rating)));
    let starred = Filter { min_rating: 4, ..Default::default() };
    assert!(!starred.matches_rating(3));
    assert!(starred.matches_rating(4) && starred.matches_rating(5));
}

/// One image's adjustments and history.
#[derive(Clone, Default)]
pub struct Edit {
    pub adj: AdjustmentSet,
    pub undo: Vec<(SharedString, AdjustmentSet)>,
    pub redo: Vec<(SharedString, AdjustmentSet)>,
    /// Set while a drag is in progress: the state before it began, and its label.
    gesture: Option<(SharedString, AdjustmentSet)>,
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
    /// The developed-frame region this frame shows.
    pub region: NormRect,
    pub image_id: ImageId,
    /// Clipped highlights (red) and shadows (blue), when the exposure warning is on.
    pub warning: Option<Arc<RenderImage>>,
    /// Sharp edges in green, when the focus mask is on.
    pub focus: Option<Arc<RenderImage>>,
    pub(crate) stamp: crate::perf::Stamp,
}

impl Frame {
    pub(crate) fn release(self, cx: &mut gpui_kit::App) {
        cx.drop_image(self.image, None);
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
    /// The portrait parameters are shown although nothing renders them yet.
    pub portrait_open: bool,
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
    pub images: Vec<ImageInfo>,
    pub nodes: Vec<CollNode>,
    pub library_document: Option<LibraryDocument>,
    pub library_busy: bool,
    pub library_offline: HashSet<ImageId>,
    pub library_include_subfolders: bool,
    pub(crate) file_status_generation: u64,
    pub recent_imports: BTreeSet<ImageId>,
    pub recent_captures: BTreeSet<ImageId>,
    pub trash: BTreeSet<ImageId>,
    pub active_collection: u64,
    pub filter: Filter,
    pub sort: SortKey,
    pub sort_desc: bool,
    pub selection: BTreeSet<ImageId>,
    pub current: Option<ImageId>,
    pub edits: HashMap<ImageId, Edit>,
    /// `None` edits the background layer.
    pub active_layer: Option<u64>,
    /// A selected mixer or the output in the node editor; layers use `active_layer`.
    pub active_node: Option<u64>,
    pub tool_tab: ToolTab,
    pub cursor: CursorTool,
    pub view: ViewPrefs,
    pub panels: PanelPrefs,
    pub zoom: ViewZoom,
    /// Viewer size in device pixels, written by the viewer each layout.
    pub viewport: (f32, f32),
    /// Window position of the viewer's top-left corner.
    pub viewport_origin: gpui_kit::Point<Pixels>,
    pub scale_factor: f32,
    pub preview: Preview,
    pub thumbs: HashMap<ImageId, Arc<RenderImage>>,
    thumb_queue: Vec<ImageId>,
    thumb_task: Option<Task<()>>,
    pub clipboard: Option<AdjustmentSet>,
    /// Tools the adjustments clipboard copies and applies.
    pub clip_groups: BTreeSet<ClipGroup>,
    pub active_collection_name: SharedString,
    pub capturing: bool,
    /// Freehand notes drawn with the annotate tool, in frame space, per image.
    pub annotations: HashMap<ImageId, Vec<Vec<(f32, f32)>>>,
    pub styles: Vec<StyleInfo>,
    pub icc_profiles: Vec<String>,
    pub lens_profiles: Vec<String>,
    pub recipes: Vec<ProcessRecipe>,
    pub selected_recipe: usize,
    pub jobs: Vec<JobView>,
    next_job: u64,
    next_layer: u64,
    pub sampled_color: Option<[f32; 3]>,
    pub picker_point: Option<[f32; 2]>,
    pub camera: CameraStatus,
    pub camera_settings: Option<CameraSettings>,
    /// What the engine's tethering backend answered; `None` until the first probe returns.
    pub tether: Option<TetherStatus>,
    pub tether_probing: bool,
    pub notice: Option<SharedString>,
    pub ai_busy: bool,
    /// The current image with a hovered style applied, shown instead of its own
    /// adjustments until the pointer leaves the style.
    pub hover_look: Option<AdjustmentSet>,
    /// AI mask kinds the engine refused this session (no model, no runtime).
    pub ai_unavailable: HashSet<AiMaskKind>,
    /// Closed-eye results for this session, shown on the thumbnails.
    pub eyes: HashMap<ImageId, EyeReport>,
    /// A selection analysis (eyes, rotation, Match Look) is running.
    pub analysing: bool,
    /// Images whose thumbnail the engine could not make.
    pub thumb_failed: HashSet<ImageId>,
    /// Why the engine shows the current image from something other than its full
    /// data (`planned::source_status`), refreshed with each preview.
    pub source_note: Option<(ImageId, SharedString)>,
    pub settings: Settings,
    pub engine_name: String,
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
    pub custom_order: Vec<ImageId>,
    pub stacks: Vec<Stack>,
    /// Picks of the stacks shown expanded.
    pub open_stacks: HashSet<ImageId>,
    pub slideshow: Option<Slideshow>,
    pub cull: Option<Cull>,
    /// The cursor before the last change, and when it changed (Speed Edit
    /// puts it back after a key that also selects a cursor tool).
    pub(crate) prev_cursor: CursorTool,
    pub(crate) cursor_changed: std::time::Instant,
}

impl EventEmitter<StateEvent> for AppState {}

impl AppState {
    pub fn new(engine: Arc<dyn ImageEngine>, source: Source, settings: Settings, cx: &mut Context<Self>) -> Self {
        let images = engine.images();
        let edits = images
            .iter()
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
        let mut this = Self {
            library_document: source.path().and_then(|p| LibraryDocument::load(if p.is_dir() { p } else { p.parent()? }).ok().flatten()),
            library_busy: false,
            library_offline: HashSet::new(),
            library_include_subfolders: true,
            file_status_generation: 0,
            styles: engine.styles(),
            icc_profiles: engine.icc_profiles(),
            lens_profiles: engine.lens_profiles(),
            nodes: collections::flatten(engine.as_ref(), &engine.collections()),
            recent_imports: BTreeSet::new(),
            recent_captures: BTreeSet::new(),
            trash: BTreeSet::new(),
            camera: engine.camera_status(),
            camera_settings: engine.camera_settings(),
            tether: None,
            ai_unavailable: HashSet::new(),
            eyes: HashMap::new(),
            analysing: false,
            thumb_failed: HashSet::new(),
            source_note: None,
            tether_probing: false,
            engine_name: engine.name(),
            engine,
            selection: current.into_iter().collect(),
            current,
            images,
            active_collection: 1,
            filter: Filter::default(),
            sort: SortKey::Name,
            sort_desc: false,
            edits,
            active_layer: None,
            active_node: None,
            tool_tab: tab_from(&settings.workspace.tab).unwrap_or(ToolTab::Exposure),
            cursor: CursorTool::Select,
            view: ViewPrefs::default(),
            settings,
            panels: PanelPrefs::default(),
            zoom: ViewZoom::default(),
            viewport: (1200.0, 800.0),
            viewport_origin: Default::default(),
            scale_factor: 1.0,
            preview: Preview::default(),
            thumbs: HashMap::new(),
            thumb_queue: Vec::new(),
            thumb_task: None,
            clipboard: None,
            clip_groups: ClipGroup::ALL.iter().copied().filter(|g| !g.is_composition()).collect(),
            active_collection_name: t("All Images", "所有图像").into(),
            capturing: false,
            annotations: HashMap::new(),
            recipes,
            selected_recipe: 0,
            jobs: Vec::new(),
            next_job: 1,
            next_layer: 1,
            sampled_color: None,
            picker_point: None,
            notice: None,
            ai_busy: false,
            hover_look: None,
            welcome: source == Source::Demo,
            source,
            opening: None,
            open_error: None,
            speed: None,
            metadata_rev: 0,
            custom_order: Vec::new(),
            stacks: Vec::new(),
            open_stacks: HashSet::new(),
            slideshow: None,
            cull: None,
            prev_cursor: CursorTool::Select,
            cursor_changed: std::time::Instant::now(),
        };
        let layout = this.settings.workspace.clone();
        this.library_include_subfolders = this.library_document.as_ref().is_none_or(|d| d.include_subfolders);
        if let Some((root, output)) = this.engine.library_root().zip(this.library_document.as_ref().and_then(|d| d.shortcuts.get("Output"))) {
            for recipe in &mut this.recipes { recipe.destination = Some(LibraryDocument::resolve(&root, output)); }
        }
        this.load_layout(&layout);
        this.load_arrangement();
        this.view.viewer_gray = this.settings.viewer_gray;
        this.view.thumb_size = this.settings.thumb_size;
        this.view.show_info_overlay = this.settings.info_overlay;
        let all: Vec<_> = this.images.iter().map(|i| i.id).collect();
        this.queue_thumbs(all, cx);
        this.request_render(RenderQuality::Final, cx);
        this.probe_tether(cx);
        this.refresh_file_status(cx);
        this
    }

    // ───────────── Catalog ─────────────

    pub fn info(&self, id: ImageId) -> Option<&ImageInfo> {
        self.images.iter().find(|i| i.id == id)
    }

    pub fn current_info(&self) -> Option<&ImageInfo> {
        self.current.and_then(|id| self.info(id))
    }

    /// Images passing the filter, in sort order.
    pub fn visible_images(&self) -> Vec<&ImageInfo> {
        let f = &self.filter;
        let search = f.search.to_lowercase();
        let in_trash = self.node(self.active_collection).is_some_and(|n| n.kind == CollectionKind::Trash);
        let members = self.members(self.active_collection);
        let mut out: Vec<&ImageInfo> = self
            .images
            .iter()
            .filter(|i| members.as_ref().is_none_or(|m| m.contains(&i.id)))
            .filter(|i| in_trash || !self.trash.contains(&i.id))
            .filter(|i| f.matches_rating(i.rating))
            .filter(|i| f.color_tag.is_none_or(|t| i.color_tag == t))
            .filter(|i| !f.edited_only || self.edits.get(&i.id).is_some_and(|e| !e.adj.is_default()))
            .filter(|i| {
                search.is_empty()
                    || i.file_name.to_lowercase().contains(&search)
                    || i.keywords.iter().any(|k| k.to_lowercase().contains(&search))
            })
            .collect();
        match self.sort {
            SortKey::Name => out.sort_by(|a, b| a.file_name.cmp(&b.file_name)),
            SortKey::CaptureDate => out.sort_by(|a, b| a.exif.captured_at.cmp(&b.exif.captured_at)),
            SortKey::Rating => out.sort_by(|a, b| b.rating.cmp(&a.rating)),
            SortKey::ColorTag => out.sort_by_key(|i| i.color_tag as u8),
            SortKey::FileSize => out.sort_by_key(|i| i.file_size),
            SortKey::Custom => {
                let ranks: HashMap<_, _> = self.custom_order.iter().enumerate().map(|(rank, id)| (*id, rank)).collect();
                out.sort_by(|a, b| (ranks.get(&a.id).copied().unwrap_or(usize::MAX), &a.file_name).cmp(&(ranks.get(&b.id).copied().unwrap_or(usize::MAX), &b.file_name)));
            }
        }
        if self.sort_desc {
            out.reverse();
        }
        self.fold_stacks(out)
    }

    pub fn select(&mut self, id: ImageId, extend: bool, range: bool, cx: &mut Context<Self>) {
        self.hover_look = None;
        if range {
            if let Some(anchor) = self.current {
                let visible: Vec<ImageId> = self.visible_images().iter().map(|i| i.id).collect();
                let a = visible.iter().position(|i| *i == anchor);
                let b = visible.iter().position(|i| *i == id);
                if let (Some(a), Some(b)) = (a, b) {
                    let (lo, hi) = (a.min(b), a.max(b));
                    self.selection = visible[lo..=hi].iter().copied().collect();
                }
            }
        } else if extend {
            if !self.selection.remove(&id) {
                self.selection.insert(id);
            }
        } else {
            self.selection.clear();
            self.selection.insert(id);
        }
        self.set_current(id, cx);
        cx.notify();
    }

    /// Makes `id` the image the tools edit, leaving the selection alone.
    pub fn set_current(&mut self, id: ImageId, cx: &mut Context<Self>) {
        if self.current != Some(id) {
            self.end_gesture(cx);
            self.current = Some(id);
            self.active_layer = None;
            self.active_node = None;
            self.zoom = ViewZoom::default();
            self.preview.before_dirty = true;
            self.sampled_color = None;
            self.picker_point = None;
            cx.emit(StateEvent::CurrentChanged);
            self.request_render(RenderQuality::Final, cx);
        }
        cx.notify();
    }

    /// The images multiple view shows, the pinned compare variant first; empty
    /// when a single image fills the viewer.
    pub fn multi_images(&self) -> Vec<ImageId> {
        if !self.view.multi {
            return Vec::new();
        }
        let pinned = self.view.compare_variant.filter(|id| self.info(*id).is_some());
        let mut ids: Vec<ImageId> = pinned.into_iter().collect();
        ids.extend(self.selection.iter().copied().filter(|id| Some(*id) != pinned).take(16));
        if ids.len() < 2 { Vec::new() } else { ids }
    }

    /// Moves the current image by `delta` within the visible order.
    pub fn step(&mut self, delta: isize, cx: &mut Context<Self>) {
        let visible: Vec<ImageId> = self.visible_images().iter().map(|i| i.id).collect();
        if visible.is_empty() {
            return;
        }
        let pos = self.current.and_then(|c| visible.iter().position(|i| *i == c)).unwrap_or(0) as isize;
        let next = (pos + delta).clamp(0, visible.len() as isize - 1) as usize;
        self.select(visible[next], false, false, cx);
    }

    pub fn select_all(&mut self, cx: &mut Context<Self>) {
        self.selection = self.visible_images().iter().map(|i| i.id).collect();
        cx.notify();
    }

    pub fn set_rating(&mut self, rating: i8, cx: &mut Context<Self>) {
        let changes = self.images.iter().filter(|i| self.selection.contains(&i.id) && i.rating != rating)
            .cloned().map(|mut info| { info.rating = rating; info }).collect();
        self.persist_infos(changes, cx);
        cx.notify();
    }

    pub fn set_color_tag(&mut self, tag: ColorTag, cx: &mut Context<Self>) {
        let changes = self.images.iter().filter(|i| self.selection.contains(&i.id) && i.color_tag != tag)
            .cloned().map(|mut info| { info.color_tag = tag; info }).collect();
        self.persist_infos(changes, cx);
        cx.notify();
    }

    fn persist_infos(&mut self, changes: Vec<ImageInfo>, cx: &mut Context<Self>) -> bool {
        if changes.is_empty() { return true; }
        if let Err(err) = self.engine.update_infos(&changes) {
            self.notify(format!("{}: {err}", t("Could not save metadata", "无法存储元数据")), cx);
            return false;
        }
        let mut changes: HashMap<_, _> = changes.into_iter().map(|i| (i.id, i)).collect();
        for info in &mut self.images {
            if let Some(updated) = changes.remove(&info.id) { *info = updated; }
        }
        true
    }

    /// Saves the current image's IPTC fields and keywords as a metadata preset.
    pub fn save_metadata_preset(&mut self, name: String, cx: &mut Context<Self>) {
        let Some(info) = self.current_info() else { return };
        let preset = MetadataPreset { name: name.clone(), iptc: info.iptc.clone(), keywords: info.keywords.clone() };
        match self.settings.metadata_presets.iter_mut().find(|p| p.name == name) {
            Some(existing) => *existing = preset,
            None => self.settings.metadata_presets.push(preset),
        }
        self.settings.save();
        self.notify(format!("{} {name}", t("Saved metadata preset", "已存储元数据预设")), cx);
    }

    /// Applies metadata preset `index` to every selected image.
    pub fn apply_metadata_preset(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(preset) = self.settings.metadata_presets.get(index).cloned() else { return };
        let fill = |dst: &mut String, src: &String| {
            if !src.trim().is_empty() {
                *dst = src.clone();
            }
        };
        let ids: HashSet<ImageId> = if self.selection.is_empty() { self.current.into_iter().collect() } else { self.selection.iter().copied().collect() };
        let mut changes: Vec<_> = self.images.iter().filter(|i| ids.contains(&i.id)).cloned().collect();
        for info in &mut changes {
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
        }
        if !self.persist_infos(changes, cx) { return; }
        // Fields keep their own text; a new revision starts them from the new values.
        self.metadata_rev += 1;
        let msg = format!("{} {} · {} {}", t("Applied", "已应用"), preset.name, ids.len(), t("images", "张图像"));
        self.notify(msg, cx);
    }

    pub fn rename_metadata_preset(&mut self, index: usize, name: String, cx: &mut Context<Self>) {
        if let Some(p) = self.settings.metadata_presets.get_mut(index) {
            p.name = name;
            self.settings.save();
            cx.notify();
        }
    }

    pub fn delete_metadata_preset(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.settings.metadata_presets.len() {
            let p = self.settings.metadata_presets.remove(index);
            self.settings.save();
            self.notify(format!("{} {}", t("Deleted preset", "已删除预设"), p.name), cx);
        }
    }

    pub fn update_current_info(&mut self, f: impl FnOnce(&mut ImageInfo), cx: &mut Context<Self>) {
        if let Some(id) = self.current
            && let Some(info) = self.images.iter().find(|i| i.id == id)
        {
            let mut info = info.clone();
            f(&mut info);
            self.persist_infos(vec![info], cx);
            cx.notify();
        }
    }

    pub fn import(&mut self, paths: Vec<std::path::PathBuf>, cx: &mut Context<Self>) {
        self.import_with(paths, ImportAfter::default(), cx);
    }

    /// Imports files, then applies metadata, a style and an album to what arrived.
    pub fn import_with(&mut self, paths: Vec<std::path::PathBuf>, after: ImportAfter, cx: &mut Context<Self>) {
        if self.library_busy { self.notify(t("A file operation is already running", "正在处理文件，请稍候"), cx); return; }
        self.library_busy = true;
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async move { engine.import(&paths).map(|added| (added, collections::flatten(engine.as_ref(), &engine.collections()))) }).await;
            this.update(cx, |this, cx| { this.library_busy = false; match result {
                Ok((added, nodes)) => {
                    let user_ids: HashSet<_> = this.nodes.iter().filter(|n| n.user).map(|n| n.id).collect();
                    this.nodes.retain(|n| n.user);
                    this.nodes.extend(nodes.into_iter().filter(|n| !user_ids.contains(&n.id)));
                    let ids: Vec<_> = added.iter().map(|i| i.id).collect();
                    for info in &added {
                        this.edits.insert(info.id, Edit::default());
                    }
                    let n = added.len();
                    this.recent_imports.extend(ids.iter().copied());
                    this.images.extend(added);
                    let mut metadata = Vec::new();
                    if !after.copyright.is_empty() || !after.creator.is_empty() || !after.keywords.is_empty() {
                        let imported: HashSet<_> = ids.iter().copied().collect();
                        for info in this.images.iter().filter(|i| imported.contains(&i.id)) {
                            let mut info = info.clone();
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
                            metadata.push(info);
                        }
                    }
                    let metadata_saved = this.persist_infos(metadata, cx);
                    if let Some(album) = after.album {
                        this.add_to_album(album, &ids, cx);
                    }
                    if let Some(style) = &after.style {
                        this.selection = ids.iter().copied().collect();
                        this.apply_style(style, false, cx);
                    }
                    this.queue_thumbs(ids.clone(), cx);
                    if let Some(first) = ids.first() {
                        this.select(*first, false, false, cx);
                    }
                    if metadata_saved {
                        this.notify(format!("{} {n} {}", t("Imported", "已导入"), t("images", "张图像")), cx);
                    }
                }
                Err(err) => this.notify(format!("{}: {err}", t("Import failed", "导入失败")), cx),
            }})
            .ok();
        })
        .detach();
    }

    pub fn new_variant(&mut self, clean: bool, cx: &mut Context<Self>) {
        let Some(id) = self.current else { return };
        let adj = if clean { AdjustmentSet::default() } else { self.adj().clone() };
        match self.engine.new_variant(id, &adj) {
            Ok(info) => {
                let new_id = info.id;
                let pos = self.images.iter().position(|i| i.id == id).map_or(self.images.len(), |p| p + 1);
                self.images.insert(pos, info);
                self.edits.insert(new_id, Edit { adj, ..Default::default() });
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
        self.current
            .and_then(|id| self.edits.get(&id))
            .map(|e| &e.adj)
            .unwrap_or_else(|| EMPTY.get_or_init(AdjustmentSet::default))
    }

    pub fn scope(&self) -> Scope {
        match self.active_layer {
            Some(id) if self.adj().layers.iter().any(|l| l.id == id) => Scope::Layer(id),
            _ => Scope::Global,
        }
    }

    pub fn active_layer_ref(&self) -> Option<&Layer> {
        let id = self.active_layer?;
        self.adj().layers.iter().find(|l| l.id == id)
    }

    pub fn can_undo(&self) -> bool {
        self.current.and_then(|id| self.edits.get(&id)).is_some_and(|e| !e.undo.is_empty())
    }

    pub fn can_redo(&self) -> bool {
        self.current.and_then(|id| self.edits.get(&id)).is_some_and(|e| !e.redo.is_empty())
    }

    pub fn history(&self) -> (Vec<SharedString>, Vec<SharedString>) {
        self.current
            .and_then(|id| self.edits.get(&id))
            .map(|e| (e.undo.iter().map(|u| u.0.clone()).collect(), e.redo.iter().map(|u| u.0.clone()).collect()))
            .unwrap_or_default()
    }

    /// Applies a change while a control is being dragged. The first call of a gesture
    /// records the state to undo to; [`Self::end_gesture`] commits it as one step.
    pub fn edit_live(&mut self, label: impl Into<SharedString>, f: impl FnOnce(&mut AdjustmentSet), cx: &mut Context<Self>) {
        let Some(id) = self.current else { return };
        let edit = self.edits.entry(id).or_default();
        if edit.gesture.is_none() {
            edit.gesture = Some((label.into(), edit.adj.clone()));
        }
        f(&mut edit.adj);
        self.request_render(RenderQuality::Draft, cx);
        cx.notify();
    }

    /// Ends the current gesture, if any, as a single undo step.
    #[track_caller]
    pub fn end_gesture(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.current else { return };
        let Some(edit) = self.edits.get_mut(&id) else { return };
        let Some((label, before)) = edit.gesture.take() else {
            // Panning has no undo step but still leaves a draft on screen.
            if self.preview.draft {
                self.request_render(RenderQuality::Final, cx);
            }
            return;
        };
        if crate::perf::enabled() {
            crate::perf::record(serde_json::json!({"event": "gesture_commit", "caller": std::panic::Location::caller().to_string()}));
        }
        if before != edit.adj {
            edit.undo.push((label, before));
            edit.redo.clear();
            if edit.undo.len() > self.settings.undo_levels.max(10) {
                edit.undo.remove(0);
            }
            self.after_commit(id, cx);
        }
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    /// A discrete change: one undo step, final-quality render.
    pub fn edit(&mut self, label: impl Into<SharedString>, f: impl FnOnce(&mut AdjustmentSet), cx: &mut Context<Self>) {
        self.end_gesture(cx);
        self.edit_live(label, f, cx);
        self.end_gesture(cx);
    }

    /// Applies one change to every selected image (Capture One's "Edit selected variants").
    pub fn edit_selected(&mut self, label: impl Into<SharedString>, f: impl Fn(&mut AdjustmentSet), cx: &mut Context<Self>) {
        let label = label.into();
        self.end_gesture(cx);
        let ids: Vec<ImageId> = self.selection.iter().copied().collect();
        let mut changed = Vec::new();
        for id in ids {
            let edit = self.edits.entry(id).or_default();
            let before = edit.adj.clone();
            f(&mut edit.adj);
            if before != edit.adj {
                edit.undo.push((label.clone(), before));
                edit.redo.clear();
                changed.push(id);
            }
        }
        self.after_commit_many(changed, cx);
        self.preview.before_dirty = true;
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    pub(crate) fn after_commit(&mut self, id: ImageId, cx: &mut Context<Self>) {
        self.after_commit_many(vec![id], cx);
    }

    fn after_commit_many(&mut self, ids: Vec<ImageId>, cx: &mut Context<Self>) {
        if ids.is_empty() { return; }
        let changes: Vec<_> = ids.iter().filter_map(|id| self.edits.get(id).map(|e| (*id, &e.adj))).collect();
        let saved = self.engine.save_adjustments_batch(&changes);
        let changed: HashMap<_, _> = changes.iter().map(|(id, adj)| (*id, !adj.is_default())).collect();
        if let Err(err) = saved {
            self.notify(format!("{}: {err}", t("Could not save adjustments", "无法存储调整")), cx);
        }
        for info in &mut self.images {
            if let Some(changed) = changed.get(&info.id) { info.has_adjustments = *changed; }
        }
        self.queue_thumbs(ids, cx);
    }

    pub fn undo(&mut self, cx: &mut Context<Self>) {
        self.end_gesture(cx);
        let Some(id) = self.current else { return };
        let Some(edit) = self.edits.get_mut(&id) else { return };
        if let Some((label, prev)) = edit.undo.pop() {
            let now = std::mem::replace(&mut edit.adj, prev);
            edit.redo.push((label.clone(), now));
            self.after_commit(id, cx);
            self.request_render(RenderQuality::Final, cx);
            self.notify(format!("{} {label}", t("Undo", "撤销")), cx);
        }
    }

    pub fn redo(&mut self, cx: &mut Context<Self>) {
        self.end_gesture(cx);
        let Some(id) = self.current else { return };
        let Some(edit) = self.edits.get_mut(&id) else { return };
        if let Some((label, next)) = edit.redo.pop() {
            let now = std::mem::replace(&mut edit.adj, next);
            edit.undo.push((label.clone(), now));
            self.after_commit(id, cx);
            self.request_render(RenderQuality::Final, cx);
            self.notify(format!("{} {label}", t("Redo", "重做")), cx);
        }
    }

    pub fn copy_adjustments(&mut self, cx: &mut Context<Self>) {
        self.clipboard = Some(self.adj().clone());
        self.notify(t("Adjustments copied", "已拷贝调整"), cx);
    }

    pub fn apply_adjustments(&mut self, cx: &mut Context<Self>) {
        let Some(clip) = self.clipboard.clone() else { return };
        let groups: Vec<ClipGroup> = self.clip_groups.iter().copied().collect();
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
        let Some(id) = self.current else { return };
        let Some(edit) = self.edits.get_mut(&id) else { return };
        while edit.undo.len() > undo_len {
            let Some((label, prev)) = edit.undo.pop() else { break };
            let now = std::mem::replace(&mut edit.adj, prev);
            edit.redo.push((label, now));
        }
        while edit.undo.len() < undo_len {
            let Some((label, next)) = edit.redo.pop() else { break };
            let now = std::mem::replace(&mut edit.adj, next);
            edit.undo.push((label, now));
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
        match self.settings.user_styles.iter_mut().find(|s| s.name == name) {
            Some(existing) => *existing = style,
            None => self.settings.user_styles.push(style),
        }
        self.settings.save();
        self.notify(format!("{} {name}", t("Saved style", "已存储风格")), cx);
    }

    /// Applies user style `index` to the selection: only its saved tools change.
    pub fn apply_user_style(&mut self, index: usize, cx: &mut Context<Self>) {
        let Some(style) = self.settings.user_styles.get(index).cloned() else { return };
        let name = style.name.clone();
        self.hover_look = None;
        self.edit_selected(format!("{} {name}", t("Style", "风格")), move |a| user_style_onto(a, &style), cx);
    }

    pub fn rename_user_style(&mut self, index: usize, name: String, cx: &mut Context<Self>) {
        if let Some(style) = self.settings.user_styles.get_mut(index) {
            style.name = name;
            self.settings.save();
            cx.notify();
        }
    }

    pub fn delete_user_style(&mut self, index: usize, cx: &mut Context<Self>) {
        if index < self.settings.user_styles.len() {
            let style = self.settings.user_styles.remove(index);
            self.settings.save();
            self.notify(format!("{} {}", t("Deleted style", "已删除风格"), style.name), cx);
        }
    }

    pub fn apply_style(&mut self, style: &StyleInfo, stack: bool, cx: &mut Context<Self>) {
        let style = style.clone();
        self.hover_look = None;
        self.edit_selected(format!("{} {}", t("Style", "风格"), style.name), move |a| style_onto(a, &style, stack), cx);
    }

    /// Previews a style on the current image while the pointer rests on it, as
    /// Lightroom does; `None` returns to the image's own adjustments. Nothing is
    /// committed or saved.
    pub fn preview_look(&mut self, look: Option<HoverLook>, cx: &mut Context<Self>) {
        let adj = look.map(|look| {
            let mut a = self.adj().clone();
            match look {
                HoverLook::Style(style, stack) => style_onto(&mut a, &style, stack),
                HoverLook::User(index) => {
                    if let Some(style) = self.settings.user_styles.get(index) {
                        user_style_onto(&mut a, style);
                    }
                }
            }
            a
        });
        if self.hover_look == adj {
            return;
        }
        self.hover_look = adj;
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    /// Whether tethered capture can be offered: the engine's tethering backend has
    /// a camera open (`tether_status`, not `camera_status`, per the engine contract).
    pub fn camera_ready(&self) -> bool {
        matches!(self.tether, Some(TetherStatus::Available { .. }))
    }

    /// Why capture is not offered right now, for disabled controls and notices.
    pub fn tether_note(&self) -> &'static str {
        match self.tether {
            None => t("Looking for a camera…", "正在检测相机…"),
            Some(TetherStatus::NoLibrary) => t("Tethering needs libgphoto2, which was not found (Linux only for now)", "联机拍摄需要 libgphoto2，本机未找到（目前仅支持 Linux）"),
            Some(TetherStatus::NoCamera) => t("No camera answered; connect one by USB and check again", "未检测到相机；请通过 USB 连接后重新检测"),
            Some(TetherStatus::Available { .. }) => t("Capture", "拍摄"),
        }
    }

    /// Asks the engine whether a camera can be opened. Opening one can take a few
    /// seconds, so it runs in the background.
    pub fn probe_tether(&mut self, cx: &mut Context<Self>) {
        if self.tether_probing {
            return;
        }
        self.tether_probing = true;
        cx.notify();
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let (tether, camera, settings) =
                cx.background_spawn(async move { (engine.tether_status(), engine.camera_status(), engine.camera_settings()) }).await;
            this.update(cx, |this, cx| {
                this.tether_probing = false;
                this.tether = Some(tether);
                this.camera = camera;
                this.camera_settings = settings;
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
                        this.edits.insert(id, Edit::default());
                        this.recent_captures.insert(id);
                        this.images.push(info);
                        this.queue_thumbs(vec![id], cx);
                        this.select(id, false, false, cx);
                        this.notify(t("Captured", "已拍摄"), cx);
                        // Capture One reviews each capture for closed eyes; it only reports.
                        this.run_eye_check(vec![id], false, cx);
                    }
                    Err(err) => this.notify(format!("{}: {err}", t("Capture failed", "拍摄失败")), cx),
                }
            })
            .ok();
        })
        .detach();
    }

    pub fn reset_all(&mut self, cx: &mut Context<Self>) {
        self.edit(t("Reset", "复位"), |a| *a = AdjustmentSet::default(), cx);
        self.active_layer = None;
        self.active_node = None;
    }

    pub fn auto_adjust(&mut self, target: AutoTarget, cx: &mut Context<Self>) {
        let Some(id) = self.current else { return };
        let engine = self.engine.clone();
        let current = self.adj().clone();
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async move { engine.auto_adjust(id, &current, target) }).await;
            this.update(cx, |this, cx| match result {
                Ok(r) => {
                    if this.current == Some(id) {
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
        self.active_layer = Some(id);
        self.active_node = None;
        cx.notify();
        id
    }

    /// Turns a sampled color into a color editor selection: the nearest basic
    /// sector, or a new advanced / skin range centered on its hue.
    pub fn range_from_sample(&mut self, rgb: [f32; 3], cx: &mut Context<Self>) {
        self.panels.pick_to_range = false;
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
        let mode = self.panels.color_editor_mode;
        if mode == 0 {
            self.panels.selected_sector = HueSector::nearest(hue) as usize;
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
            self.panels.selected_color_range = Some(id);
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

    pub fn compute_ai_mask(&mut self, kind: AiMaskKind, cx: &mut Context<Self>) {
        let Some(image) = self.current else { return };
        let engine = self.engine.clone();
        self.ai_busy = true;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async move { engine.compute_ai_mask(image, kind) }).await;
            this.update(cx, |this, cx| {
                this.ai_busy = false;
                match result {
                    Ok(key) => {
                        this.ai_unavailable.remove(&kind);
                        let shape = MaskShape::Ai(AiMask { op: MaskOp::Add, kind, cache_key: Some(key) });
                        match this.scope() {
                            Scope::Layer(id) => this.edit_layer(t("AI mask", "AI 蒙版"), id, false, |l| nodes::add_shape(&mut l.mask, shape), cx),
                            Scope::Global => {
                                let mask = Mask { shapes: vec![shape], ..Default::default() };
                                this.add_layer(LayerKind::Adjustment, Some(format!("AI · {kind:?}")), Some(mask), cx);
                            }
                        }
                    }
                    Err(err) => {
                        this.ai_unavailable.insert(kind);
                        this.notify(format!("{}: {err}", t("AI mask unavailable", "AI 蒙版不可用")), cx);
                    }
                }
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
        if tool == CursorTool::Crop || was_crop {
            // The crop tool shows the uncropped frame with an overlay.
            self.zoom = ViewZoom::default();
            self.request_render(RenderQuality::Final, cx);
        }
        if tool.is_mask_tool() && self.scope() == Scope::Global {
            self.add_layer(LayerKind::Adjustment, None, None, cx);
        }
        cx.notify();
    }

    pub fn set_zoom(&mut self, scale: Option<f32>, center: Option<(f32, f32)>, cx: &mut Context<Self>) {
        self.zoom.scale = scale.map(|s| s.clamp(0.05, 8.0));
        if let Some(c) = center {
            self.zoom.center = c;
        }
        if scale.is_none() {
            self.zoom.center = (0.5, 0.5);
        }
        self.clamp_center();
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    pub fn pan_by(&mut self, dx: f32, dy: f32, cx: &mut Context<Self>) {
        if self.zoom.scale.is_none() {
            return;
        }
        self.zoom.center.0 += dx;
        self.zoom.center.1 += dy;
        self.clamp_center();
        self.request_render(RenderQuality::Draft, cx);
        cx.notify();
    }

    pub fn set_viewport(&mut self, size: (Pixels, Pixels), scale_factor: f32, cx: &mut Context<Self>) {
        let size = (f32::from(size.0), f32::from(size.1));
        if (size.0 - self.viewport.0).abs() > 1.0 || (size.1 - self.viewport.1).abs() > 1.0 || scale_factor != self.scale_factor {
            self.viewport = size;
            self.scale_factor = scale_factor;
            self.preview.before_dirty = true;
            self.request_render(RenderQuality::Final, cx);
        }
    }

    pub fn toggle_panel(&mut self, id: &'static str, cx: &mut Context<Self>) {
        if !self.panels.collapsed.remove(id) {
            self.panels.collapsed.insert(id);
        }
        cx.notify();
    }

    /// Opens `id` and folds every other panel; when it is already the only
    /// open one, opens them all again (Alt-click on a panel header).
    pub fn solo_panel(&mut self, id: &'static str, cx: &mut Context<Self>) {
        let others = || crate::tools::QUICK_PANELS.iter().copied().filter(|p| *p != id);
        let alone = !self.panels.collapsed.contains(id) && others().all(|p| self.panels.collapsed.contains(p));
        if alone {
            self.set_all_panels(true, cx);
        } else {
            self.panels.collapsed.extend(others());
            self.panels.collapsed.remove(id);
            cx.notify();
        }
    }

    /// Opens or folds every adjustment panel at once.
    pub fn set_all_panels(&mut self, open: bool, cx: &mut Context<Self>) {
        for id in crate::tools::QUICK_PANELS {
            if open {
                self.panels.collapsed.remove(id);
            } else {
                self.panels.collapsed.insert(id);
            }
        }
        cx.notify();
    }

    pub fn filter_active(&self) -> bool {
        self.filter != Filter::default()
    }

    pub fn clear_filter(&mut self, cx: &mut Context<Self>) {
        self.filter = Filter::default();
        cx.notify();
    }

    // ───────────── Output ─────────────

    pub fn process_selection(&mut self, cx: &mut Context<Self>) {
        let recipes: Vec<ProcessRecipe> = self.recipes.iter().filter(|r| r.enabled).cloned().collect();
        self.process(recipes, cx);
    }

    /// Queues every selected image through each of `recipes`.
    pub fn process(&mut self, recipes: Vec<ProcessRecipe>, cx: &mut Context<Self>) {
        let images: Vec<ImageId> = self.selection.iter().copied().collect();
        if recipes.is_empty() {
            self.notify(t("Enable at least one process recipe", "请至少启用一个处理配方"), cx);
            return;
        }
        for image in images {
            let name = self.info(image).map(|i| i.file_name.clone()).unwrap_or_default();
            for recipe in &recipes {
                let id = self.next_job;
                self.next_job += 1;
                let job = ExportJob { job_id: id, image, recipe: recipe.clone() };
                if let Err(err) = self.engine.start_export(job) {
                    self.notify(err.to_string(), cx);
                    continue;
                }
                self.jobs.push(JobView { id, name: name.clone(), recipe: recipe.name.clone(), status: JobStatus::Queued });
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
                        for job in &mut this.jobs {
                            if matches!(job.status, JobStatus::Queued | JobStatus::Running { .. }) {
                                job.status = this.engine.job_status(job.id);
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
        let layout = self.current_layout(self.settings.workspace.name.clone());
        let st = &mut self.settings;
        st.workspace = layout;
        st.viewer_gray = self.view.viewer_gray;
        st.thumb_size = self.view.thumb_size;
        st.info_overlay = self.view.show_info_overlay;
        st.chinese = crate::i18n::locale() == crate::i18n::Locale::ZhCn;
        st.motion_speed = crate::motion::speed();
        st.save();
    }

    pub fn current_layout(&self, name: String) -> WorkspaceLayout {
        WorkspaceLayout {
            name,
            inspector_right: self.view.inspector_right,
            show_inspector: self.view.show_inspector,
            browser: browser_name(self.view.browser).into(),
            hidden_tabs: self.settings.workspace.hidden_tabs.clone(),
            tab_order: self.settings.workspace.tab_order.clone(),
            quick_tools: self.settings.workspace.quick_tools.clone(),
            tab: tab_id(self.tool_tab),
            inspector_width: self.view.inspector_width,
            filmstrip_height: self.view.filmstrip_height,
            multi: self.view.multi,
        }
    }

    fn load_layout(&mut self, l: &WorkspaceLayout) {
        self.view.inspector_right = l.inspector_right;
        self.view.show_inspector = l.show_inspector;
        self.view.browser = browser_from(&l.browser);
        self.view.inspector_width = l.inspector_width.clamp(260.0, 640.0);
        self.view.filmstrip_height = l.filmstrip_height.clamp(96.0, 420.0);
        self.view.multi = l.multi;
        self.settings.workspace = l.clone();
        if let Some(tab) = tab_from(&l.tab) {
            self.tool_tab = tab;
        }
        if self.tab_hidden(self.tool_tab) {
            self.tool_tab = self.visible_tabs().first().copied().unwrap_or(ToolTab::Exposure);
        }
    }

    /// Built-in workspaces followed by the user's saved ones.
    pub fn workspace_list(&self) -> Vec<WorkspaceLayout> {
        let mut all = builtin_workspaces();
        all.extend(self.settings.workspaces.iter().cloned());
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
        self.settings.workspaces.retain(|w| w.name != name);
        self.settings.workspaces.push(layout.clone());
        self.settings.workspace = layout;
        self.save_settings();
        self.notify(format!("{} · {name}", t("Workspace saved", "工作区已保存")), cx);
    }

    pub fn delete_workspace(&mut self, name: &str, cx: &mut Context<Self>) {
        self.settings.workspaces.retain(|w| w.name != name);
        self.save_settings();
        cx.notify();
    }

    pub fn tab_hidden(&self, tab: ToolTab) -> bool {
        self.settings.workspace.hidden_tabs.iter().any(|h| *h == tab_id(tab))
    }

    /// The shown tabs in the workspace's order; tabs the order does not name
    /// yet keep their default place.
    pub fn visible_tabs(&self) -> Vec<ToolTab> {
        self.tab_order().into_iter().filter(|t| !self.tab_hidden(*t)).collect()
    }

    pub fn tab_order(&self) -> Vec<ToolTab> {
        let mut order: Vec<ToolTab> = self.settings.workspace.tab_order.iter().filter_map(|id| tab_from(id)).collect();
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
        self.settings.workspace.tab_order = order.into_iter().map(tab_id).collect();
        self.save_settings();
        cx.notify();
    }

    pub fn in_quick(&self, panel: &str) -> bool {
        self.settings.workspace.quick_tools.iter().any(|p| p == panel)
    }

    /// Adds a tool panel to the Quick tab, or takes it off.
    pub fn toggle_quick(&mut self, panel: &str, cx: &mut Context<Self>) {
        let tools = &mut self.settings.workspace.quick_tools;
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

    /// Moves a panel up (`-1`) or down (`1`) within the Quick tab.
    pub fn move_quick(&mut self, panel: &str, delta: isize, cx: &mut Context<Self>) {
        let tools = &mut self.settings.workspace.quick_tools;
        let Some(i) = tools.iter().position(|p| p == panel) else { return };
        let j = i as isize + delta;
        if (0..tools.len() as isize).contains(&j) {
            tools.swap(i, j as usize);
            self.save_settings();
            cx.notify();
        }
    }

    pub fn set_tab_hidden(&mut self, tab: ToolTab, hidden: bool, cx: &mut Context<Self>) {
        let id = tab_id(tab);
        self.settings.workspace.hidden_tabs.retain(|h| *h != id);
        if hidden && self.visible_tabs().len() > 1 {
            self.settings.workspace.hidden_tabs.push(id);
            if self.tool_tab == tab {
                self.tool_tab = self.visible_tabs().first().copied().unwrap_or(ToolTab::Exposure);
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

fn user_style_onto(a: &mut AdjustmentSet, style: &UserStyle) {
    for g in style.groups.iter().filter_map(|g| ClipGroup::from_id(g)) {
        g.copy(a, &style.adjustments);
    }
    a.style = Some(style.name.clone());
}
