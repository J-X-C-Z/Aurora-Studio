//! One module per tool tab. Each exposes `render(state, window, cx)`.

pub mod adjustments;
pub mod capture;
pub mod color;
pub mod details;
pub mod exposure;
pub mod grading;
pub mod layers;
pub mod lens;
pub mod library;
pub mod metadata;
pub mod nodes;
pub mod output;
pub mod portrait;
pub mod quick;

use gpui_kit::{AnyElement, App, Entity, Window};

use crate::state::{AppState, ToolTab};

pub fn render_tab(tab: ToolTab, state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    match tab {
        ToolTab::Grading => grading::render(state, window, cx),
        ToolTab::Lens => lens::render(state, window, cx),
        ToolTab::Details => details::render(state, window, cx),
        ToolTab::Portrait => portrait::render(state, window, cx),
        ToolTab::Library => library::render(state, window, cx),
        ToolTab::Capture => capture::render(state, window, cx),
        ToolTab::Adjustments => adjustments::render(state, window, cx),
        ToolTab::Metadata => metadata::render(state, window, cx),
        ToolTab::Output => output::render(state, window, cx),
        ToolTab::Batch => output::render_batch(state, window, cx),
        ToolTab::Quick | ToolTab::User(_) => quick::render(tab, state, window, cx),
    }
}

/// Tool panels that can go on the Quick tab, by the id `ToolPanel` uses.
pub const QUICK_PANELS: [&str; 26] = [
    "histogram", "scopes", "exposure", "hdr", "levels", "curve", "clarity", "dehaze", "vignetting",
    "base", "wb", "color-balance", "color-editor", "bw", "readout",
    "lens", "rotation", "crop", "keystone",
    "navigator", "sharpening", "noise", "moire", "grain", "lens-blur", "spots",
];

/// Panels from the other tabs that can also go on the Quick and user tabs.
/// Nodes, masks and the portrait panels depend on their own tab's context and stay there.
pub const EXTRA_TAB_PANELS: [&str; 20] = [
    "collections", "filters", "catalog",
    "camera", "camera-settings", "capture-naming", "capture-adjust",
    "user-styles", "styles", "clipboard", "history",
    "metadata-presets", "rating", "file", "exif", "iptc", "keywords",
    "recipes", "recipe", "summary",
];

/// The static id of a panel that can go on the Quick and user tabs.
pub fn tab_panel(id: &str) -> Option<&'static str> {
    QUICK_PANELS.iter().chain(EXTRA_TAB_PANELS.iter()).find(|p| **p == id).copied()
}

/// Renders one panel by id wherever it lives; `None` for an unknown id or a
/// panel the current scope does not have (a global-only tool while a node is selected).
pub fn render_panel(id: &str, state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> Option<AnyElement> {
    exposure::panel(id, state, window, cx)
        .or_else(|| color::panel(id, state, window, cx))
        .or_else(|| lens::panel(id, state, window, cx))
        .or_else(|| details::panel(id, state, window, cx))
        .or_else(|| library::panel(id, state, window, cx))
        .or_else(|| capture::panel(id, state, window, cx))
        .or_else(|| adjustments::panel(id, state, window, cx))
        .or_else(|| metadata::panel(id, state, window, cx))
        .or_else(|| output::panel(id, state, window, cx))
}
