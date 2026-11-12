//! One module per tool tab. Each exposes `render(state, window, cx)`.

pub mod adjustments;
pub mod capture;
pub mod color;
pub mod details;
pub mod exposure;
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
        ToolTab::Exposure => exposure::render(state, window, cx),
        ToolTab::Color => color::render(state, window, cx),
        ToolTab::Lens => lens::render(state, window, cx),
        ToolTab::Details => details::render(state, window, cx),
        ToolTab::Layers => layers::render(state, window, cx),
        ToolTab::Portrait => portrait::render(state, window, cx),
        ToolTab::Library => library::render(state, window, cx),
        ToolTab::Capture => capture::render(state, window, cx),
        ToolTab::Adjustments => adjustments::render(state, window, cx),
        ToolTab::Metadata => metadata::render(state, window, cx),
        ToolTab::Output => output::render(state, window, cx),
        ToolTab::Batch => output::render_batch(state, window, cx),
        ToolTab::Quick => quick::render(state, window, cx),
    }
}

/// Tool panels that can go on the Quick tab, by the id `ToolPanel` uses.
pub const QUICK_PANELS: [&str; 26] = [
    "histogram", "scopes", "exposure", "hdr", "levels", "curve", "clarity", "dehaze", "vignetting",
    "base", "wb", "color-balance", "color-editor", "bw", "readout",
    "lens", "rotation", "crop", "keystone",
    "navigator", "sharpening", "noise", "moire", "grain", "lens-blur", "spots",
];

/// Renders one panel by id wherever it lives; `None` for an unknown id or a
/// panel the current scope does not have (a global-only tool while a node is selected).
pub fn render_panel(id: &str, state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> Option<AnyElement> {
    exposure::panel(id, state, window, cx)
        .or_else(|| color::panel(id, state, window, cx))
        .or_else(|| lens::panel(id, state, window, cx))
        .or_else(|| details::panel(id, state, window, cx))
}
