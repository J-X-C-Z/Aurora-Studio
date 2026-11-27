//! Color Grading tab: combines Histogram, Scopes, Layers (Nodes + Masks),
//! Exposure, and Color tools into one unified Capture One-style grading workspace.
//! Every panel can be dragged into a new place or pinned to the top.

use gpui_kit::{AnyElement, App, Entity, IntoElement, Window};

use crate::state::{AppState, Scope};
use crate::widgets::PanelStack;

use super::{color, exposure, layers};

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let scope = state.read(cx).scope();
    let global = scope == Scope::Global;
    let layer = state.read(cx).active_layer_ref().cloned();
    let mut stack = PanelStack::new(state)
        .panel("histogram", exposure::histogram_panel(state, cx))
        .panel("scopes", exposure::scopes_panel(state, cx))
        // Nodes & Masks act as C1-style adjustment layers inside the grading tab.
        .panel("nodes", layers::node_panel(state, window, cx));
    if let Some(l) = &layer {
        stack = stack
            .panel("mask", layers::mask_panel(state, l, cx))
            .panel("luma", layers::luma_panel(state, l, cx))
            .panel("mask-color", layers::color_range_panel(state, l, cx))
            .panel("brush", layers::brush_panel(state, cx));
    }
    // Capture One's Adjust order: color balance of the shot first (base
    // characteristics, white balance), then tone, then color, then local contrast.
    stack
        .panel_when(global, "base", || color::base_panel(state, cx).into_any_element())
        .panel("wb", color::white_balance_panel(state, scope, cx))
        .panel("exposure", exposure::exposure_panel(state, scope, cx))
        .panel("hdr", exposure::hdr_panel(state, scope, cx))
        .panel("levels", exposure::levels_panel(state, scope, window, cx))
        .panel("curve", exposure::curve_panel(state, scope, cx))
        .panel("color-editor", color::color_editor_panel(state, scope, cx))
        .panel("color-balance", color::color_balance_panel(state, scope, cx))
        .panel_when(global, "bw", || color::black_white_panel(state, cx).into_any_element())
        .panel("clarity", exposure::clarity_panel(state, scope, cx))
        .panel("dehaze", exposure::dehaze_panel(state, scope, cx))
        .panel_when(global, "vignetting", || exposure::vignetting_panel(state, cx).into_any_element())
        .panel("readout", color::readout_panel(state, cx))
        .render(cx)
}
