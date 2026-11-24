//! Color Grading tab: combines Histogram, Scopes, Layers (Nodes + Masks),
//! Exposure, and Color tools into one unified Capture One-style grading workspace.

use gpui_kit::{AnyElement, App, Entity, IntoElement, ParentElement, Styled, Window, div, prelude::FluentBuilder as _};

use crate::state::{AppState, Scope};

use super::{color, exposure, layers};

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let scope = state.read(cx).scope();
    div()
        .w_full()
        .child(exposure::histogram_panel(state, cx))
        .child(exposure::scopes_panel(state, cx))
        // Nodes & Masks act as C1-style adjustment layers inside the grading tab.
        .child(layers::render(state, window, cx))
        // Exposure & Tone
        .child(exposure::exposure_panel(state, scope, cx))
        .child(exposure::hdr_panel(state, scope, cx))
        .child(exposure::levels_panel(state, scope, window, cx))
        .child(exposure::curve_panel(state, scope, cx))
        .child(exposure::clarity_panel(state, scope, cx))
        .child(exposure::dehaze_panel(state, scope, cx))
        .when(scope == Scope::Global, |d| d.child(exposure::vignetting_panel(state, cx)))
        // Color
        .when(scope == Scope::Global, |d| d.child(color::base_panel(state, cx)))
        .child(color::white_balance_panel(state, scope, cx))
        .child(color::color_balance_panel(state, scope, cx))
        .child(color::color_editor_panel(state, scope, cx))
        .when(scope == Scope::Global, |d| d.child(color::black_white_panel(state, cx)))
        .child(color::readout_panel(state, cx))
        .into_any_element()
}
