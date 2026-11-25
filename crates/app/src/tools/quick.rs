//! Quick tab: the tool panels the user picked from the other tabs, in their
//! order. A panel is added or removed from its header's right-click menu and
//! dragged by its header into a new place.

use gpui_kit::component::v_flex;
use gpui_kit::{AnyElement, App, Entity, IntoElement, ParentElement, Styled, Window, div};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::pal;
use crate::widgets::PanelStack;

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let p = pal(cx).clone();
    let ids = state.read(cx).prefs.settings.workspace.quick_tools.clone();
    let panels: Vec<(&'static str, AnyElement)> = ids
        .iter()
        .filter_map(|id| super::QUICK_PANELS.iter().find(|q| **q == id.as_str()).copied())
        .filter_map(|id| super::render_panel(id, state, window, cx).map(|el| (id, el)))
        .collect();
    if panels.is_empty() {
        return v_flex()
            .w_full()
            .p_4()
            .gap_1()
            .text_xs()
            .text_color(p.text_subtle)
            .child(div().text_color(p.text_muted).child(t("No tools here yet", "这里还没有工具")))
            .child(t(
                "Right-click the title of any tool panel and choose Add to Quick Tab.",
                "在任意工具面板的标题上右键，选择“添加到快速页签”。",
            ))
            .into_any_element();
    }
    panels.into_iter().fold(PanelStack::new(state), |stack, (id, el)| stack.panel(id, el)).render(cx)
}
