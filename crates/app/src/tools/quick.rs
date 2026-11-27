//! Quick tab and the user's own tabs: the tool panels the user picked from the
//! other tabs, in their order. A panel is added or removed from its header's
//! right-click menu or dropped on the tab's icon, and dragged by its header
//! into a new place.

use gpui_kit::component::v_flex;
use gpui_kit::{AnyElement, App, Entity, IntoElement, ParentElement, Styled, Window, div};

use crate::i18n::t;
use crate::state::{AppState, ToolTab};
use crate::theme::pal;
use crate::widgets::PanelStack;

pub fn render(tab: ToolTab, state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let p = pal(cx).clone();
    let ids = state.read(cx).tab_panels(tab);
    let panels: Vec<(&'static str, AnyElement)> = ids
        .iter()
        .filter_map(|id| super::tab_panel(id))
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
                "Drag a tool panel by its title onto this tab's icon, or right-click the title and choose Add to Tab.",
                "把任意工具面板的标题拖到这个页签的图标上，或在标题上右键选择“添加到页签”。",
            ))
            .into_any_element();
    }
    panels.into_iter().fold(PanelStack::new(state), |stack, (id, el)| stack.panel(id, el)).render(cx)
}
