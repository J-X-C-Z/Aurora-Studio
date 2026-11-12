//! Saving the window arrangement as a named workspace.

use gpui_kit::component::{
    WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{App, Entity, ParentElement, Styled, Window, div, px};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::pal;
use crate::widgets::text_field;

pub fn save_dialog(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let state = state.clone();
    let name = std::rc::Rc::new(std::cell::RefCell::new(String::new()));
    window.open_dialog(cx, move |dialog, window, cx| {
        let p = pal(cx).clone();
        let (st, n1, n2) = (state.clone(), name.clone(), name.clone());
        let save = move |window: &mut Window, cx: &mut App| {
            let name = n2.borrow().trim().to_string();
            if name.is_empty() {
                return;
            }
            st.update(cx, |s, cx| s.save_workspace_as(name, cx));
            window.close_dialog(cx);
        };
        let field = text_field("ws-name", "", t("Workspace name", "工作区名称"), move |v, enter, _| {
            *n1.borrow_mut() = v;
            let _ = enter;
        }, window, cx);
        dialog
            .title(t("Save Workspace", "存储工作区"))
            .w(px(400.0))
            .child(
                v_flex()
                    .gap_3()
                    .child(div().text_xs().text_color(p.text_muted).child(t(
                        "Saves where the tools are, which tabs and browser show, and their sizes.",
                        "存储工具栏位置、显示的页签与浏览器，以及它们的尺寸。",
                    )))
                    .child(field)
                    .child(
                        h_flex()
                            .justify_end()
                            .gap_2()
                            .child(Button::new("ws-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                            .child(Button::new("ws-save").primary().label(t("Save", "存储")).on_click(move |_, window, cx| save(window, cx))),
                    ),
            )
    });
}
