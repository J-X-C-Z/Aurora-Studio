//! Saving the window arrangement as a named workspace.

use gpui_kit::component::{
    Sizable as _, WindowExt as _,
    input::{Input, SelectAll},
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{App, Entity, Focusable as _, ParentElement, Styled, Window, div, px};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::pal;
use crate::widgets::text_field;
use crate::widgets::field_state;

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

/// Names a new tool tab of the user's own, holding `panels` to start with.
pub fn new_tab_dialog(state: &Entity<AppState>, panels: Vec<String>, window: &mut Window, cx: &mut App) {
    let n = state.read(cx).prefs.settings.workspace.user_tabs.len() + 1;
    let initial = format!("{} {n}", t("Custom", "自定义"));
    let st = state.clone();
    name_dialog(t("New Tab", "新建页签"), initial, t("Create", "创建"), window, cx, move |name, cx| {
        st.update(cx, |s, cx| {
            s.new_user_tab(name, panels.clone(), cx);
        })
    });
}

/// Renames a tool tab of the user's own.
pub fn rename_tab_dialog(state: &Entity<AppState>, key: u8, window: &mut Window, cx: &mut App) {
    let Some(initial) = state.read(cx).user_tab(key).map(|u| u.name.clone()) else { return };
    let st = state.clone();
    name_dialog(t("Rename Tab", "重命名页签"), initial, t("Rename", "重命名"), window, cx, move |name, cx| {
        st.update(cx, |s, cx| s.rename_user_tab(key, name, cx))
    });
}

fn name_dialog(
    title: &'static str,
    initial: String,
    confirm: &'static str,
    window: &mut Window,
    cx: &mut App,
    done: impl Fn(String, &mut App) + 'static,
) {
    let name = std::rc::Rc::new(std::cell::RefCell::new(initial.clone()));
    let done = std::rc::Rc::new(done);
    let first = std::rc::Rc::new(std::cell::Cell::new(true));
    let handle = window.window_handle();
    window.open_dialog(cx, move |dialog, window, cx| {
        let (typed, read, done) = (name.clone(), name.clone(), done.clone());
        let submit = std::rc::Rc::new(move |window: &mut Window, cx: &mut App| {
            let name = read.borrow().trim().to_string();
            if !name.is_empty() {
                done(name, cx);
                window.close_dialog(cx);
            }
        });
        let on_enter = submit.clone();
        let current = name.borrow().clone();
        // Enter confirms; the window is reached after the input's event is handled.
        let input = field_state("tab-name", current, t("Tab name", "页签名称"), move |v, enter, cx| {
            *typed.borrow_mut() = v;
            if enter {
                let on_enter = on_enter.clone();
                cx.defer(move |cx| {
                    let _ = handle.update(cx, |_, window, cx| on_enter(window, cx));
                });
            }
        }, window, cx);
        // Ready to type over the suggested name as soon as the dialog opens.
        if first.replace(false) {
            let input = input.clone();
            window.defer(cx, move |window, cx| {
                input.read(cx).focus_handle(cx).focus(window, cx);
                window.dispatch_action(Box::new(SelectAll), cx);
            });
        }
        dialog
            .title(title)
            .w(px(360.0))
            .child(
                v_flex()
                    .gap_3()
                    .child(Input::new(&input).xsmall())
                    .child(
                        h_flex()
                            .justify_end()
                            .gap_2()
                            .child(Button::new("tab-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                            .child(Button::new("tab-ok").primary().label(confirm).on_click(move |_, window, cx| submit(window, cx))),
                    ),
            )
    });
}
