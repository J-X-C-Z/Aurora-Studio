//! Small dialogs that ask for a name: renaming a style or a preset, saving a
//! user style with the tools it should include.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::rc::Rc;

use gpui_kit::component::{
    Disableable as _, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    checkbox::Checkbox,
    h_flex,
    input::Input,
    v_flex,
};
use gpui_kit::{AnyElement, App, Entity, Focusable as _, IntoElement, ParentElement, SharedString, Styled, Window, div, px};

use crate::i18n::t;
use crate::state::{AppState, ClipGroup};
use crate::theme::pal;
use crate::widgets::field_state;

thread_local! {
    /// Each dialog gets fresh field state, so it never shows a previous entry.
    static SERIAL: Cell<u64> = const { Cell::new(0) };
}

fn next_key(base: &str) -> SharedString {
    let n = SERIAL.with(|s| {
        s.set(s.get() + 1);
        s.get()
    });
    format!("{base}-{n}").into()
}

type Submit = Rc<dyn Fn(&mut Window, &mut App)>;

/// The name field: focused when the dialog opens, Enter submits.
fn name_field(
    key: SharedString,
    initial: String,
    placeholder: &'static str,
    name: Rc<RefCell<String>>,
    submit: Submit,
    focused: &Cell<bool>,
    window: &mut Window,
    cx: &mut App,
) -> AnyElement {
    let handle = window.window_handle();
    let input = field_state(
        key,
        initial,
        placeholder,
        move |v, enter, cx| {
            *name.borrow_mut() = v;
            if enter {
                let submit = submit.clone();
                cx.defer(move |cx| {
                    let _ = handle.update(cx, |_, window, cx| submit(window, cx));
                });
            }
        },
        window,
        cx,
    );
    if !focused.replace(true) {
        let focus = input.read(cx).focus_handle(cx);
        window.focus(&focus, cx);
    }
    Input::new(&input).xsmall().into_any_element()
}

/// Asks for a name. `on_save` runs with the trimmed, non-empty name.
pub fn prompt(
    title: &'static str,
    hint: &'static str,
    initial: String,
    placeholder: &'static str,
    window: &mut Window,
    cx: &mut App,
    on_save: impl Fn(String, &mut App) + 'static,
) {
    let key = next_key("name-prompt");
    let name = Rc::new(RefCell::new(initial.clone()));
    let n2 = name.clone();
    let save: Submit = Rc::new(move |window, cx| {
        let name = n2.borrow().trim().to_string();
        if !name.is_empty() {
            on_save(name, cx);
            window.close_dialog(cx);
        }
    });
    let focused = Cell::new(false);
    window.open_dialog(cx, move |dialog, window, cx| {
        let p = pal(cx).clone();
        let field = name_field(key.clone(), initial.clone(), placeholder, name.clone(), save.clone(), &focused, window, cx);
        let save = save.clone();
        dialog.title(title).w(px(400.0)).child(
            v_flex()
                .gap_3()
                .when_hint(hint, p.text_muted)
                .child(field)
                .child(
                    h_flex()
                        .justify_end()
                        .gap_2()
                        .child(Button::new("name-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                        .child(Button::new("name-save").primary().label(t("Save", "存储")).on_click(move |_, window, cx| save(window, cx))),
                ),
        )
    });
}

trait Hint {
    fn when_hint(self, hint: &'static str, color: gpui_kit::Hsla) -> Self;
}

impl Hint for gpui_kit::Div {
    fn when_hint(self, hint: &'static str, color: gpui_kit::Hsla) -> Self {
        if hint.is_empty() { self } else { self.child(div().text_xs().text_color(color).child(hint)) }
    }
}

/// Saves the current image's adjustments as a user style. The tools that differ
/// from their defaults are ticked to start with; composition tools never are.
pub fn save_style(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let key = next_key("style-name");
    let (preset, count) = {
        let s = state.read(cx);
        let adj = s.adj();
        let set: BTreeSet<ClipGroup> =
            ClipGroup::ALL.into_iter().filter(|g| !g.is_composition() && *g != ClipGroup::Style && *g != ClipGroup::Layers && g.is_set(adj)).collect();
        (set, s.settings.user_styles.len())
    };
    let groups = Rc::new(RefCell::new(preset));
    let name = Rc::new(RefCell::new(format!("{} {}", t("My Style", "我的风格"), count + 1)));
    let initial = name.borrow().clone();
    let state = state.clone();
    let save: Submit = {
        let (st, n2, g2) = (state.clone(), name.clone(), groups.clone());
        Rc::new(move |window, cx| {
            let name = n2.borrow().trim().to_string();
            let groups: Vec<ClipGroup> = g2.borrow().iter().copied().collect();
            if name.is_empty() || groups.is_empty() {
                return;
            }
            st.update(cx, |s, cx| s.save_user_style(name, groups, cx));
            window.close_dialog(cx);
        })
    };
    let focused = Cell::new(false);
    window.open_dialog(cx, move |dialog, window, cx| {
        let p = pal(cx).clone();
        let field = name_field(key.clone(), initial.clone(), t("Style name", "风格名称"), name.clone(), save.clone(), &focused, window, cx);
        let chosen = groups.borrow().clone();
        let adj_groups: Vec<ClipGroup> = ClipGroup::ALL.into_iter().filter(|g| *g != ClipGroup::Style).collect();
        let checks = div().flex().flex_wrap().gap_x_3().gap_y_1().children(adj_groups.into_iter().map(|g| {
            let groups = groups.clone();
            div().w(px(150.0)).child(
                Checkbox::new(SharedString::from(format!("style-g-{g:?}")))
                    .xsmall()
                    .checked(chosen.contains(&g))
                    .label(g.label())
                    .on_change(move |v: &bool, window, _| {
                        if *v {
                            groups.borrow_mut().insert(g);
                        } else {
                            groups.borrow_mut().remove(&g);
                        }
                        window.refresh();
                    }),
            )
        }));
        let save = save.clone();
        let empty = chosen.is_empty();
        dialog.title(t("Save User Style", "存储用户风格")).w(px(520.0)).child(
            v_flex()
                .gap_3()
                .child(field)
                .child(div().text_xs().text_color(p.text_muted).child(t(
                    "Include these tools. Applying the style changes only them.",
                    "包含以下工具。套用风格时只改变这些工具。",
                )))
                .child(checks)
                .child(
                    h_flex()
                        .gap_2()
                        .child(div().flex_1().text_xs().text_color(p.text_subtle).child(format!("{} {}", chosen.len(), t("tools", "个工具"))))
                        .child(Button::new("style-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                        .child(Button::new("style-save").primary().label(t("Save Style", "存储风格")).disabled(empty).on_click(move |_, window, cx| save(window, cx))),
                ),
        )
    });
}
