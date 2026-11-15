//! Modal dialogs: preferences, about and the command palette.

use gpui_kit::component::{WindowExt as _, v_flex};
use gpui_kit::{App, Entity, ParentElement, Styled, Window, px};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::pal;

pub fn about(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let engine = state.read(cx).engine_name.clone();
    window.open_dialog(cx, move |dialog, _, cx| {
        let p = pal(cx).clone();
        dialog.title("Aurora Studio").w(px(420.0)).child(
            v_flex()
                .gap_2()
                .text_sm()
                .text_color(p.text_muted)
                .child(format!("{} 0.1.0", t("Version", "版本")))
                .child(format!("{}: {engine}", t("Engine", "引擎")))
                .child(t("Interface built with GPUI, GPUI Kit and Ely.", "界面基于 GPUI、GPUI Kit 与 Ely 构建。")),
        )
    });
}

pub fn preferences(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    super::prefs::open(state, window, cx);
}

pub fn command_palette(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    super::palette::open(state, window, cx);
}
