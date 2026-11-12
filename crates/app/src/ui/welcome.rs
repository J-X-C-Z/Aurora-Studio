//! The welcome screen: what Aurora shows when started without a folder or
//! catalog. It offers to open one, lists the recent ones, and only shows the
//! procedural demo images when asked to, labelled as such.
//!
//! Opening from here replaces this window in place: the engine scans off the
//! main thread while the screen shows progress, then a window for the opened
//! catalog takes over.

use std::path::PathBuf;

use gpui_kit::assets::IconName;
use gpui_kit::component::{Disableable as _, Icon, Sizable as _, button::{Button, ButtonVariants as _}, h_flex, spinner::Spinner, v_flex};
use gpui_kit::{
    AnyElement, AnyWindowHandle, App, AppContext as _, Entity, InteractiveElement, IntoElement, ParentElement, PathPromptOptions, SharedString, StatefulInteractiveElement,
    Styled, Window, div, prelude::FluentBuilder as _, px,
};

use crate::contract;
use crate::i18n::t;
use crate::motion;
use crate::state::{AppState, open as source};
use crate::theme::{hexa, pal};
use crate::widgets::switch_row;

/// Opens `path` in place of the window `replace`; on failure the welcome
/// screen of `state` shows why.
pub fn open_path(state: Entity<AppState>, path: PathBuf, replace: AnyWindowHandle, cx: &mut App) {
    state.update(cx, |s, cx| {
        s.opening = Some(path.clone());
        s.open_error = None;
        cx.notify();
    });
    cx.spawn(async move |cx| {
        let target = path.clone();
        let result = cx.background_spawn(async move { source::open(&target) }).await;
        cx.update(|cx| match result {
            Ok((engine, source)) => {
                let next = crate::state::init_state(engine, source, cx);
                super::workspace::open(next, cx);
                let _ = replace.update(cx, |_, window, _| window.remove_window());
            }
            Err(err) => state.update(cx, |s, cx| {
                s.opening = None;
                s.welcome = true;
                s.open_error = Some(format!("{}: {err}", path.display()));
                cx.notify();
            }),
        })
    })
    .detach();
}

/// Asks for a folder (or a catalog) and opens it in place of `replace`.
pub fn prompt_and_open(state: Entity<AppState>, folder: bool, replace: AnyWindowHandle, cx: &mut App) {
    let rx = cx.prompt_for_paths(PathPromptOptions {
        files: !folder,
        directories: true,
        multiple: false,
        prompt: Some(if folder { t("Open", "打开") } else { t("Open Catalog", "打开目录") }.into()),
    });
    cx.spawn(async move |cx| {
        if let Ok(Ok(Some(paths))) = rx.await
            && let Some(path) = paths.into_iter().next()
        {
            cx.update(|cx| open_path(state, path, replace, cx));
        }
    })
    .detach();
}

fn card(id: &'static str, icon: IconName, title: &'static str, detail: &'static str, chip: Option<&'static str>, disabled: bool, cx: &App) -> gpui_kit::Stateful<gpui_kit::Div> {
    let p = pal(cx).clone();
    div()
        .id(id)
        .flex_1()
        .min_w_0()
        .p_4()
        .gap_2()
        .flex()
        .flex_col()
        .rounded(px(8.0))
        .bg(p.panel)
        .border_1()
        .border_color(p.edge)
        .when(!disabled, |d| d.cursor_pointer().hover(|d| d.border_color(p.accent.opacity(0.7)).bg(p.hover)))
        .when(disabled, |d| d.opacity(0.5))
        .child(
            h_flex()
                .gap_2()
                .child(div().size(px(32.0)).rounded(px(7.0)).bg(p.accent.opacity(0.14)).flex().items_center().justify_center().child(Icon::new(icon).text_color(p.accent)))
                .child(div().flex_1())
                .when_some(chip, |d, chip| d.child(div().px_1p5().rounded(px(3.0)).bg(p.well).text_size(px(10.0)).text_color(p.warning).child(chip))),
        )
        .child(div().text_sm().text_color(p.text).child(title))
        .child(div().text_xs().text_color(p.text_subtle).child(detail))
}

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let p = pal(cx).clone();
    let s = state.read(cx);
    let opening = s.opening.clone();
    let error = s.open_error.clone();
    let recent = s.settings.recent.clone();
    let reopen = s.settings.reopen_last;
    let busy = opening.is_some();
    let handle = window.window_handle();

    let logo = div()
        .size(px(56.0))
        .rounded(px(14.0))
        .bg(gpui_kit::linear_gradient(135.0, gpui_kit::linear_color_stop(p.accent, 0.0), gpui_kit::linear_color_stop(hexa(0xd05f9a, 1.0), 1.0)))
        .shadow_lg()
        .flex()
        .items_center()
        .justify_center()
        .child(Icon::new(IconName::Aperture).size(px(30.0)).text_color(gpui_kit::white()));

    let (st_folder, st_catalog, st_demo) = (state.clone(), state.clone(), state.clone());
    let cards = h_flex()
        .gap_3()
        .child(motion::stagger(
            "welcome-card-0",
            0,
            card("welcome-folder", IconName::FolderOpen, t("Open Folder…", "打开文件夹…"), t("Browse and edit the photos in a folder.", "浏览并编辑一个文件夹中的照片。"), None, busy, cx)
                .when(!busy, |d| d.on_click(move |_, _, cx| prompt_and_open(st_folder.clone(), true, handle, cx))),
            10.0,
        ))
        .child(motion::stagger(
            "welcome-card-1",
            1,
            card(
                "welcome-catalog",
                IconName::Library,
                t("Open Capture One Catalog…", "打开 Capture One 目录…"),
                t("Import a .cocatalog or session read-only.", "只读导入 .cocatalog 目录或会话。"),
                contract::C1_CATALOG.chip(),
                busy,
                cx,
            )
            .when(!busy, |d| d.on_click(move |_, _, cx| prompt_and_open(st_catalog.clone(), false, handle, cx))),
            10.0,
        ))
        .child(motion::stagger(
            "welcome-card-2",
            2,
            card("welcome-demo", IconName::Sparkles, t("Explore the Demo", "浏览演示图像"), t("Generated sample images, not real photos.", "程序生成的示例图像，并非真实照片。"), None, busy, cx)
                .when(!busy, |d| {
                    d.on_click(move |_, _, cx| {
                        st_demo.update(cx, |s, cx| {
                            s.welcome = false;
                            cx.notify();
                        })
                    })
                }),
            10.0,
        ));

    let mut list = v_flex().gap_0p5();
    for (i, path) in recent.iter().enumerate() {
        let exists = path.exists();
        let is_catalog = source::is_catalog(path);
        let name = source::display_name(path);
        let parent = path.parent().map(|p| p.display().to_string()).unwrap_or_default();
        let st_open = state.clone();
        let st_forget = state.clone();
        let (open_path_buf, forget_path) = (path.clone(), path.clone());
        let enabled = exists && !busy;
        let row = h_flex()
            .id(SharedString::from(format!("recent-{i}")))
            .group("recent-row")
            .h(px(40.0))
            .px_2()
            .gap_2()
            .rounded(px(6.0))
            .when(enabled, |d| d.cursor_pointer().hover(|d| d.bg(p.hover)))
            .when(enabled, |d| d.on_click(move |_, _, cx| open_path(st_open.clone(), open_path_buf.clone(), handle, cx)))
            .child(Icon::new(if is_catalog { IconName::Library } else { IconName::Folder }).small().text_color(if exists { p.accent } else { p.text_disabled }))
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .child(div().text_sm().text_color(if exists { p.text } else { p.text_disabled }).overflow_hidden().whitespace_nowrap().text_ellipsis().child(name))
                    .child(div().text_xs().text_color(p.text_subtle).overflow_hidden().whitespace_nowrap().text_ellipsis().child(parent)),
            )
            .when(!exists, |d| d.child(div().text_xs().text_color(p.text_disabled).child(t("Not found", "已不存在"))))
            .when(i == 0 && exists, |d| d.child(div().text_xs().text_color(p.text_subtle).child(t("Last opened", "上次打开"))))
            .child(
                Button::new(SharedString::from(format!("recent-forget-{i}")))
                    .xsmall()
                    .ghost()
                    .icon(Icon::new(IconName::X))
                    .tooltip(t("Remove from list", "从列表中移除"))
                    .disabled(busy)
                    .on_click(move |_, _, cx| {
                        cx.stop_propagation();
                        st_forget.update(cx, |s, cx| {
                            s.settings.forget(&forget_path);
                            s.settings.save();
                            cx.notify();
                        })
                    }),
            );
        list = list.child(motion::stagger(SharedString::from(format!("recent-in-{i}")), i + 3, row, 6.0));
    }

    let st_reopen = state.clone();
    let recent_block = v_flex()
        .gap_1()
        .child(
            h_flex()
                .px_2()
                .text_xs()
                .text_color(p.text_subtle)
                .child(div().flex_1().child(t("Recent", "最近打开")))
                .when(!recent.is_empty(), |d| d.child(format!("{}", recent.len()))),
        )
        .child(if recent.is_empty() {
            div()
                .px_2()
                .py_4()
                .text_xs()
                .text_color(p.text_disabled)
                .child(t("Folders and catalogs you open appear here.", "打开过的文件夹和目录会出现在这里。"))
                .into_any_element()
        } else {
            list.into_any_element()
        })
        .child(div().pt_2().child(switch_row(
            "welcome-reopen",
            t("Reopen the last one at startup", "启动时打开上次的目录"),
            reopen,
            move |v, _, cx| {
                st_reopen.update(cx, |s, cx| {
                    s.settings.reopen_last = v;
                    s.settings.save();
                    cx.notify();
                })
            },
            cx,
        )));

    let status = match &opening {
        Some(path) => h_flex()
            .gap_2()
            .px_3()
            .py_2()
            .rounded(px(6.0))
            .bg(p.well)
            .text_xs()
            .text_color(p.text_muted)
            .child(Spinner::new().small())
            .child(format!("{} {}…", t("Opening", "正在打开"), source::display_name(path)))
            .into_any_element(),
        None => match error {
            Some(err) => h_flex()
                .gap_2()
                .px_3()
                .py_2()
                .rounded(px(6.0))
                .bg(p.danger.opacity(0.12))
                .border_1()
                .border_color(p.danger.opacity(0.4))
                .text_xs()
                .text_color(p.text)
                .child(Icon::new(IconName::TriangleAlert).xsmall().text_color(p.danger))
                .child(div().flex_1().min_w_0().child(format!("{} {err}", t("Could not open", "无法打开"))))
                .into_any_element(),
            None => div().into_any_element(),
        },
    };

    // Sit a little above the middle of the window, where the eye lands first;
    // on a short window fall back to a fixed margin and scroll.
    let top = ((f32::from(window.viewport_size().height) - 520.0) * 0.38).max(40.0);
    div()
        .id("welcome")
        .size_full()
        .overflow_y_scroll()
        // A faint wash of the accent from the top keeps the page from being a flat slab.
        .bg(gpui_kit::linear_gradient(180.0, gpui_kit::linear_color_stop(p.chrome.blend(p.accent.opacity(0.07)), 0.0), gpui_kit::linear_color_stop(p.chrome, 0.55)))
        .flex()
        .justify_center()
        .child(
            v_flex()
                .w(px(720.0))
                .pt(px(top))
                .pb_8()
                .gap_6()
                .child(motion::enter(
                    "welcome-head",
                    h_flex()
                        .gap_4()
                        .child(logo)
                        .child(
                            v_flex()
                                .child(div().text_2xl().text_color(p.text).child("Aurora Studio"))
                                .child(div().text_sm().text_color(p.text_subtle).child(t("Raw processing, cataloguing and retouching", "RAW 处理、图库管理与人像精修"))),
                        ),
                    12.0,
                ))
                .child(status)
                .child(cards)
                .child(motion::enter("welcome-recent", div().p_2().rounded(px(8.0)).bg(p.panel).border_1().border_color(p.edge).child(recent_block), 8.0)),
        )
        .into_any_element()
}
