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

/// Prompts for a folder and initializes it as an Aurora Session or Catalog, then opens it.
pub fn prompt_and_create(state: Entity<AppState>, is_session: bool, replace: AnyWindowHandle, cx: &mut App) {
    let rx = cx.prompt_for_paths(PathPromptOptions {
        files: false,
        directories: true,
        multiple: false,
        prompt: Some(if is_session {
            t("Choose Session Folder", "选择 Session 会话文件夹")
        } else {
            t("Choose Catalog Folder", "选择 Catalog 目录文件夹")
        }.into()),
    });
    cx.spawn(async move |cx| {
        if let Ok(Ok(Some(paths))) = rx.await
            && let Some(path) = paths.into_iter().next()
        {
            cx.update(|cx| {
                state.update(cx, |s, cx| {
                    s.opening = Some(path.clone());
                    s.open_error = None;
                    cx.notify();
                });
            });
            let target = path.clone();
            let result = cx.background_spawn(async move {
                if is_session {
                    source::create_session(&target, None)
                } else {
                    source::create_catalog(&target, None)
                }
            }).await;
            cx.update(|cx| match result {
                Ok((engine, src)) => {
                    let next = crate::state::init_state(engine, src, cx);
                    super::workspace::open(next, cx);
                    let _ = replace.update(cx, |_, window, _| window.remove_window());
                }
                Err(err) => state.update(cx, |s, cx| {
                    s.opening = None;
                    s.welcome = true;
                    s.open_error = Some(format!("{}: {err}", path.display()));
                    cx.notify();
                }),
            });
        }
    })
    .detach();
}

/// One action in the left column: an icon tile, a title and one line on what it does.
fn action(id: &'static str, icon: IconName, title: &'static str, detail: &'static str, chip: Option<&'static str>, disabled: bool, cx: &App) -> gpui_kit::Stateful<gpui_kit::Div> {
    let p = pal(cx).clone();
    h_flex()
        .id(id)
        .w_full()
        .px_2()
        .py_2()
        .gap_3()
        .rounded(px(8.0))
        .when(!disabled, |d| d.cursor_pointer().hover(|d| d.bg(p.hover)))
        .when(disabled, |d| d.opacity(0.5))
        .child(
            div()
                .size(px(32.0))
                .flex_shrink_0()
                .rounded(px(8.0))
                .bg(p.well)
                .flex()
                .items_center()
                .justify_center()
                .child(Icon::new(icon).text_color(p.text_muted)),
        )
        .child(
            v_flex()
                .flex_1()
                .min_w_0()
                .child(
                    h_flex()
                        .gap_1p5()
                        .child(div().text_sm().text_color(p.text).child(title))
                        .when_some(chip, |d, chip| d.child(div().px_1().rounded(px(3.0)).bg(p.well).text_size(px(10.0)).text_color(p.warning).child(chip))),
                )
                .child(div().text_xs().text_color(p.text_subtle).overflow_hidden().whitespace_nowrap().text_ellipsis().child(detail)),
        )
}

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let p = pal(cx).clone();
    let s = state.read(cx);
    let opening = s.opening.clone();
    let error = s.open_error.clone();
    let recent = s.prefs.settings.recent.clone();
    let reopen = s.prefs.settings.reopen_last;
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

    let (st_folder, st_catalog, st_demo, st_new_session, st_new_catalog) =
        (state.clone(), state.clone(), state.clone(), state.clone(), state.clone());
    // One primary action, the rest as a quiet list under it.
    let open_folder = Button::new("welcome-open-folder")
        .primary()
        .large()
        .w_full()
        .icon(Icon::new(IconName::FolderOpen))
        .label(t("Open Folder…", "打开文件夹…"))
        .disabled(busy)
        .on_click(move |_, _, cx| prompt_and_open(st_folder.clone(), true, handle, cx));
    let actions = v_flex()
        .gap_0p5()
        .child(motion::stagger(
            "welcome-act-0",
            0,
            action("welcome-new-session", IconName::FolderPlus, t("New Session…", "新建会话…"), t("A folder that keeps its photos, edits and output together.", "照片、调整与输出都放在一个文件夹里。"), None, busy, cx)
                .when(!busy, |d| d.on_click(move |_, _, cx| prompt_and_create(st_new_session.clone(), true, handle, cx))),
            6.0,
        ))
        .child(motion::stagger(
            "welcome-act-1",
            1,
            action("welcome-new-catalog", IconName::Database, t("New Catalog…", "新建图库…"), t("One library for photos from many folders.", "把多个文件夹的照片管理在一个图库里。"), None, busy, cx)
                .when(!busy, |d| d.on_click(move |_, _, cx| prompt_and_create(st_new_catalog.clone(), false, handle, cx))),
            6.0,
        ))
        .child(motion::stagger(
            "welcome-act-2",
            2,
            action(
                "welcome-catalog",
                IconName::Library,
                t("Open Capture One Catalog…", "打开 Capture One 目录…"),
                t("Read a .cocatalog, session or Lightroom .lrcat without changing it.", "只读打开 .cocatalog 目录、会话或 Lightroom .lrcat。"),
                contract::C1_CATALOG.chip(),
                busy,
                cx,
            )
            .when(!busy, |d| d.on_click(move |_, _, cx| prompt_and_open(st_catalog.clone(), false, handle, cx))),
            6.0,
        ));
    let demo = h_flex()
        .id("welcome-demo")
        .gap_1p5()
        .px_2()
        .py_1()
        .rounded(px(5.0))
        .text_xs()
        .text_color(p.text_subtle)
        .when(!busy, |d| {
            d.cursor_pointer().hover(|d| d.bg(p.hover).text_color(p.text)).on_click(move |_, _, cx| {
                st_demo.update(cx, |s, cx| {
                    s.welcome = false;
                    cx.notify();
                })
            })
        })
        .child(Icon::new(IconName::Sparkles).xsmall())
        .child(t("Explore the demo (generated images)", "浏览演示（程序生成的示例图像）"));

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
            .h(px(52.0))
            .px_3()
            .gap_3()
            .rounded(px(8.0))
            .when(enabled, |d| d.cursor_pointer().hover(|d| d.bg(p.hover)))
            .when(enabled, |d| d.on_click(move |_, _, cx| open_path(st_open.clone(), open_path_buf.clone(), handle, cx)))
            .child(
                div()
                    .size(px(32.0))
                    .flex_shrink_0()
                    .rounded(px(8.0))
                    .bg(if exists { p.accent.opacity(0.14) } else { p.well })
                    .flex()
                    .items_center()
                    .justify_center()
                    .child(Icon::new(if is_catalog { IconName::Library } else { IconName::Folder }).small().text_color(if exists { p.accent } else { p.text_disabled })),
            )
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
                            s.prefs.settings.forget(&forget_path);
                            s.prefs.settings.save();
                            cx.notify();
                        })
                    }),
            );
        list = list.child(motion::stagger(SharedString::from(format!("recent-in-{i}")), i + 3, row, 6.0));
    }

    let st_reopen = state.clone();
    let reopen_row = switch_row(
        "welcome-reopen",
        t("Reopen the last one at startup", "启动时打开上次的目录"),
        reopen,
        move |v, _, cx| {
            st_reopen.update(cx, |s, cx| {
                s.prefs.settings.reopen_last = v;
                s.prefs.settings.save();
                cx.notify();
            })
        },
        cx,
    );
    let recent_block = v_flex()
        .size_full()
        .p_2()
        .gap_1()
        .child(
            h_flex()
                .px_3()
                .pt_2()
                .pb_1()
                .child(div().flex_1().text_sm().text_color(p.text).child(t("Recent", "最近打开")))
                .when(!recent.is_empty(), |d| d.child(div().text_xs().text_color(p.text_subtle).child(format!("{}", recent.len())))),
        )
        .child(if recent.is_empty() {
            v_flex()
                .flex_1()
                .items_center()
                .justify_center()
                .gap_2()
                .text_color(p.text_subtle)
                .child(
                    div()
                        .size(px(56.0))
                        .rounded_full()
                        .bg(p.well)
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(Icon::new(IconName::FolderClock).size(px(26.0)).text_color(p.text_subtle)),
                )
                .child(div().text_sm().text_color(p.text_muted).child(t("Nothing opened yet", "还没有打开过文件夹或目录")))
                .child(div().text_xs().child(t("Open a folder to start; it will be listed here next time.", "先打开一个文件夹，下次就会出现在这里。")))
                .into_any_element()
        } else {
            // A long history scrolls instead of stretching the page.
            div().id("welcome-recent-list").flex_1().min_h_0().max_h(px(420.0)).overflow_y_scroll().child(list).into_any_element()
        });

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

    let brand = h_flex()
        .gap_3()
        .child(logo)
        .child(
            v_flex()
                .child(div().text_xl().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(p.text).child("Aurora Studio"))
                .child(div().text_xs().text_color(p.text_subtle).child(t("Raw processing, cataloguing and retouching", "RAW 处理、图库管理与人像精修"))),
        );
    let left = v_flex()
        .w(px(300.0))
        .flex_shrink_0()
        .gap_5()
        .child(motion::enter("welcome-head", brand, 10.0))
        .child(open_folder)
        .child(actions)
        .child(div().flex_1())
        .child(v_flex().gap_2().child(div().h(px(1.0)).bg(p.divider)).child(demo).child(div().pl_0p5().child(reopen_row)));
    let right = motion::enter(
        "welcome-recent",
        div().flex_1().min_w_0().min_h(px(320.0)).rounded(px(12.0)).bg(p.panel).border_1().border_color(p.edge).child(recent_block),
        8.0,
    );

    // A two-column start page sized by its content and centred in the window:
    // what to do on the left, where the user was on the right. The columns
    // stretch to the taller one, so neither leaves a hole under its content.
    div()
        .id("welcome")
        .size_full()
        .overflow_y_scroll()
        // A faint wash of the accent from the top keeps the page from being a flat slab.
        .bg(gpui_kit::linear_gradient(180.0, gpui_kit::linear_color_stop(p.chrome.blend(p.accent.opacity(0.07)), 0.0), gpui_kit::linear_color_stop(p.chrome, 0.55)))
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .py_8()
        .child(
            v_flex()
                .w(px(960.0))
                .gap_4()
                .child(status)
                .child(h_flex().gap_8().items_stretch().child(left).child(right)),
        )
        .into_any_element()
}
