//! The top toolbar: catalog commands, history, adjustment clipboard, cursor tools,
//! view toggles and zoom.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    menu::{DropdownMenu as _, PopupMenuItem},
};
use gpui_kit::{App, Entity, IntoElement, ParentElement, Styled, Window, div, px};
use lumen_engine::AutoTarget;

use crate::actions;
use crate::i18n::t;
use crate::state::{AppState, BrowserMode, CompareMode, CursorTool};
use crate::theme::{metrics, pal};
use crate::widgets::IconBtn;

fn sep(cx: &App) -> impl IntoElement + use<> {
    div().w(px(1.0)).h(px(20.0)).mx_1p5().bg(pal(cx).divider)
}

fn group(cx: &App) -> gpui_kit::Div {
    h_flex().gap_0p5().p(px(2.0)).rounded(px(6.0)).bg(pal(cx).well.opacity(0.6))
}

pub const CURSOR_TOOLS: [CursorTool; 15] = [
    CursorTool::Select,
    CursorTool::Pan,
    CursorTool::Zoom,
    CursorTool::Crop,
    CursorTool::Straighten,
    CursorTool::Keystone,
    CursorTool::Spot,
    CursorTool::Brush,
    CursorTool::Eraser,
    CursorTool::LinearGradient,
    CursorTool::RadialGradient,
    CursorTool::ColorPicker,
    CursorTool::WhiteBalance,
    CursorTool::Annotate,
    CursorTool::Loupe,
];

pub fn toolbar(state: &Entity<AppState>, _window: &mut Window, cx: &App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let s = state.read(cx);
    let has_image = s.current.is_some();
    let cursor = s.cursor;
    let view = s.view.clone();
    let zoom_pct = (s.effective_scale() / s.scale_factor.max(0.5) * 100.0).round();
    let fit = s.zoom.scale.is_none();
    let can_undo = s.can_undo();
    let camera_ready = s.camera_ready();
    let tether_note = s.tether_note();
    let can_redo = s.can_redo();
    // Name the step, so undo is never a guess.
    let (undo_steps, redo_steps) = s.history();
    let undo_tip = match undo_steps.last() {
        Some(step) => format!("{} · {step}", t("Undo", "撤销")),
        None => t("Undo", "撤销").to_string(),
    };
    let redo_tip = match redo_steps.last() {
        Some(step) => format!("{} · {step}", t("Redo", "重做")),
        None => t("Redo", "重做").to_string(),
    };
    let has_clip = s.clipboard.is_some();

    let st = state.clone();
    let tool = move |tool: CursorTool| {
        let st = st.clone();
        IconBtn::new(format!("cursor-{tool:?}"), tool.icon(), tool.label())
            .shortcut(tool.key())
            .active(cursor == tool)
            .disabled(!has_image)
            .on_click(move |_, cx| st.update(cx, |s, cx| s.set_cursor(tool, cx)))
    };

    let dispatch = |id: &'static str, icon: IconName, tip: &'static str, key: &'static str, action: Box<dyn gpui_kit::Action>| {
        IconBtn::new(id, icon, tip).shortcut(key).on_click(move |window, cx| window.dispatch_action(action.boxed_clone(), cx))
    };

    let st = state.clone();
    let zoom_menu = Button::new("zoom-menu")
        .xsmall()
        .ghost()
        .label(if fit { format!("{} · {zoom_pct}%", t("Fit", "适合")) } else { format!("{zoom_pct}%") })
        .dropdown_caret(true)
        .dropdown_menu(move |mut menu, _, _| {
            let st2 = st.clone();
            menu = menu.item(PopupMenuItem::new(t("Fit", "适合窗口")).checked(fit).on_click(move |_, _, cx| {
                st2.update(cx, |s, cx| s.set_zoom(None, None, cx))
            }));
            for pct in [25.0, 50.0, 100.0, 200.0, 400.0, 800.0] {
                let st = st.clone();
                menu = menu.item(PopupMenuItem::new(format!("{pct:.0}%")).checked(!fit && (zoom_pct - pct).abs() < 0.5).on_click(
                    move |_, _, cx| {
                        st.update(cx, |s, cx| {
                            let sf = s.scale_factor;
                            let c = s.zoom.center;
                            s.set_zoom(Some(pct / 100.0 * sf), Some(c), cx)
                        })
                    },
                ));
            }
            menu
        });

    let st_auto = state.clone();
    let st_browser = state.clone();
    let browser_btn = |mode: BrowserMode, icon: IconName, tip: &'static str| {
        let st = st_browser.clone();
        IconBtn::new(format!("browser-{mode:?}"), icon, tip)
            .active(view.browser == mode)
            .on_click(move |_, cx| {
                st.update(cx, |s, cx| {
                    s.view.browser = if s.view.browser == mode { BrowserMode::Hidden } else { mode };
                    cx.notify();
                })
            })
    };
    let st_view = state.clone();
    let view_toggle = |id: &'static str, icon: IconName, tip: &'static str, key: &'static str, on: bool, f: fn(&mut AppState)| {
        let st = st_view.clone();
        IconBtn::new(id, icon, tip).shortcut(key).active(on).on_click(move |_, cx| {
            st.update(cx, |s, cx| {
                f(s);
                s.preview.before_dirty = true;
                s.request_render(lumen_engine::RenderQuality::Final, cx);
                cx.notify();
            })
        })
    };

    h_flex()
        .h(px(metrics::TOOLBAR))
        .px_2()
        .gap_1()
        .bg(p.chrome)
        .border_b_1()
        .border_color(p.divider)
        // Catalog.
        .child(
            group(cx)
                .child(dispatch("tb-import", IconName::ImageDown, t("Import", "导入"), "Ctrl+Shift+I", Box::new(actions::Import)))
                .child(if camera_ready {
                    dispatch("tb-capture", IconName::Camera, t("Capture", "拍摄"), "Ctrl+K", Box::new(actions::Capture))
                } else {
                    IconBtn::new("tb-capture", IconName::CameraOff, tether_note).disabled(true)
                })
                .child(dispatch("tb-export", IconName::Upload, t("Process / Export", "处理 / 导出"), "Ctrl+D", Box::new(actions::OpenExportDialog))),
        )
        .child(sep(cx))
        .child(
            group(cx)
                .child(
                    IconBtn::new("tb-undo", IconName::Undo2, undo_tip)
                        .shortcut("Ctrl+Z")
                        .disabled(!can_undo)
                        .on_click(|window, cx| window.dispatch_action(Box::new(actions::Undo), cx)),
                )
                .child(
                    IconBtn::new("tb-redo", IconName::Redo2, redo_tip)
                        .shortcut("Ctrl+Y")
                        .disabled(!can_redo)
                        .on_click(|window, cx| window.dispatch_action(Box::new(actions::Redo), cx)),
                ),
        )
        .child(
            group(cx)
                .child(dispatch("tb-reset", IconName::RotateCcw, t("Reset adjustments", "复位调整"), "Ctrl+R", Box::new(actions::ResetAdjustments)))
                .child(
                    IconBtn::new("tb-auto", IconName::WandSparkles, t("Auto adjust", "自动调整"))
                        .shortcut("Ctrl+L")
                        .disabled(!has_image)
                        .on_click(move |_, cx| st_auto.update(cx, |s, cx| s.auto_adjust(AutoTarget::All, cx))),
                )
                .child(dispatch("tb-copy", IconName::Copy, t("Copy adjustments", "拷贝调整"), "Ctrl+Shift+C", Box::new(actions::CopyAdjustments)))
                .child(
                    IconBtn::new("tb-apply", IconName::ClipboardPaste, t("Apply adjustments", "应用调整"))
                        .shortcut("Ctrl+Shift+V")
                        .disabled(!has_clip)
                        .on_click(|window, cx| window.dispatch_action(Box::new(actions::ApplyAdjustments), cx)),
                ),
        )
        .child(sep(cx))
        .child(group(cx).children(CURSOR_TOOLS[..3].iter().map(|t| tool(*t))))
        .child(group(cx).children(CURSOR_TOOLS[3..6].iter().map(|t| tool(*t))))
        .child(group(cx).children(CURSOR_TOOLS[6..11].iter().map(|t| tool(*t))))
        .child(group(cx).children(CURSOR_TOOLS[11..].iter().map(|t| tool(*t))))
        .child(div().flex_1())
        .child(
            group(cx)
                .child(view_toggle("tb-compare", IconName::Columns2, t("Before / after", "前后对比"), "Y", view.compare != CompareMode::Off, |s| {
                    s.view.compare = if s.view.compare == CompareMode::Off { CompareMode::Split } else { CompareMode::Off }
                }))
                .child(view_toggle("tb-exposure", IconName::TriangleAlert, t("Exposure warning", "曝光警告"), "Alt+E", view.exposure_warning, |s| {
                    s.view.exposure_warning = !s.view.exposure_warning
                }))
                .child(view_toggle("tb-focus", IconName::Focus, t("Focus mask", "焦点蒙版"), "", view.focus_mask, |s| s.view.focus_mask = !s.view.focus_mask))
                .child(view_toggle("tb-grid", IconName::Grid3x3, t("Grid overlay", "网格叠加"), "Alt+G", view.grid, |s| s.view.grid = !s.view.grid))
                .child(view_toggle("tb-proof", IconName::Frame, t("Proof margin", "校样边距"), "Alt+P", view.proof, |s| s.view.proof = !s.view.proof)),
        )
        .child(sep(cx))
        .child(
            group(cx)
                .child(browser_btn(BrowserMode::Filmstrip, IconName::PanelBottom, t("Filmstrip", "胶片条")))
                .child(browser_btn(BrowserMode::Grid, IconName::LayoutGrid, t("Grid browser", "网格浏览")))
                .child(browser_btn(BrowserMode::List, IconName::List, t("List browser", "列表浏览")))
                .child(view_toggle("tb-inspector", IconName::PanelLeft, t("Show tools", "显示工具"), "Ctrl+T", view.show_inspector, |s| {
                    s.view.show_inspector = !s.view.show_inspector
                })),
        )
        .child(sep(cx))
        .child(zoom_menu)
}
