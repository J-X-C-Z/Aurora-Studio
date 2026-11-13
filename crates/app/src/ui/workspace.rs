//! The window's root view: title bar, toolbar, inspector, viewer, browser and
//! status bar, plus every workspace-level command.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, TitleBar, WindowExt as _, h_flex,
    menu::AppMenuBar,
    notification::Notification,
    v_flex,
};
use gpui_kit::{
    App, AppContext as _, Context, Entity, FocusHandle, Focusable, InteractiveElement, IntoElement, ParentElement,
    PathPromptOptions, Render, Styled, Subscription, Window, div, prelude::FluentBuilder as _, px,
};
use lumen_engine::{AutoTarget, ColorTag, RenderQuality};

use super::browser::Browser;
use super::viewer::Viewer;
use super::{dialogs, inspector, statusbar, toolbar};
use crate::actions::*;
use crate::i18n::{self, t};
use crate::motion;
use crate::state::{AppState, BrowserMode, CompareMode, NewNode, Source, StateEvent, ToolTab};

/// Height of the node editor docked under the viewer.
const NODE_DOCK_H: f32 = 300.0;
use crate::theme::{hexa, metrics, pal};
use crate::widgets::surface::Surface;

/// The workspace's focus handle, so a control that borrowed the keyboard (a
/// typed slider value) can hand it back and shortcuts keep working.
pub struct WorkspaceFocus(pub FocusHandle);

impl gpui_kit::Global for WorkspaceFocus {}

/// The latest message, shown briefly over the viewer where the eyes already
/// are; the status bar keeps it afterwards.
pub(crate) fn notice_hud(text: gpui_kit::SharedString, serial: u64, p: &crate::theme::Palette) -> impl IntoElement + use<> {
    use gpui_kit::AnimationExt as _;
    let secs = (1.5 + text.chars().count() as f32 * 0.04).min(4.5);
    let (fade_in, fade_out) = (0.16 / secs, 0.4 / secs);
    div().absolute().left_0().right_0().bottom(px(46.0)).flex().justify_center().child(
        h_flex()
            .max_w(px(560.0))
            .px_3()
            .py_1p5()
            .rounded(px(8.0))
            .bg(hexa(0x101011, 0.86))
            .border_1()
            .border_color(p.edge)
            .shadow_lg()
            .text_sm()
            .text_color(p.text)
            .child(text)
            .with_animation(
                gpui_kit::SharedString::from(format!("notice-hud-{serial}")),
                gpui_kit::Animation::new(std::time::Duration::from_secs_f32(secs)),
                move |el, t| {
                    let rise = (t / fade_in).min(1.0);
                    let rise = 1.0 - (1.0 - rise).powi(3);
                    let o = if t > 1.0 - fade_out { (1.0 - t) / fade_out } else { rise };
                    el.opacity(o.clamp(0.0, 1.0)).mb(px(-8.0 * (1.0 - rise)))
                },
            ),
    )
}

pub struct Workspace {
    pub state: Entity<AppState>,
    viewer: Entity<Viewer>,
    browser: Entity<Browser>,
    menu_bar: Entity<AppMenuBar>,
    focus: FocusHandle,
    notice_serial: u64,
    /// The latest notice went to a notification card, so the viewer HUD skips it.
    notice_carded: bool,
    menu_key: Option<(bool, Vec<String>, Vec<String>)>,
    _subs: Vec<Subscription>,
}

impl Focusable for Workspace {
    fn focus_handle(&self, _: &gpui_kit::App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Workspace {
    pub fn new(state: Entity<AppState>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let viewer = cx.new(|cx| Viewer::new(&state, cx));
        let browser = cx.new(|cx| Browser::new(&state, window, cx));
        let menu_bar = AppMenuBar::new(cx);
        let names = Self::workspace_names(&state, cx);
        install_menus(&menu_bar, &names, &Self::recent_names(&state, cx), cx);
        let focus = cx.focus_handle();
        window.focus(&focus, cx);
        cx.set_global(WorkspaceFocus(focus.clone()));
        // Speed Edit: holding Q W E R A S D F arms a parameter for the viewer's
        // scroll wheel and drag. Interceptors run before key bindings and do not
        // consume the keystroke, so a tap still selects the key's cursor tool.
        let handle = window.window_handle();
        let speed_state = state.clone();
        let speed_keys = cx.intercept_keystrokes(move |e, window, cx| {
            let m = &e.keystroke.modifiers;
            if window.window_handle() != handle || m.control || m.alt || m.shift || m.platform {
                return;
            }
            // Culling takes its keys before any binding.
            if speed_state.read(cx).cull.is_some() {
                let key = e.keystroke.key.clone();
                let handled = match key.as_str() {
                    "escape" => {
                        speed_state.update(cx, |s, cx| s.stop_cull(window, cx));
                        true
                    }
                    "right" | "down" => {
                        speed_state.update(cx, |s, cx| s.cull_step(1, cx));
                        true
                    }
                    "left" | "up" => {
                        speed_state.update(cx, |s, cx| s.cull_step(-1, cx));
                        true
                    }
                    "0" | "1" | "2" | "3" | "4" | "5" => {
                        let r = key.parse::<i8>().unwrap_or(0);
                        speed_state.update(cx, |s, cx| s.cull_rate(r, cx));
                        true
                    }
                    "x" | "-" => {
                        speed_state.update(cx, |s, cx| s.cull_rate(-1, cx));
                        true
                    }
                    "a" => {
                        speed_state.update(cx, |s, cx| s.toggle_cull_auto_advance(cx));
                        true
                    }
                    "u" => {
                        speed_state.update(cx, |s, cx| s.toggle_cull_unrated(cx));
                        true
                    }
                    "z" => {
                        speed_state.update(cx, |s, cx| s.toggle_cull_zoom(cx));
                        true
                    }
                    _ => false,
                };
                if handled {
                    cx.stop_propagation();
                }
                return;
            }
            // A slideshow takes Esc, Space and the arrows before any binding.
            if speed_state.read(cx).slideshow.is_some() {
                let handled = match e.keystroke.key.as_str() {
                    "escape" => {
                        speed_state.update(cx, |s, cx| s.stop_slideshow(window, cx));
                        true
                    }
                    "space" => {
                        speed_state.update(cx, |s, cx| s.toggle_slideshow_play(cx));
                        true
                    }
                    "right" | "down" => {
                        speed_state.update(cx, |s, cx| s.slide(1, cx));
                        true
                    }
                    "left" | "up" => {
                        speed_state.update(cx, |s, cx| s.slide(-1, cx));
                        true
                    }
                    _ => false,
                };
                if handled {
                    cx.stop_propagation();
                }
                return;
            }
            // Letters typed into a field or a dialog are text, not Speed Edit.
            let typing = window.context_stack().iter().any(|c| c.contains("Input")) || window.has_active_dialog(cx);
            if let Some(param) = crate::state::SpeedParam::from_key(&e.keystroke.key).filter(|_| !typing) {
                speed_state.update(cx, |s, cx| s.speed_begin(param, cx));
            }
        });
        let subs = vec![
            speed_keys,
            cx.observe(&state, |this: &mut Self, state, cx| {
                // Menus carry the language and the workspace list; rebuild them when either changes.
                let names = Self::workspace_names(&state, cx);
                let recent = Self::recent_names(&state, cx);
                let key = (i18n::locale() == i18n::Locale::ZhCn, names.clone(), recent.clone());
                if this.menu_key.as_ref() != Some(&key) {
                    this.menu_key = Some(key);
                    install_menus(&this.menu_bar, &names, &recent, cx);
                }
                cx.notify()
            }),
            cx.subscribe_in(&state, window, |this, _, e: &StateEvent, window, cx| match e {
                StateEvent::Notice(msg) => {
                    this.notice_serial += 1;
                    let important = msg.contains(t("Imported", "已导入")) || msg.contains(t("failed", "失败"));
                    this.notice_carded = important;
                    if important {
                        window.push_notification(Notification::new().message(msg.clone()), cx);
                    }
                }
                StateEvent::CurrentChanged => {}
            }),
        ];
        Self { state, viewer, browser, menu_bar, focus, notice_serial: 0, notice_carded: false, menu_key: None, _subs: subs }
    }

    fn workspace_names(state: &Entity<AppState>, cx: &App) -> Vec<String> {
        state.read(cx).workspace_list().into_iter().map(|w| w.name).collect()
    }

    fn recent_names(state: &Entity<AppState>, cx: &App) -> Vec<String> {
        state.read(cx).settings.recent.iter().map(|p| p.display().to_string()).collect()
    }

    fn with_state(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&mut AppState, &mut Context<AppState>)) {
        self.state.update(cx, f);
    }

    /// Opens a folder or a Capture One catalog in a new Aurora window, the way
    /// Capture One opens a second catalog: one engine per process.
    fn open_in_new_window(&mut self, folder: bool, window: &mut Window, cx: &mut Context<Self>) {
        // The demo window has nothing worth keeping: replace it in place.
        if self.state.read(cx).source == Source::Demo {
            super::welcome::prompt_and_open(self.state.clone(), folder, window.window_handle(), cx);
            return;
        }
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: !folder,
            directories: true,
            multiple: false,
            prompt: Some(if folder { t("Open", "打开") } else { t("Open Catalog", "打开目录") }.into()),
        });
        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return };
            let Some(path) = paths.into_iter().next() else { return };
            let _ = this.update(cx, |this, cx| this.spawn_window(path, !folder, cx));
        })
        .detach();
    }

    /// Opens `path` in a new Aurora process, so each catalog keeps its own engine.
    fn spawn_window(&mut self, path: std::path::PathBuf, catalog: bool, cx: &mut Context<Self>) {
        let spawned = std::env::current_exe().and_then(|exe| std::process::Command::new(exe).arg(&path).spawn());
        self.state.update(cx, |s, cx| match spawned {
            Ok(_) => {
                let note = if catalog { format!(" · {}", crate::contract::C1_CATALOG.note().unwrap_or_default()) } else { String::new() };
                s.settings.remember(&path);
                s.notify(format!("{} {}{note}", t("Opening", "正在打开"), path.display()), cx)
            }
            Err(e) => s.notify(format!("{}: {e}", t("Cannot open", "无法打开")), cx),
        })
    }

    fn register_actions<E: InteractiveElement>(&self, el: E, cx: &mut Context<Self>) -> E {
        el.on_action(cx.listener(|_, _: &Quit, _, cx| cx.quit()))
            .on_action(cx.listener(|this, _: &Import, window, cx| super::import::open(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &OpenFolder, window, cx| this.open_in_new_window(true, window, cx)))
            .on_action(cx.listener(|this, _: &OpenCatalog, window, cx| this.open_in_new_window(false, window, cx)))
            .on_action(cx.listener(|this, a: &OpenRecent, window, cx| {
                let Some(path) = this.state.read(cx).settings.recent.get(a.0).cloned() else { return };
                if this.state.read(cx).source == Source::Demo {
                    super::welcome::open_path(this.state.clone(), path, window.window_handle(), cx);
                } else {
                    this.spawn_window(path, false, cx);
                }
            }))
            .on_action(cx.listener(|this, _: &NewSession, window, cx| super::files::new_document(&this.state, crate::state::DocumentKind::Session, window, cx)))
            .on_action(cx.listener(|this, _: &NewCatalog, window, cx| super::files::new_document(&this.state, crate::state::DocumentKind::Catalog, window, cx)))
            .on_action(cx.listener(|this, _: &SyncXmp, _, cx| {
                this.with_state(cx, |s, cx| s.notify(crate::contract::SESSIONS.note().unwrap_or_default(), cx))
            }))
            .on_action(cx.listener(|this, _: &Undo, _, cx| this.with_state(cx, |s, cx| s.undo(cx))))
            .on_action(cx.listener(|this, _: &Redo, _, cx| this.with_state(cx, |s, cx| s.redo(cx))))
            .on_action(cx.listener(|this, _: &CopyAdjustments, _, cx| this.with_state(cx, |s, cx| s.copy_adjustments(cx))))
            .on_action(cx.listener(|this, _: &ApplyAdjustments, _, cx| this.with_state(cx, |s, cx| s.apply_adjustments(cx))))
            .on_action(cx.listener(|this, _: &ResetAdjustments, _, cx| this.with_state(cx, |s, cx| s.reset_all(cx))))
            .on_action(cx.listener(|this, _: &AutoAdjust, _, cx| this.with_state(cx, |s, cx| s.auto_adjust(AutoTarget::All, cx))))
            .on_action(cx.listener(|this, _: &NewVariant, _, cx| this.with_state(cx, |s, cx| s.new_variant(true, cx))))
            .on_action(cx.listener(|this, _: &AutoRotate, _, cx| this.with_state(cx, |s, cx| s.auto_rotate_selection(cx))))
            .on_action(cx.listener(|this, _: &MergeHdr, window, cx| super::merge::open(&this.state, crate::state::MergeKind::Hdr, window, cx)))
            .on_action(cx.listener(|this, _: &MergePanorama, window, cx| super::merge::open(&this.state, crate::state::MergeKind::Panorama, window, cx)))
            .on_action(cx.listener(|this, _: &CheckClosedEyes, _, cx| this.with_state(cx, |s, cx| s.check_closed_eyes(cx))))
            .on_action(cx.listener(|this, _: &MatchLook, _, cx| this.with_state(cx, |s, cx| s.match_look_selection(cx))))
            .on_action(cx.listener(|this, _: &CloneVariant, _, cx| this.with_state(cx, |s, cx| s.new_variant(false, cx))))
            .on_action(cx.listener(|this, _: &ToggleOverlay, window, cx| {
                if this.state.read(cx).settings.overlay_image.is_none() {
                    window.dispatch_action(Box::new(ChooseOverlay), cx);
                } else {
                    this.with_state(cx, |s, cx| {
                        s.view.overlay = !s.view.overlay;
                        cx.notify();
                    })
                }
            }))
            .on_action(cx.listener(|this, _: &ChooseOverlay, _, cx| {
                let rx = cx.prompt_for_paths(PathPromptOptions {
                    files: true,
                    directories: false,
                    multiple: false,
                    prompt: Some(t("Use as Overlay", "用作叠加图").into()),
                });
                let state = this.state.clone();
                cx.spawn(async move |_, cx| {
                    let Ok(Ok(Some(paths))) = rx.await else { return };
                    let Some(path) = paths.into_iter().next() else { return };
                    cx.update(|cx| {
                        state.update(cx, |s, cx| {
                            s.settings.overlay_image = Some(path);
                            s.settings.save();
                            s.view.overlay = true;
                            cx.notify();
                        })
                    });
                })
                .detach();
            }))
            .on_action(cx.listener(|this, _: &SecondViewer, _, cx| super::second_viewer::open(&this.state, cx)))
            .on_action(cx.listener(|this, _: &Slideshow, window, cx| this.state.update(cx, |s, cx| s.start_slideshow(window, cx))))
            .on_action(cx.listener(|this, _: &CullMode, window, cx| this.state.update(cx, |s, cx| s.start_cull(window, cx))))
            .on_action(cx.listener(|this, _: &StackSelected, _, cx| this.with_state(cx, |s, cx| s.stack_selection(cx))))
            .on_action(cx.listener(|this, _: &UnstackSelected, _, cx| this.with_state(cx, |s, cx| s.unstack_selection(cx))))
            .on_action(cx.listener(|this, _: &AutoStack, _, cx| this.with_state(cx, |s, cx| s.auto_stack(2, cx))))
            .on_action(cx.listener(|this, _: &ToggleStacksOpen, _, cx| this.with_state(cx, |s, cx| s.toggle_stacks_open(cx))))
            .on_action(cx.listener(|this, _: &SaveUserStyle, window, cx| crate::ui::naming::save_style(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &AddSerialNode, _, cx| this.with_state(cx, |s, cx| s.add_node(NewNode::Serial, cx))))
            .on_action(cx.listener(|this, _: &AddParallelNode, _, cx| this.with_state(cx, |s, cx| s.add_node(NewNode::Parallel, cx))))
            .on_action(cx.listener(|this, _: &AddLayerNode, _, cx| this.with_state(cx, |s, cx| s.add_node(NewNode::Layer, cx))))
            .on_action(cx.listener(|this, _: &AddOutsideNode, _, cx| this.with_state(cx, |s, cx| s.add_node(NewNode::Outside, cx))))
            .on_action(cx.listener(|this, _: &ToggleNode, _, cx| {
                this.with_state(cx, |s, cx| {
                    if let Some(id) = s.active_layer {
                        s.toggle_node(id, cx)
                    }
                })
            }))
            .on_action(cx.listener(|this, _: &DeleteNode, _, cx| this.with_state(cx, |s, cx| s.delete_node(s.selected_node(), cx))))
            .on_action(cx.listener(|this, _: &TidyNodes, _, cx| this.with_state(cx, |s, cx| s.tidy_nodes(cx))))
            .on_action(cx.listener(|this, _: &ToggleNodeDock, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.node_dock = !s.view.node_dock;
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &SelectAll, _, cx| this.with_state(cx, |s, cx| s.select_all(cx))))
            .on_action(cx.listener(|this, _: &SelectCurrentOnly, _, cx| {
                // Collapses the multiple view back to the current image.
                this.with_state(cx, |s, cx| {
                    if let Some(id) = s.current
                        && s.selection.len() > 1
                    {
                        s.select(id, false, false, cx);
                    }
                })
            }))
            .on_action(cx.listener(|this, _: &NextImage, _, cx| this.with_state(cx, |s, cx| s.step(1, cx))))
            .on_action(cx.listener(|this, _: &PrevImage, _, cx| this.with_state(cx, |s, cx| s.step(-1, cx))))
            .on_action(cx.listener(|this, _: &ZoomFit, _, cx| this.with_state(cx, |s, cx| s.set_zoom(None, None, cx))))
            .on_action(cx.listener(|this, _: &Zoom100, _, cx| {
                this.with_state(cx, |s, cx| {
                    let sf = s.scale_factor;
                    let c = s.zoom.center;
                    s.set_zoom(Some(sf), Some(c), cx)
                })
            }))
            .on_action(cx.listener(|this, _: &ZoomIn, _, cx| {
                this.with_state(cx, |s, cx| {
                    let next = s.effective_scale() * 1.5;
                    let c = s.zoom.center;
                    s.set_zoom(Some(next), Some(c), cx)
                })
            }))
            .on_action(cx.listener(|this, _: &ZoomOut, _, cx| {
                this.with_state(cx, |s, cx| {
                    let next = s.effective_scale() / 1.5;
                    if next <= s.fit_scale() {
                        s.set_zoom(None, None, cx)
                    } else {
                        let c = s.zoom.center;
                        s.set_zoom(Some(next), Some(c), cx)
                    }
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleBrowser, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.browser = if s.view.browser == BrowserMode::Hidden { BrowserMode::Filmstrip } else { BrowserMode::Hidden };
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleGridView, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.browser = if s.view.browser == BrowserMode::Grid { BrowserMode::Filmstrip } else { BrowserMode::Grid };
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleInspector, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.show_inspector = !s.view.show_inspector;
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleCompare, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.compare = if s.view.compare == CompareMode::Off { CompareMode::Split } else { CompareMode::Off };
                    s.preview.before_dirty = true;
                    s.request_render(RenderQuality::Final, cx);
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleExposureWarning, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.exposure_warning = !s.view.exposure_warning;
                    s.request_render(RenderQuality::Final, cx);
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleSoftProof, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.soft_proof = !s.view.soft_proof;
                    s.request_render(lumen_engine::RenderQuality::Final, cx);
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleProof, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.proof = !s.view.proof;
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleGrid, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.grid = !s.view.grid;
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|_, _: &ToggleFullscreen, window, _| window.toggle_fullscreen()))
            .on_action(cx.listener(|this, _: &ToggleMaskOverlay, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.panels.mask_overlay = !s.panels.mask_overlay;
                    s.request_render(RenderQuality::Final, cx);
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &RotateLeft, _, cx| {
                this.with_state(cx, |s, cx| s.edit_selected(t("Rotate left", "向左旋转"), |a| a.geometry.orientation = (a.geometry.orientation + 3) % 4, cx))
            }))
            .on_action(cx.listener(|this, _: &RotateRight, _, cx| {
                this.with_state(cx, |s, cx| s.edit_selected(t("Rotate right", "向右旋转"), |a| a.geometry.orientation = (a.geometry.orientation + 1) % 4, cx))
            }))
            .on_action(cx.listener(|this, _: &FlipHorizontal, _, cx| {
                this.with_state(cx, |s, cx| s.edit_selected(t("Flip", "翻转"), |a| a.geometry.flip_horizontal = !a.geometry.flip_horizontal, cx))
            }))
            .on_action(cx.listener(|this, _: &NewLayer, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.add_layer(lumen_engine::LayerKind::Adjustment, None, None, cx);
                    s.tool_tab = ToolTab::Layers;
                })
            }))
            .on_action(cx.listener(|this, a: &SetRating, _, cx| {
                let rating = a.0;
                this.with_state(cx, |s, cx| {
                    s.set_rating(rating, cx);
                    // Say what happened and to how many, where the eyes are.
                    let n = s.selection.len();
                    let what = match rating {
                        r if r < 0 => t("Rejected", "已拒绝").to_string(),
                        0 => t("Rating cleared", "已清除评级").to_string(),
                        r => "★".repeat(r as usize),
                    };
                    let note = if n > 1 { format!("{what} · {n} {}", t("images", "张图像")) } else { what };
                    if n > 0 {
                        s.notify(note, cx);
                    }
                })
            }))
            .on_action(cx.listener(|this, a: &SetColorTag, _, cx| {
                let tag = ColorTag::ALL[(a.0 as usize).min(7)];
                this.with_state(cx, |s, cx| {
                    s.set_color_tag(tag, cx);
                    let n = s.selection.len();
                    let what = if tag == ColorTag::None {
                        t("Color tag cleared", "已清除色标").to_string()
                    } else {
                        format!("{} · {}", t("Color tag", "色标"), super::collections::tag_name(tag))
                    };
                    let note = if n > 1 { format!("{what} · {n} {}", t("images", "张图像")) } else { what };
                    if n > 0 {
                        s.notify(note, cx);
                    }
                })
            }))
            .on_action(cx.listener(|this, a: &SelectToolTab, _, cx| {
                this.with_state(cx, |s, cx| {
                    if let Some(tab) = ToolTab::ALL.get(a.0) {
                        s.tool_tab = *tab;
                        cx.notify();
                    }
                })
            }))
            .on_action(cx.listener(|this, a: &SelectCursor, _, cx| {
                this.with_state(cx, |s, cx| {
                    if let Some(tool) = toolbar::CURSOR_TOOLS.get(a.0 as usize) {
                        s.set_cursor(*tool, cx)
                    }
                })
            }))
            .on_action(cx.listener(|this, _: &ExportSelected, _, cx| this.with_state(cx, |s, cx| s.process_selection(cx))))
            .on_action(cx.listener(|this, _: &Capture, _, cx| {
                this.with_state(cx, |s, cx| {
                    if s.camera_ready() {
                        s.capture(cx)
                    } else {
                        s.tool_tab = ToolTab::Capture;
                        let note = s.tether_note();
                        s.notify(note, cx);
                    }
                })
            }))
            .on_action(cx.listener(|this, _: &LiveView, _, cx| {
                if this.state.read(cx).camera_ready() {
                    super::live_view::open(&this.state, cx)
                } else {
                    this.with_state(cx, |s, cx| {
                        let note = s.tether_note();
                        s.notify(note, cx)
                    });
                }
            }))
            .on_action(cx.listener(|this, _: &MoveToTrash, _, cx| this.with_state(cx, |s, cx| s.toggle_trash_selection(cx))))
            .on_action(cx.listener(|this, _: &ClearAnnotations, _, cx| {
                this.with_state(cx, |s, cx| {
                    if let Some(id) = s.current {
                        s.annotations.remove(&id);
                    }
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &OpenExportDialog, window, cx| super::export::open(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &ToggleMultiView, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.multi = !s.view.multi;
                    if s.view.multi && s.multi_images().is_empty() {
                        s.notify(t("Select several images to see them side by side", "选择多张图像即可并排查看"), cx);
                    }
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &SetCompareVariant, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.compare_variant = s.current;
                    s.view.multi = true;
                    s.notify(t("Set as compare variant", "已设为对比变体"), cx);
                })
            }))
            .on_action(cx.listener(|this, _: &ClearCompareVariant, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.compare_variant = None;
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, _: &ToggleInspectorSide, _, cx| {
                this.with_state(cx, |s, cx| {
                    s.view.inspector_right = !s.view.inspector_right;
                    s.save_settings();
                    cx.notify();
                })
            }))
            .on_action(cx.listener(|this, a: &SetWorkspace, _, cx| {
                let i = a.0;
                this.with_state(cx, |s, cx| s.apply_workspace(i, cx))
            }))
            .on_action(cx.listener(|this, _: &SaveWorkspace, window, cx| super::workspaces::save_dialog(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &Print, window, cx| super::print::open(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &ShowShortcuts, window, cx| super::prefs::open_shortcuts(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &ShowAbout, window, cx| dialogs::about(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &ShowPreferences, window, cx| dialogs::preferences(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &CommandPalette, window, cx| dialogs::command_palette(&this.state, window, cx)))
            .on_action(cx.listener(|this, _: &ToggleLanguage, window, cx| {
                i18n::set_locale(if i18n::locale() == i18n::Locale::ZhCn { i18n::Locale::En } else { i18n::Locale::ZhCn });
                let names = Self::workspace_names(&this.state, cx);
                install_menus(&this.menu_bar, &names, &Self::recent_names(&this.state, cx), cx);
                window.refresh();
            }))
    }

    fn title_bar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let title = if s.welcome {
            String::new()
        } else {
            let file = s.current_info().map(|i| i.file_name.clone()).unwrap_or_default();
            if file.is_empty() { s.source.title() } else { format!("{} · {file}", s.source.title()) }
        };
        let camera = s.camera_ready();
        TitleBar::new()
            .bg(p.chrome)
            .border_color(p.divider)
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        div()
                            .ml_1()
                            .size(px(18.0))
                            .rounded(px(5.0))
                            .bg(gpui_kit::linear_gradient(
                                135.0,
                                gpui_kit::linear_color_stop(p.accent, 0.0),
                                gpui_kit::linear_color_stop(hexa(0xd05f9a, 1.0), 1.0),
                            ))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(Icon::new(IconName::Aperture).size(px(12.0)).text_color(gpui_kit::white())),
                    )
                    .child(self.menu_bar.clone()),
            )
            .child(
                h_flex()
                    .flex_1()
                    .justify_center()
                    .gap_2()
                    .text_xs()
                    .text_color(p.text_muted)
                    .child("Aurora Studio")
                    .when(!title.is_empty(), |d| d.child(div().text_color(p.text_subtle).child("—")).child(div().text_color(p.text).child(title))),
            )
            .child(
                h_flex()
                    .gap_2()
                    .pr_2()
                    .when(camera, |d| {
                        d.child(
                            h_flex()
                                .gap_1()
                                .px_2()
                                .h(px(20.0))
                                .rounded(px(10.0))
                                .bg(p.well)
                                .text_xs()
                                .text_color(p.success)
                                .child(div().size(px(6.0)).rounded_full().bg(p.success))
                                .child(t("Tethered", "联机拍摄")),
                        )
                    }),
            )
    }
}

impl Render for Workspace {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let view = s.view.clone();
        let grid = view.browser == BrowserMode::Grid;
        let film = matches!(view.browser, BrowserMode::Filmstrip | BrowserMode::List);
        let list = view.browser == BrowserMode::List;
        let inspector_target = if view.show_inspector { view.inspector_width } else { 0.0 };
        let film_target = if film { if list { view.filmstrip_height.max(220.0) } else { view.filmstrip_height } } else { 0.0 };
        let notice = s.notice.clone();
        let welcome = s.welcome;
        let culling = s.cull.is_some();
        let showing = s.slideshow.is_some() || culling;

        let insp_w = motion::value(motion::key("ws", "inspector-w"), inspector_target, motion::PANEL, window, cx);
        let film_h = motion::value(motion::key("ws", "film-h"), film_target, motion::PANEL, window, cx);
        let grid_t = motion::toggle(motion::key("ws", "grid"), grid, motion::PANEL, window, cx);
        // The node editor docked wide under the viewer, as on Resolve's Color page.
        let dock_t = motion::toggle(motion::key("ws", "node-dock"), view.node_dock, motion::PANEL, window, cx);
        let node_dock = (dock_t > 0.01).then(|| {
            v_flex()
                .h(px(NODE_DOCK_H * dock_t))
                .flex_shrink_0()
                .overflow_hidden()
                .border_t_1()
                .border_color(p.divider)
                .bg(p.panel)
                .pt_2()
                .child(crate::tools::nodes::toolbar(&self.state, true, cx))
                .child(div().px_2().child(crate::tools::nodes::editor(&self.state, "node-dock", NODE_DOCK_H - 52.0, window, cx)))
        });

        let state = self.state.clone();
        let inspector_panel = (insp_w > 1.0).then(|| {
            h_flex()
                .h_full()
                .w(px(insp_w))
                .flex_shrink_0()
                .overflow_hidden()
                .child(
                    h_flex()
                        .h_full()
                        .w(px(view.inspector_width))
                        .flex_shrink_0()
                        .when(!view.inspector_right, |d| d.child(inspector::tool_strip(&state, window, cx)))
                        .child(div().flex_1().h_full().min_w_0().child(inspector::inspector(&state, window, cx)))
                        .when(view.inspector_right, |d| d.child(inspector::tool_strip(&state, window, cx))),
                )
        });

        let st_w = self.state.clone();
        let st_h = self.state.clone();
        let inspector_handle = Surface::new("inspector-resize")
            .w(px(5.0))
            .h_full()
            .flex_shrink_0()
            .cursor_ew_resize()
            .bg(p.chrome)
            .hover(|d| d.bg(p.accent.opacity(0.5)))
            .child(div().absolute().top_0().bottom_0().when(view.inspector_right, |d| d.right_0()).when(!view.inspector_right, |d| d.left_0()).w(px(1.0)).bg(p.divider))
            .on_drag(move |ptr, window, cx| {
                let total = f32::from(window.viewport_size().width);
                st_w.update(cx, |s, cx| {
                    let x = f32::from(ptr.point.x);
                    let w = if s.view.inspector_right { total - x } else { x };
                    s.view.inspector_width = w.clamp(260.0, 640.0);
                    cx.notify();
                })
            });
        let film_handle = Surface::new("film-resize")
            .h(px(5.0))
            .w_full()
            .flex_shrink_0()
            .cursor_ns_resize()
            .bg(p.chrome)
            .hover(|d| d.bg(p.accent.opacity(0.5)))
            .child(div().absolute().left_0().right_0().bottom_0().h(px(1.0)).bg(p.divider))
            .on_drag(move |ptr, window, cx| {
                let total = f32::from(window.viewport_size().height);
                st_h.update(cx, |s, cx| {
                    let h = total - f32::from(ptr.point.y) - metrics::STATUS_BAR;
                    s.view.filmstrip_height = h.clamp(96.0, 420.0);
                    cx.notify();
                })
            });

        let center = v_flex()
            .flex_1()
            .h_full()
            .min_w_0()
            .relative()
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .child(
                        div()
                            .absolute()
                            .size_full()
                            // Clicking the image leaves any text field, so the
                            // keyboard goes back to shortcuts and Speed Edit.
                            .capture_any_mouse_down(cx.listener(|this, _, window, cx| window.focus(&this.focus, cx)))
                            .child(self.viewer.clone()),
                    )
                    .when(grid_t > 0.01, |d| {
                        d.child(
                            div()
                                .absolute()
                                .size_full()
                                .opacity(grid_t)
                                .mt(px((1.0 - grid_t) * 18.0))
                                .child(self.browser.clone()),
                        )
                    })
                    .when_some(notice.clone().filter(|_| self.notice_serial > 0 && !self.notice_carded), |d, n| d.child(notice_hud(n, self.notice_serial, &p))),
            )
            .children(node_dock)
            .when(film_h > 1.0 && !grid, |d| {
                d.child(film_handle).child(div().h(px(film_h)).flex_shrink_0().overflow_hidden().child(self.browser.clone()))
            });

        let root = v_flex()
            .id("workspace")
            .size_full()
            .bg(p.chrome)
            .text_color(p.text)
            .key_context(CONTEXT)
            .track_focus(&self.focus)
            .capture_any_mouse_up({
                let state = self.state.clone();
                move |_, _, cx| {
                    if crate::perf::enabled() {
                        cx.stop_propagation();
                    } else {
                        state.update(cx, |s, cx| s.end_gesture(cx));
                    }
                }
            })
            // Speed Edit ends when its key is released (armed by the keystroke
            // interceptor in `new`, since bindings see key-down first).
            .capture_key_up({
                let state = self.state.clone();
                move |e: &gpui_kit::KeyUpEvent, _, cx| {
                    let held = state.read(cx).speed.as_ref().map(|s| s.param.key());
                    if held == Some(e.keystroke.key.as_str()) {
                        state.update(cx, |s, cx| s.speed_end(cx));
                    }
                }
            })
            .when(showing && !culling, |d| d.child(super::slideshow::render(&self.state, &self.viewer, window, cx)))
            .when(culling, |d| d.child(super::cull::render(&self.state, &self.viewer, window, cx)))
            .when(!showing, |d| d.child(self.title_bar(cx)))
            .map(|d| {
                if showing {
                    d
                } else if welcome {
                    d.child(div().flex_1().min_h_0().child(super::welcome::render(&self.state, window, cx)))
                } else {
                    d.child(toolbar::toolbar(&self.state, window, cx)).child(
                        motion::enter(
                            "workspace-main",
                            h_flex().flex_1().min_h_0().map(|d| match (view.inspector_right, inspector_panel) {
                                (true, Some(panel)) => d.child(center).child(inspector_handle).child(panel),
                                (false, Some(panel)) => d.child(panel).child(inspector_handle).child(center),
                                (_, None) => d.child(center),
                            }),
                            6.0,
                        ),
                    )
                }
            })
            .when(!showing, |d| d.child(statusbar::status_bar(&self.state, notice, self.notice_serial, window, cx)));
        self.register_actions(root, cx)
    }
}

pub fn open(state: Entity<AppState>, cx: &mut gpui_kit::App) {
    let mut options = gpui_kit::WindowOptions {
        window_bounds: Some(gpui_kit::WindowBounds::centered(gpui_kit::size(px(1600.0), px(1000.0)), cx)),
        window_min_size: Some(gpui_kit::size(px(1100.0), px(700.0))),
        ..TitleBar::window_options()
    };
    if crate::perf::enabled() {
        // Callback replay must not inherit the inactive-window frame cap.
        options.inactive_frame_interval = None;
    }
    let replay_state = state.clone();
    let (window, _) = gpui_kit::open_window(options, cx, move |window, cx| {
        window.set_window_title("Aurora Studio");
        if crate::perf::enabled() {
            window.activate_window();
        }
        cx.new(|cx| Workspace::new(state, window, cx))
    })
    .expect("failed to open the main window");
    crate::perf::replay(replay_state, window, cx);
}
